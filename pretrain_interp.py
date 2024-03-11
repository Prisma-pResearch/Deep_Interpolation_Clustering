#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
interp_v6.py: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger
from interpolation_layer import SingleChannelInterp, CrossChannelInterp
from rbf import RBF, basis_func_dict

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        output, (hidden, cell_state) = self.lstm(x)
        return output, hidden, cell_state

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x, hidden, context):  # LSTM
        x = F.relu(x)
        x, (hidden, cell_state) = self.lstm(x, (hidden, context))
        return x, (hidden, cell_state)

class AuxFc(nn.Module):
    # Build it as linear classifier to reduce its complexity, and increase the hidden representation.
    def __init__(self, idim, odim, dropout):
        super(AuxFc, self).__init__()
        nhidden = 128
        self.model = nn.Sequential(
            nn.Linear(idim, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.Dropout(dropout),
            nn.Linear(nhidden, odim)
        )

    def forward(self, x):
        return self.model(x)    # (B, #aux_task)

class FuturePredFc(nn.Module):
    # Build it as linear classifier to reduce its complexity, and increase the hidden representation.
    def __init__(self, idim, odim, dropout):
        super(FuturePredFc, self).__init__()
        nhidden = 128
        self.model = nn.Sequential(
            nn.Linear(idim, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.Dropout(dropout),
            nn.Linear(nhidden, odim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)    # (B, #aux_task)

class FakeDetFc(nn.Module):
    def __init__(self, idim, odim, dropout):
        super(FakeDetFc, self).__init__()
        nhidden = 128
        self.model = nn.Sequential(
            nn.Linear(idim, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.Dropout(dropout),
            nn.Linear(nhidden, odim),
            nn.LogSoftmax(dim=1)     # Add logSoftmax, then use NLLLoss, otherwise CrossEntropyLoss without LogSoftmax
        )

    def forward(self, x):
        return self.model(x)    # (B, #aux_task)


class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()
        self.device = device
        self.args = args
        self.num_variables = args.num_variables
        self.num_timestamps = args.num_timestamps
        self.nhidden = 128
        self.nlstm = 1
        self.bidirectional = True
        self.dim_enc_hidden = self.nlstm * self.nhidden * 2 if self.bidirectional else self.nlstm * self.nhidden
        self.dim_dec_out = self.nhidden * 2 if self.bidirectional else self.nhidden     # Out last layer

        self.sci = SingleChannelInterp(args.ref_points, args.hours_from_admission, self.num_variables,
                                       self.num_timestamps, self.device)

        self.cci = CrossChannelInterp(self.num_variables, self.num_timestamps, self.device)

        self.encoder = EncoderRNN(3*self.num_variables, self.nhidden, num_layers=self.nlstm, dropout=0,
                                  bidirectional=self.bidirectional, device=self.device)
        self.decoder = DecoderRNN(input_size=self.nhidden * 2, hidden_size=self.nhidden,
                                  num_layers=self.nlstm, dropout=0,
                                  bidirectional=self.bidirectional, device=self.device)

        self.rbf = RBF(hours_look_ahead=args.hours_from_admission, ref_points=args.ref_points, 
                       in_dim=self.dim_dec_out, out_dim=self.num_variables,
                       dropout=args.dropout, basis_func=basis_func_dict()['gaussian'], device=self.device)

        num_aux_tasks = len(self.args.aux_tasks)
        if 'future_vital' in self.args.aux_tasks:
            self.predict_future = FuturePredFc(self.dim_enc_hidden, args.num_variables, args.dropout)
            num_aux_tasks -= 1 
            
        if num_aux_tasks > 0:
            self.aux_head = AuxFc(self.dim_enc_hidden, num_aux_tasks, args.dropout)
            
        if self.args.fake_detection:
            self.fake_det_head = FakeDetFc(self.dim_enc_hidden, 2, args.dropout)

    ###return hidden_embedding, reconstructed vital, and aux_predict label(multi_task, future_vital, and fake_detection)
    def forward(self, x, fake_x=None, fake_perm_idx=None, positive_x=None):
        interp_x = self.sci(x)     # (batch, time_step, channel), channel = num_variable * 3
        interp_x = self.cci(interp_x)       # (batch, time_step, channel)
        interp_x = interp_x.permute(1, 0, 2)  # (time_step, batchsize, channel), as lstm requires time_step first

        ###encoder_output: time_step, batch_size, hidden_size * (2 if bidrection else 1), last lstm layer output
        ###hidden: num_layers *  (2 if bidrection else 1), batch_size, hidden_size, last time stamp hidden output
        ###cell_state: num_layers *  (2 if bidrection else 1), batch_size, hidden_size, last time stamp cell status output
        encoder_output, hidden, cell_state = self.encoder(interp_x)  # LSTM
        cat_hidden = torch.cat([x for x in hidden], dim=-1)
        y, _ = self.decoder(encoder_output, hidden, cell_state)

        y = y.permute(1, 2, 0)  # (b, hidden, ref_t)
        y = self.rbf(y, x)       # (b, num_variables, all_timetamps)

        num_aux_tasks = len(self.args.aux_tasks)
        aux_pred_dict = dict()
        if 'future_vital' in self.args.aux_tasks:
            aux_pred_dict['future_vital'] = self.predict_future(cat_hidden)
            num_aux_tasks -= 1 
        
        if num_aux_tasks > 0:
            aux_pred = self.aux_head(cat_hidden)
            all_tasks = [task for task in self.args.aux_tasks.keys() if task != 'future_vital']
            for i, task in enumerate(all_tasks):
                aux_pred_dict[task] = aux_pred[:, i]

        if self.args.fake_detection:
            fake_interp_x = self.sci(fake_x)  # (batch, time_step, channel)
            fake_interp_x = self.cci(fake_interp_x)  # (batch, time_step, channel)

            fake_interp_x = fake_interp_x.permute(1, 0, 2)  # (time_step, batchsize, channel)
            fake_context, fake_hidden, fake_cell_state = self.encoder(fake_interp_x)  # LSTM
            fake_cat_hidden = torch.cat([x for x in fake_hidden], dim=-1)
            pos_neg_feat = torch.cat([cat_hidden, fake_cat_hidden], dim=0)[fake_perm_idx]
            fake_det_pred = self.fake_det_head(pos_neg_feat)
            aux_pred_dict['fake_det'] = fake_det_pred
        return cat_hidden, y, aux_pred_dict

    def rec_loss(self, org_ob, rec_ob, padding_mask):
        rec_true = org_ob * padding_mask
        rec_pred = rec_ob * padding_mask
        num_rec = (padding_mask == 1.0).sum()
        mse = F.mse_loss(rec_pred, rec_true, reduction='sum')
        mse = mse / num_rec
        return {'loss': mse, 'ae_mse': mse}

    def sup_aux_loss(self, aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask=None):
        aux_loss_dict = dict()
        
        if 'future_vital' in aux_tasks:
            aux_true = aux_label_dict['future_vital'] * future_vital_mask 
            aux_pred = aux_pred_dict['future_vital'] * future_vital_mask  
            num_rec = (future_vital_mask == 1.0).sum() 
            mse = F.mse_loss(aux_pred, aux_true, reduction='sum')
            mse = mse / num_rec  
            aux_loss_dict['future_vital'] = mse 
        
        for aux_task in aux_tasks:
            if aux_task == 'future_vital':
                continue
            aux_true = aux_label_dict[aux_task]
            aux_pred = aux_pred_dict[aux_task]
            pos_weight = torch.tensor(self.args.aux_pos_weights[aux_task]).to(self.device)
            aux_loss = F.binary_cross_entropy_with_logits(aux_pred, aux_true, pos_weight=pos_weight)
            aux_loss_dict[aux_task] = aux_loss
        return aux_loss_dict

    def fake_det_loss(self, label, pred):
        nll = F.nll_loss(pred, label, reduction='mean')
        return {'fake_detection': nll}

    def triplet_loss(self):
        return None

    # Combine all the loss terms with weights
    def multi_task_loss(self, aux_tasks, rec_loss_dict, aux_loss_dict):
        ae_mse = rec_loss_dict['ae_mse']
        loss = ae_mse
        for loss_name, loss_value in aux_loss_dict.items():
            loss = loss + aux_tasks[loss_name] * loss_value
            logger.debug('Aux loss {}:{}, w:{}'.format(loss_name, loss_value, aux_tasks[loss_name]))
        rec_loss_dict['loss'] = loss
        rec_loss_dict.update(aux_loss_dict)

        return rec_loss_dict

if __name__ == '__main__':
    from p1_pretrain_main import get_arguments
    args = get_arguments()
    args.fake_detection=False
    args.aux_tasks = {}
    device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
    net = Net(args, device=device).to(device)
    logger.info(net)

    feat = torch.randn(args.batch_size, args.num_variables, args.num_timestamps)
    mask = torch.randint(0, 2, size=(args.batch_size, args.num_variables, args.num_timestamps), dtype=torch.float32)
    timestamp = args.ref_points * torch.rand((args.batch_size, args.num_variables, args.num_timestamps))
    hold_out = torch.randint(0, 2, size=(args.batch_size, args.num_variables, args.num_timestamps), dtype=torch.float32)
    x = torch.cat([feat, mask, timestamp, hold_out], dim=1).to(device)

    hidden, rec, aux_pred = net(x)
    logger.info('rec.size: {}'.format(rec.size()))