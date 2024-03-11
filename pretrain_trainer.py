#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created by yanjun.li at 12/3/19
"""
from utils import logger, create_weight_dir, pytorch_optimizer, pytorch_lr_scheduler, create_flag_dict 
from utils import Summary, timer, format_metric_dict, reduce_lr_on_plateau, save_model_update_flag, early_stop
from info import COHORT2SCOPE, MIN_MAX_VALUES, METRICS, MIN_METRICS, MAX_METRICS,  SUMMARY_ITEMS
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import defaultdict
from copy import deepcopy
import os
import numpy as np

class Trainer(object):
    def __init__(self, args, model, dl_dict, exp_path, device, **kwags):
        self.args = args
        self.device = device
        self.model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus)) if args.num_gpus > 0 else model
        self.model.to(self.device)
        
        ###get loss function pointer of model
        self.rec_loss_f = model.rec_loss
        self.sup_aux_loss_f = model.sup_aux_loss
        self.fake_det_loss_f = model.fake_det_loss
        self.triplet_loss_f = model.triplet_loss
        self.multi_task_loss_f = model.multi_task_loss

        self.exp_path = exp_path
        ###directory to save the weight of model
        self.weight_path = os.path.join(self.exp_path, "weight")
        ##create the dictionary of directory of weights (loss, delta, ae_mse)
        self.weight_path_dict = create_weight_dir(self.weight_path, METRICS)
        self.summary_path = os.path.join(self.exp_path, "summary")
        os.makedirs(self.weight_path, exist_ok=True)

        ###create the folder to save the hidden features of cohorts
        self.out_feat_path = os.path.join(self.exp_path, 'out_feat', self.args.restore_metric)
        os.makedirs(self.out_feat_path, exist_ok=True)

        self.dl_dict = dl_dict
        self.train_dl = self.dl_dict['training']
        self.valid_dl = self.dl_dict['validation']
        self.test_dl = self.dl_dict['testing']

        self.epoch = 1
        logger.info(self.model)

        # Optimizer
        # Note: construct optimizer after moving model to cuda
        self.optimizer = pytorch_optimizer(self.model, args.optimizer, args.init_lr, args.weight_decay_rate)
        self.lr_scheduler = pytorch_lr_scheduler(self.optimizer, args.lr_decay_mode, args.lr_decay_step_or_patience,
                                                 args.lr_decay_rate)

        self.flag_dict = create_flag_dict(METRICS, MIN_METRICS, MAX_METRICS)
        ###create the summaryWriter used for tensorboard
        summary_writer = SummaryWriter(self.summary_path, filename_suffix=datetime.now().strftime('_%m-%d-%y_%H-%M-%S'))
        self.summary = Summary(summary_writer, METRICS, SUMMARY_ITEMS)

        torch.set_printoptions(precision=10)

    def train(self):
        logger.info('*******Building the model*******')

        if self.args.restore:
            self.load_weight()

        with timer('Duration of training'):
            for epoch in range(1, self.args.max_epochs):
                train_metrics_dict = self.train_one_epoch(self.train_dl, denoise=self.args.denoise)
                logger.info('==> Epoch: {}, Train, {}'.format(epoch, format_metric_dict(train_metrics_dict)))
                logger.info('==================')

                valid_metrics_dict, _ = self.eval_one_epoch('valid', self.valid_dl, denoise=self.args.denoise)
                logger.debug('{}'.format(format_metric_dict(valid_metrics_dict)))
                # test_data_dict = self.eval_one_epoch("test", self.test_data_loader)

                valid_aly_result_dict = self.aly_pred('valid', valid_metrics_dict)
                # _ = self.aly_pred("test", test_data_dict)
                logger.info('==================')

                self.epoch += 1
                if valid_aly_result_dict['early_stop']:
                    logger.info('========Best model=========')
                    logger.info('{}'.format(self.flag_dict))
                    break

    def eval(self, cohort, generate_feat=False, viz_feat=False, denoise=False):
        logger.info("*******Evaluating the model*******")
        self.load_weight()
        dl = self._get_dl(cohort)

        scope = COHORT2SCOPE[cohort]
        metrics_dict, ob_pred_lst = self.eval_one_epoch(scope, dl, denoise)
        logger.info("{}, {}".format(scope, format_metric_dict(metrics_dict)))
        ob_pred_dict = self.merge_ob_pred(ob_pred_lst)
        ob_pred_dict = self.re_norm_data(ob_pred_dict)      # Re-normalize to real range

        if generate_feat:
            folder_path = os.path.join(self.exp_path, 'out_feat', self.args.restore_metric)
            os.makedirs(folder_path, exist_ok=True)
            if self.args.evaluate_interpolation:
                np_f = os.path.join(folder_path, "{}_interp_eval.npy".format(cohort))
            else:
                np_f = os.path.join(folder_path, "{}.npy".format(cohort))            
            if os.path.exists(np_f):
                logger.info('No save, the npy file exists. {}'.format(np_f))
            else:
                logger.info("*******Generate the features for {}*******".format(cohort))
                np.save(np_f, ob_pred_dict)
                logger.info('The npy is saved to {}'.format(self.out_feat_path))

        if viz_feat:
            logger.info("*******Visualize the features for {}*******".format(cohort))
            self.summary.summary_writer.add_embedding(ob_pred_dict['hidden'], global_step=self.epoch, tag=cohort)

    def _get_dl(self, cohort):
        if cohort == "training":
            return self.train_dl
        elif cohort == "validation":
            return self.valid_dl
        elif cohort == "testing":
            return self.test_dl

    def train_one_epoch(self, dl, denoise=True):
        self.model.train()
        metrics_dict = defaultdict(list)
        for i_batch, (batch_sample, fake_batch_sample) in enumerate(dl, start=1):
            encounter_ids = batch_sample['encounter_id']
            ob = batch_sample['ob'].to(self.device)
            padding_mask = batch_sample['padding_mask'].to(self.device)
            timestamp = batch_sample['timestamp'].to(self.device)
            ae_mask = batch_sample['ae_mask'].to(self.device)
            ob = ob * padding_mask
            
            ###if denoise, the randomly dropped input will be used 
            if denoise:
                ae_masked_ob = ob * ae_mask
                stacked_input = torch.cat([ae_masked_ob, padding_mask, timestamp, ae_mask], dim=1)
            else:
                stacked_input = torch.cat([ob, padding_mask, timestamp, ae_mask], dim=1)
            # logger.debug(stacked_input.size())

            if self.args.fake_detection:
                fake_encounter_ids = fake_batch_sample['encounter_id']
                fake_ob = fake_batch_sample['ob'].to(self.device)
                fake_padding_mask = fake_batch_sample['padding_mask'].to(self.device)
                fake_timestamp = fake_batch_sample['timestamp'].to(self.device)
                fake_ae_mask = fake_batch_sample['ae_mask'].to(self.device)
                fake_ob = fake_ob * fake_padding_mask
                assert (fake_encounter_ids == encounter_ids), 'Encounter_id dose not match.'
                if denoise:
                    fake_ae_masked_ob = fake_ob * fake_ae_mask
                    fake_stacked_input = torch.cat([fake_ae_masked_ob, fake_padding_mask, fake_timestamp, fake_ae_mask], dim=1)
                else:
                    fake_stacked_input = torch.cat([fake_ob, fake_padding_mask, fake_timestamp, fake_ae_mask], dim=1)

                # generate the label for fake_detection, and shuffle the pos and neg samples
                pos_cls, neg_cls = torch.ones(ob.size()[0], device=self.device), \
                                   torch.zeros(fake_ob.size()[0], device=self.device)
                fake_det_label = torch.cat([pos_cls, neg_cls])
                fake_perm_idx = torch.randperm(fake_det_label.size()[0])    
                fake_det_label = fake_det_label[fake_perm_idx].to(torch.int64)
            else:
                fake_stacked_input, fake_perm_idx, fake_det_label = None, None, None

            if self.args.triple_margin != 0. and self.args.fake_detection:
                assert self.args.scale == 20, 'The noise gaussian config should be adjusted.'
                triple_pos_ob = self.add_gaussian_noise(ob, padding_mask, {'std': self.args.triple_pos_std})
                triple_pos_timestamp = self.add_gaussian_noise(timestamp, padding_mask, {'std': 0.01})
                triple_pos_stacked_input = torch.cat([triple_pos_ob, padding_mask, triple_pos_timestamp, ae_mask], dim=1)
            else:
                triple_pos_stacked_input = None

            # Fetch the labels of sup_aux_task
            aux_label_dict = dict()
            future_vital_mask = None
            if self.args.aux_tasks:
                if 'future_vital' in self.args.aux_tasks:
                    future_vital_mask = batch_sample['future_vital_mask'].to(self.device)
                
                for aux_task in self.args.aux_tasks.keys():
                    aux_label_dict[aux_task] = batch_sample[aux_task].to(self.device)
            elif self.args.aux_tasks:
                raise Exception('The aux_tasks setting is wrong.')

            self.optimizer.zero_grad()
            ###the model will output hidden feature, reconstructed observation, and predicted label for aux tasks
            hidden, rec_ob, aux_pred_dict = self.model(stacked_input, fake_stacked_input, fake_perm_idx, triple_pos_stacked_input)
            # logger.debug('ref_y: {}, rec_ob: {}'.format(ref_y.size(), rec_ob.size()))

            # Reconstruct all the input
            ##calculate loss
            rec_loss_dict = self.rec_loss_f(ob, rec_ob, padding_mask.to(self.device))
            if self.args.loss == 'ae_mse':
                loss_dict = rec_loss_dict
            elif self.args.loss == 'ae_mse_sup':
                aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)
                loss_dict = self.multi_task_loss_f(self.args.aux_tasks, rec_loss_dict, aux_loss_dict)
            elif self.args.loss == 'ae_mse_fake_detect':
                fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
            elif self.args.loss == 'ae_mse_fake_detect_triplet':
                fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                triplet_loss_dict = self.triplet_loss_f(hidden, aux_pred_dict['positive'], aux_pred_dict['negative'],
                                                       self.args.triple_margin)
                fake_det_loss_dict.update(triplet_loss_dict)
                loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
            elif self.args.loss == 'ae_mse_sup_fake_detect':
                aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)
                fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                
                tasks_dict = self.args.aux_tasks.copy()
                tasks_dict.update(self.args.unsup_aux_tasks)
                task_loss_dict = aux_loss_dict.copy()
                task_loss_dict.update(fake_det_loss_dict)
                loss_dict = self.multi_task_loss_f(tasks_dict, rec_loss_dict, task_loss_dict)
            else:
                raise NotImplementedError
            ###add current loss of each batch to metrics dict, format is as {'loss' : [1, 2, xx]}
            [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

            total_loss = loss_dict['loss']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            if i_batch % self.args.log_train_freq == 1:
                logger.info('{}-[{}/{} ({:.0f}%)]: train-{}'.format(
                    self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                    {k: v.item() for k, v in loss_dict.items()}))
                loss_dict['scope'] = 'train_batch'
                self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)
        
        ###as one epoch has many batches, so the loss for one epoch will be the average value
        train_metrics_dict = {'scope': 'train'}
        train_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})
        return train_metrics_dict

    def eval_one_epoch(self, scope, dl, denoise=True):
        self.model.eval()
        ob_pred_lst = []
        metrics_dict = defaultdict(list)
        with torch.no_grad():
            for i_batch, (batch_sample, fake_batch_sample) in enumerate(dl, start=1):
                encounter_ids = batch_sample['encounter_id']
                ob = batch_sample['ob'].to(self.device)
                padding_mask = batch_sample['padding_mask'].to(self.device)
                timestamp = batch_sample['timestamp'].to(self.device)
                ae_mask = batch_sample['ae_mask'].to(self.device)
                
                if self.args.evaluate_interpolation:
                    denoise = True
                
                ob = ob * padding_mask

                if denoise:
                    ae_masked_ob = ob * ae_mask
                    stacked_input = torch.cat([ae_masked_ob, padding_mask, timestamp, ae_mask], dim=1)
                else:
                    stacked_input = torch.cat([ob, padding_mask, timestamp, ae_mask], dim=1)

                if self.args.fake_detection:
                    fake_encounter_ids = fake_batch_sample['encounter_id']
                    fake_ob = fake_batch_sample['ob'].to(self.device)
                    fake_padding_mask = fake_batch_sample['padding_mask'].to(self.device)
                    fake_timestamp = fake_batch_sample['timestamp'].to(self.device)
                    fake_ae_mask = fake_batch_sample['ae_mask'].to(self.device)
                    fake_ob = fake_ob * padding_mask
                    assert (fake_encounter_ids == encounter_ids), 'Encounter_id dose not match.'

                    if denoise:
                        fake_ae_masked_ob = fake_ob * fake_ae_mask
                        fake_stacked_input = torch.cat(
                            [fake_ae_masked_ob, fake_padding_mask, fake_timestamp, fake_ae_mask], dim=1)
                    else:
                        fake_stacked_input = torch.cat([fake_ob, fake_padding_mask, fake_timestamp, fake_ae_mask],dim=1)

                    # generate the label for fake_detection
                    pos_cls, neg_cls = torch.ones(ob.size()[0], device=self.device), \
                                       torch.zeros(fake_ob.size()[0], device=self.device)
                    fake_det_label = torch.cat([pos_cls, neg_cls])
                    fake_perm_idx = torch.randperm(fake_det_label.size()[0])  # shuffle the pos and neg samples
                    fake_det_label = fake_det_label[fake_perm_idx].to(torch.int64)
                else:
                    fake_stacked_input, fake_perm_idx, fake_det_label = None, None, None

                if self.args.triple_margin != 0. and self.args.fake_detection:
                    assert self.args.scale == 20, 'The noise gaussian config should be adjusted.'
                    triple_pos_ob = self.add_gaussian_noise(ob, padding_mask, {'std': self.args.triple_pos_std})
                    triple_pos_timestamp = self.add_gaussian_noise(timestamp, padding_mask, {'std': 0.01})
                    triple_pos_stacked_input = torch.cat([triple_pos_ob, padding_mask, triple_pos_timestamp, ae_mask],
                                                         dim=1)
                else:
                    triple_pos_stacked_input = None

                # Fetch the labels of sup_aux_task
                aux_label_dict = dict()
                future_vital_mask = None
                if self.args.aux_tasks:
                    if 'future_vital' in self.args.aux_tasks:
                        future_vital_mask = batch_sample['future_vital_mask'].to(self.device)
                    
                    for aux_task in self.args.aux_tasks.keys():
                        aux_label_dict[aux_task] = batch_sample[aux_task].to(self.device)
                elif self.args.aux_tasks:
                    raise Exception('The aux_pred_dict setting is wrong.')

                hidden, rec_ob, aux_pred_dict = self.model(stacked_input, fake_stacked_input, fake_perm_idx, triple_pos_stacked_input)

                # Reconstruct all the input, based on the full ob, instead of ae_masked_ob
                rec_loss_dict = self.rec_loss_f(ob, rec_ob, padding_mask.to(self.device))
                if self.args.loss == 'ae_mse':
                    loss_dict = rec_loss_dict
                elif self.args.loss == 'ae_mse_sup':
                    aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)
                    loss_dict = self.multi_task_loss_f(self.args.aux_tasks, rec_loss_dict, aux_loss_dict)
                elif self.args.loss == 'ae_mse_fake_detect':
                    fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                    loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
                elif self.args.loss == 'ae_mse_fake_detect_triplet':
                    fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                    triplet_loss_dict = self.triplet_loss_f(hidden, aux_pred_dict['positive'],
                                                            aux_pred_dict['negative'],
                                                            self.args.triple_margin)
                    fake_det_loss_dict.update(triplet_loss_dict)
                    loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
                elif self.args.loss == 'ae_mse_sup_fake_detect':
                    aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)
                    fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                    
                    tasks_dict = self.args.aux_tasks.copy()
                    tasks_dict.update(self.args.unsup_aux_tasks)
                    task_loss_dict = aux_loss_dict.copy()
                    task_loss_dict.update(fake_det_loss_dict)
                    loss_dict = self.multi_task_loss_f(tasks_dict, rec_loss_dict, task_loss_dict)
                else:
                    raise NotImplementedError

                [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

                ###save all batch_data, predicted labels, hidden_features and recon_observations to a dict, and append to the list
                batch_sample_copy = deepcopy(batch_sample)
                for k, v in batch_sample_copy.items():
                    if k != 'encounter_id':
                        batch_sample_copy[k] = v.numpy()
                batch_sample_copy.update({k: v.detach().cpu().numpy() for k, v in aux_pred_dict.items()})
                batch_sample_copy['hidden'] = hidden.detach().cpu().numpy()
                batch_sample_copy['rec_ob'] = rec_ob.detach().cpu().numpy()
                ob_pred_lst.append(batch_sample_copy)

                if i_batch % self.args.log_valid_freq == 1:
                    logger.info('{}-[{}/{} ({:.0f}%)]: {}-{}'.format(
                        self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                        scope, {k: v.item() for k, v in loss_dict.items()}))
                    loss_dict['scope'] = '{}_batch'.format(scope)
                    logger.debug('Batch: {}'.format(self.epoch * len(dl) + i_batch))
                    if self.args.mode == 'train':
                        self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)
                logger.debug('Eval_batch: {}'.format(i_batch))
            valid_metrics_dict = {'scope': scope}
            valid_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})
            return valid_metrics_dict, ob_pred_lst

    def aly_pred(self, scope, metric_dict):
        logger.debug('Epoch: {}'.format(self.epoch))

        if scope == 'valid':
            if self.args.lr_decay_mode in ['step', 'warmup']:
                self.lr_scheduler.step()
            elif self.args.lr_decay_mode == 'plateau':
                reduce_lr_on_plateau(self.lr_scheduler, metric_dict, 'loss')

            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < self.args.min_lr:
                    param_group['lr'] = self.args.min_lr
                metric_dict.update({'lr': param_group['lr']})

            # save_weight_update_flag(model, weight_dir_dict, flag_dict, metric_dict, epoch)
            save_model_update_flag(self.model, self.optimizer, self.weight_path_dict, self.flag_dict,
                                   metric_dict, MIN_METRICS, MAX_METRICS, self.epoch)
            logger.debug('Flag dict: {}'.format(self.flag_dict))
        self.summary.add_summary(self.epoch, **metric_dict)
        logger.info(metric_dict)

        result_dict = dict()
        result_dict['early_stop'] = early_stop(self.flag_dict, self.epoch, self.args.early_stopping, scope)
        return result_dict

    ###load checkpoint model to restore based on the loss
    def load_weight(self):
        logger.info("*******Restoring the model weight based on {}*******".format(self.args.restore_metric))
        restore_file = os.path.join(self.weight_path_dict[self.args.restore_metric], 'model.pth.tar')
        if restore_file.endswith('.tar'):
            checkpoint = torch.load(restore_file)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> loaded checkpoint from model.path.tar')
        else:
            logger.error('==> Load fail: no checkpoint for {}'.format(restore_file))

    ###merge all batch data to a single list 
    def merge_ob_pred(self, ob_pred_lst):
        super_dict = defaultdict(list)
        for d in ob_pred_lst:
            for k, v in d.items():
                super_dict[k].extend(v)
        for k, v in super_dict.items():
            super_dict[k] = np.array(v)
        return super_dict

    def re_norm_data(self, ob_pred_dict):
        ks = ['ob', 'rec_ob']
        if self.args.norm_method == 'minmax':
            for k in ks:
                normed_data = ob_pred_dict[k]
                renorm = (normed_data + self.args.scale / 2) / self.args.scale  # Back to [0, 1]

                for i, (feat_name, val_range) in enumerate(MIN_MAX_VALUES.items()):
                    min_val, max_val = val_range
                    normed_data[:, i, :] = renorm[:, i, :] * (max_val - min_val) + min_val
                ob_pred_dict[k] = normed_data
            return ob_pred_dict
        else:
            raise NotImplementedError
    
    ###add gaussian noise to the input, only work for triplet loss
    def add_gaussian_noise(self, tensor, padding_mask, gaussian_config_dict):
        mean = gaussian_config_dict.get('mean', 0.0)
        std = gaussian_config_dict.get('std', .1)
        noise = torch.randn(tensor.size(), device=self.device) * std + mean
        noise_tensor = tensor + noise
        noise_tensor *= padding_mask
        return noise_tensor
