#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
train_cluster.py: 
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
import os.path as osp
from sklearn.cluster import KMeans

class TrainerCluster(object):
    def __init__(self, args, model, dl_dict, exp_path, pretrain_exp_path, device, **kwags):
        self.args = args
        self.device = device
        self.model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus)) if args.num_gpus > 0 else model
        # self.model = model
        self.model.to(self.device)

        ###get the functions pf current model
        self.init_cluster_center = model.init_cluster_center
        self.get_cluster_center = model.get_cluster_center
        self.rec_loss_f = model.rec_loss
        self.sup_aux_loss_f = model.sup_aux_loss
        self.fake_det_loss_f = model.fake_det_loss
        self.triplet_loss_f = model.triplet_loss
        self.kl_loss_f = model.kl_loss
        self.multi_task_loss_f = model.multi_task_loss

        self.pretrain_exp_path = pretrain_exp_path
        self.exp_path = exp_path
        self.weight_path = osp.join(self.exp_path, "weight")
        self.weight_path_dict = create_weight_dir(self.weight_path, METRICS)
        self.summary_path = osp.join(self.exp_path, "summary")
        os.makedirs(self.weight_path, exist_ok=True)

        self.out_feat_path = osp.join(self.exp_path, 'out_feat', self.args.dc_restore_metric)
        os.makedirs(self.out_feat_path, exist_ok=True)

        self.dl_dict = dl_dict
        self.train_dl = self.dl_dict['training']
        self.valid_dl = self.dl_dict['validation']
        self.test_dl = self.dl_dict['testing']

        # Record the model and config
        self.epoch = 1
        logger.info(self.model)

        # Optimizer
        # Note: construct optimizer after moving model to cuda
        self.optimizer = pytorch_optimizer(self.model, args.optimizer, args.init_lr, args.weight_decay_rate)
        self.lr_scheduler = pytorch_lr_scheduler(self.optimizer, args.lr_decay_mode, args.lr_decay_step_or_patience,
                                                 args.lr_decay_rate)

        self.flag_dict = create_flag_dict(METRICS, MIN_METRICS, MAX_METRICS)
        summary_writer = SummaryWriter(self.summary_path, filename_suffix=datetime.now().strftime('_%m-%d-%y_%H-%M-%S'))
        self.summary = Summary(summary_writer, METRICS, SUMMARY_ITEMS)

        torch.set_printoptions(precision=5)

    def train(self):
        logger.info('*******Building the model*******')
        if self.args.init_cluster_center == 'kmeans':
            self.load_pretrain_weight()
            ob_pred_dict = self.generate_pretrain_feat('training')
            kmeans = KMeans(n_clusters=self.args.cluster_number, n_init=20)
            predicted = kmeans.fit_predict(ob_pred_dict['hidden'])
            train_prev_cluster_pred = predicted
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True, device=self.device)
            self.init_cluster_center(cluster_centers)

            valid_ob_pred_dict = self.generate_pretrain_feat('validation')
            valid_prev_cluster_pred = kmeans.predict(valid_ob_pred_dict['hidden'])

        elif self.args.init_cluster_center == 'random':
            # uniform random in the same range of per-dim of the train_hidden feat
            self.load_pretrain_weight()
            ob_pred_dict = self.generate_pretrain_feat('training')
            train_hidden = ob_pred_dict['hidden']
            max_val, min_val = np.max(train_hidden, axis=0), np.min(train_hidden, axis=0)
            cluster_centers = np.random.uniform(low=min_val, high=max_val, size=(self.args.cluster_number, max_val.shape[-1]))
            cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True, device=self.device)
            self.init_cluster_center(cluster_centers)
            valid_prev_cluster_pred = None

        elif self.args.init_cluster_center == 'none':
            valid_prev_cluster_pred = None

        logger.info('*****Cluster initialize {} is done.*****'.format(self.args.init_cluster_center))

        with timer('Duration of training'):
            for epoch in range(1, self.args.max_epochs):
                train_metrics_dict = self.train_one_epoch(self.train_dl, denoise=self.args.denoise)
                logger.info('==> Epoch: {}, Train, {}'.format(epoch, format_metric_dict(train_metrics_dict)))

                # train_delta, train_cluster_pred, _ = \
                #     self.generate_pred_cluster('train', self.train_dl, train_prev_cluster_pred)
                # self.summary.add_summary(epoch, **{'scope': 'train', 'delta': train_delta})
                # logger.info('Epoch: {}: Train delta of cluster label change: {}'.format(epoch, train_delta))

                valid_delta, valid_cluster_pred, valid_metrics_dict = \
                    self.generate_pred_cluster('valid', self.valid_dl, valid_prev_cluster_pred)
                logger.info('Epoch: {}: valid delta of cluster label change: {}'.format(epoch, valid_delta))
                valid_metrics_dict['delta'] = valid_delta

                # Save model and adjust lr.
                valid_aly_result_dict = self.aly_pred('valid', valid_metrics_dict)

                if epoch % self.args.update_interval == 0:
                    # valid_delta, valid_cluster_pred = \
                    #     self.generate_pred_cluster('valid', self.valid_dl, valid_prev_cluster_pred)

                    if self.args.stopping_delta is not None and valid_delta < self.args.stopping_delta:
                        logger.info('Early stopping as label delta "%1.5f" less than "%1.5f".' % (valid_delta, self.args.stopping_delta))
                        break
                    valid_prev_cluster_pred = valid_cluster_pred
                    # train_prev_cluster_pred = train_cluster_pred
                    logger.info('=== Update the cluster centers for train and valid.')

                # TODO: generate the t-sne viz_label

                self.epoch += 1
                # test_data_dict = self.eval_one_epoch("test", self.test_data_loader)
                logger.info('==================')

    def eval(self, cohort, generate_feat=False, viz_feat=False, denoise=False):
        logger.info("*******Evaluating the model*******")
        self.load_weight()
        dl = self._get_dl(cohort)

        scope = COHORT2SCOPE[cohort]
        metrics_dict, ob_pred_lst = self.eval_one_epoch(scope, dl, denoise)
        logger.info("{}, {}".format(scope, format_metric_dict(metrics_dict)))
        ob_pred_dict = self.merge_ob_pred(ob_pred_lst)
        ob_pred_dict = self.re_norm_data(ob_pred_dict)  # Re-normalize to real range

        if generate_feat:
            folder_path = os.path.join(self.exp_path, 'out_feat', self.args.dc_restore_metric)
            os.makedirs(folder_path, exist_ok=True)
            np_f = osp.join(folder_path, "{}.npy".format(cohort))
            if osp.exists(np_f):
                logger.info('No save, the npy file exists. {}'.format(np_f))
            else:
                logger.info("*******Generate the features for {}*******".format(cohort))
                np.save(np_f, ob_pred_dict)
                logger.info('The npy is saved to {}'.format(np_f))

        if viz_feat:
            logger.info("*******Visualize the features for {}*******".format(cohort))
            self.summary.summary_writer.add_embedding(ob_pred_dict['hidden'], global_step=self.epoch, tag=cohort)

    def train_one_epoch(self, dl, denoise=True):
        self.model.train()
        metrics_dict = defaultdict(list)
        for i_batch, (batch_sample, fake_batch_sample) in enumerate(dl, start=1):
            encounter_ids = batch_sample['encounter_id']
            ob = batch_sample['ob'].to(self.device)
            padding_mask = batch_sample['padding_mask'].to(self.device)
            timestamp = batch_sample['timestamp'].to(self.device)
            ae_mask = batch_sample['ae_mask'].to(self.device)
            ob *= padding_mask

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
                fake_ob *= fake_padding_mask
                assert (fake_encounter_ids == encounter_ids), 'Encounter_id dose not match.'
                if denoise:
                    fake_ae_masked_ob = fake_ob * fake_ae_mask
                    fake_stacked_input = torch.cat([fake_ae_masked_ob, fake_padding_mask, fake_timestamp, fake_ae_mask], dim=1)
                else:
                    fake_stacked_input = torch.cat([fake_ob, fake_padding_mask, fake_timestamp, fake_ae_mask], dim=1)

                # generate the label for fake_detection
                pos_cls, neg_cls = torch.ones(ob.size()[0], device=self.device), \
                                   torch.zeros(fake_ob.size()[0], device=self.device)
                fake_det_label = torch.cat([pos_cls, neg_cls])
                fake_perm_idx = torch.randperm(fake_det_label.size()[0])    # shuffle the pos and neg samples
                fake_det_label = fake_det_label[fake_perm_idx].to(torch.int64)
            else:
                fake_stacked_input, fake_perm_idx, fake_det_label = None, None, None

            if self.args.triple_margin != 0. and self.args.fake_detection:
                assert self.args.scale == 20, 'The noise gaussian config should be adjusted.'
                noise_ob = self.add_gaussian_noise(ob, padding_mask, {'std': self.args.triple_pos_std})
                noise_timestamp = self.add_gaussian_noise(timestamp, padding_mask, {'std': 0.01})
                noise_stacked_input = torch.cat([noise_ob, padding_mask, noise_timestamp, ae_mask], dim=1)
            else:
                noise_stacked_input = None

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
            hidden, rec_ob, aux_pred_dict = self.model(stacked_input, fake_stacked_input, fake_perm_idx, noise_stacked_input)
            # logger.debug('ref_y: {}, rec_ob: {}'.format(ref_y.size(), rec_ob.size()))

            # Reconstruct all the input
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
            elif self.args.loss == 'ae_mse_kl':
                kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, kl_loss_dict)
            elif self.args.loss == 'ae_mse_fake_detect_kl':
                fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                fake_det_loss_dict.update(kl_loss_dict)
                loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
            elif self.args.loss == 'ae_mse_sup_fake_detect_kl':
                fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)

                tasks_dict = self.args.aux_tasks.copy()
                tasks_dict.update(self.args.unsup_aux_tasks)
                task_loss_dict = aux_loss_dict.copy()
                task_loss_dict.update(fake_det_loss_dict)
                task_loss_dict.update(kl_loss_dict)

                loss_dict = self.multi_task_loss_f(tasks_dict, rec_loss_dict, task_loss_dict)
            else:
                raise NotImplementedError
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

        train_metrics_dict = {'scope': 'train'}
        train_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})
        return train_metrics_dict

    def eval_one_epoch(self, scope, dl, denoise=False):
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
                ob *= padding_mask

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
                    fake_ob *= padding_mask
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
                    noise_ob = self.add_gaussian_noise(ob, padding_mask, {'std': self.args.triple_pos_std})
                    noise_timestamp = self.add_gaussian_noise(timestamp, padding_mask, {'std': 0.01})
                    noise_stacked_input = torch.cat([noise_ob, padding_mask, noise_timestamp, ae_mask], dim=1)
                else:
                    noise_stacked_input = None

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

                hidden, rec_ob, aux_pred_dict = self.model(stacked_input, fake_stacked_input, fake_perm_idx, noise_stacked_input)

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
                elif self.args.loss == 'ae_mse_kl':
                    kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                    loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, kl_loss_dict)
                elif self.args.loss == 'ae_mse_fake_detect_kl':
                    fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                    kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                    fake_det_loss_dict.update(kl_loss_dict)
                    loss_dict = self.multi_task_loss_f(self.args.unsup_aux_tasks, rec_loss_dict, fake_det_loss_dict)
                elif self.args.loss == 'ae_mse_sup_fake_detect_kl':
                    fake_det_loss_dict = self.fake_det_loss_f(fake_det_label, aux_pred_dict['fake_det'])
                    kl_loss_dict = self.kl_loss_f(aux_pred_dict['cluster_label'], aux_pred_dict['cluster_pred'])
                    aux_loss_dict = self.sup_aux_loss_f(self.args.aux_tasks, aux_label_dict, aux_pred_dict, future_vital_mask)
    
                    tasks_dict = self.args.aux_tasks.copy()
                    tasks_dict.update(self.args.unsup_aux_tasks)
                    task_loss_dict = aux_loss_dict.copy()
                    task_loss_dict.update(fake_det_loss_dict)
                    task_loss_dict.update(kl_loss_dict)
    
                    loss_dict = self.multi_task_loss_f(tasks_dict, rec_loss_dict, task_loss_dict)
                else:
                    raise NotImplementedError

                [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

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

    def load_pretrain_weight(self):
        logger.info("*******Restoring the pretrain model weight based on {}*******".format(self.args.restore_metric))
        restore_file = osp.join(self.pretrain_exp_path, 'weight', '{}'.format(self.args.restore_metric), 'model.pth.tar')
        if restore_file.endswith('.tar'):
            checkpoint = torch.load(restore_file,map_location='cuda:0')
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(pretrained_dict, strict=False)
            logger.info('=> loaded checkpoint from model.path.tar')
        else:
            logger.error('==> Load fail: no checkpoint for {}'.format(restore_file))

    def _get_dl(self, cohort):
        if cohort == "training":
            return self.train_dl
        elif cohort == "validation":
            return self.valid_dl
        elif cohort == "testing":
            return self.test_dl

    def generate_feat(self, cohort):
        logger.info("*******Generate the features for {}*******".format(cohort))
        dl = self._get_dl(cohort)

        _, features = self.model.predict(dl.feed_data)
        np.save(osp.join(self.out_feat_path, "{}.npy".format(cohort)), features)
        return features

    def generate_pretrain_feat(self, cohort, denoise=False):
        dl = self._get_dl(cohort)
        scope = COHORT2SCOPE[cohort]
        metrics_dict, ob_pred_lst = self.eval_one_epoch(scope, dl, denoise)
        logger.info("{}, {}".format(scope, format_metric_dict(metrics_dict)))
        ob_pred_dict = self.merge_ob_pred(ob_pred_lst)
        return ob_pred_dict

    def generate_pred_cluster(self, scope, dl, prev_pred, denoise=False):
        metrics_dict, ob_pred_lst = self.eval_one_epoch(scope, dl, denoise=denoise)
        logger.info('{}'.format(format_metric_dict(metrics_dict)))
        ob_pred_dict = self.merge_ob_pred(ob_pred_lst)
        cluster_pred = np.argmax(ob_pred_dict['cluster_pred'], axis=1)
        if prev_pred is None:
            delta_label = 1.0
        else:
            logger.debug(ob_pred_dict['cluster_pred'][0:10], cluster_pred[:10], ob_pred_dict['cluster_label'],
                         prev_pred[:10])
            delta_label = np.sum((cluster_pred != prev_pred)) / prev_pred.shape[0]
        return delta_label, cluster_pred, metrics_dict

    def merge_ob_pred(self, ob_pred_lst):
        super_dict = defaultdict(list)
        for d in ob_pred_lst:
            for k, v in d.items():
                super_dict[k].extend(v)
        for k, v in super_dict.items():
            super_dict[k] = np.array(v)
        return super_dict

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

    def load_weight(self):
        logger.info("*******Restoring the model weight based on {}*******".format(self.args.dc_restore_metric))
        restore_file = osp.join(self.exp_path, 'weight', self.args.dc_restore_metric, 'model.pth.tar')
        if restore_file.endswith('.tar'):
            checkpoint = torch.load(restore_file,map_location='cuda:0')
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> loaded checkpoint from model.path.tar')
        else:
            logger.error('==> Load fail: no checkpoint for {}'.format(restore_file))

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
