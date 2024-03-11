#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created by yanjun.li at 12/2/19
"""
from clustering_trainer import TrainerCluster
from torch.utils.data import DataLoader
from dataloader import DataSet
from utils import set_seed, count_parameters, logger
from info import COHORTS, METRICS
import os
import random
import torch
from clustering_interp import Net
import argparse

def get_arguments():
    description = "Implementation of deep clustering for time series data"
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default="DEBUG", choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    general.add_argument('-s', "--seed", type=float, default=7529)  # 7529

    general.add_argument('--cluster_number', type=int, default=4, help='The number of cluster.')
    ###the metric used for load trained clustering model
    general.add_argument('--dc_restore_metric', type=str, default='ae_mse', help='The restore metric of deep cluster.')
    general.add_argument('--init_cluster_center', type=str, default='kmeans',
                         help='The method for initializing the cluster center. e.g., kmeans, random, none')
    general.add_argument('--num_gpus', type=int, default=1)
    general.add_argument('--mode', type=str, default='train',
                         choices=['train', 'eval'],
                         help='Train or predict')
    general.add_argument("--restore", action='store_true', help="Whether to restore or not.")
    ###the metric used for load pretrained model
    general.add_argument('--restore_metric', type=str, default='ae_mse', help='The metric used for restoring the weight',
                         choices=['loss', 'ae_mse', 'ae_mse_sup', 'ae_mse_fake_detect', 'ae_mse_sup_fake_detect'])
    general.add_argument('--log_train_freq', default=20, help='The log frequency for training.')
    general.add_argument('--log_valid_freq', default=20, help='The log frequency for testing.')

    # Data Options
    ##The store_true option automatically creates a default value of False.
    data = parser.add_argument_group('Data specific options')
    data.add_argument('--hours_from_admission', type=int, default=6, help='Hours of record to look at')
    data.add_argument('--num_workers', type=int, default=3, help='The number of workers used for loading data.')
    data.add_argument('--batch_size', type=int, default=256, help='batch size for the lstm training')
    data.add_argument('--norm_method', type=str, default='minmax', choices=['minmax'],
                      help='The type of normalization method to preprocess data')
    data.add_argument('--aug_input', action='store_true', help='whether add gaussian noise to input ob and time point.')
    data.add_argument('--aug_std', type=float, default=0.1, help='The std of gaussian noise to generate aug input.')
    data.add_argument('--scale', type=float, default=5,
                      help='0: No scale, keep original [0, 1]; Otherwise scale the input to [-scale/2, +scale/2]')
    data.add_argument('--denoise', default=False, help='Whether to denoise the input.')
    data.add_argument('--num_variables', type=int, default=6, help='The number of observation variables.')
    data.add_argument('--num_timestamps', type=int, default=354)
    data.add_argument('--data_filter', action='store_true',
                      help='If yes, align the data as same as Ren by removing some samples')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('--ref_points', type=int, default=6, help='Number of reference points')
    model.add_argument("--dropout", type=float, default=0.2, help="The dropout ratio for recurrent state and inputs.")
    model.add_argument('--fake_detection', default=True, help='Generate the fake samples and detect them')
    model.add_argument('--triple_margin', type=float, default=0.,
                       help='The margin for triplet loss, if 0, no triple loss is applied. And triple loss is valid '
                            'only when triple_margin > 0. and fake_detection is true.')
    model.add_argument('--triple_pos_std', type=float, default=0.1,
                       help='The std used for generating positive sample for triplet loss')
    model.add_argument('--stopping_delta', type=float, default=0.0001,
                       help='The stop criterion for clustering, i.e., '
                            'less than tol%of points change cluster assignment between two consecutive iterations.')
    model.add_argument('--update_interval', type=int, default=1,
                       help='The epoch number for updating cluster target labels.')
    # Learning options
    learning = parser.add_argument_group('Training specific options')
    learning.add_argument('--loss', default='ae_mse_sup_fake_detect_kl', help='The name for loss.',
                          choices=['ae_mse', 'ae_mse_sup', 'ae_mse_fake_detect', 'ae_mse_fake_detect_triplet', 'ae_mse_sup_fake_detect',
                                   'ae_mse_kl', 'ae_mse_fake_detect_kl', 'ae_mse_sup_kl', 'ae_mse_sup_fake_detect_kl'])
    learning.add_argument('--aux_tasks', default={"future_vital" : .5},
                          help="The auxiliary tasks used for training, including AKI_overall, mort_status_30d, ICU, ICU_24h, future_vital")
    learning.add_argument('--aux_pos_weights', default={"AKI_overall": 1, "mort_status_30d": 1, "ICU": 1, "future_vital" : 1},
                          help='The positive weight for different aux task: neg/pos (6, 25) for imbalance problem.')
    learning.add_argument('--unsup_aux_tasks', default={'fake_detection': 1., 'triplet': 1., 'kl': 10.},
                          help='The unsupervised task and its weight')
    learning.add_argument('--max_epochs', type=int, default=8000,
                          help='Number of epochs to run the training')
    learning.add_argument('--optimizer', default='Adam')
    learning.add_argument('--init_lr', '-l', type=float, default=0.003,
                          help='The current learning rate.(SGD/Mom: 0.01, Adam: 0.001)')
    learning.add_argument('--min_lr', '-mlr', type=float, default=1e-6,
                          help='The minimum learning rate for training.')
    learning.add_argument('--lr_decay_mode', '-lm', type=str, default='step',
                          choices=['exp', 'anneal', 'plateau', 'step', 'warmup'])
    learning.add_argument('--lr_decay_step_or_patience', type=int, default=20,
                          help='learning rate decay patience on plateau')
    learning.add_argument('--lr_decay_rate', '-a', type=float, default=0.2, help='The learning rate decay speed.')
    learning.add_argument('--grad_clip', type=float, default=15)
    learning.add_argument('--weight_decay_rate', '-wd', type=float, default=0.0004,
                          help='The weight decay rate for l2 loss.')
    learning.add_argument('--early_stopping', default=50, help='The early stopping step.')

    args = parser.parse_args()
    return args

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    
    pretrain_exp_path = os.path.join(os.getcwd(), "Results", "Pretrain")
    exp_path = os.path.join(os.getcwd(), 'Results', 'Clustering')
    os.makedirs(exp_path, exist_ok=True)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_path))

    device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
    model = Net(args, device=device)    

    dl_dict = dict()
    num_train_sample = 1
    for cohort in COHORTS:
        ds = DataSet(args, cohort)
        if cohort == 'training':
            shuffle = True
            num_train_sample = len(ds)
        else:
            shuffle = False
        dl_dict[cohort] = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)

    trainable_count = count_parameters(model)
    logger.info('The ratio is {} ({} / {})'.format(trainable_count / num_train_sample, trainable_count, num_train_sample))

    trainer = TrainerCluster(args, model, dl_dict, exp_path, pretrain_exp_path, device)
    if args.mode == "train":
        trainer.train()
        trainer.args.mode = 'eval'

    # Generate the feat, and update dataloader
    for metric in METRICS: 
        trainer.args.dc_restore_metric = metric
        for cohort in COHORTS:
            trainer.eval(cohort, generate_feat=True, viz_feat=True, denoise=False)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
