#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created by yanjun.li at 12/3/19
"""
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import copy
from info import BASE_PATH, USE_FEATURES
import os
import numpy as np
from utils import logger

class DataSet(Dataset):
    def __init__(self, args, cohort):
        self.use_features = USE_FEATURES
        self.num_features = len(self.use_features)
        self.logger = logger
        self.hours_from_admission = args.hours_from_admission
        self.cohort = cohort
        self.scale = args.scale
        
        ###other supervised tasks
        self.aux_tasks = args.aux_tasks
        self.fake_detection = args.fake_detection

        self.data_path = os.path.join(BASE_PATH, 'Data')
        self.model_data_path = os.path.join(self.data_path, "model_data")
        self.processed_data_path = os.path.join(self.model_data_path, "split_processed")
        self.auxiliary_data_path = os.path.join(self.data_path, "analysis_data")
        self.future_vital_path = os.path.join(self.data_path, 'vital_data')

        ###load real time series data
        self._read_data()
        self._fix_input_format()
        self._scale_data(self.scale)

        if self.aux_tasks:
            self.auxiliary_dict = self._load_auxiliary_data()
            self.logger.debug(self.auxiliary_dict)
            
        if args.aug_input and cohort == 'training':
            self.transform = Transform(aug=True, **{'ob_std': args.aug_std})
        else:
            self.transform = Transform(aug=False)

    def _read_data(self):
        self.data_f = os.path.join(self.processed_data_path, "{}.pickle".format(self.cohort))
        with open(self.data_f, "rb") as f:
            self.data_dict = pickle.load(f)

    def _fix_input_format(self):
        """Return the input in the proper format
        # change the self.data_dict to the numpy input with input_shape [batch, features * 4, time_stamp]
        # The order matters!
        # [:, :features, :] observed value
        # [:, features : 2*features, :] padding_mask, if no observed value, then padding zero
        # [:, 2*features : 3*features, :] actual time stamps for each observation
        # [:, 3*features : 4*features, :] hold out, if is 0, then this observation is taken out, used for autoencoder
        """
        self.logger.info(self.data_dict.keys())
        self.encounter_ids = self.data_dict["encounter_id"]
        self.encounter_ids_df = pd.DataFrame(data={"encounter_deiden_id": self.encounter_ids})

        input_lst = [self.data_dict["feat"], self.data_dict["padding_mask"],
                     self.data_dict["time_step"], self.data_dict["drop_mask"]]
        self.feed_data = np.concatenate(input_lst, axis=1)
        self.num_timestep = self.feed_data.shape[-1]
        self.logger.info("{} data shape: {}".format(self.cohort, self.feed_data.shape))
        return

    def _scale_data(self, scale):
        if scale != 0:
            self.feed_data[:, 0:self.num_features, :] = scale * self.feed_data[:, 0:self.num_features, :] - scale/2            # print(self.feed_data[0, 0:self.num_features, :])
            self.logger.info("Scale input data to {}".format(scale * np.array([0, 1]) - scale/2))
        else:
            self.logger.info('No scale input, keep [0, 1]')

    def _load_auxiliary_data(self):
        outcome_df = pd.read_csv(os.path.join(self.auxiliary_data_path, "table_data.csv"))
        mortality_df = pd.read_csv(os.path.join(self.auxiliary_data_path, "mortality_summary.csv"))
        future_vital_df = pd.read_csv(os.path.join(self.future_vital_path, 'next_hour_abnormal_norm_val.csv'))
        auxiliary_df = self.encounter_ids_df.merge(outcome_df)
        auxiliary_df = auxiliary_df.merge(mortality_df)
        auxiliary_df = auxiliary_df.merge(future_vital_df)
        
        auxiliary_dict = dict()
        auxiliary_dict['encounter_deiden_id'] = self.encounter_ids_df['encounter_deiden_id'].values
        
        if 'future_vital' in self.aux_tasks:
            future_vital_data = auxiliary_df[USE_FEATURES]
            mask_df = pd.DataFrame()
            for i, value in enumerate(future_vital_data.columns):
                mask_df[value + '_mask'] = future_vital_data[value].notnull().astype(int)
            mask_arr = np.asarray(mask_df)
            auxiliary_dict['future_vital_mask'] = mask_arr 
            
            value_df = future_vital_data.fillna(0)
            auxiliary_dict['future_vital'] = np.asarray(value_df)
            self.logger.info('Finish loading future vital data and mask.')
        
        for auxiliary_k in self.aux_tasks:
            if auxiliary_k == 'future_vital':
                continue 
            
            row_data = auxiliary_df[auxiliary_k].values
            processed_data = self._process_auxiliary_data(auxiliary_k, row_data)
            auxiliary_dict[auxiliary_k] = processed_data
            num_pos = np.count_nonzero(processed_data)
            self.logger.info('For {}, neg/pos={}'.format(auxiliary_k, len(processed_data)/num_pos))
        return auxiliary_dict

    def _process_auxiliary_data(self, auxiliary_task, row_data):
        if auxiliary_task in ["AKI_overall", "ICU_24h", "ICU", "mort_status_30d", 'mort_status_3y']:
            numerical_data = (row_data == "Y").astype(int)
        return numerical_data

    def __getitem__(self, index):
        sample_feat = self.feed_data[index]
        encounter_id = self.encounter_ids[index]

        ob = sample_feat[0: self.num_features]
        padding_mask = sample_feat[self.num_features: 2*self.num_features]
        timestamp = sample_feat[2*self.num_features: 3*self.num_features]
        ae_mask = sample_feat[3*self.num_features:]

        sample = {'encounter_id': encounter_id, 'ob': ob, 'padding_mask': padding_mask,
                  'timestamp': timestamp, 'ae_mask': ae_mask}

        if self.fake_detection:
            fake_ob = self._generate_fake_data(ob, padding_mask)
            fake_sample = {'encounter_id': encounter_id, 'ob': fake_ob, 'padding_mask': padding_mask,
                           'timestamp': timestamp, 'ae_mask': ae_mask}
        else:
            fake_sample = sample

        if self.aux_tasks:
            for task in self.aux_tasks:
                if task == 'future_vital': 
                    sample.update({'future_vital' : self.auxiliary_dict['future_vital'][index]}) 
                    sample.update({'future_vital_mask' : self.auxiliary_dict['future_vital_mask'][index]})                     
                else:
                    sample.update({task: self.auxiliary_dict[task][index]}) 
        
        sample = self.transform(sample)
        fake_sample = self.transform(fake_sample)
        return sample, fake_sample

    def __len__(self):
        return len(self.feed_data)

    # def _generate_fake_data(self, ob, padding_mask):
    #     fake_ob = copy.deepcopy(ob)
    #     for var_val, padding_val in zip(fake_ob, padding_mask):
    #         num_valid_val = int(np.sum(padding_val))
    #         num_perm = int(num_valid_val * 0.2)
    #         if num_perm == 0:
    #             # TODO: refine more decent noise way when num_valid_val is small.
    #             # Add large random noise to the org value. Because num_valid_val is very small, std is small.
    #             mean, std = np.mean(var_val[:num_valid_val]), np.std(var_val[:num_valid_val])
    #             var_val[:num_valid_val] = var_val[:num_valid_val] + 10 * std * np.random.rand(num_valid_val)
    #         else:
    #             perm_idx = np.random.permutation(num_valid_val)
    #             perm_mask = np.zeros(num_valid_val)
    #             perm_mask_idx = np.random.choice(num_valid_val, size=num_perm, replace=False)
    #             perm_mask[perm_mask_idx] = 1
    #             var_val[:num_valid_val] = var_val[perm_idx] * perm_mask + var_val[:num_valid_val] * (1 - perm_mask)
    #     return fake_ob
    
    # def _generate_fake_data(self, ob, padding_mask):
    #     fake_ob = copy.deepcopy(ob)
    #     for var_val, padding_val in zip(fake_ob, padding_mask):
    #         num_valid_val = int(np.sum(padding_val))
    #         if self.scale == 0:
    #             var_val[:num_valid_val] = np.random.rand(num_valid_val)
    #         else:
    #             var_val[:num_valid_val] = np.random.rand(num_valid_val) * self.scale - self.scale/2
    #     return fake_ob 
    
    def _generate_fake_data(self, ob, padding_mask):
        fake_ob = copy.deepcopy(ob)
        for var_val, padding_val in zip(fake_ob, padding_mask):
            num_valid_val = int(np.sum(padding_val))
            num_perm = max(1, int(num_valid_val * 0.5))
            
            perm_mask_idx = np.random.choice(num_valid_val, size=num_perm, replace=False)
            if self.scale == 0:
                var_val[perm_mask_idx] = np.random.rand(num_perm)
            else:
                var_val[perm_mask_idx] = np.random.rand(num_perm) * self.scale - self.scale/2
        return fake_ob


class Transform(object):
    def __init__(self, aug, **aug_config):
        self.aug = aug
        self.ob_std = aug_config.get('ob_std', 0)

    def __call__(self, sample):
        for k, v in sample.items():
            if k != 'encounter_id':
                sample[k] = torch.as_tensor(v, dtype=torch.float32)
        if self.aug:
            sample['ob'] = self.add_gaussian_noise(sample['ob'], sample['padding_mask'], {'mean': 0., 'std': self.ob_std})
            sample['timestamp'] = self.add_gaussian_noise(sample['timestamp'], sample['padding_mask'],
                                                          {'mean': 0., 'std': .01})
        return sample

    def add_gaussian_noise(self, tensor, padding_mask, gaussian_config_dict):
        mean = gaussian_config_dict.get('mean', 0.0)
        std = gaussian_config_dict.get('std', .1)
        noise = torch.randn(tensor.size()) * std + mean
        noise_tensor = tensor + noise
        noise_tensor *= padding_mask
        return noise_tensor
    
if __name__ == "__main__":
    from p1_pretrain_main import get_arguments 
    from torch.utils.data import DataLoader
    from utils import set_seed
    
    args = get_arguments()
    set_seed(args.seed)
    ds = DataSet(args, 'validation')  # training, validation, testing
    logger.info(len(ds))
    dl = DataLoader(ds, batch_size=20, num_workers=0)

    for batch_id, (batch_samples, batch_fake_samples) in enumerate(dl):
        logger.debug(batch_samples)
        logger.debug(batch_fake_samples)
        logger.info('===')
        break