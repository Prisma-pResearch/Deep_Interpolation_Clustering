#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created by yanjun.li at 12/3/19
modified by Yuanfang Ren
"""
import pickle
from collections import defaultdict
import os
import pandas as pd
import numpy as np 
import random  
import tensorflow as tf
import argparse
from info import BASE_PATH, USE_FEATURES, COHORTS, MIN_MAX_VALUES

def set_seed(seed):
    """Set all random seeds."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def parse_args():
    description = "Implementation of deep clustering for processing time series data"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--hours_from_admission', type=int, default=6,
                         help='Hours of record to look at')
    parser.add_argument('--norm_method', type=str, default='minmax',
                         choices=['minmax'],
                         help='The type of normalization method to preprocess data')  
    args = parser.parse_args()
    return args

def generate_data(encounter, vital_data):
    """
    mainly used for interpolation_LSTM method, generate a python dict with:
        1. feat: observed value
        2. time_step: time stamps of observed value
        3. padding: mask of the data, 1 indicates having value, 0 indicates no value
        4. idx
    """
    
    # 1. first calculate the max number of time stamps across vitals
    max_length = 0
    for feature_name, subdf in vital_data.items():
        max_len = subdf.groupby('encounter_deiden_id')['time_stamp'].count().max()
        if max_len > max_length:
            max_length = max_len
    print('max_length', max_length)

    # 2. generate the data with shape [num_samples, num_features, num_stamps (max_length)]
    feat = np.zeros((len(encounter), len(vital_data), max_length))
    padding_mask = np.zeros_like(feat, dtype=np.int8)
    time_step = np.zeros_like(feat)
    encounter_id_lst = encounter['encounter_deiden_id'].tolist()   
    encounter_idx_map = {eid : i for i, eid in enumerate(encounter_id_lst)}

    count = 0
    for feature_name, df in vital_data.items():
        print('process data frame {}'.format(feature_name))
        for encounter_id, sub_df in df.groupby(['encounter_deiden_id']):
            encounter_idx = encounter_idx_map[encounter_id]
            num_records = len(sub_df)
            padding_mask[encounter_idx, count, : num_records] = 1
            feat[encounter_idx, count, : num_records] = sub_df.measurement.values
            time_step[encounter_idx, count, : num_records] = sub_df.time_stamp.values
        count += 1

    return dict(feat=feat, time_step=time_step, padding_mask=padding_mask, encounter_id=encounter_id_lst)

def mean_imputation(vitals, mask, pre_mean=None):
    """For the time series missing entirely, our interpolation network
    assigns the starting point (time t=0) value of the time series to
    the global mean before applying the two-layer interpolation network.
    In such cases, the first interpolation layer just outputs the global
    mean for that channel, but the second interpolation layer performs
    a more meaningful interpolation using the learned correlations from
    other channels.
    pre_mean: mean value of training.
    The value of vital and mask will be changed inplace
    """
    if pre_mean is not None:
        mean_values = pre_mean
    else:
        counts = np.sum(np.sum(mask, axis=2), axis=0)
        mean_values = np.sum(np.sum(vitals*mask, axis=2), axis=0)/counts
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i, j]) == 0:
                mask[i, j, 0] = 1
                vitals[i, j, 0] = mean_values[j]
    return mean_values

def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we remove the data point as an input to the interpolation network,
    and include the predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask

def normalize_data(split_dict, norm_method='minmax'):
    """
    Normalize the data into range [0,1], the previous loaded clean_data only removes the outliers.
    Change the split_dict inplace
    TODO: This method does not consider the normalization for FF_LSTM data.
    :param data:
    :return:
    """

    if norm_method == 'minmax':
        for i, feature in enumerate(USE_FEATURES):
            min_val, max_val = MIN_MAX_VALUES[feature]
            for cohort in COHORTS:
                split_dict[cohort]["feat"][:, i, :] = (split_dict[cohort]["feat"][:, i, :] - min_val) / (max_val - min_val)
    return

set_seed(7529)
args = parse_args()
num_features = len(USE_FEATURES)
hours_from_admission = args.hours_from_admission 
norm_method = args.norm_method

data_path = os.path.join(BASE_PATH, 'Data')
vital_path = os.path.join(data_path, 'vital_data')
encounter_path = os.path.join(data_path, 'encounter_data')
model_data_path = os.path.join(data_path, 'model_data')

# 1.encounter file
encounter_f = os.path.join(encounter_path, 'encounters_6h_lastTwoYears_admit_simple.csv')
encounter = pd.read_csv(encounter_f, index_col=False)

# 2.load vital data
vital_data_path = os.path.join(vital_path, f'original_data_{hours_from_admission}h.pickle')
with open(vital_data_path, 'rb') as f:
    data = pickle.load(f)
f.close()

data_dict = generate_data(encounter, data)

# 3.load split index data
idx_path = os.path.join(model_data_path, 'idx.pickle')
with open(idx_path, 'rb') as f:
    split_encounter_id = pickle.load(f)
f.close()

indices = dict()
for cohort in COHORTS:
    index = np.asarray([data_dict["encounter_id"].index(x) for x in split_encounter_id[cohort + '_idx']])
    indices[cohort + '_index'] = index

# 4.Prepossing data
# save split original data before pre-processing, e.g., imputation and normalization; and store them in split_dict
split_dict = defaultdict(dict)

for cohort in COHORTS:
    split_org_path = os.path.join(model_data_path, "split_org")
    os.makedirs(split_org_path, exist_ok=True)
    for k, v in data_dict.items():
        tmp_data_f = os.path.join(split_org_path, "{}_{}.pickle".format(cohort, k))
        if not os.path.exists(tmp_data_f):
            tmp_data = np.asarray(v)[indices["{}_index".format(cohort)]]
            with open(tmp_data_f, 'wb') as f:
                pickle.dump(tmp_data, f)
        else:
            with open(tmp_data_f, 'rb') as f:
                tmp_data = pickle.load(f)
        split_dict[cohort][k] = tmp_data

##Mean imputation, if one vital is entirely missing, the value at the starting point will be imputed with mean value  
##randomly mask out 20% data to assess the performance of autoencoder      
train_mean = mean_imputation(split_dict["training"]["feat"], split_dict["training"]["padding_mask"], pre_mean=None)
for cohort in COHORTS:
    if cohort in ["validation", "testing"]:
        _ = mean_imputation(split_dict[cohort]["feat"], split_dict[cohort]["padding_mask"], pre_mean=train_mean)
        _ = mean_imputation(split_dict[cohort]["feat"], split_dict[cohort]["padding_mask"], pre_mean=train_mean)
    split_dict[cohort]["drop_mask"] = hold_out(split_dict[cohort]["padding_mask"])
    
##normalize dataset
processed_data_path = os.path.join(model_data_path, "split_processed")
normalize_data(split_dict)
os.makedirs(processed_data_path, exist_ok=True)
for cohort, cohort_dict in split_dict.items():
    tmp_data_f = os.path.join(processed_data_path, "{}.pickle".format(cohort))
    if not os.path.exists(tmp_data_f):
        with open(tmp_data_f, 'wb') as f:
            pickle.dump(cohort_dict, f)