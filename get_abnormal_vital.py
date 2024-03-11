# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:57:53 2022

@author: renyuanfang
"""

import os
import pandas as pd
import numpy as np
import pickle
from info import BASE_PATH, USE_FEATURES, COHORTS, MIN_MAX_VALUES
import random  
import tensorflow as tf
import argparse

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

###get the 12 hour dataset
set_seed(7529)
args = parse_args()
num_features = len(USE_FEATURES)
hours_from_admission = args.hours_from_admission 
norm_method = args.norm_method

data_path = os.path.join(BASE_PATH, 'Data')
vital_path = os.path.join(data_path, 'vital_data')
encounter_path = os.path.join(data_path, 'encounter_data')

# 1.encounter file
encounter_f = os.path.join(encounter_path, 'encounters_6h_lastTwoYears_admit_simple.csv')
encounter = pd.read_csv(encounter_f, index_col=False)

vital_data_path = os.path.join(vital_path, 'original_data_24h.pickle')
with open(vital_data_path, 'rb') as f:
    data = pickle.load(f)
f.close()

next_hour_data = {}
for vital in data.keys():
    temp_data = data[vital]
    temp_data = temp_data[(temp_data['time_stamp'] >= args.hours_from_admission) & (temp_data['time_stamp'] < args.hours_from_admission + 1)]
    next_hour_data[vital] = temp_data

##sbp 
sbp = next_hour_data['sbp'].groupby(['encounter_deiden_id'], as_index=False)['measurement'].min()
dbp = next_hour_data['dbp'].groupby(['encounter_deiden_id'], as_index=False)['measurement'].min()
spo2 = next_hour_data['spo2'].groupby(['encounter_deiden_id'], as_index=False)['measurement'].min()

temperature = next_hour_data['temperature'].groupby(['encounter_deiden_id'],as_index=False)['measurement'].max()
heartRate = next_hour_data['heartRate'].groupby(['encounter_deiden_id'],as_index=False)['measurement'].max()
respiratory = next_hour_data['respiratory'].groupby(['encounter_deiden_id'],as_index=False)['measurement'].max()

for vital, df in zip(next_hour_data.keys(), [sbp, dbp, heartRate, temperature, respiratory, spo2]):
    df = df.rename(columns = {'measurement' : vital})
    encounter = encounter.merge(df, on='encounter_deiden_id', how='left')
    
for vital in MIN_MAX_VALUES.keys():
    min_v, max_v = MIN_MAX_VALUES[vital]
    encounter[vital] = (encounter[vital] - min_v) / (max_v - min_v)

encounter.to_csv(os.path.join(vital_path, 'next_hour_abnormal_norm_val.csv'), index=False)

