import os
BASE_PATH = os.path.dirname(os.getcwd())
USE_FEATURES = ['sbp', 'dbp', 'heartRate', 'temperature', 'spo2', 'respiratory']
COHORTS = ['training', 'validation', 'testing']
DATA_DICT_KEYS = ["feat", "time_step", "padding_mask", "encounter_id"]
MIN_MAX_VALUES = {'sbp': [20, 300], 'dbp': [5, 225], 'heartRate': [0, 300],
                  'temperature': [24, 45], 'spo2': [0, 100], 'respiratory': [0, 60]}

COHORT2SCOPE = {
    'training': 'train',
    'validation': 'valid',
    'testing': 'test'
}

LEGEND_INFO = {
    "9": "Phenotype J",
    "8": "Phenotype I",
    "7": "Phenotype H",
    "6": "Phenotype G",
    "5": "Phenotype F",
    "4": "Phenotype E",
    "3": "Phenotype D",
    "2": "Phenotype C",
    "1": "Phenotype B",
    "0": "Phenotype A"
}

PALETTE_INFO = {
    0 : "#9b59b6", 
    1 : "#3498db",  
    2 : "#8de5a1", 
    3 : "#e74c3c", 
    4 : "#34495e", 
    5 : "#2ecc71"
}

###min_metrics: all metrics that we want to minimize
METRICS = ['loss', 'ae_mse', 'delta']     # 'AKI_overall', 'mort_status_30d',
MIN_METRICS = ['loss', 'ae_mse', 'delta']     # 'AKI_overall', 'mort_status_30d',
MAX_METRICS = []
SUMMARY_ITEMS = ['lr', 'kl', 'fake_detection']
AUX_POS_WEIGHT = {''}