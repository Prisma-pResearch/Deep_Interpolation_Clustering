# Deep_Interpolation_Clustering
This is the repo for the implementation of paper 'Identifying acute illness phenotypes via deep temporal interpolation and clustering network on physiologic signatures'.

![image](https://github.com/Prisma-pResearch/Deep_Interpolation_Clustering/assets/31426497/ee517db4-4990-400b-b4c3-77971124ec8e)

## Prerequisites
* torch <br />
* tensorboardX <br />
* scikit-learn <br />
* numpy <br />
* pickle <br />
* pandas <br />
* tensorflow <br />
* seaborn <br />
* matplotlib <br />
* scipy <br />
* kneed <br />
* warmup-scheduler <br />

## Input data format
We require three inputs: <br />
* csv file: column 'encounter_deiden_id' lists the all encounter ids
* pickle file: a dictionary file records the time series vital signs within the first 24 hours (at least 7 hours) of hospital admission. For each <Key, Value> pair in the dictionary, Key represents the vital name and Value represents the dataframe of corresponding time series vital sign data.  Six vitals including 'sbp' (systolic blood pressure), 'dbp' (diastolic blood pressure), 'heartRate' (heart rate), 'temperature', 'spo2' (saturation of peripheral oxygen) and 'respiratory' (respiratory rate) were used. Each time series vital sign dataframe contains three columns: encounter_deiden_id, time_stamp, and measurement, where time_stamp represents the hours from admission, and measurement represents the vital sign value at that time stamp.
* pickle file: a dictionary file contains the ids of three cohorts (training, validation and testing). For each <Key, Value> pair in the dictionary, Key represents the cohort name and Value represents the list of encounter ids.

## How to run
* Step 1: process data. Run p0_data_process.py to format and standardize the input time series data. Run get_abnormal_vital.py to get the maximum/minimum vital values in the 7th hour. This is used for auxiliary prediction tasks. Change the path of input data in the script based on your own folder structure.<br />
   * python p0_data_process.py <br />
   * python get_abnormal_vital.py <br />
* Step 2: run p1_pretrain_main.py to pretrain the interpolation natwork to generate the feature represention of the time series data. <br />
   * python p1_pretrain_main.py
* Step 3: run p2_clustering_optK.py to help determine the optimal number of clusters. Supporting figures and scores(i.e., gap_statistic and elblow figures) are generated. Optimal number is determined manually. <br />
   * python p2_clustering_optK.py
* Step 4: run p3_clustering_main.py to jointly learn the feature representation and clustering. <br />
   * python p3_clustering_main.py
* Step 5: run p4_clustering_final.py to obtain the final cluster labels. <br />
   * python p4_clustering_final.py









