# Deep_Interpolation_Clustering
This is the repo for the implementation of paper 'Identifying acute illness phenotypes via deep temporal interpolation and clustering network on physiologic signatures'.

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
* pickle file: a dictionary file records the time series vital signs within the first 24 hours (at least 7 hours) of hospital admission. For each <Key, Value> pair in the dictionary, Key represents the vital name and Value represents the dataframe of corresponding time series vital sign data.  
  Six vitals including 'sbp' (systolic blood pressure), 'dbp' (diastolic blood pressure),
  'heart'











