# SHL2023

## Directory Structure

```
./main Functions are given for data generation, for calculating the results of the test set we constructed and for evaluating the metrics, and for obtaining the test files for the original test set.

./notebooks  Files for model optimization, including feature selection, data enhancement, etc  are given here.

./road_network  Documents used to derive road-network features.

./traing_result  The obtained training results are placed here. 

./utils It includes data pre-processing, calculation of time and frequency domain features, and the integration of road network features,based on these three types of base features, window features and change features are calculated. Finally, the construction of an integrated machine learning model is given.
```

## Getting started

### Requirement:

* Ubuntu (we have tested Ubuntu 18.04)
* python3,
* klearn.model_selection, sklearn.ensemble, sklearn.tree, lightgbm, xgboost, catboost

## Running Example:



```
python3 ./main/data_generate.py
python3 ./main/get_result_all.py
python3 ./main/get_result_all_win.py
python3 ./main/get_result_all_feture_selection.py
python3 ./main/get_result_all_without_road_feature.py
python3 ./main/get_testdata_result.py
```

