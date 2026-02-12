#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:06:46 2023

@author: veiga5
"""

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# =============================================================================
# 
# -----------------------------------main函数-------------------------------

import time
start = time.time()    
BASE_PATH = "/home/dongyf/data/SM_data/CN/merge"
DATA_PATH = "/data1/user_data1/dongyf/SM_data/CN/merge"
OUTPUT_FOLDER = "default"
feature_index = 1
NTRIAL=50

SCALE_FACTOR =0.01
DataFRACT = 1
SCORE = 'KGE'
feature_selection_strategy = "RFE"
CAL_IMP_METHOD = 'shap'
CAL_STATE = True
SOIL_DEPTH = "10cm"
BASIN_NAME = "CN"
MODELS = [ "LGBM", "XGB", "CB", "RF"]
modelNCPU,optunaNCPU = 10,10
# =============================================================================
import sys
sys.path.append(f"{BASE_PATH}/code/Library")
from MergeSM import save_files, save_model_scaler, calculate_metrics, get_DEFmodel, preprocess_features,evaluate_model_byCV,suggest_hyperparameters,objective,optuna_MLmodel,load_data
score_df = pd.DataFrame()
MODEL_NAME = "XGB" 
for MODEL_NAME in MODELS:
    print(f"#----------------{MODEL_NAME}---------------#")
    DB_PATH = os.path.join(DATA_PATH , "csv", "database","CN")
    SAVE_PATH = os.path.join(DATA_PATH, "train_output", SOIL_DEPTH, f"CN/output_{OUTPUT_FOLDER}")
    TRAIN_DATA_FILENAME = f"SM_db_train_by_Date.h5"
    FEATURE_PATH = f"{SAVE_PATH}/FeatureSelection"
    OPTUNA_PATH = f"{SAVE_PATH}/Optuna"
    # 按照"Date"列进行排序
    train_data = load_data(TRAIN_DATA_FILENAME, DB_PATH, DataFRACT)
    target = f'OBS_SM_{SOIL_DEPTH}'
    # -----------------------------------特征选择---------------------------------
    features = pd.read_csv(os.path.join(FEATURE_PATH, "csv", f"{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{CAL_STATE}_subset_feature.csv"))["Feature"].tolist()
    # 重命名规则
    rename_dict = {
        'ERA5': 'ERA5_SM',
        'ERA5_Land': 'ERA5_Land_SM',
        'GLDAS_Noah': 'GLDAS_Noah_SM'
    }

    # 使用列表推导式进行重命名
    print(features)
    # -------------------------------提取特征和目标变量-------------------------------
    X_train_scaled, y_train, scaler = preprocess_features(train_data, features, target, SCALE_FACTOR)
    # ---------------------------------超参数优化-------------------------------
    study, optuna_score, DEFscore = optuna_MLmodel(X_train_scaled, y_train, SCORE, BASIN_NAME, MODEL_NAME, NTRIAL, modelNCPU,optunaNCPU)
    template_score_df_list = {
            "BASIN":BASIN_NAME,
            "MODEL":MODEL_NAME,
            "Optuna Score":optuna_score,
            "Def Score":DEFscore
    }
    print(template_score_df_list)
    template_score_df = pd.DataFrame([template_score_df_list])
    score_df = pd.concat([score_df, template_score_df], ignore_index=True)
    # -----------------------------保存history-----------------------------------
    # ...（其余代码略）
    data = []
    for trial in study.trials:
        data.append([trial.number, trial.value])
    optuna_history_df = pd.DataFrame(data, columns=['Iteration', f'Objective Score: {SCORE}'])
    # 将DataFrame存储为CSV文件
    save_files(os.path.join(OPTUNA_PATH, "csv"), f'{BASIN_NAME}_{MODEL_NAME}_{feature_selection_strategy}_{CAL_IMP_METHOD}_optimization_history_{SCORE}.csv', optuna_history_df)    
    # -----------------------------保存study---------------------------------------
    study_file = os.path.join(OPTUNA_PATH , f'{BASIN_NAME}_{MODEL_NAME}_{feature_selection_strategy}_{CAL_IMP_METHOD}_study.pkl')
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
save_files(os.path.join(BASE_PATH, "code/optuna_code", "csv"), f"CN1_optuna_metric.csv", score_df)   

end = time.time()
print(f"Elapse Time: {end - start}Seconds")

