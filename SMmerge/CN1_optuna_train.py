#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import pickle
from datetime import datetime, timedelta
# Constants
BASE_PATH = "/home/dongyf/data/SM_data/CN/merge"
DATA_PATH = "/data1/user_data1/dongyf/SM_data/CN/merge"
OUTPUT_FOLDER = "default"

SCALE_FACTOR = 0.01
NCPU = -1
feature_selection_strategy = "RFE"
CAL_IMP_METHOD = "shap"
CAL_STATE = "True"
IF_FS = False
SCORE="KGE"
SOIL_DEPTH = "10cm"
BASIN_NAME = "CN"
MODELS = ["CB", "LGBM", "XGB",  "RF"]
MODELS = ["LGBM"]
import sys
sys.path.append(f"{BASE_PATH}/code/Library")
from MergeSM import save_files, save_model_scaler, calculate_metrics, get_DEFmodel, preprocess_features, evaluate_model_byCV, get_OPTUNAmodel, load_data, train_model
def main():
    start = time.time()
    for MODEL_NAME in MODELS:
        print(f"#----------------{MODEL_NAME}---------------#")
        print(f"#----------------{SOIL_DEPTH}---------------#")
        # =============================================================================
        # File paths
        DB_PATH = os.path.join(DATA_PATH , "csv", "database","CN")
        SAVE_PATH = os.path.join(DATA_PATH, "train_output", '10cm', f"CN/output_{OUTPUT_FOLDER}")
        STUDY_PATH = os.path.join(SAVE_PATH, "Optuna")
        FEATURE_PATH = f"{SAVE_PATH}/FeatureSelection"
        TRAIN_DATA_FILENAME = "SM_db_train_by_Date.h5"
        TEST_DATA_FILENAME = "SM_db_test_by_Date.h5"
        TEST_COMPARISON_FILENAME = f"SM_comparison_test.csv"
        STUDY_FILENAME = f'{BASIN_NAME}_{MODEL_NAME}_{feature_selection_strategy}_{CAL_IMP_METHOD}_study.pkl'
        #STUDY_FILENAME = f'{BASIN_NAME}_{MODEL_NAME}_{feature_selection_strategy}_{CAL_IMP_METHOD}_study.StandardScaler.pkl'
        print("SAVE_PATH", SAVE_PATH)
        # =============================================================================
    
        # Load data
        train_data = load_data(TRAIN_DATA_FILENAME, DB_PATH, datafrac=1, LST_threshold=None)
        test_data = load_data(TEST_DATA_FILENAME, DB_PATH, datafrac=1, LST_threshold=None)
        print(DB_PATH, TRAIN_DATA_FILENAME)
        with open(os.path.join(STUDY_PATH, STUDY_FILENAME), 'rb') as f:
            study = pickle.load(f)
            print(os.path.join(STUDY_PATH, STUDY_FILENAME))
            print(study.best_params)
        # Feature selection
        # 
        if IF_FS:
            features = pd.read_csv(os.path.join(FEATURE_PATH, "csv", f"{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{CAL_STATE}_subset_feature.csv"))["Feature"].tolist()
        else:
            # 预训练：使用全部特征，超参数优化后的超参数集合训练模型。用于计算特征重要性排序
            features =[
        'ERA5','ERA5_Land', 'GLDAS_Noah','GLDAS_CLSM',  'SMC_SM', 'GDS_Monthly_SM', 'GDS_Daily_SM',
        'GLDAS_Noahlag3', 'GLDAS_CLSMlag3','ERA5_Landlag3', "ERA5lag3" ,'SMC_SMlag3', 'GDS_Monthly_SMlag3', 'GDS_Daily_SMlag3',
        'GLDAS_Noahlag7', 'GLDAS_CLSMlag7','ERA5_Landlag7', "ERA5lag7" ,'SMC_SMlag7', 'GDS_Monthly_SMlag7', 'GDS_Daily_SMlag7',
        'Prec', 'ET', 'LST', 'SRF','Lrad', 'Srad', 'Wind', 'Shum','SUBRF','PET','SH','LH',
        'NDVI', 'DEM','SLOPE',
        'bd', 'btcly', 'btslt', 'btsnd', 'cec', 'cf', 'ph', 'soc', 'texcls','thickness', 'tk', 'tn', 'tp'
        ]
        rename_dict = {
            'ERA5': 'ERA5_SM',
            'ERA5_Land': 'ERA5_Land_SM',
            'GLDAS_Noah': 'GLDAS_Noah_SM'
        }

    # 使用列表推导式进行重命名
        #features = [rename_dict.get(feature, feature) for feature in features]  

        target = f'OBS_SM_{SOIL_DEPTH}'
        print(features)
        # Preprocess data
        X_train_scaled, y_train, scaler = preprocess_features(train_data, features, target, SCALE_FACTOR)
        DEFmodel = get_DEFmodel(MODEL_NAME, n_cpu=NCPU, SimpleModel=False)
        OPTUNAmodel = get_OPTUNAmodel(MODEL_NAME, study, n_cpu=NCPU)
        # Train and cross-validation model
        DEFscore = evaluate_model_byCV(X_train_scaled, y_train,  DEFmodel, score =SCORE,KF=3,n_cpu =NCPU)
        OPTUNAscore = evaluate_model_byCV(X_train_scaled, y_train,  OPTUNAmodel, score =SCORE,KF=3,n_cpu =NCPU)
        
        print(f"start train ML {MODEL_NAME} model based on default hyperparameter combination ......")
        StartTime = time.time()
        DEFmodel = train_model(X_train_scaled, y_train,  DEFmodel)
        EndTime = time.time()
        TimeFormatted= str(timedelta(seconds=EndTime - StartTime))
        print(f"Training the ML model {MODEL_NAME} on the train set took {TimeFormatted}. Training Set Size: {len(X_train_scaled)}")
        
        print(f"start train ML {MODEL_NAME} model based on Opt. hyperparameter combination ...... ")
        StartTime = time.time()
        OPTUNAmodel = train_model(X_train_scaled, y_train,  OPTUNAmodel)
        EndTime = time.time()
        TimeFormatted= str(timedelta(seconds=EndTime - StartTime))
        print(f"Training the ML model {MODEL_NAME} on the train set took {TimeFormatted}. Training Set Size: {len(X_train_scaled)} .")
        
        
        # Predict and evaluate on test set
        X_test = test_data[features]
        y_test = test_data[target] * SCALE_FACTOR
        X_test_scaled = scaler.transform(X_test)
        
        # Predict and evaluate on train set
        print(f"start predicte ML model {MODEL_NAME} based on default hyperparameter combination ......")
        StartTime = time.time()
        DEFpredicted_test = DEFmodel.predict(X_test_scaled)
        EndTime = time.time()
        TimeFormatted= str(timedelta(seconds=EndTime - StartTime))
        print(f"Predicte the ML model {MODEL_NAME} took {TimeFormatted}. Testing Set Size: {len(X_test_scaled)} .")

        print(f"start predicte ML model {MODEL_NAME} based on Opt. hyperparameter combination ......")
        StartTime = time.time()
        OPTUNApredicted_test = OPTUNAmodel.predict(X_test_scaled)
        EndTime = time.time()
        TimeFormatted= str(timedelta(seconds=EndTime - StartTime))
        print(f"Predicte the ML model {MODEL_NAME} took {TimeFormatted}. Testing Set Size: {len(X_test_scaled)} .")        
        
        print(f"*******************DEF************************")
        calculate_metrics(DEFpredicted_test, y_test)
        print(f"*****************OPTUNA************************")
        calculate_metrics(OPTUNApredicted_test, y_test)
        print("OPTUNA score: ", OPTUNAscore, "DEF score: ", DEFscore)   
        if OPTUNAscore > DEFscore:
            print(f"**************************************{BASIN_NAME}**************************************")
            
            print("The results of optuna hyperparameter optimization are better than the default model:", MODEL_NAME)
            save_model_scaler(os.path.join(SAVE_PATH, "model"), f"{MODEL_NAME}_{CAL_IMP_METHOD}_OPT_FS{IF_FS}", OPTUNAmodel, scaler)
            test_comparison = pd.DataFrame({'Predicted': OPTUNApredicted_test, 'Actual': y_test})
            save_files(os.path.join(SAVE_PATH, "csv"), TEST_COMPARISON_FILENAME, test_comparison)
        elif OPTUNAscore < DEFscore:
            # Save model and results
            save_model_scaler(os.path.join(SAVE_PATH, "model"), MODEL_NAME, DEFmodel, scaler)
            test_comparison = pd.DataFrame({'Predicted': DEFpredicted_test, 'Actual': y_test})
            save_files(os.path.join(SAVE_PATH, "csv"), TEST_COMPARISON_FILENAME, test_comparison)
            
        # Save merged test file
        merged_test = test_data.copy()
        merged_test[MODEL_NAME] = test_comparison["Predicted"]
        
        if IF_FS: 
            save_files(os.path.join(SAVE_PATH, "csv"), f"merged_test_{MODEL_NAME}_{feature_selection_strategy}_{CAL_IMP_METHOD}_{SOIL_DEPTH}.csv", merged_test)
        else:
            pass # 预训练模型不保存预测数据集
        end = time.time()
        print(f"Elapsed Time: {end - start} seconds")

if __name__ == "__main__":
    main()
