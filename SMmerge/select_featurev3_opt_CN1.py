# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:54:42 2023

@author: user
"""

# =============================================================================
# 
import os
import netCDF4 as nc
import pandas as pd
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset
import shapefile
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# =============================================================================

start = time.time()
# =============================================================================
# 


import numpy as np
import pandas as pd

def find_best_feature_set(BASIN_NAME, MODEL_NAME, INDEXthreshold, feature_score_df, all_features, score):
    """
    找到最优 score 对应的特征数量以及最优 score 值。
    
    参数：
    - feature_score_df: 包含特征数量和 score 评分的 DataFrame
    - threshold: 阈值，用于筛选有效的最优特征数量（默认为 0.001）
    
    返回：
    - min_score_x1: 最优特征数量
    - min_score_y1: 最优 score 值
    """

    # 找到最优 score 的索引
    if score=="RMSE":
        best_score_index = np.argmin(feature_score_df[f"Cross Validation Score:{score}"].values)
        best_score_x0 = feature_score_df["Number of Features"].values[best_score_index]
        best_score_y0 = feature_score_df[f"Cross Validation Score:{score}"].values[best_score_index]
        threshold =best_score_y0*INDEXthreshold
    elif score=="KGE":
        best_score_index = np.argmax(feature_score_df[f"Cross Validation Score:{score}"].values)
        best_score_x0 = feature_score_df["Number of Features"].values[best_score_index]
        best_score_y0 = feature_score_df[f"Cross Validation Score:{score}"].values[best_score_index]
        threshold =best_score_y0*INDEXthreshold

    best_score_y1 = best_score_y0
    best_score_x1 = best_score_x0
    print("ORI", best_score_y1, best_score_x1)
    # 找到与最优 score 之差小于阈值的最优特征数量
    for index in np.arange(0,best_score_index+1,1):
        print(index)
        temp_rmse_y = feature_score_df[f"Cross Validation Score:{score}"].values[index]
        print(temp_rmse_y, best_score_y0 ,threshold)
        if abs(temp_rmse_y - best_score_y0) < threshold:
            best_score_y1 = temp_rmse_y
            best_score_x1 = feature_score_df["Number of Features"].values[index]
            print("Number of Features",best_score_x1)
            break
    
    #print(best_score_x1)
    BASINlist.append(BASIN_NAME)
    MODELlist.append(MODEL_NAME)
    MinFeature.append(best_score_x1)
    BestScore.append(best_score_y1)
    FS_list = {
        "MODEL":MODELlist,
        "BASIN":BASINlist,
        f"Best features: ":MinFeature,
        f"Best Score: ":BestScore
        }
    print(FS_list)
    subset_feature = all_features[:best_score_x1]
    print(all_features)
    return FS_list ,subset_feature

def get_feature(feature_selection_strategy):
    if feature_selection_strategy =="HRFE":
        #HRFE
        print(f"feature_selection_strategy: {feature_selection_strategy}")
        all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"{MODEL_METHOD}_{CAL_IMP_METHOD}_total_imp_df.xlsx"))["Feature"].tolist()  
    elif feature_selection_strategy =="RFE":
        if CAL_IMP_METHOD=="shap":
            print(f"Feature selection strategy: {feature_selection_strategy}",f"Calculate feature importance by: {CAL_IMP_METHOD}")
            all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"{MODEL_METHOD}_shap_XGB_OPTFalse_FSFalse_importance_df.xlsx"))["Feature"].tolist() 
        # RFE
        elif CAL_IMP_METHOD=="MDI":
            print(f"Feature selection strategy: {feature_selection_strategy}",f"Calculate feature importance by: {CAL_IMP_METHOD}")
            all_features = pd.read_excel(os.path.join(FEATURE_PATH ,f"{MODEL_METHOD}_MDI_{MODEL_NAME}_importance_df.xlsx"))["Feature"].tolist() 
    return all_features
# =============================================================================
# =============================================================================
# 
import sys
import pickle
BASE_PATH = "/home/dongyf/data/SM_data/CN/merge"
DATA_PATH = "/data1/user_data1/dongyf/SM_data/CN/merge"
SCORE_PATH = os.path.join(BASE_PATH, "code/feature_selection/csv","CN")
import sys
sys.path.append(f"{BASE_PATH}/code/Library")
from MergeSM import evaluate_model_byCV, load_data, evaluate_features_serial,evaluate_features_mpi, set_dynamic_yticks, get_DEFmodel, save_files
BASIN_NAME = "CN"
MODELS = ["LGBM", "XGB", "CB", "RF"]
feature_selection_strategy = "RFE"
# Constants
SOIL_DEPTH = "10cm"
OUTPUT_FOLDER = "default"
MODEL_METHOD = "multiple_model"
CAL_IMP_METHOD = "shap"
SCORE = 'KGE'
IFopt = False
SCALE_FACTOR = 0.01
NCPU = -1
DataFRACT = 1
INDEXthreshold = 0.005
BASINlist = []
MODELlist = []
MinFeature = []
BestScore = []
# for feature_selection_strategy in ["RFE","HRFE"]:
for feature_selection_strategy in ["RFE"]:
    FS_strategy_df = pd.DataFrame()
    feature_score_df=pd.DataFrame()
    
    print(f"*******************{BASIN_NAME}*******************")
    for i, MODEL_NAME in enumerate(MODELS):
        print(f"#----------------{MODEL_NAME}---------------#")
        DB_PATH = os.path.join(DATA_PATH , "csv", "database","CN")
        SAVE_PATH = os.path.join(DATA_PATH, "train_output", SOIL_DEPTH, f"CN/output_{OUTPUT_FOLDER}")
        TRAIN_DATA_FILENAME = f"SM_db_train_by_Date.h5"
        FEATURE_PATH = f"{SAVE_PATH}/FeatureSelection"
        STUDY_PATH = os.path.join(SAVE_PATH, "Optuna")
        STUDY_FILENAME = f'{BASIN_NAME}_{MODEL_NAME}_{feature_selection_strategy}_shap_study.pkl'
        with open(os.path.join(STUDY_PATH, STUDY_FILENAME), 'rb') as f:
            study = pickle.load(f)
        print(SAVE_PATH)
        # 读取数据库
        train_data = load_data(TRAIN_DATA_FILENAME, DB_PATH, DataFRACT)
        print(train_data.columns)
        # 获取特征
        all_features = get_feature(feature_selection_strategy)
        all_columns = ['ID',  "Date" , f"OBS_SM_{SOIL_DEPTH}"  ] + all_features
        print((train_data.columns), all_features)
        db_df = train_data.reindex(columns = all_columns)
        basic_columns = 3 #开始迭代的特征数量
        # -----------------------------评估不同特征子集的表现-----------------------
        feature_list = evaluate_features_serial(MODEL_NAME, BASIN_NAME,  study, IFopt, train_data, all_features, basic_columns, target=f'OBS_SM_{SOIL_DEPTH}', score =SCORE, scale_factor=SCALE_FACTOR, kf=3, n_cpu =NCPU)
        MODEL_feature_score_df = pd.DataFrame(feature_list)
        feature_score_df =  pd.concat([feature_score_df, MODEL_feature_score_df], ignore_index=True)
        print(feature_score_df)
        # ------------------------------找到最优值点的坐标-------------------------
        FS_list ,subset_feature= find_best_feature_set(BASIN_NAME, MODEL_NAME,INDEXthreshold, feature_score_df, all_features,score =SCORE)
        # ---------------------添加到列表--------------------------------
        FS_strategy_df = pd.DataFrame(FS_list)
        print(pd.DataFrame(FS_list))
        subset_feature_df = pd.DataFrame(subset_feature, columns=['Feature'])
        print(subset_feature_df)
        save_files(os.path.join(FEATURE_PATH,"csv"), f"{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{IFopt}_subset_feature.csv", subset_feature_df)
        save_files(SCORE_PATH, f"{feature_selection_strategy}_{CAL_IMP_METHOD}_{MODEL_NAME}_{BASIN_NAME}_{IFopt}_feature_score.csv", feature_score_df)
        


end = time.time()
print(f"Elapse Time: {end - start}Seconds")
