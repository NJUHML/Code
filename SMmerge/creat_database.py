#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:54:22 2023

@author: veiga5
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.interpolate import griddata
import os
import glob
from itertools import zip_longest
import time
start = time.time()
def save_files(save_path, filename, dataframe):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file = os.path.join(save_path, filename)
    extension = filename.split('.')[-1]
    
    if extension == 'txt':
        dataframe.to_csv(file, index=False, sep='\t', header=False)
    elif extension == 'csv':
        dataframe.to_csv(file, index=False)
    elif extension == 'xlsx':
        dataframe.to_excel(file, index=False)
    elif extension == 'h5':
        dataframe.to_hdf(file, key='data', mode='w', format='table')
    else:
        print("Unsupported file format")
# =============================================================================
# 
start_year = 2010
end_year = 2017
# =============================================================================
DB_PATH = "/data1/user_data1/dongyf/SM_data/CN/merge/csv/database/CN"
merge_data_ORI = pd.read_hdf(f"{DB_PATH}/2010_2017_SM_db.NEW5.h5", key='data') # 20250618~
print(merge_data_ORI.columns)
ID_df = pd.read_csv(f"{DB_PATH}/ID_station.csv")
# 按照"ID"列进行排列
merge_data_ORI.sort_values(by='ID', inplace=True)
# 按照"Date"列进行排序
merge_data_ORI = merge_data_ORI.sort_values(by='Date')
# 重置行索引
merge_data_ORI.reset_index(drop=True, inplace=True)
# ---------------------------------质量控制------------------------------------
# =============================================================================
#------------------剔除观测周期小于180天的站点--------------------
# 1. 计算每个站点的观测周期
merge_data_ORI['Date'] = pd.to_datetime(merge_data_ORI['Date'])  # 确保日期列是datetime格式
# 计算每个站点的最早和最晚观测日期
site_observation_dates = merge_data_ORI.groupby('ID')['Date'].agg(['min', 'max'])
# 计算观测周期长度（天数）
site_observation_dates['observation_period'] = (site_observation_dates['max'] - site_observation_dates['min']).dt.days
# 2. 筛选出观测周期大于等于180天的站点
valid_sites = site_observation_dates[site_observation_dates['observation_period'] >= 180].index.tolist()
# 3. 根据筛选结果过滤原始DataFrame
merge_data_ORI = merge_data_ORI[merge_data_ORI['ID'].isin(valid_sites)]

# --------------Observed Frequency must Greater than 0.5-------------------
# 将"Date"列转换为日期时间类型
merge_data_ORI['Date'] = pd.to_datetime(merge_data_ORI['Date'])
# 找到每个ID的起始日期
start_dates = merge_data_ORI.groupby('ID')['Date'].min().reset_index()
# 找到每个ID的结束日期
end_dates = merge_data_ORI.groupby('ID')['Date'].max().reset_index()
# 合并起始日期和结束日期
date_range = start_dates.merge(end_dates, on='ID', suffixes=('_start', '_end'))
# 计算每个ID存在的天数
date_range['total_exist'] = (date_range['Date_end'] - date_range['Date_start']).dt.days + 1
# 计算每个ID的实际观测天数
date_range['total_obs'] = merge_data_ORI.groupby('ID').size().reset_index(name='count')['count']
# 计算观测频率
date_range['FR_ID'] = date_range['total_obs'] / date_range['total_exist']
# 根据观测频率进行筛选
filtered_data = date_range[date_range['FR_ID'] >= 0.5]
# 如果需要，你可以将筛选后的数据与原始数据合并
final_data = merge_data_ORI.merge(filtered_data['ID'], on='ID', how='inner')
# =============================================================================
merge_data = final_data.copy()
# ----------------识别缺测值------------------
# --------土壤湿度--------

cols_SM = ['OBS_SM_10cm',
           'GLDAS_Noah', 'GLDAS_CLSM','ERA5_Land', "ERA5" ,'SMC_SM', 'GDS_Monthly_SM', 'GDS_Daily_SM',
           'GLDAS_Noahlag3', 'GLDAS_CLSMlag3','ERA5_Landlag3', "ERA5lag3" ,'SMC_SMlag3', 'GDS_Monthly_SMlag3', 'GDS_Daily_SMlag3',
           'GLDAS_Noahlag7', 'GLDAS_CLSMlag7','ERA5_Landlag7', "ERA5lag7" ,'SMC_SMlag7', 'GDS_Monthly_SMlag7', 'GDS_Daily_SMlag7',
           ] 

merge_data['GDS_Daily_SM'] = merge_data['GDS_Daily_SM']*100
merge_data['GDS_Daily_SMlag3'] = merge_data['GDS_Daily_SMlag3']*100
merge_data['GDS_Daily_SMlag7'] = merge_data['GDS_Daily_SMlag7']*100
for col_SM in cols_SM:    
    merge_data.loc[(merge_data[col_SM] < 0) | (merge_data[col_SM] > 50) , col_SM] = np.nan
# --------CMFD----------
cols_CMFD = ["Prec","ET",'LST','SRF','Lrad', 'Srad','Wind', 'Shum']
#cols_CMFD = ["Prec", 'LST']
for col_CMFD in cols_CMFD:   
    merge_data.loc[(merge_data[col_CMFD] < 0) | (merge_data[col_CMFD] > 1000) , col_CMFD] = np.nan
# --------ERA5-Land----------
cols_ERA5_Land = ["SH","LH", "PET", "SUBRF"]
#cols_ERA5_Land = [ "SUBRF"]
for col_ERA5_Land in cols_ERA5_Land:   
    merge_data.loc[(abs(merge_data[col_ERA5_Land]) > 1e+10) , col_ERA5_Land] = np.nan
# --------USGS----------
merge_data.loc[(merge_data["NDVI"] < -2000) | (merge_data["NDVI"] > 10000) , "NDVI"] = np.nan
merge_data.loc[(merge_data["SLOPE"] < 0) | (merge_data["SLOPE"] > 180) , "NDVI"] = np.nan
merge_data.loc[(merge_data["DEM"] < 0) | (merge_data["DEM"] > 10000) , "DEM"] = np.nan
# --------ISSCAS----------
soil_vars = ["bd" , "btcly" , "btslt" , "btsnd" , "cec" ,"cf" , "ph" ,"soc" , "texcls" , "thickness" , "tk" , "tn" , "tp" ]
for soil_var in soil_vars: 
    merge_data.loc[(merge_data[soil_var] < 0) | (merge_data[soil_var] > 10000) , soil_var] = np.nan
merge_data.loc[(merge_data['cf'] < 0) | (merge_data['cf'] > 100) , soil_var] = np.nan
# ----------------填充缺测值 中位数填充------------------
# --------土壤湿度--------
for col_SM in cols_SM:    
    merge_data[col_SM].fillna(merge_data[col_SM].median(),inplace=True)
# --------CMFD----------
for col_CMFD in cols_CMFD: 
    merge_data[col_CMFD].fillna(merge_data[col_CMFD].median(),inplace=True)
# --------ERA5-Land----------
for col_ERA5_Land in cols_ERA5_Land: 
    merge_data[col_ERA5_Land].fillna(merge_data[col_ERA5_Land].median(),inplace=True)
# --------USGS----------
merge_data["NDVI"].fillna(merge_data["NDVI"].median(),inplace=True)
merge_data["DEM"].fillna(merge_data["DEM"].median(),inplace=True)
merge_data["SLOPE"].fillna(merge_data["SLOPE"].median(),inplace=True)
# --------ISSCAS----------
for soil_var in soil_vars: 
    merge_data[soil_var].fillna(merge_data[soil_var].median(),inplace=True)
    print(soil_var, np.min(merge_data[soil_var]), np.max(merge_data[soil_var]))
# 保存文件
Save_path = DB_PATH
#save_files(Save_path,"SM_db_allData.csv", merge_data)
save_files(Save_path,"SM_db_allData.h5", merge_data)
print(merge_data.columns)

# ******************* 提取测试集数据和训练集数据 ********************
# ------------------- 根据时序区分训练集和测试集 ------------------------
train_data = merge_data[merge_data["Date"]<"2016"]
test_data = merge_data[merge_data["Date"]>"2016"]

save_files(Save_path,"SM_db_train_by_Date.h5", train_data)
save_files(Save_path,"SM_db_test_by_Date.h5", test_data)

end = time.time()
print(f"Elapse Time: {end - start}Seconds")

