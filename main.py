# -*- coding: utf-8 -*-
# @Author: zhushuai
# @Date:   2019-04-02 11:53:48
# @Last Modified by:   zhushuai
# @Last Modified time: 2019-04-02 13:11:40
import os
import numpy as np 
import pandas as pd 
import xgboost as xgb 
import lightgbm as lgb 
import catboost 
from catboost import CatBoostRegressor, Pool
from collections import Counter
from sklearn.metrics import mean_absolute_error

from betaencoder import BetaEncoder


# 设置一些文件的路径
train_path = "../origin/Metro_train"
test_path = "../origin/Metro_testA"

test_file = "testA_record_2019-01-28.csv"
sub_file = "testA_submit_2019-01-29.csv"
roadfile = "../origin/Metro_roadMap.csv"



# 数据清洗
# 表示2019年本应该是工作日，却是节假日的情况
w2h = ["1-1", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "2-10", "4-5", "5-1", "6-7", "9-13",
  "10-1", "10-2", "10-3", "10-4", "10-5", "10-6", "10-7"]
# 表示本应该是节假日，调休为工作日的情况
h2w = ["2-2", "2-3", "9-29", "10-12"]

def get_base_features(df_):
    train_df = df_.copy()
    
    # 加入日期的信息
    train_df['day'] = train_df['time'].apply(lambda x: int(x[8:10]))
    train_df['minute'] = train_df['time'].apply(lambda  x: int(x[14:15] + '0'))

    # 将时间对应的列转换为pandas可以处理的类型
    train_df['time'] = pd.to_datetime(train_df['time'])
    # 得到月的信息
    train_df['month'] = train_df['time'].dt.month
    # 得到对应的周次、小时和分钟
    train_df['weekday'] = train_df['time'].dt.weekday + 1
    #train_df['weekend'] = (train_df.weekday >=6 ).astype(int)
    train_df['hour'] = train_df['time'].dt.hour
    date = str(train_df.month.values[0])+'-'+str(train_df.day.values[0])
    if date in w2h or (date not in h2w and train_df.weekday.values[0] in [6,7]):
        train_df['is_holiday'] = 1
    else:
        train_df['is_holiday'] = 0
    
    train_final = train_df.groupby(['stationID', 'weekday', 'is_holiday', 'day', 'hour', 'minute']).status.agg(['count', 'sum']).reset_index()
    # 这一段参考yu的代码
    # 考虑和刷卡设备相关的特征
    # 每个站点的刷卡设备数量与该站点的人流量存在一定的关系
    # 每个时段的刷卡设备数与流量也存在一定的关系
    tmp = train_df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    train_final = train_final.merge(tmp, on=['stationID'], how='left')
    tmp = train_df.groupby(['stationID','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    train_final = train_final.merge(tmp, on=['stationID', 'hour'], how='left')
    tmp = train_df.groupby(['stationID','hour','minute'])['deviceID'].nunique().\
                                           reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    train_final  = train_final.merge(tmp, on=['stationID','hour','minute'], how='left')
    
    train_final['time_cut'] = train_final['hour'] * 6 + train_final['minute'] // 10
    train_final['inNums'] = train_final['sum']
    train_final['outNums'] = train_final['count'] - train_final['sum']
    del train_final['sum'], train_final['count']
    
    return train_final

# 特征工程

## 增加同站点相邻时间段的信息
def add_neighbor_time(df_):
    train_df = df_.copy()  
    # 生成一个用于中间转换的DataFrame
    train_now = train_df[['stationID', 'day', 'time_cut', 'inNums', 'outNums']]
    
    train_df.rename(columns={'inNums': 'inNums_now', 'outNums': 'outNums_now'}, inplace=True)
    
    # 考虑前多少个时间段，默认考虑前两个时间段
    for i in range(2, 0, -1):
        train_before = train_now.copy()
        train_before['time_cut'] = train_before['time_cut'] + i
        train_df = train_df.merge(train_before, how='left', on = ["stationID","day", "time_cut"])
        train_df.rename(columns = {'inNums': f'inNums_before{i}', 'outNums': f'outNums_before{i}'}, inplace=True)
        train_df.fillna(0, inplace=True)
    
		# 考虑后多少个时间段，默认考虑前两个时间段
    for j in range(2, 0, -1):
        train_after = train_now.copy()
        train_after['time_cut'] = train_after['time_cut'] - j
        train_df = train_df.merge(train_after, how='left', on = ["stationID", "day", "time_cut"])
        train_df.rename(columns = {'inNums': f'inNums_after{j}', 'outNums': f'outNums_after{j}'}, inplace=True)
        train_df.fillna(0, inplace=True)
    
    return train_df

## 增加乘车高峰时段相关的特征

def add_peak_type(df_):
    
    train_df = df_.copy()
    
    ### 7:00-9:00、17:00-19:00为高峰时段，7:30-8:30、17:30-18:30为特殊高峰
    w_time = ['6:00', '7:00', '7:30', '8:30', '9:00', '17:00', '17:30', '18:30', '19:00', '0:00']

    w_time_cut = [10*pd.to_datetime(ti).hour+pd.to_datetime(ti).minute//10 for ti in w_time]

    # 节假日的高峰时间段
    h_time = ['6:00','7:00', '9:00', '12:00', '15:00', '18:00', '20:00', '0:00']

    h_time_cut = [10*pd.to_datetime(ti).hour+pd.to_datetime(ti).minute//10 for ti in h_time]

    # 每个时间点对应的time_cut
    temp_array = {'temp': [i*10+j for i in range(24) for j in range(6)], 'time_cut': list(range(0, 144))}

    # 工作日
    ## 早高峰的开始和结束时间
    ### 运营开始时间
    w_start = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[0])]

    mp_start_1 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[1])]
    mp_end_1 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[2])]

    mp_start_2 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[3])]
    mp_end_2 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[4])]

    ## 早特殊高峰开始结束时间
    msp_start = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[2])]
    msp_end = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[3])]

    ## 晚高峰开始和结束时间
    ap_start_1 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[5])]
    ap_end_1 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[6])]

    ap_start_2 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[7])]
    ap_end_2 = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[8])]

    ## 晚特殊高峰开始结束时间
    asp_start = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[6])]
    asp_end = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[7])]

    w_end = temp_array['time_cut'][temp_array['temp'].index(w_time_cut[9])]

    ## 工作日peak类型映射表
    ### 0表示停运期间，1表示运行开始到高峰前和高峰后到运营结束，2表示高峰时间，3表示特殊高峰时间

    w_peak = {0: list(range(w_end, w_start)),
            2: list(range(mp_start_1, mp_end_1))
              + list(range(mp_start_2, mp_end_2))
              + list(range(ap_start_1, ap_end_1))
              + list(range(ap_start_2, ap_end_2)),
              3: list(range(msp_start, msp_end))
              + list(range(asp_start, asp_end))}

    peak_workday = {}

    for key, value in w_peak.items():
        for t in value:
            peak_workday[t] = key


    # 节假日

    ## 早高峰的开始和结束时间

    h_start = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[0])]
    h_end = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[7])]

    h_mp_start_1 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[1])]
    h_mp_end_1 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[2])]

    h_mp_start_2 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[3])]
    h_mp_end_2 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[4])]

    ## 早特殊高峰开始结束时间
    h_msp_start = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[2])]
    h_msp_end = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[3])]

    ## 晚高峰开始和结束时间
    h_ap_start_1 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[5])]
    h_ap_end_1 = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[6])]

    ## 晚特殊高峰开始结束时间
    h_asp_start = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[4])]
    h_asp_end = temp_array['time_cut'][temp_array['temp'].index(h_time_cut[5])]

    ## 节假日日peak类型映射表

    h_peak = {0: list(range(h_end, h_start)),
            2: list(range(h_mp_start_1, h_mp_end_1))
              + list(range(h_mp_start_2, h_mp_end_2))
              + list(range(h_ap_start_1, h_ap_end_1)) ,
              3: list(range(h_msp_start, h_msp_end))
              + list(range(h_asp_start, h_asp_end))}

    peak_holiday = {}

    for key, value in h_peak.items():
        for t in value:
            peak_holiday[t] = key
		
    # 转换成用于连接的DataFrame
    wo_peak = pd.DataFrame({'time_cut':list(peak_workday.keys()) , 'peak_type': list(peak_workday.values()) })
    wo_peak['is_holiday'] = 0

    ho_peak = pd.DataFrame({'time_cut': list(peak_holiday.keys()), 'peak_type': list(peak_holiday.values())})
    ho_peak['is_holiday'] = 1

    to_peak = pd.concat([wo_peak, ho_peak], axis=0, sort=False)


    
    train_df = train_df.merge(to_peak, on=['is_holiday', 'time_cut'], how='left')
    train_df['peak_type'].fillna(1, inplace=True)
    
    return train_df

## 增加同周次进出站流量的信息
def add_week_flow(df_):
    
    train_df = df_.copy()
    
    tmp = train_df.groupby(['stationID','weekday','hour','minute'], as_index=False)['inNums_now'].agg({
                                                                        'inNums_whm_max'    : 'max',
                                                                        'inNums_whm_min'    : 'min',
                                                                        'inNums_whm_mean'   : 'mean'
                                                                            })
    train_df = train_df.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

    tmp = train_df.groupby(['stationID','weekday','hour','minute'], as_index=False)['outNums_now'].agg({
                                                                            'outNums_whm_max'    : 'max',
                                                                            'outNums_whm_min'    : 'min',
                                                                            'outNums_whm_mean'   : 'mean'
                                                                            })
    train_df = train_df.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

    tmp = train_df.groupby(['stationID','weekday','hour'], as_index=False)['inNums_now'].agg({
                                                                            'inNums_wh_max'    : 'max',
                                                                            'inNums_wh_min'    : 'min',
                                                                            'inNums_wh_mean'   : 'mean'
                                                                            })
    train_df = train_df.merge(tmp, on=['stationID','weekday','hour'], how='left')

    tmp = train_df.groupby(['stationID','weekday','hour'], as_index=False)['outNums_now'].agg({
                                                                            #'outNums_wh_max'    : 'max',
                                                                            #'outNums_wh_min'    : 'min',
                                                                            'outNums_wh_mean'   : 'mean'
                                                                            })
    train_df = train_df.merge(tmp, on=['stationID','weekday','hour'], how='left')
    
    return train_df

## 增加线路信息
def add_line(df_):
    def station_line(record):
        station_line = record[['lineID', 'stationID']]
        station_line = station_line.drop_duplicates().reset_index(drop=True)
        station_line = station_line.sort_values(by='stationID').reset_index(drop=True)
        return station_line

    train_df = df_.copy()

    # 这里的file表示原始文件
    train_files = [f for f in sorted(os.listdir(train_path)) if f.endswith("csv")]
    record = pd.read_csv(os.path.join(train_path, train_files[0]))
    line = station_line(record)
    line_pad = pd.DataFrame(Counter(line['lineID']), index=line.lineID.unique())
    line_ID = (line_pad.T)['B'].reset_index().rename(columns={'index': 'lineID', 'B': 'line'})
    line = line.merge(line_ID, on='lineID', how='left')
    line.drop('lineID', axis=1, inplace=True)
    train_df = train_df.merge(line, on=['stationID'], how='left')
    return train_df

## 增加车站类型信息
def add_station_type(df_):
    
    def get_map(roadmap):
        roadmap.rename(columns={"Unnamed: 0": 'stationID'}, inplace=True)
        tmp = roadmap.drop(['stationID'], axis=1)
        roadmap['station_type'] = tmp.sum(axis=1)
        return roadmap[['stationID', 'station_type']]
    
    
    roadmap = pd.read_csv(roadfile)
    map_pad = get_map(roadmap)
    
    train_df = df_.copy()
    train_df = train_df.merge(map_pad, on=['stationID'], how='left')
    return train_df

## 增加特征站点标记

def add_special_station(df_):
    train_df = df_.copy()
    
    train_df['is_special'] = np.nan
    train_df.loc[train_df['stationID']==15, 'is_special'] = 1 
    train_df['is_special'].fillna(0, inplace=True)
    return train_df


## 连接训练特征和目标值
# 定义增加周末信息的函数
def add_is_holiday(test):
    date = str(test.month.values[0])+'-'+str(test.day.values[0])
    if date in w2h or (date not in h2w and test.weekday.values[0] in [6,7]):
        test['is_holiday'] = 1
    else:
        test['is_holiday'] = 0
    return test


# 获取最终的测试日期数据
def read_test(test):
    test['weekday']    = pd.to_datetime(test['startTime']).dt.dayofweek + 1
    #test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >=5).astype(int)
    test['month'] = pd.to_datetime(test['startTime']).dt.month
    test['day']     = test['startTime'].apply(lambda x: int(x[8:10]))
    test['hour']    = test['startTime'].apply(lambda x: int(x[11:13]))
    test['minute']  = test['startTime'].apply(lambda x: int(x[14:15]+'0'))
    test = test.drop(['startTime', 'endTime'], axis=1)

    test = add_is_holiday(test)
    test['time_cut'] = test['hour'] * 6 + test['minute'] // 10
    test.drop(['inNums', 'outNums', 'month'], axis=1, inplace=True)
    return test

# 用于对weekday的信息进行修正
def fix_weekday(w):
    if w == 7:
        return 1 
    else:
        return w+1

# 合并得到训练集和测试集的数据
def merge_train_test(df_):
    train_df = df_.copy()
    
    # 读取最后要提交的数据
    test = pd.read_csv(os.path.join(test_path, sub_file))
    test_df = read_test(test)
    all_data = pd.concat([train_df, test_df], axis=0, sort=False)

    # 将当天对应的节假日信息取出备用
    th = all_data[['day', 'is_holiday']].drop_duplicates().sort_values(by=['day']).reset_index(drop=True).rename(columns={'is_holiday': 'today_is_holiday'})
    # 提取用于合并为target的部分
    train_target = all_data[['stationID','day', 'hour','minute', 'time_cut', 'inNums_now', 'outNums_now']]

    train_target.rename(columns={'inNums_now': 'inNums', 'outNums_now': 'outNums'}, inplace=True)
    # 将所有数据的节假日信息名称改为前一天的节假日信息
    all_data.rename(columns={'is_holiday': 'yesterday_is_holiday'}, inplace=True)
    # 为了之后合并，将day的特征加1
    all_data['day'] += 1
    # 对周的信息进行修正
    all_data['weekday'] = all_data['weekday'].apply(fix_weekday)
    
    # 需要将训练集和测试集单独合并

    train_df = all_data[(all_data.day != 29) & (all_data.day != 26) & (all_data.day != 30)]

    test_df = all_data[all_data.day == 29]
    
    # 首先对生成训练集数据
    train_df = train_df.merge(train_target, on=['stationID', 'day', 'hour', 'minute', 'time_cut'], how='left')
    ## 对预测目标值的缺失值补0
    train_df['inNums'].fillna(0, inplace=True)
    train_df['outNums'].fillna(0, inplace=True)
    
    ## 补充当天是否周末的信息
    train_df = train_df.merge(th, on='day', how='left')
    
    # 生成测试集的数据
    test_target = train_target[train_target.day == 29]
    # 对测试值进行连接
    test_df = test_df.merge(test_target, on=['stationID', 'day', 'hour', 'minute', 'time_cut'], how='outer')
    test_df = test_df.sort_values(by=['stationID', 'hour', 'minute']).reset_index(drop=True)
    use_fe = ['stationID', 'weekday', 'yesterday_is_holiday', 'nuni_deviceID_of_stationID', 'line', 'station_type', 'is_special']
    test_merge = test_df[use_fe].drop_duplicates().dropna()
    test_df = test_df.drop(use_fe[1:], axis=1)
    test_df = test_df.merge(test_merge, on=['stationID'], how='left')
    
    test_df = test_df.merge(th, on='day', how='left')
    test_df.fillna(0, inplace=True)
    all_data = pd.concat([train_df, test_df], axis=0, sort=False)
    return all_data, train_df

# 考虑进出站流量和时间段密切相关
# 针对time_cut进行目标编码
def target_encoding(all_data, train_df):
    # 设置参数
    N_min = 300
    fe = 'time_cut'
    
    # 针对进站流量的目标编码
    te_in = BetaEncoder(fe)
    te_in.fit(train_df, 'inNums')
    all_data['in_time_cut'] = te_in.transform(all_data, 'mean', N_min=N_min)
    
    # 针对出站流量的目标编码
    te_out = BetaEncoder(fe)
    te_out.fit(train_df, 'outNums')
    all_data['out_time_cut'] = te_out.transform(all_data, 'mean', N_min = N_min)
    
    return all_data


def train(all_data):
    # 设置需要使用的特征
    features = [f for f in all_data.columns if f not in ['inNums', 'outNums', 'time_cut']]

    # 提取训练集和测试集

    # 所有数据
    # 由于1号是元旦节，情况相对特殊，所以去掉了该极端值
    # 也可以使用一下看看效果
    train_data = all_data[(all_data.day != 29) & (all_data.day != 2)]

    # 用于训练的数据
    test = all_data[all_data.day == 29]
    X_test = test[features].values

    # 设置滑动窗口
    # 这里A榜要预测的是29号的信息，所以设置同是周二的日期作为滑动窗口的末尾
    # 而B榜要预测的是27号的信息，所以设置周末的日期作为滑窗末尾，即13，20
    slip = [15, 22]

    n = len(slip)


    ## 搭建LGB模型
    # 设置模型参数
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed':0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha':1,
        'reg_lambda':2,
        
        # 设置GPU 
        'device' : 'gpu',
        'gpu_platform_id':1,  
        'gpu_device_id':0 
    }

    ## 预测进站流量
    in_lgb_pred = np.zeros(len(X_test))
    X_data = train_data[features].values
    y_data = train_data['inNums'].values
    for i, date in enumerate(slip):
        train = train_data[train_data.day<date]
        valid = train_data[train_data.day==date]
        X_train = train[features].values
        X_eval =  valid[features].values
        y_train = train['inNums'].values
        y_eval = valid['inNums'].values
        
        print("\n Fold ", i)    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        
        lgb_model = lgb.train(lgb_params, 
                             lgb_train,
                             num_boost_round=10000,
                             valid_sets=[lgb_train, lgb_eval],
                             valid_names=['train', 'valid'],
                             early_stopping_rounds=200,
                             verbose_eval=1000,)
        
        # 用所有的数据训练再预测
        all_train = lgb.Dataset(X_data, y_data)
        lgb_model = lgb.train(lgb_params, 
                             all_train,
                             num_boost_round=lgb_model.best_iteration,
                             valid_sets=[all_train],
                             valid_names=['train'],
                             verbose_eval=1000
                             )
        
        in_lgb_pred += lgb_model.predict(X_test) / n
        print("第%d轮训练结束"%(i+1))
    print("完成！")


    ## 预测出站流量
    out_lgb_pred = np.zeros(len(X_test))
    X_data = train_data[features].values
    y_data = train_data['outNums'].values
    for i, date in enumerate(slip):
        train = train_data[train_data.day<date]
        valid = train_data[train_data.day==date]
        X_train = train[features].values
        X_eval =  valid[features].values
        y_train = train['outNums'].values
        y_eval = valid['outNums'].values
        
        print("\n Fold ", i)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        
        lgb_model = lgb.train(lgb_params, 
                             lgb_train,
                             num_boost_round=10000,
                             valid_sets=[lgb_train, lgb_eval],
                             valid_names=['train', 'valid'],
                             early_stopping_rounds=200,
                             verbose_eval=1000,)
        
        # 用所有的数据训练再预测
        all_train = lgb.Dataset(X_data, y_data)
        lgb_model_out = lgb.train(lgb_params, 
                             all_train,
                             num_boost_round=lgb_model.best_iteration,
                             valid_sets=[all_train],
                             valid_names=['train'],
                             verbose_eval=1000
                             )
        
        out_lgb_pred += lgb_model_out.predict(X_test) / n
        print("第%d轮训练结束"%(i+1))
    print("完成！")

        ## 搭建CatBoost模型
        # 设置模型参数
    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        
        'learning_rate': 0.08,
        
        'max_depth': 5,
        'max_leaves_count': 60, 

        'reg_lambda': 2,
        
        'verbose': 1,
        
        'od_type': 'Iter',
        'task_type': 'GPU'
    }

    ## 预测进站流量
    in_cat_pred = np.zeros(len(X_test))
    X_data = train_data[features].values
    y_data = train_data['inNums'].values


    for i, date in enumerate(slip):
        train = train_data[train_data.day<date]
        valid = train_data[train_data.day==date]
        X_train = train[features].values
        X_eval =  valid[features].values
        y_train = train['inNums'].values
        y_eval = valid['inNums'].values
        
        print("\n Fold ", i)
        cat_train = Pool(X_train, y_train)
        cat_eval = Pool(X_eval, y_eval)
        
        cat_model = catboost.train(
            pool = cat_train, params=cat_params,
            eval_set=cat_eval, num_boost_round=50000,
            verbose_eval=5000, early_stopping_rounds=200,)
        
        eval_pred = cat_model.predict(X_eval)
        print("MAE = ", mean_absolute_error(y_eval, eval_pred))
        
        # all data
        all_train = Pool(X_data, y_data)
        cat_model = catboost.train(
            pool = all_train, params=cat_params,
            eval_set= all_train, num_boost_round=cat_model.best_iteration_,
            verbose_eval=5000, early_stopping_rounds=200,)
        in_cat_pred += cat_model.predict(X_test) / n
        print("第%d轮完成"%(i+1))
    print("全部结束！")

    ## 预测出站流量
    out_cat_pred = np.zeros(len(X_test))
    X_data = train_data[features].values
    y_data = train_data['outNums'].values


    for i, date in enumerate(slip):
        train = train_data[train_data.day<date]
        valid = train_data[train_data.day==date]
        X_train = train[features].values
        X_eval =  valid[features].values
        y_train = train['outNums'].values
        y_eval = valid['outNums'].values
        
        print("\n Fold ", i)
        cat_train = Pool(X_train, y_train)
        cat_eval = Pool(X_eval, y_eval)
        
        out_cat_model = catboost.train(
            pool = cat_train, params=cat_params,
            eval_set=cat_eval, num_boost_round=50000,
            verbose_eval=5000, early_stopping_rounds=200,)
        
        eval_pred = out_cat_model.predict(X_eval)
        print("MAE = ", mean_absolute_error(y_eval, eval_pred))
        
        # all data
        all_train = Pool(X_data, y_data)
        out_cat_model = catboost.train(
            pool = all_train, params=cat_params,
            eval_set= all_train, num_boost_round=out_cat_model.best_iteration_,
            verbose_eval=5000, early_stopping_rounds=200,)
        out_cat_pred += out_cat_model.predict(X_test) / n
        print("第%d轮完成"%(i+1))
    print("全部结束！")


    ## 对模型结果进行平均
    inNums = (in_cat_pred + in_lgb_pred) / 2
    outNums = (out_cat_pred + out_lgb_pred) / 2
    return inNums, outNums


# 定义运行的主函数
def main():

    test_28 = pd.read_csv(os.path.join(test_path, test_file))

    data = get_base_features(test_28)

    train_files = [f for f in sorted(os.listdir(train_path)) if f.endswith("csv")]

    for file in train_files:
        print("正在处理...", file)
        tmp = pd.read_csv(os.path.join(train_path, file))

        tmp = get_base_features(tmp)
        data = pd.concat([data, tmp], axis=0, sort=False)
    print("数据清洗完成！")

    # 加入相邻时间的流量信息
    data = add_neighbor_time(data)
    # 增加高峰时间信息
    data = add_peak_type(data)
    # 增加同周次的流量信息
    data = add_week_flow(data)
    # 增加线路信息
    data = add_line(data)
    # 增加站点类型信息
    data = add_station_type(data)
    # 增加特殊站点的标记
    data = add_special_station(data)
    # 连接训练特征和目标值
    all_data, train_df = merge_train_test(data)
    # 进行目标编码
    all_data = target_encoding(all_data, train_df)
    # 训练模型
    inNums, outNums = train(all_data)

    # 添加进要提交的文件中
    submission = pd.read_csv(os.path.join(test_path, sub_file))
    submission['inNums'] = inNums
    submission['outNums'] = outNums
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()