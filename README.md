# <center>地铁乘客流量预测</center>

> 竞赛题目

&emsp;通过分析地铁站的历史刷卡数据，预测站点未来的客流量变化。开放了20190101至20190125共25天的刷卡记录，共涉及3条线路81个地铁站约7000万条数据作为训练数据`Metro_train.zip`，供选手搭建地铁站点乘客流量预测模型。同时大赛提供了路网地图，即各个地铁站之间的连接关系表，存储在文件`Metro_roadMap.csv`文件中。

&emsp;测试阶段，提供某天所有线路所有站点的刷卡记录数据，预测未来一天00时至24时以10分钟为单位的各时段各站点的进站和出站人次。

&emsp;测试集A集上，提供2019年1月28日的刷卡数据`testA_record_2019-01-28.csv`，选手需对2019年1月29日全天各地铁站以10分钟为单位的人流量进行预测。

&emsp;评估指标采用**平均绝对误差`Mean Absolute Error, MAE`**，分别对入站人数和出站人数预测结果进行评估，然后在对两者取平均，得到最终评分。

&emsp;关于数据的具体描述以及说明[详见天池官网](<https://tianchi.aliyun.com/competition/entrance/231708/information>)。



# <font size=4>1. 赛题分析和前期思路</font>

&emsp;比赛提供了1号到25号共25天的刷卡记录数据，所以第一步就是对每一天的文件进行处理。原始数据集中包含了`time`, `lineID`, `stationID`, `deviceID`, `status`, `userID`, `payType`这几个列，根据题目要求要预测进站和出站的人流量，所以要先统计出每一天的进站和出站流量。

## <font size=3>1.1 数据清洗</font>

>  **提取基础信息**

&emsp;首先对时间信息进行处理，提取出日、周、时、分、秒的信息，由于是按照10分钟为间隔统计，所以在提取分钟信息的时候只需要取整十。接着总计80个站点(除去缺失数据的54站)，每个站点从0点到24点，以10分钟为一次单位，总计144段时间间隔。根据站点、日、时、分进行分组统计，得到每个时段的进站人数和出站人数。

&emsp;经过第一轮处理之后，得到了每一天的文件包含的`columns`有：`stationID`,  `weekday`, `is_holiday`, `day`,  `hour`,  `minute`,  `time_cut`,  `inNums`以及`outNums`这几列。

&emsp;增加了一些和刷卡设备相关的特征，包括`nuni_deveiceID_of_stationID`, `nuni_deviceID_of_stationID_hour`, `nuni_deviceID_of_stationID_hour_minute`。

```python
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
```



## <font size=3>1.2 特征工程</font>

> **增加同一站点相邻时间段的进出站流量信息**

&emsp;考虑到当前时刻的流量信息与前后时刻的流量信息存在一定的关系，所以将当前时间段的前两个时段以及后两个时段流量信息作为特征。

&emsp;增加的特征包括`inNums_before1`, `inNums_before2`, `inNums_after1`, `inNums_after2`, `outNums_before1`, `outNums_before2`, `outNums_after1`, `outNums_after2`。

```python
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
```



> **增加乘车高峰时段相关的特征**

&emsp;根据杭州地铁的运营时段信息，将高峰时段分为四类，0表示非运营是简单，1表示非高峰是简单，2表示高峰是简单，3表示特殊高峰是简单。由于周末和非周末的高峰时间存在一定的差异，所以需要分别计算。

&emsp;增加了特征`peak_type`。

```python
# 这里我的实现方法可能有点复杂了
# 将时间段信息映射为高峰时间段信息

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
```



> **增加同周次的进出站流量信息**

&emsp;均值、最大值以及最小值在一定程度上反映了数据的分布信息，所以增加同周次进站流量和出站流量的均值、最大值以及最小值作为特征。

&emsp;增加的特征包括`inNums_whm_max`, `inNums_whm_min`, `inNums_whm_mean`, `outNums_whm_max`, `outNums_whm_min`, `outNums_whm_mean`, `inNums_wh_max`, `inNums_wh_min`, `inNums_wh_mean`, `outNums_wh_mean`。

```python
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
```



> **增加线路信息**

&emsp;根据某一天的刷卡记录表统计每条线路和站点的对应信息，并计算各条线路的站点数，用站点数量代表该站的线路信息。

&emsp;增加特征`line`。

```python
from collections import Counter

def add_line(df_):
    def station_line(record):
        station_line = record[['lineID', 'stationID']]
        station_line = station_line.drop_duplicates().reset_index(drop=True)
        station_line = station_line.sort_values(by='stationID').reset_index(drop=True)
        return station_line

    train_df = df_.copy()

    # 这里的file表示原始文件
    file = "../origin/Metro_train/record_2019-01-01.csv"
    record = pd.read_csv(file)
    line = station_line(record)
    line_pad = pd.DataFrame(Counter(line['lineID']), index=line.lineID.unique())
    line_ID = (line_pad.T)['B'].reset_index().rename(columns={'index': 'lineID', 'B': 'line'})
    line = line.merge(line_ID, on='lineID', how='left')
    line.drop('lineID', axis=1, inplace=True)
    train_df = train_df.merge(line, on=['stationID'], how='left')
    return train_df
```



> **增加站点的类型信息**

&emsp;不同的站点属于不同的类型，比如起点站、终点站、换乘站、普通站等，而这些站点的类别信息可以通过邻站点的数量表示，所以根据路网图对邻站点的数量进行统计，表示各个站点的类别。

&emsp;增加特征`station_type`。

```python
def add_station_type(df_):
    
    def get_map(roadmap):
        roadmap.rename(columns={"Unnamed: 0": 'stationID'}, inplace=True)
        tmp = roadmap.drop(['stationID'], axis=1)
        roadmap['station_type'] = tmp.sum(axis=1)
        return roadmap[['stationID', 'station_type']]
    
    roadpath = "../origin/Metro_roadMap.csv"
    roadmap = pd.read_csv(roadpath)
    map_pad = get_map(roadmap)
    
    train_df = df_.copy()
    train_df = train_df.merge(map_pad, on=['stationID'], how='left')
    return train_df
```



> **增加特殊站点的标记**

&emsp;对站点流量的分析过程中，发现第15站的流量与其他站点存在明显的区别，全天都处于高峰状态，因此给15站添加特别的标记。

&emsp;增加特征`is_special`。

```python
def add_special_station(df_):
    train_df = df_.copy()
    
    train_df['is_special'] = np.nan
    train_df.loc[train_df['stationID']==15, 'is_special'] = 1 
    train_df['is_special'].fillna(0, inplace=True)
    return train_df
```



> **连接训练特征和目标值**

&emsp;本次建模的思想使用前一天的流量特征和时间特征，以及预测当天的时间特征，来预测进站流量和出站流量。所以要对之前处理好的数据集进行拼接。

&emsp;增加新的特征`yesterday_is_holiday`以及`today_is_holiday`，增加目标值列`inNums`和`outNums`。

```python
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
    test = pd.read_csv(path + 'Metro_testA/testA_submit_2019-01-29.csv')
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
  
```



> **对时间间隔进行目标编码**

&emsp;考虑时间间隔信息与进站、出站流量的相关性，对时间间隔信息针对`inNums`和`outNums`进行目标编码。(这一步并非必须的，一定程度上可能会导致过拟合，所以可以考虑加入和不加入的情况都测试一下)

&emsp;目标编码后得到了`in_time_cut`和`out_time_cut`。

```python
# 考虑进出站流量和时间段密切相关
# 针对time_cut进行目标编码
# 定义目标编码函数

class BetaEncoder(object):

    def __init__(self, group):

        self.group = group
        self.stats = None

    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats

    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):

        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()

        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0

        # prior parameters
        N_prior = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior = (1 - self.prior_mean) * N_prior

        # posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + N - n

        # calculate statistics
        if stat_type == 'mean':
            num = alpha
            dem = alpha + beta

        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2

        elif stat_type == 'median':
            num = alpha - 1 / 3
            dem = alpha + beta - 2 / 3

        elif stat_type == 'var':
            num = alpha * beta
            dem = (alpha + beta) ** 2 * (alpha + beta + 1)

        elif stat_type == 'skewness':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)

        elif stat_type == 'kurtosis':
            num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        else:
            num = self.prior_mean
            dem = np.ones_like(N_prior)

        # replace missing
        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value

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
```



## <font size=3>1.3 划分训练集和测试集</font>

&emsp;根据上面数据清洗以及特征工程得到的结果对数据集进行划分。

```python
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
```



## <font size=3>1.4 搭建模型预测</font>

&emsp;我们使用了LightGBM和CatBoost两个模型预测并取其均值，其实也可以尝试加入XGBoost，然后取3个模型的加权平均，但是我们当时训练时发现XGBoost得到的结果不是很好，所以直接丢掉了。其实，通过加权平均，给XGBoost的结果一个比较好的权重，也有可能会得到比较不错的结果。

> **LightGBM模型**

```python
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
```



> **CatBoost模型**

```python
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
```



> **XGBoost模型**  —— `并没有用到它的结果`

```python
# 设置模型参数
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    
    'eval_metric': 'mae',
    
    'learning_rate': 0.0894,
    'max_depth': 9,
    'max_leaves': 20,
    
    'lambda': 2,
    'alpha': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'silent': 1,
    
    'gpu_id': 0,
    'tree_method': 'gpu_hist'
}

## 预测进站流量
in_xgb_pred = np.zeros(len(X_test))
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
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_eval = xgb.DMatrix(X_eval,y_eval)
    
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=10000,
        evals=[(xgb_eval, 'evals')],
        early_stopping_rounds=200,
        verbose_eval=1000
    )
    
    temp_eval = xgb.DMatrix(X_eval)
    eval_pred = xgb_model.predict(temp_eval)
    print("MAE = ", mean_absolute_error(y_eval, eval_pred))
    
    ## all_data
    all_train = xgb.DMatrix(X_data, y_data)
    xgb_model = xgb.train(
        xgb_params,
        all_train,
        num_boost_round=xgb_model.best_iteration,
        evals=[(all_train, 'train')],
        verbose_eval=1000
    )
    
    xgb_test = xgb.DMatrix(X_test)
    in_xgb_pred += xgb_model.predict(xgb_test) / n
    print("第%d轮完成"%(i+1))
print("全部结束！")

## 预测出站流量
out_xgb_pred = np.zeros(len(X_test))
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
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_eval = xgb.DMatrix(X_eval,y_eval)
    
    out_xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=10000,
        evals=[(xgb_eval, 'evals')],
        early_stopping_rounds=200,
        verbose_eval=1000
    )
    
    temp_eval = xgb.DMatrix(X_eval)
    eval_pred = out_xgb_model.predict(temp_eval)
    print("MAE = ", mean_absolute_error(y_eval, eval_pred))
    
    ## all_data
    all_train = xgb.DMatrix(X_data, y_data)
    out_xgb_model = xgb.train(
        xgb_params,
        all_train,
        num_boost_round = out_xgb_model.best_iteration,
        evals=[(all_train, 'train')],
        verbose_eval=1000
    )
    
    xgb_test = xgb.DMatrix(X_test)
    out_xgb_pred += out_xgb_model.predict(xgb_test) / n
    print("第%d轮完成"%(i+1))
print("全部结束！")
```



最后，<font color=blue>**对模型结果平均**</font>

```python
inNums = (in_cat_pred + in_lgb_pred) / 2
outNums = (out_cat_pred + out_lgb_pred) / 2
```



# <font size=4>**2. 其他的一些想法**</font>

(1) 由于官方给了路网图，所以我们尝试将路网图拼接在特征后面，表示各个站点之间的连接关系，但是这样反而降低了模型最终的性能。

<br/>

(2) 过程中我们一直想充分利用邻站点的信息，想在邻站点上提取尽可能多的特征，包括邻站点相同时刻以及相邻时刻的流量信息，但是这样都会降低模型的性能，也在这上面浪费了不少的时间。

<br/>

(3) 除了从以后的数据中提取特征之外，我们还对提出出来的特征做了一些特征工程，包括计算一些可能有一定关联的特征**计算加减以及比率信息**，但是这些工作都没有能够提升模型的性能。

<br/>

(4) 根据官方交流群的讨论，我们在A榜的时候尝试去掉周末的信息，只讲工作日的信息进行提取和拼接，这在一定程度上提升了模型的效果，后来我们在[鱼的代码](<https://zhuanlan.zhihu.com/p/59998657>)基础上加入了我们之前找到的一些特征。之后，仔细推敲了一下鱼的代码，发现在进行特征`feature`和目标值`target`拼接的时候，和`deviceID`相关的特征使用的是预测当天的，这样一方面会导致leak，另一方面就是最后的测试集这些特征都是`nan`值，也就是说在最终的预测中没有起到作用。于是，我们改变了拼接的方法，再次加入了自己提取的特征，最终在A榜跑出的成绩是`12.99`。注意，**这里使用的数据并不是全部的数据，而只是使用了工作日的数据拼接**。

<br/>

&emsp;在B榜的时候，我们还是希望能够训练一个通用的模型，所以这次把所有的数据放在一起训练，并没有将周末的数据单独提取出来，但是在滑窗的时候使用了`13`和`20`两天的数据作为验证，最终跑出了`12.57`的成绩，说明这种想法是可行的。为了验证一下只提取周末信息的效果，我们也尝试把周末的信息单独提取出来，最后得分一直在`14`以上，可能也是因为我们提的特征不太适用于这种场景。

<br/>

&emsp;最终，我们使用了全部的数据通过Stacking的方法进行了训练，将三个梯度提升树模型进行了堆叠，最后得到的结果也是`14`多一点。Stacking的方法如下：

```python
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class MyStacking(BaseEstimator, RegressorMixin, TransformerMixin):
    # base_models表示基准模型，meta_model表示元模型
    # slip表示滑窗的预测值
    # time_fe表示时间对应的特征
    def __init__(self, base_models, 
                 meta_model, slip, time_fe, features, target):
        self.base_models = base_models
        self._slip = slip
        self.fe_ = time_fe
        self.features = features
        self.target_ = target
        
        self.meta_model = meta_model
        
    
    # 定义拟合函数
    # 这里的X和y是DataFrame形式的
    def fit(self, train_data):
        # 定义基本模型
        self.base_models_ = [list() for x in self.base_models]
        # 定义元模型
        self.meta_model_ = clone(self.meta_model)
        
        
        shape_ = [train_data[train_data[self.fe_] == d].shape[0] for d in self._slip]
        y_true = np.array([])
        for d in self._slip:
            y_true = np.hstack((y_true, train_data.loc[train_data.day == d, self.target_].values))
        
        index = []
        for k, sh in enumerate(shape_):
            if k == 0:
                index.append(list(range(sh)))
            else:
                index.append(list(range(index[-1][-1], index[-1][-1]+shape_[k])))
        
        # 设置用于元模型的特征大小
        oof_pred = np.zeros((sum(shape_), 
                             len(self.base_models)))
        # 训练基础模型
        for i, model_name in enumerate(self.base_models):
            for j, date in enumerate(self._slip):
                # 设置训练集和验证集
                train = train_data[train_data[self.fe_]<date]
                valid = train_data[train_data[self.fe_]==date]
                
                X_train = train[self.features].values
                X_eval = valid[self.features].values
                y_train = train[self.target_].values
                y_eval = valid[self.target_].values
                
                if model_name =='lgb':
                    lgb_train = lgb.Dataset(X_train, y_train)
                    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
                    print("开始训练{}_{}".format(i, j))
                    model = lgb.train(lgb_params, 
                                         lgb_train,
                                         num_boost_round=10000,
                                         valid_sets=[lgb_train, lgb_eval],
                                         valid_names=['train', 'valid'],
                                         early_stopping_rounds=200,
                                         verbose_eval=1000,)
                    y_pred = model.predict(X_eval)
                    print("结束本次训练！")
                if model_name == 'cat':
                    cat_train = Pool(X_train, y_train)
                    cat_eval = Pool(X_eval, y_eval)
                    print("开始训练{}_{}".format(i, j))
                    model = catboost.train(
                                pool = cat_train, params=cat_params,
                                eval_set=cat_eval, num_boost_round=50000,
                                verbose_eval=5000, early_stopping_rounds=200,)
                    y_pred = model.predict(X_eval)
                    print("结束本次训练！")
                    
                if model_name == 'xgb':
                    
                    print("开始训练{}_{}".format(i, j))
                    model = xgb.XGBRegressor(**xgb_params)
                    #print(X_train.shape)
                    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=400, verbose=1000)
                    y_pred = model.predict(X_eval)
                    print("结束本次训练！")
                    
                self.base_models_[i].append(model)
                oof_pred[index[j], i] = y_pred
        
        self.meta_model_.fit(oof_pred, y_true)
        return self

    # 预测结果
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
        for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
```



# <font size=4>3. 总结与思考</font>

(1) 首先是对数据的EDA做的不够，包括对各个站点的分析，各个时间段综合分析，对特征重要性的分析等等。主要还是因为经验不够，不知道该怎么做，甚至15站点的特殊性也是从交流群里得到的信息。另外，就是调参做的有问题，反而把模型的性能调低了，说明对参数的理解不够。

<br/>

(2) 第一次团队作战，不知道该怎么协作，分工不是很明确，所以效率不是很高，没能有机会尝试更多的模型。代码写的不规范，导致后面修改的时候浪费了比较多的时间，包括整理也花了不少的时间。

<br/>

(3) 知道的模型太少，只使用了梯度提升树模型，其实还有很多可能有效的模型可以尝试，包括图神经网络，时空模型以及LSTM，但是都因为不够熟悉而无从入手。

<br/>

(4) 总结：

- 了解自己能做的事情，明确分工；
- 做好EDA，做到对数据的充分理解；
- 代码书写规范，每一个功能模块应该定义为一个函数；
- 熟悉不同模型的功能以及试用场景。

