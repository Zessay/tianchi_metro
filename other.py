# -*- coding: utf-8 -*-
# @Author: zhushuai
# @Date:   2019-04-02 13:08:20
# @Last Modified by:   zhushuai
# @Last Modified time: 2019-04-02 13:11:13

import xgboost as xgb 
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# 定义XGB模型

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


# 定义Stacking模型
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