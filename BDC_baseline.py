#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14  2020
@author: zhaoxx17
"""
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
from pre_process_new import get_feature, get_data
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
path = 'E:/HuaweiBDC/'
# baseline只用到gps定位数据，即train_gps_path
train_gps_path = path + 'data/train0523.csv'
test_data_path = path + 'data/A_testData0531.csv'
order_data_path = path + 'data/loadingOrderEvent.csv'
port_data_path = path + 'data/port.csv'

# 取前1000000行
debug = False
is_kfold = True
data_process = False      # 将train0523.csv分割百万量级的chunk，数据清理后保存
data_concat_chunk = False # 读取chunk_X.csv,合并得到数据清理后train_data_processed.csv
NDATA = 1000000


if debug:
    train_data = pd.read_csv( path+'data/train_data_processed.csv',nrows=NDATA,header=None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_rows', None)
elif data_process:
    train_data_df=pd.read_csv(train_gps_path, chunksize=1000000)
    ind = 0
    num = 0
    print('data number after dropna, ')
    for train_data_chunk in train_data_df:
        ind = ind +1
        chunk_name = path + 'data/chunk_'+str(ind)+'.csv'
        train_data_chunk = train_data_chunk.dropna(axis =0, how='any')
        train_data_chunk.to_csv(chunk_name, index=False)
        num = num + train_data_chunk.shape[0]
        print(num)
    exit(0)  # attention, 0 means successful exit
elif data_concat_chunk:
    train_data = pd.DataFrame()
    data_folder_names = os.listdir(path + 'data')
    print('data number concatenated, ')
    for name in data_folder_names:
        if name.startswith('chunk_') and name.endswith('.csv'):
            chunk_data = pd.read_csv(path + 'data/' + name,header=None)
            train_data = pd.concat([train_data, chunk_data])
            print(train_data.shape[0])
        if train_data.shape[0] >10000000:    #本地内存跑不动，只留了一千万多一点点的数据，全部大概是二千多万
            break
    print('save processed data, ')
    train_data.to_csv(path+'data/train_data_processed.csv',index = False,header=0)
    exit(0)
else:
    train_data = pd.read_csv(path+'data/train_data_processed.csv', header=None)
    print('load processed data, ')
    print(train_data.shape[0])


train_data.columns = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
test_data = pd.read_csv( test_data_path)
port_data = pd.read_csv( port_data_path)

train_data = get_data(train_data, mode='train')
test_data = get_data(test_data, mode='test')
train_data = train_data.drop(index=(train_data.loc[(train_data['vesselNextport']=='Unnamed: 8')].index))



train = get_feature(train_data, mode='train')
test = get_feature(test_data, mode='test')
features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count']]


def mse_score_eval(preds, valid):
    labels = valid.get_label()
    scores = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse_score', scores


def build_model(train, test, pred, label, seed=1080, is_shuffle=True, is_kfold=False):
    train_pred = np.zeros((train.shape[0],))
    test_pred = np.zeros((test.shape[0],))

    # features
    print(len(pred), pred)
    # params
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 36,
        'metric': 'mse',
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'seed': 8,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': 1,
    }
    # Kfold
    if is_kfold:
        fold = KFold(n_splits=10, shuffle=is_shuffle, random_state=seed)
        kf_way = fold.split(train[pred])
        # train
        for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
            train_x, train_y = train[pred].iloc[train_idx], train[label].iloc[train_idx]
            valid_x, valid_y = train[pred].iloc[valid_idx], train[label].iloc[valid_idx]
            # 数据加载
            n_train = lgb.Dataset(train_x, label=train_y)
            n_valid = lgb.Dataset(valid_x, label=valid_y)

            clf = lgb.train(
                params=params,
                train_set=n_train,
                num_boost_round=3000,
                valid_sets=[n_valid],
                early_stopping_rounds=100,
                verbose_eval=100
            )
            train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
            test_pred += clf.predict(test[pred], num_iteration=clf.best_iteration) / fold.n_splits
    else:
        train_index = int(0.8*train.shape[0])
        train_x, train_y = train[pred].iloc[:train_index], train[label].iloc[:train_index]
        valid_x, valid_y = train[pred].iloc[train_index:], train[label].iloc[train_index:]
        # 数据加载

        lgb_model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31,
                                    max_depth=- 1, learning_rate=0.05,
                                    n_estimators=3000, objective='regression',
                                    min_child_samples=20, subsample=0.8,
                                    subsample_freq=1,   colsample_bytree=0.8,
                                    reg_alpha=0.0, reg_lambda=0.01, n_jobs=- 1)
        lgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                      eval_metric= 'mse',
                      verbose=10, early_stopping_rounds=100)
        train_pred[train_index:] = lgb_model.predict(valid_x)
        test_pred += lgb_model.predict(test[pred])

        imp = pd.DataFrame()
        imp['fea'] = pred
        imp['imp'] = lgb_model.feature_importances_
        imp = imp.sort_values('imp', ascending=False)
        print(imp)
        plt.figure(figsize=[20, 10])
        sns.barplot(x='fea', y='imp', data=imp)

    test['label'] = test_pred

    return test[['loadingOrder', 'label']]

print('start to train ')
result = build_model(train, test, features, 'label', is_shuffle=True, is_kfold=is_kfold)
print('finish training')
test_data = test_data.merge(result, on='loadingOrder', how='left')
test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
test_data['timestamp'] = test_data['temp_timestamp']
# 整理columns顺序
result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]


result.to_csv('result' + pd.datetime.now().strftime('_%Y%m%d_%H%M%S') + '_n{}.csv'.format(train_data.shape[0]), index=False)

result