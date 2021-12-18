# -*- codeing = utf-8 -*-
# @Time : 2021-12-02 15:55
# @Author 胡智

import datetime
import os, sys
from collections import Counter

import numpy as np
import pandas as pd
import gresearch_crypto

env = gresearch_crypto.make_env()
iter_test = env.iter_test()


class IterativeImputer(object):
    pass


import plotly.graph_objects as go


class gresearch_lgbm():
    """docstring for gresearch_guada"""

    def __init__(self):
        # super(gresearch_guada, self).__init__()
        ### 训练集
        self.train = pd.read_csv('E:\pythonProject\pythonProject1\crypto forecasting\g-research-crypto-forecasting/train.csv')
        ### 补充训练数据集——（验证集）
        self.supplemental_train = pd.read_csv('E:\pythonProject\pythonProject1\crypto forecasting\g-research-crypto-forecasting/supplemental_train.csv')
        ### 资产信息数据集，包含14个虚拟货币资产
        self.asset_details = pd.read_csv('E:\pythonProject\pythonProject1\crypto forecasting\g-research-crypto-forecasting/asset_details.csv')
        ### 测试数据样例
        self.example_test = 'E:\pythonProject\pythonProject1\crypto forecasting\g-research-crypto-forecasting/example_test.csv'
        # self.env = gresearch_crypto.make_env()
        # self.iter_test = self.env.iter_test()

    def dataReader(self, datasetName):
        ### 数据集读取
        if datasetName == 'train':
            ### 获取train数据集
            df = pd.read_csv(self.train, usecols=['Target', 'Asset_ID', 'timestamp'], dtype={'Asset_ID': 'int8'})
        elif datasetName == 'supplemental_train':
            ### 获取supplemental_train
            df = pd.read_csv(self.supplemental_train, usecols=['Target', 'Asset_ID', 'timestamp'],
                             dtype={'Asset_ID': 'int8'})
        else:
            print("ERROR [1018] - message: 数据集传入参数错误!")
        return df

    def dataFillNan(self, train_data, columns, fillType):
        ## 空缺值填充
        if fillType == "0":
            ### fixed value fillnan
            fixed_value = input(
                "请输入" + columns + "列固定填充值：(eg: 推荐值：" + str(Counter(train_data[columns]).most_common(3)) + ")：")
            # self.logging.info("kaggle." + sys._getframe().f_code.co_name + ".service Message: 已获取缺失值填充参数：{'缺失值填充方法':'固定值填充','固定值':" + fixed_value + "},准备开始缺失值填充......")
            train_data[columns].fillna(fixed_value, inplace=True)
            if train_data[columns].isnull().any():
                print("仍旧有空值")
                # self.logging.error("kaggle." + sys._getframe().f_code.co_name + ".service Message: 数据集列" + columns +"缺失值以固定值方式填充失败，请查看原因！")
            else:
                print("该列无空值")
                # self.logging.info("kaggle." + sys._getframe().f_code.co_name + ".service Message: 数据集列" + columns +"缺失值以固定值方式填充成功！")
            return train_data
        elif fillType == "1":
            ### before value fillnan
            return "前值填充法 暂未开放......"
        elif fillType == "2":
            ### 中位数填补
            if train_data[columns].dtypes == 'float64':
                train_data[columns].fillna(train_data[columns].median(), inplace=True)
            else:
                print("该列数据类型不支持中位数填充！")
            return train_data
        elif fillType == "3":
            ## 众数填补法
            if train_data[columns].dtypes == 'float64':
                train_data[columns].fillna(train_data[columns].mode(), inplace=True)
            else:
                print("该列数据类型不支持众数填充！")
                # continue
            return train_data
            # train_data[columns].fillna(train_data[columns].mode(),inplace=True)
        elif fillType == "4":
            ## 回归填补法
            # self.logging.info("kaggle." + sys._getframe().f_code.co_name + ".service Message: 已获取缺失值填充参数：{'缺失值填充方法':'回归填充','当前列回归':" + str(train_data[columns].mode()) + "},准备开始缺失值填充......")
            if train_data[columns].dtypes == 'float64':
                # self.logging.info("kaggle." + sys._getframe().f_code.co_name + ".service Message: 已获取缺失值填充参数：{'缺失值填充方法':'回归填补法','填充值':}")
                imp = IterativeImputer(max_iter=10, random_state=0)
                imp.fit(train_data)
                np.round(imp.transform(train_data))
            else:
                print("数据类型不支持回归填充法！")
                # self.logging.info("kaggle." + sys._getframe().f_code.co_name + ".service Message: 数据类型不支持回归填充法！（说明*：该方法仅适用于缺失值为定量的数据类型）")
        else:
            ### 其他填充法
            return "其他填充法 暂未开放......"

    def datetimeProc(self, datasetName):
        ## 数据集时间处理
        datasetName['datetime'] = pd.to_datetime(datasetName['timestamp'], unit='s')
        # print(datasetName['datetime'])
        datasetName = datasetName.set_index('datetime').drop('timestamp', axis=1)
        datasetName = datasetName[(datasetName.index.year == 2021) & (datasetName.index.month > 5)]
        # print(datasetName)

        dfs = {asset_id: datasetName[datasetName['Asset_ID'] == asset_id].resample('1min').interpolate().copy() for
               asset_id in datasetName['Asset_ID'].unique()}
        # print(dfs)
        ## delete $datasetName dataset
        del datasetName
        return dfs


## 类实例化
gresearch_lgbm = gresearch_lgbm()

## 读取train.csv
df = gresearch_lgbm.dataReader("train")

## 初步处理datetime字段
dfs = gresearch_lgbm.datetimeProc(df)

for df_test, df_pred in iter_test:
    print(df_pred)
    df_test['datetime'] = pd.to_datetime(df_test['timestamp'], unit='s')
    print(df_test)
    for _, row in df_test.iterrows():
        try:
            df = dfs[row['Asset_ID']]
            closest_train_sample = df.iloc[df.index.get_loc(row['datetime'], method='nearest')]
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = closest_train_sample['Target']
        except:
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
    # df_pred['Target'] = df_pred['Target'].fillna(0)
    df_pred = gresearch_lgbm.dataFillNan(df_pred, 'Target', '2')
    print(df_pred)
    env.predict(df_pred)