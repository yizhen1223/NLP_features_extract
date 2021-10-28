#!/usr/bin/env python
# coding: utf-8
# 檔案目的為切分訓練集與測試集，訓練集又分成原始未平衡與隨機抽樣平衡，SMOTE平衡資料集會另外處理
# 測試集兩種標籤數目為1:1，兩邊數量皆為少數樣本的20%資料數
# 原始未平衡的訓練集，即為測試集剩餘的資料集合
# 隨機抽樣平衡的訓練集，兩標籤的數量皆為訓練集中少數樣本的數量。(抽樣出跟少數樣本同量的多數樣本)
import pandas as pd
import numpy as np
import os


# Read in saved w2c files.
dataset_Name = 'Data'
use_file = 'Dataset\\(all)Data'
file_type = '.csv'

df_all = pd.read_csv(use_file + file_type, header=0, index_col = 0)
# df_all['harmful_label_binary'].unique()
# df_all['harmful_label_binary'].value_counts()



#### 正規化情感分數ave_sentiment
from sklearn import preprocessing
data = np.array(df_all['ave_sentiment'])
data = data.reshape(-1, 1)
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# print(scaler.fit(data))
# print(scaler.data_max_)
df_all['ave_sentiment'] = scaler.transform(data)



df_harmful = df_all[df_all['harmful_label_binary']==1]
df_non_harmful = df_all[df_all['harmful_label_binary']==0]
# df_harmful['harmful_label_binary'].unique()
# df_harmful['harmful_label_binary'].value_counts()
# df_non_harmful['harmful_label_binary'].unique()
# df_non_harmful['harmful_label_binary'].value_counts()




# 先根據少數樣本測試集的數量去隨機抽樣出一樣數目的多數樣本做為測試集
df_harmful_test = df_harmful.sample(frac=0.2, replace=False)
test_len = len(df_harmful_test)
# print(test_len)

df_non_harmful_test = df_non_harmful.sample(n=test_len, replace=False)
# print(len(df_non_harmful_test))



# 再根據測試集中的id去刪除原樣本中的資料列，確保訓練集不會出現測試集內容。
drop_test_index_list = df_non_harmful_test.index.values.tolist()
df_non_harmful_train = df_non_harmful.drop(index=drop_test_index_list)
# print(len(drop_test_index_list))
# print(len(df_non_harmful_train))
# df_non_harmful_train.head()


drop_test_index_list = df_harmful_test.index.values.tolist()
df_harmful_train = df_harmful.drop(index=drop_test_index_list)
# print(len(drop_test_index_list))
# print(len(df_harmful_train))
# df_harmful_train.head()




# 兩種標籤的訓練集合併並打散，測試集同樣做法
df_train = pd.concat([df_non_harmful_train, df_harmful_train], axis=0)
df_train = df_train.sample(frac=1)   #這樣可以隨機打散
# df_train.head()
# df_train['harmful_label_binary'].unique()
# df_train['harmful_label_binary'].value_counts()

df_test = pd.concat([df_non_harmful_test, df_harmful_test], axis=0)
df_test = df_test.sample(frac=1)   #這樣可以隨機打散
# df_train.head()
# df_test['harmful_label_binary'].unique()
# df_test['harmful_label_binary'].value_counts()


# 在這個程式檔中，索引值即為id需要連帶儲存起來。
path_out = 'Dataset\\'
df_train.to_csv(path_out+dataset_Name+'_train.csv' ,index=True)
df_test.to_csv(path_out+dataset_Name+'_test.csv' ,index=True)




