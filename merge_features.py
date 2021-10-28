#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os

dir_path = 'Dataset\\'
# w2c, d2c files without columns names -> header is none.
df_features = pd.read_csv(os.path.join(dir_path, '(tidytext)(sentimentr)(simple)(clean)Data.csv'))
df_w_m_d2c= pd.read_csv(os.path.join(dir_path, '(word_mean_d2c)(tidytext)(sentimentr)(simple)(clean)Data.csv'), header=None)
df_d2c = pd.read_csv(os.path.join(dir_path, '(dm_d2c)(tidytext)(sentimentr)(simple)(clean)Data.csv'), header=None)
df_bert= pd.read_csv(os.path.join(dir_path, '(bert_vec)(clean)Data.csv'))
df_target = pd.read_csv(os.path.join(dir_path, '(target_labels)(tidytext)(sentimentr)(simple)(clean)Data.csv'))



## (1)為w_m_d2c、d2c添增標題列
w2c_header_name=[]
d2c_header_name=[]
# range(1, 51):實際會從1-50
for i in range(1, 51):
    w2c_header_str = 'w2c_%d' % (i) 
    d2c_header_str = 'd2c_%d' % (i) 
    w2c_header_name.append(w2c_header_str)
    d2c_header_name.append(d2c_header_str)

#columns可以定義標題列，輸入格式是列表
df_w_m_d2c.columns=w2c_header_name
df_d2c.columns=d2c_header_name



## (2)用數據集的索引值來幫w_m_d2c、d2c添增key欄位('comment_counter' / 'id')
df_w_m_d2c_key = pd.concat([df_features['id'], df_w_m_d2c], axis=1)
df_d2c_key = pd.concat([df_features['id'], df_d2c], axis=1)



## (3)依據key欄位合併所有欄位，key欄位('comment_counter' / 'id')
# id+orinal_text+label+clear*3+simple*6+sentimentr*2+emotion*8+w2c*50+d2c*50+bert*768=890
df_features_merge = pd.merge(df_features, df_w_m_d2c_key, on='id')
df_features_merge = pd.merge(df_features_merge, df_d2c_key, on='id')
df_features_merge = pd.merge(df_features_merge, df_bert, on='id')

# 包含文本內容與標籤的資料也保存起來
df_features_merge.to_csv(os.path.join(dir_path, '(all)Data.csv') ,index=False )




