#!/usr/bin/env python
# coding: utf-8
import nltk
import pandas as pd
import numpy as np
import os, sys, re, string, time



# 讀取csv
folerName = 'Dataset\\'
fileName = '(clean)Data.csv'
df = pd.read_csv(folerName + fileName, encoding='latin1')



# (A)統計總字數
def get_txt_len(txt):
    txt_list = txt.split(' ')
    return len(txt_list)

# (Boriginal_text)找全英大寫字詞
def get_cap_n(txt):
    big=0
    txt_list = txt.split()
    # 用for loop把輸入的字串跑一遍
    for t in txt_list:
        if t.isupper():
#             print(t)
            big+=1
    return big

# (D)找數值
def get_num_n(txt):
    num_count=0
    txt_list = txt.split()
    # 用for loop把輸入的字串跑一遍
    for t in txt_list:
        if t.isdigit():
            num_count+=1
    return num_count

# (F)統計句子數量
def get_sen_n(org_txt):
    # 會有人用多種符號來區隔句子，因此使用多種符號分割評論
    sen_list = re.split(r';|/|\!|\n|\?|\.', org_txt)
    # 過濾掉空字串
    sen_list = list(filter(None, sen_list))
    return len(sen_list)



# 先轉換資料型態為字串
df['clear_text_spell'] = df['clear_text_spell'].astype(str)
# df['text'] = df['text'].astype(str)

# (A)txt_len:文本長度 ; (F)sen_n:句子數量
# (B)cap_n:大寫字詞數量 ; (C)cap_mean:大寫字詞平均(cap_mean=cap_n/txt_len)
# (D)num_n:數字數量 ; (E)num_mean:數字平均數(num_mean=num_n/txt_len)

df['txt_len'] = df['clear_text_spell'].apply(lambda x: get_txt_len(x))
df['cap_n'] = df['clear_text_spell'].apply(lambda x: get_cap_n(x))
df['cap_mean'] = df['cap_n']/df['txt_len']
df['num_n'] = df['clear_text_spell'].apply(lambda x: get_num_n(x))
df['num_mean'] = df['num_n']/df['txt_len']
# 特別注意要使用清理前的字串作為輸入欄位
df['sen_n'] = df['comment_text'].apply(lambda x: get_sen_n(x))
df.head()



# 儲存檔案
fileName = '(simple)'+fileName
df.to_csv(folerName+fileName, index=False)




