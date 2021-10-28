#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from numpy import random
import os
import joblib

# 讀出特徵欄位名稱檔案並存放在列表
features_columns_ls = []
features_columns_path = 'dataset_features_columns.txt'
f = open(features_columns_path)
text = f.read()
f.close()
features_columns_ls = text.split()

# simple features
# used_feat = features_columns_ls[13:19]
# used_features_str = 'simple'

# # # 只取情緒特徵
# used_feat = features_columns_ls[4:13]
# used_features_str = 'sentiment'

# # 只取詞向量&段落向量特徵
# used_feat = features_columns_ls[19:119]
# used_features_str = 'wordvec'

# # # # 只取BERT向量特徵
# used_feat = features_columns_ls[119:]
# used_features_str = 'bert'

# # simple+情緒特徵
# used_feat = features_columns_ls[4:19]
# used_features_str = 'simple&sentiment'

# # simple+情緒+詞向量&段落向量(115)
# used_feat = features_columns_ls[4:119]
# used_features_str = 'simple&sentiment&wordvec'

# # all features(883)
used_feat = features_columns_ls[4:]
used_features_str = 'all_883'


dataset_Name = 'Data'
train_file = 'Data_train'
test_file = 'Data_test'
file_type = '.csv'
input_path = 'Dataset\\'

train_X = pd.read_csv(input_path+train_file+file_type, header=0, usecols=used_feat)
train_Y = pd.read_csv(input_path+train_file+file_type, header=0, usecols=['harmful_label_binary']).values.tolist()
test_X = pd.read_csv(input_path+test_file+file_type, header=0, usecols=used_feat)
test_Y = pd.read_csv(input_path+test_file+file_type, header=0, usecols=['harmful_label_binary']).values.tolist()



# 設定分類模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

model = LogisticRegression(random_state=1, multi_class='ovr', solver='sag', max_iter=10000, n_jobs=10)
target_ls = ['Harmful', 'non-Harmful']

# min_feat_select_int = 1  # simple feat
# step_int = 1  # (6-1)/5 = 1

# min_feat_select_int = 3  # sentiment feat 
# step_int = 1  # (9-3)/5 = 1

# min_feat_select_int = 10  # w2c feat 
# step_int = 18  #  (100-10)/5 = 18

# min_feat_select_int = 84  # bert feat 
# step_int = 140  # (784-84)/5 = 140

# min_feat_select_int = 3  # 15 feat
# step_int = 2  #  (883-100)/5 = 156.6

# min_feat_select_int = 5  # 115 feat 
# step_int = 18  #  (115-25)/5 = 18

min_feat_select_int = 50  # 883 feat
step_int = 157  #  (883-100)/5 = 156.6


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score,recall_score , f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=model, step = step_int, min_features_to_select = min_feat_select_int, 
              cv=StratifiedKFold(5), n_jobs=10)
rfecv.fit(train_X, train_Y)


## features selection list
features_select_ls = []

for i, f in enumerate(used_feat):
    if rfecv.support_[i] :
        features_select_ls.append(f)

# print(used_feat)
# print(rfecv.support_)
# print(features_select_ls)
path_out = 'RFECV_result\\{0}_{1}_{2}'.format(dataset_Name, used_features_str, model.__class__.__name__)
path_feat = '%s_OptimalFeature_list.csv' % path_out
df_feat_select = pd.DataFrame(columns=['Feature_Name'])
df_feat_select['Feature_Name'] = features_select_ls
df_feat_select.to_csv(path_feat, index=True, index_label='No')




## report
report_str = 'RFECV Use Dataset : %s \n' % dataset_Name
report_str = report_str + 'Use features : %s feature \n' % used_features_str
report_str = report_str + "Use classifer: %s \n\n" % model
report_str = report_str + 'Optimal number of features : %d \n------------------------------------------------------\n' % rfecv.n_features_

test_path = '\\Dataset'
test_files_ls = ['Data_test']
target_ls = ['Harmful', 'non-Harmful']
for f in test_files_ls :
    test_X = pd.read_csv(test_path+f+file_type, header=0, usecols=used_feat)
    test_Y = pd.read_csv(test_path+f+file_type, header=0, usecols=['harmful_label_binary']).values.tolist()
    pred_Y = rfecv.predict(test_X)
    
    report_str = report_str + 'Use test Dataset : %s.csv \n' % f
    report_str = report_str + 'Classification report : \n %s \n' % classification_report(test_Y, pred_Y, target_names=target_ls)
    report_str = report_str + 'Accuracy_score : {:.2} \n'.format(accuracy_score(test_Y, pred_Y))
    report_str = report_str + 'Precision_score :\t {:.2} \n'.format(precision_score(test_Y, pred_Y, labels=np.unique(test_Y)))
    report_str = report_str + 'Recall_score :\t\t {:.2} \n'.format(recall_score(test_Y, pred_Y, labels=np.unique(test_Y)))    
    report_str = report_str + 'F1 score :\t\t {:.2} \n'.format(f1_score(test_Y, pred_Y, labels=np.unique(test_Y)))
    report_str = report_str + 'Confusion matrix: \n %s \n------------------------------------------------------\n' % confusion_matrix(test_Y, pred_Y, labels=[1, 0])
print(report_str)
# print(rfecv.n_features_)
path_report = '%s_test_dataset_Pred_report.txt' % path_out
report_f = open(path_report, mode='w')
report_f.write(report_str)
report_f.close()


## save model
path_model = '%s_OptimalFeature_model.pkl' % path_out
joblib.dump(rfecv, path_model)