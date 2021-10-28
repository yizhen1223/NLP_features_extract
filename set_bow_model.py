#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


dataset_Name = 'Data'
train_file = 'Data_train'
test_file = 'Data_test'
file_type = '.csv'


input_path = 'Dataset\\'
df_train = pd.read_csv(os.path.join(input_path, train_file+file_type), header=0)
# print(len(df_train))

df_test = pd.read_csv(os.path.join(input_path, test_file+file_type), header=0)
# print(len(df_test))


train_text = df_train['clear_text_spell'].values.astype('U').tolist()
test_text = df_test['clear_text_spell'].values.astype('U').tolist()



bow_vectorizer = CountVectorizer()
train_bow = bow_vectorizer.fit_transform(train_text)
test_bow = bow_vectorizer.transform(test_text)
print("Shape of train matrix after BOW : ",train_bow.shape)
print("Shape of test matrix after BOW : ",test_bow.shape) 



tfidf_vectorizer = TfidfVectorizer(min_df = 10)
train_tfidf = tfidf_vectorizer.fit_transform(train_text)
test_tfidf = tfidf_vectorizer.transform(test_text)
print("Shape of train matrix after Tfidf : ",train_tfidf.shape)
print("Shape of test matrix after Tfidf : ",test_tfidf.shape) 



X_tr = train_bow
y_tr = df_train['harmful_label_binary']
X_ts = test_bow
y_ts = df_test['harmful_label_binary']




from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

models_list = [LogisticRegression(random_state=1, multi_class='ovr', solver='sag', max_iter=10000, n_jobs=3)]
label_names_list = ['non-Harmful', 'Harmful']

# Model on BOW
for model in models_list:
    score = cross_val_score(model, X_tr, y_tr, cv=5, n_jobs=3)
    
    report_str = 'Use Dataset : %s \n' % dataset_Name
    report_str = report_str + 'Use model : BOW \n'
    model_name = model.__class__.__name__
    
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.2f} "
           "(+/- {2:.3f})".format(model_name,
                                  score.mean(),
                                  score.std()))
    print(msg)
    report_str = report_str + msg + '\n'
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = balanced_accuracy_score(y_ts, pred_eval)
    pre_sc = precision_score(y_ts, pred_eval, average='weighted', labels=np.unique(y_ts))
    recall_sc = recall_score(y_ts, pred_eval, average='weighted', labels=np.unique(y_ts))
    f1_sc = f1_score(y_ts, pred_eval, average='weighted')
    
    report_str = report_str + '\nclassification report: \n %s \n' % classification_report(y_ts, pred_eval, target_names=label_names_list)
    report_str = report_str + "\t'balanced_Accuracy on evaluation set\t\t= {0:.2f}\n".format(acc_eval)
    report_str = report_str + "\t(weighted)Precision on evaluation set\t\t= {0:.2f}\n".format(pre_sc)
    report_str = report_str + "\t(weighted)Recall on evaluation set\t\t= {0:.2f}\n".format(recall_sc)
    report_str = report_str + "\t(weighted)F1_score on evaluation set\t\t= {0:.2f}\n".format(f1_sc)
    report_str = report_str + "\tconfusion matrix: \n %s \n" % confusion_matrix(y_ts, pred_eval)
    report_str = report_str + '------------------------------------------------------------------\n\n'
    
    path_out = '{0}_{1}_BOW_report.txt '.format(dataset_Name, model_name)
    report_f = open(path_out, mode='w')
    report_f.write(report_str)
    report_f.close()
    print(report_str)




# Model on Tf-idf
X_tr = train_tfidf
y_tr = df_train['harmful_label_binary']
X_ts = test_tfidf
y_ts = df_test['harmful_label_binary']
# print(X_tr.shape)
# print(y_tr.shape)
# print(X_ts.shape)
# print(y_ts.shape)


report_str = report_str + 'Use model : TF-IDF \n'

for model in models_list:
    score = cross_val_score(model, X_tr, y_tr, cv=5, n_jobs=3)
    
    report_str = 'Use Dataset : %s \n' % dataset_Name
    report_str = report_str + 'Use model : TFIDF \n'
    model_name = model.__class__.__name__
    
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.2f} "
           "(+/- {2:.3f})".format(model_name,
                                  score.mean(),
                                  score.std()))
    print(msg)
    report_str = report_str + msg + '\n'
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = balanced_accuracy_score(y_ts, pred_eval)
    pre_sc = precision_score(y_ts, pred_eval, average='weighted', labels=np.unique(y_ts))
    recall_sc = recall_score(y_ts, pred_eval, average='weighted', labels=np.unique(y_ts))
    f1_sc = f1_score(y_ts, pred_eval, average='weighted')
    
    report_str = report_str + '\nclassification report: \n %s \n' % classification_report(y_ts, pred_eval, target_names=label_names_list)
    report_str = report_str + "\t'balanced_Accuracy on evaluation set\t\t= {0:.2f}\n".format(acc_eval)
    report_str = report_str + "\t(weighted)Precision on evaluation set\t\t= {0:.2f}\n".format(pre_sc)
    report_str = report_str + "\t(weighted)Recall on evaluation set\t\t= {0:.2f}\n".format(recall_sc)
    report_str = report_str + "\t(weighted)F1_score on evaluation set\t\t= {0:.2f}\n".format(f1_sc)
    report_str = report_str + "\tconfusion matrix: \n %s \n" % confusion_matrix(y_ts, pred_eval)
    report_str = report_str + '------------------------------------------------------------------\n\n'
    
    path_out = '{0}_{1}_TFIDF_report.txt '.format(dataset_Name, model_name)
    report_f = open(path_out, mode='w')
    report_f.write(report_str)
    report_f.close()
    print(report_str)

