# 文本特徵擷取技術筆記－建立文本分類模型
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：[切分訓練集與測試集](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_split_train_test.md)

---
**目的**
使用包含所有特徵值的資料集進行機器學習分類器建模，利用選定特定索引範圍，來指定模型訓練的特徵類別。

## 使用資料集
*經**合併特徵**後的資料集*
* **(all)Data.csv**
* 資料架構
    | 欄位名稱 | 說明定義 | 資料型態 |
    | -------- | -------- | -------- |
    | ID               | 文本編號(key)         | String |
    | Comment_text     | 文本原始內容           | String |
    | clear_text_spell | 經資料前處理後的文本內容 | String |
    | harmful_label_binary | 標註是否為具傷害意圖的文本內容 | String |
    | **simple features** | 6種基礎特徵，共占**6個欄位** | 整數、浮點數 |
    | **sentiment features** | 2種情感特徵，共占**9個欄位** | 整數、浮點數 |
    | **w2c&d2c features** | 2種詞向量特徵，共占**100個欄位** | 浮點數 |
    | **BERT features** | BERT向量特徵，共占**768個欄位** | 浮點數 |

## 機器學習架構: 分類器建模
使用Scikit-learn套件進行建模任務，並使用REFCV的特徵選擇方法建模，以找到較有鑑別度之特徵內容。

> 使用檔案：get_simple_features.py
* 事先安裝
```
pip install sklearn
pip install numpy
pip install pandas
```
1. 讀出特徵欄位名稱檔案並存放在列表
    ```python=
    features_columns_ls = []
    features_columns_path = '\\Dataset\\dataset_features_columns.txt'
    f = open(features_columns_path)
    text = f.read()
    f.close()
    features_columns_ls = text.split()
    ```
3. 選定此次欲訓練的特徵類別
    ```python=
    # simple features
    # used_feat = features_columns_ls[13:19]
    # used_features_str = 'simple'

    # # # 只取情緒特徵
    # used_feat = features_columns_ls[4:13]
    # used_features_str = 'sentiment'

    # # 只取向量特徵
    # used_feat = features_columns_ls[19:119]
    # used_features_str = 'wordvec'

    # # # # 只取BERT向量特徵
    # used_feat = features_columns_ls[119:]
    # used_features_str = 'bert'

    # # simple+情緒特徵
    # used_feat = features_columns_ls[4:19]
    # used_features_str = 'simple&sentiment'

    # # simple+情緒特徵+詞向量(115)
    # used_feat = features_columns_ls[4:119]
    # used_features_str = 'simple&sentiment&wordvec'

    # # all features(883)
    used_feat = features_columns_ls[4:]
    used_features_str = 'all_883'
    ```
5. usecols可以讀取資料集的特定欄位，以降低多餘效能消耗
    ```python
    train_file = '(all)Data_train'
    test_file = '(all)Data_test'
    file_type = '.csv'
    
    input_path = '\\Dataset\\'
    train_X = pd.read_csv(input_path+train_file+file_type, header=0, usecols=used_feat)
    train_Y = pd.read_csv(input_path+train_file+file_type, header=0, usecols=['harmful_label_binary']).values.tolist()

    test_X = pd.read_csv(input_path+test_file+file_type, header=0, usecols=used_feat)
    test_Y = pd.read_csv(input_path+test_file+file_type, header=0, usecols=['harmful_label_binary']).values.tolist()
    ```
7. 設定分類模型與參數
    ```python=
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=1, multi_class='ovr', solver='sag', max_iter=10000, n_jobs=10)
    ```
9. 設定REFCV與相關參數:
    最小特徵選取數(min_feat_select_int)、批次刪除特徵數(step_int)
    ```python=
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score,recall_score , f1_score, classification_report, confusion_matrix
    from sklearn.feature_selection import RFECV
    target_ls = ['Harmful', 'non-Harmful']
    min_feat_select_int = 50  # 883 feat
    step_int = 157  #  (883-100)/5 = 156.6
    ```
11. 開始訓練，cv=StratifiedKFold(5)表示交叉驗證分成5組
    ```python=
    rfecv = RFECV(estimator=model, step = step_int, min_features_to_select = min_feat_select_int, 
              cv=StratifiedKFold(5), n_jobs=10)
    rfecv.fit(train_X, train_Y)
    ```
11. 輸出最終被模型選擇使用的特徵值列表
    ```python=
    features_select_ls = []
    for i, f in enumerate(used_feat):
        if rfecv.support_[i] :
            features_select_ls.append(f)
    
    path_out = 'RFECV_result\\{0}_{1}'.format(used_features_str, model.__class__.__name__)
    path_feat = '%s_OptimalFeature_list.csv' % path_out
    df_feat_select = pd.DataFrame(columns=['Feature_Name'])
    df_feat_select['Feature_Name'] = features_select_ls
    df_feat_select.to_csv(path_feat, index=True, index_label='No') 
    ```

1. 撰寫分類報告
    ```python=
    pred_Y = rfecv.predict(test_X)
    report_str = 'Use features : %s feature \n' % used_features_str
    report_str = report_str + "Use classifer: %s \n\n" % model
    report_str = report_str + 'Optimal number of features : %d \n------------------------------------------------------\n' % rfecv.n_features_
    report_str = report_str + 'Classification report : \n %s \n' % classification_report(test_Y, pred_Y, target_names=target_ls)
    report_str = report_str + 'Accuracy_score : {:.2} \n'.format(accuracy_score(test_Y, pred_Y))
    report_str = report_str + 'Precision_score :\t {:.2} \n'.format(precision_score(test_Y, pred_Y, labels=np.unique(test_Y)))
    report_str = report_str + 'Recall_score :\t\t {:.2} \n'.format(recall_score(test_Y, pred_Y, labels=np.unique(test_Y)))    
    report_str = report_str + 'F1 score :\t\t {:.2} \n'.format(f1_score(test_Y, pred_Y, labels=np.unique(test_Y)))
    report_str = report_str + 'Confusion matrix: \n %s \n------------------------------------------------------\n' % confusion_matrix(test_Y, pred_Y, labels=[1, 0])
    ```
1. 儲存分類報告與模型
    ```python=
    path_out = 'CV_BEST_parameters_report\\{0}_report.txt '.format(used_features_str)
    path_report = '%s_test_dataset_Pred_report.txt' % path_out
    report_f = open(path_report, mode='w')
    report_f.write(report_str)
    report_f.close()
    
    path_model = '%s_OptimalFeature_model.pkl' % path_out
    joblib.dump(rfecv, path_model)
    ```

* **輸出檔案**
1. all_883_LogisticRegression_OptimalFeature_list.csv
2. all_883_test_dataset_Pred_report.txt
3. all_883_OptimalFeature_model.pkl**


