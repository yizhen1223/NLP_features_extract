# 文本特徵擷取技術筆記－合併特徵類別
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：取得[情感](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_tidy_text_sentimentr.md)、[詞向量](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_get_w2c_d2c.md)、[BERT向量](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_get_BERT_representationModel.md)特徵類別

---
**目的**
將上一階段所取得的特徵類別進行合併。

## 使用資料集
1. (tidytext)(sentimentr)(simple)(clean)Data.csv
2. (word_mean_d2c)Data.csv
3. (dm_d2c)Data.csv
4. (bert_vec)(clean)Data.csv

## 合併後資料集

* **資料架構**
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
* **資料集欄位數=id+orinal_text+label+883個特徵欄位(維度)=890**

> 使用檔案：merge_features.py

1. 導入所需資料集
    ```python=
    dir_path = 'Dataset\\'
    df_features = pd.read_csv(dir_path+'(tidytext)(sentimentr)(simple)(clean)Data.csv')
    df_w_m_d2c= pd.read_csv(dir_path+ '(word_mean_d2c)Data.csv', header=None)
    df_d2c = pd.read_csv(dir_path+'(dm_d2c)Data.csv', header=None)
    df_bert= pd.read_csv(dir_path+'(bert_vec)(clean)Data.csv')
    ```
3. 為w_m_d2c、d2c添增標題列
    ```python=
    w2c_header_name=[]
    d2c_header_name=[]
    # range(1, 51):實際會從1-50
    for i in range(1, 51):
        w2c_header_str = 'w2c_%d' % (i) 
        d2c_header_str = 'd2c_%d' % (i) 
        w2c_header_name.append(w2c_header_str)
        d2c_header_name.append(d2c_header_str)

    df_w_m_d2c.columns=w2c_header_name
    df_d2c.columns=d2c_header_name
    ```
5. 用數據集的索引值來幫w_m_d2c、d2c添增key欄位
    ```python=
    df_w_m_d2c_key = pd.concat([df_features['id'], df_w_m_d2c], axis=1)
    df_d2c_key = pd.concat([df_features['id'], df_d2c], axis=1)
    ```
7. 依據key欄位合併所有欄位
    ```python=
    df_features_merge = pd.merge(df_features, df_w_m_d2c_key, on='id')
    df_features_merge = pd.merge(df_features_merge, df_d2c_key, on='id')
    df_features_merge = pd.merge(df_features_merge, df_bert, on='id')
    ```
9. 儲存: 包含文本內容與標籤的資料也保存起來(890個欄位)
    ```python=
    df_features_merge.to_csv(dir_path+'(all)Data.csv') ,index=False )
    ```


* **輸出檔案**
    **(all)Data.csv**
---

### 下一階段：[建立傷害性文本分類模型](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_ml_setup_RFECV.md)
