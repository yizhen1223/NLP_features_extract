# 文本特徵擷取技術筆記－BERT向量特徵類別
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：[資料清理](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_dataset_preprocess.md)（基礎斷詞與標點符號清除）

---
**目的**
BERT利用注意力機制來處理輸入和輸出之間的依賴性。由於消除了對先前單詞的順序依賴性，因此能有效提升訓練模型的效能，從而提升效率於長期建模的依賴性。BERT的部署十分複雜，但所幸現在有許多已經完成預訓練的BERT模型，只要對模型進行微調（fine-tuned）就可以直接承接於各種任務上。使用預設的BERTBASE模型，該預訓練模型已經由Google團隊先使用Wikipedia和BookCorpus數據集來進行預訓練。

## 使用資料集
*經**資料前處理**後的文本資料集*
* **(clean)Data.csv**
* 資料架構
    | 欄位名稱 | 說明定義 | 資料型態 |
    | -------- | -------- | -------- |
    | ID               | 文本編號(key)         | String |
    | Comment_text     | 文本原始內容           | String |
    | clear_text_spell | 經資料前處理後的文本內容 | String |
    | harmful_label_binary | 標註是否為具傷害意圖的文本內容 | String |

## BERT向量特徵類別
本研究利用Simple Transformers套件(Rajapakse, 2019)取得BERTBASE-CASED來生成每筆文本內容的向量表示法作為特徵，BERTBASE-CASED此預訓練模型架構由768個隱藏單元所組成，因此每筆文本內容經BERTBASE-CASED生成的向量其特徵維度皆為768。


| 特徵欄位名稱 | 說明定義 | 資料型態 |
| -------- | -------- | -------- |
| bert_1至bert_768 | BERT的向量表示法，輸出長度為768個。 | 浮點數 |


> 使用檔案：get_BERT_representationModel.py
> 請於支援**GPU、CUDA**的環境上執行
* 事先安裝
```
pip install simpletransformers
pip install numpy
pip install pandas
pip install torch
```
1. 導入資料集
    ```python=    
    folerName = os.path.abspath(os.path.join(os.getcwd(), "../")) + '\\Dataset\\'
    fileName = '(clean)Data'
    fileType = '.csv'
    df = pd.read_csv(folerName + fileName + fileType, encoding='latin1')
    ```
3. sentence_list: 存放所有清理後的文本內容
    ```python=
    df['clear_text_spell'] = df['clear_text_spell'].astype(str)
    sentence_list = df['clear_text_spell']
    ```
5. 設定BERT預訓練模型參數
    ```python=
    model_args = ModelArgs(max_seq_length=156)
    model = RepresentationModel('bert', 'bert-base-cased',
                                args=model_args, use_cuda=True)     ```
7. 使用BERT模型取得向量表達式，並轉換格式
    ```python=
    word_embeddings = model.encode_sentences(sentence_list, combine_strategy='mean')
    np_word_embeddings = np.array(word_embeddings) 
    ```
9. 最後要以Dataframe形式儲存，因此先為768個bert_vec添增標題列(欄位名)
    ```python=
    bert_header_name=[]
    # range(1, 51):實際會從1-768
    for i in range(1, 769):
        bert_header_str = 'bert_%d' % (i)
        bert_header_name.append(bert_header_str)
    ```
11. 儲存檔案
    ```python=
    df_bert_vec = pd.DataFrame(np_word_embeddings, columns=bert_header_name)
    df_bert_vec_key = pd.concat([df['ID'], df_bert_vec], axis=1)

    out_name = '(bert_vec)'+fileName
    df_bert_vec_key.to_csv(folerName + out_name + fileType ,index=False)
    ```


* **輸出檔案**
    **(bert_vec)(clean)Data.csv**
---

### 下一階段：[合併特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_merge_features.md)

