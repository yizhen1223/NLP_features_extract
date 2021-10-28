# 文本特徵擷取技術筆記－詞向量特徵類別
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：建構二分類標籤

---
**目的**
隨著技術發展，文本特徵表示形式的發展也跟著有所突破，深度學習技術可以有效提升特徵概念，產生的複雜特徵帶來更好的預測效果。


## 使用資料集
*建構二分類標籤後的文本資料集*
* **Data.csv**
* 資料架構
    | 欄位名稱 | 說明定義 | 資料型態 |
    | -------- | -------- | -------- |
    | ID               | 文本編號(key)         | String |
    | Comment_text     | 文本原始內容           | String |
    | harmful_label_binary | 標註是否為具傷害意圖的文本內容 | String |

## 詞向量特徵(Word2vec)
詞向量這種基於分布的特徵表示方法可以體現單詞在上下文的重要性，使用gensim套件作為取得詞向量特徵之工具，為每筆文本內容取得**50個Word2vec特徵**。

## 段落向量特徵(Doc2vec)
段落向量（Sentence embeddings，又稱Doc2vec、Paragraph2vec）為詞向量的延伸技術，可以將它理解成代表一個句子的向量表示，詞向量是將每個單詞映射到向量空間中，讓相似語意的單詞分配到相似空間位置上，而段落向量則是將每個句子映射到向量空間，讓相似語義的段落或句子也會落在相似的向量空間位置。因此段落向量適合用於文本長度較長與段落較多的內容上，於是同樣利用gensim套件來為每筆資料**取得50個Doc2vec特徵**，作為該筆資料的段落向量特徵內容。

| 特徵欄位名稱 | 說明定義 | 資料型態 |
| -------- | -------- | -------- |
| w2c_1至w2c_50| 單詞的向量表示法，共取得50個。| 浮點數 | 
| d2c_1至d2c_50| 段落的向量表示法，共取得50個。| 浮點數 | 


> 使用檔案：get_w2c_d2c.py
* 事先安裝
```
pip install gensim
pip install spacy
pip install numpy
pip install pandas
```
* 正確引用套件
```python=
# 讓UtilWordEmbedding可以正確被引用，在sys.path加入套件所在資料夾
import os, sys, re, string
sys.path.append('package')
from UtilWordEmbedding import MeanEmbeddingVectorizer
from UtilWordEmbedding import DocPreprocess
from UtilWordEmbedding import DocModel
```
1. 讀取資料集檔案
    ```python=
    folerName = 'Dataset\\'
    fileName = 'Data.csv'
    df = pd.read_csv(folerName + fileName, encoding='latin1')

    # 先將要使用到的文本與標籤，轉換資料型態為字串
    df['comment_text'] = df['comment_text'].astype(str)
    df['harmful_label_binary'] = df['harmful_label_binary'].astype(str)
    ```
    
3. 利用spacy載入相對應語言的模型，以便後續過濾停用詞
    ```python=
    # 如果spacy.load出錯，改用import en_core_web_sm 並使用en_core_web_sm.load()
    nlp = en_core_web_sm.load()
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    all_docs = DocPreprocess(nlp, stop_words, df['comment_text'], df['harmful_label_binary'])
    ```
3. 用gensim的Word2Vec函數來訓練詞向量
    ```python=
    workers = multiprocessing.cpu_count()
    word_model = Word2Vec(all_docs.doc_words,
                          min_count=2,
                          size=50,
                          window=5,
                          workers=workers,
                          iter=100)
    ```
5. **詞向量**的取得: 平均每個文檔的詞嵌入(doc_vec)
    ```python=
    # 將word2vec模型作引數，代入MeanEmbedding-Vectorizer
    # 待類別實例化後(instantiated)，再使用class.transform，並代入預存文本的詞陣列(list of tokens for each document)，
    # 它的output就是我們要的文本向量。
    mean_vec_tr = MeanEmbeddingVectorizer(word_model)
    doc_vec = mean_vec_tr.transform(all_docs.doc_words)
    ```
7. 將詞向量特徵存檔
    ```python=
    out_fileName = '(word_mean_d2c)'+fileName
    np.savetxt(folerName+out_fileName, doc_vec, delimiter=',')
    ```
    
9. **段落向量**的取得
    ```python=
    # 建立Doc2vec
    # Configure keyed arguments for Doc2Vec model.
    dm_args = {
        'dm': 1,
        'dm_mean': 1,
        'vector_size': 50,
        'window': 5,
        'negative': 5,
        'hs': 0,
        'min_count': 2,
        'sample': 0,
        'workers': workers,
        'alpha': 0.025,
        'min_alpha': 0.025,
        'epochs': 100,
        'comment': 'alpha=0.025'
    }
    
    dm = DocModel(docs=all_docs.tagdocs, **dm_args)
    dm.custom_train()
    ```
11. 將段落向量以Dataframe形式儲存
    ```python=
    dm_doc_vec_ls = []
    for i in range(len(dm.model.docvecs)):
        dm_doc_vec_ls.append(dm.model.docvecs[i])
        
    dm_doc_vec = pd.DataFrame(dm_doc_vec_ls)
    ```
7. 將詞向量特徵存檔
    ```python=
    out_fileName = '(dm_d2c)'+fileName
    dm_doc_vec.to_csv(folerName+out_fileName, index=False, header=False)
    ```
1. 建立標籤csv檔案
    ```python=
    target_labels = all_docs.labels
    out_fileName = '(target_labels)'+fileName
    target_labels.to_csv(folerName+out_fileName, index=False, header=True)
    ```

* **輸出檔案**
    1. **(word_mean_d2c)Data.csv**
    2. **(dm_d2c)Data.csv**
    3. **(target_labels)Data.csv**
---
### 參考來源
兩種詞向量特徵表示形式的取得，很大程度參考Tom Lin文章以其[GitHub範例Code](https://github.com/TomLin/MeetUp)，其中使用的套件工具與流程，在此表達萬分的感謝！
[Tom Lin - [NLP]不同詞向量在文本分類上的表現與實作](https://medium.com/ai-academy-taiwan/nlp-%E4%B8%8D%E5%90%8C%E8%A9%9E%E5%90%91%E9%87%8F%E5%9C%A8%E6%96%87%E6%9C%AC%E5%88%86%E9%A1%9E%E4%B8%8A%E7%9A%84%E8%A1%A8%E7%8F%BE%E8%88%87%E5%AF%A6%E4%BD%9C-e72a2daecfc)



---

### 下一階段：[合併特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_merge_features.md)

