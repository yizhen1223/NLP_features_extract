# 文本特徵擷取技術筆記－基礎特徵類別
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：[資料清理](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_dataset_preprocess.md)（基礎斷詞與標點符號清除）

---
**目的**
文本屬於非結構化資料，若要讓機器得以進行使用文本資料，就需要將其轉換成機器可理解的表示法，從而實踐機器對文本內容的理解。

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

## 基礎特徵類別
針對文本結構來取得的特徵值，為普通的文本分類任務中最基礎的特徵，這些特徵值是依據文本長度跟字符種類進行簡單的計量與運算來取得特徵值，基礎文本結構的特徵定義內容詳見下表。


| 特徵欄位名稱 | 說明定義 | 資料型態 |
| -------- | -------- | -------- |
| txt_len | 文本長度，即文本總字數。 | 整數 |
| cap_n | 文本中的全英大寫字詞出現次數。 | 整數 |
| cap_mean | 文本中大寫字詞出現次數除以文本長度。 | 浮點數 |
| num_n | 文本中出現數字的數量。 | 整數 |
| num_mean | 文本中出現數字的數量除以文本長度。 | 浮點數 |
| sen_n | 文本中的句子總數。 | 整數 |

> 使用檔案：get_simple_features.py
* 事先安裝
```
pip install nltk
pip install numpy
pip install pandas
```
1. txt_len: 文本長度
    ```python    
    def get_txt_len(txt):
        txt_list = txt.split(' ')
        return len(txt_list)
        
    df['txt_len'] = df['clear_text_spell'].apply(lambda x: get_txt_len(x))
    ```
3. cap_n
    ```python
    def get_cap_n(txt):
        big=0
        txt_list = txt.split()
        for t in txt_list:
            if t.isupper():
                big+=1
        return big
        
    df['cap_n'] = df['clear_text_spell'].apply(lambda x: get_cap_n(x))
    ```
5. cap_mean
    ```python
    df['cap_mean'] = df['cap_n']/df['txt_len']
    ```
7. num_n
    ```python
    def get_num_n(txt):
        num_count=0
        txt_list = txt.split()
        # 用for loop把輸入的字串跑一遍
        for t in txt_list:
            if t.isdigit():
                num_count+=1
        return num_count
    
    df['num_n'] = df['clear_text_spell'].apply(lambda x: get_num_n(x))
    ```
9. num_mean
    ```python
    df['num_mean'] = df['num_n']/df['txt_len']
    ```
11. sen_n
    ```python
    def get_sen_n(org_txt):
        # 會有人用多種符號來區隔句子，因此使用多種符號分割評論
        sen_list = re.split(r';|/|\!|\n|\?|\.', org_txt)
        # 過濾掉空字串
        sen_list = list(filter(None, sen_list))
        return len(sen_list)
    
    # 特別注意要使用清理前的字串作為輸入欄位
    df['sen_n'] = df['comment_text'].apply(lambda x: get_sen_n(x))
    ```


* **輸出檔案**
    **(simple)(clean)Data.csv**
---

### 下一階段：[取得情感特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_tidy_text_sentimentr.md)

