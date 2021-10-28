# 文本特徵擷取技術筆記－資料清理
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：建構二分類標籤

---
**目的**
將取得的資料集進行**資料清理**，針對文本資料的資料清理將進行英文斷詞，以及過濾標點符號、表情符號、超連結標籤、停用詞（Stop Words）等等處理作業，另外也將進行**拼寫檢查（Spell Check）**，還原那些因為使用者輸入錯誤而拼錯的單詞。資料清理目的在於降低資料集中的雜訊數量，盡力去避免模型會因為雜訊資料而導致預測錯誤提升的機會。


## 使用資料集
*建構二分類標籤後的文本資料集*
* **Data.csv**
* 資料架構
    | 欄位名稱 | 說明定義 | 資料型態 |
    | -------- | -------- | -------- |
    | ID               | 文本編號(key)         | String |
    | Comment_text     | 文本原始內容           | String |
    | harmful_label_binary | 標註是否為具傷害意圖的文本內容 | String |


## 資料清理


> 使用檔案：dataset_preprocess.py
* 事先安裝
```
pip install nltk
pip install symspellpy
pip install pandas
```
1. pbx_into_sp: 簡易斷詞(單詞之間以空格區隔，移除多餘空格與標點符號)
    ```python    
    def pbx_into_sp(org_str) : 
        org_str = re.sub('\W+', ' ', org_str).replace("_", '').strip()
        return org_str
    ```
3. mark_remove: 刪除標記(刪除@與#符號開頭到空白處)、刪除超連結(刪除至結尾)
    ```python
    def mark_remove(cleanText):
        while ('@' in cleanText) | ('#' in cleanText) | ('http' in cleanText):
            if '@' in cleanText :
                atIndex = cleanText.find('@')
                backStr = cleanText[atIndex:]  # 定義符號後面的字為新字串，因為要從@後頭的字開始找第一個空白
                if backStr.find(' ') == -1 :
                    cleanText = cleanText.replace('@','').strip()
                    break
                spIndex = backStr.find(' ')
                atStr = cleanText[atIndex:atIndex+spIndex+1]   # +1才能包含到空白
                cleanText = cleanText.replace(atStr,'').strip()

            if '#' in cleanText :
                hashIndex = cleanText.find('#')
                backStr = cleanText[hashIndex:]
                if backStr.find(' ') == -1 :
                    cleanText = cleanText.replace('#','').strip()
                    break
                spIndex = backStr.find(' ')
                hashStr = cleanText[hashIndex:hashIndex+spIndex+1]
                cleanText = cleanText.replace(hashStr,'').strip()

            if 'http' in cleanText:
                httpIndex = cleanText.index('http')
                backStr = cleanText[httpIndex:]
                if backStr.find(' ') == -1 :
                    cleanText = cleanText[:httpIndex].strip()  # 只保留前面
                    break
                spIndex = backStr.find(' ')
                httpStr = cleanText[httpIndex:httpIndex+spIndex+1]
                cleanText = cleanText.replace(httpStr,'').strip()
        return cleanText
    ```
5. clear_text: 評論文字前處理(清除標點符號與超連結，以空格區隔單詞)
    ```python
    def clear_text(comment) : 
        clear_comment = mark_remove(comment)
        clear_comment = pbx_into_sp(comment)
        return clear_comment
    
    # 建立新欄位'clear_comment'存放清理好的評論
    df['clear_text'] = df['comment_text'].apply(lambda x: clear_text(x))
    ```
7. rmsw_function: 輸入文本與使用的停用詞列表進行過濾，回傳過濾後文本
    ```python
    # 進行停用詞過濾
    # 下載NLTK的停用詞列表
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    
    def rmsw_function(text, stopword_list):
        return ' '.join([word for word in word_tokenize(text) if word not in stopword_list])
    
    # 使用前面初步清理過的clear_comment來過濾停用詞
    # 建立新欄位'clear_comment_sw'存放過濾停用詞的評論
    df['clear_text_sw'] = df['clear_text'].apply(lambda x: rmsw_function(x, nltk_stopwords))
    ```
    
9. spell_check: 拼寫檢查函式，輸入文本回傳替換新單詞的文本
    ```python
    def spell_check(intput_str) : 
        input_term = intput_str.split()
        for i in input_term:
            suggestions = sym_spell.lookup(i, Verbosity.CLOSEST, 
                                           max_edit_distance=2, 
                                           include_unknown=True, 
                                           ignore_token=r"\w+\d", 
                                           transfer_casing=True)
            if len(suggestions) > 1:
                word_index = input_term.index(i)   # 找到錯別字在原字串中的索引位置
                new_word = str(suggestions[0]).split(', ')  # 選擇第一個建議的單詞
                input_term[word_index]=new_word[0]   # 替換原單字成新單詞

        return space.join(input_term)
    
    # 載入SymSpell套件，下載辭典
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    space = ' '
    # 使用前面過濾停用詞的clear_text_sw來進行拼寫檢查
    # 建立新欄位'clear_comment_sw'存放過濾停用詞的評論
    df['clear_text_spell'] = df['clear_text_sw'].apply(lambda x: spell_check(x))
    ```

*  **三種資料清理後的文本欄位，僅保存clear_text_spell欄位即可**
* **輸出檔案**
    **(clean)Data.csv**
---

### 下一階段：[取得基礎特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_get_simple_features.md)、[取得BERT向量特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_get_BERT_representationModel.md)、[取得詞袋特徵類別](https://github.com/yizhen1223/NLP_features_extract/blob/main/README_set_bow_model.md)

