# 文本特徵擷取技術筆記－情感特徵類別
#### 論文名稱：應用機器學習與深度學習方法在網路傷害性新聞與惡意評論偵測辨識上之研究
### 上一階段：[取得基礎特徵類別](https://hackmd.io/@yizhenchen/r1ZhDXBIF)

---
**目的**
針對網路霸凌、仇恨評論等帶有傷害意圖的文本內容，這些內容存在的負面情感特徵（例：情感極性分類）對於接收者產生的負面影響有顯著關係。主要擷取出**兩種情感特徵：情感極性分數、八種情緒類別**。

## 使用資料集
*經**基礎特徵類別**擷取後的文本資料集*
* **(simple)(clean)Data.csv**
* 資料架構
    | 欄位名稱 | 說明定義 | 資料型態 |
    | -------- | -------- | -------- |
    | ID               | 文本編號(key)         | String |
    | Comment_text     | 文本原始內容           | String |
    | clear_text_spell | 經資料前處理後的文本內容 | String |
    | harmful_label_binary | 標註是否為具傷害意圖的文本內容 | String |
    | txt_len | 文本長度，即文本總字數。 | 整數 |
    | cap_n | 文本中的全英大寫字詞出現次數。 | 整數 |
    | cap_mean | 文本中大寫字詞出現次數除以文本長度。 | 浮點數 |
    | num_n | 文本中出現數字的數量。 | 整數 |
    | num_mean | 文本中出現數字的數量除以文本長度。 | 浮點數 |
    | sen_n | 文本中的句子總數。 | 整數 |


## 情感極性分數
將針對惡意評論資料集中的文本內容，使用**R語言中的情感分析套件Sentimentr** 計算每個句子的情感分數，最終計算文本中所有句子情感分數的平均值，作為該文本的情感分數。情感分數的數值範圍介於-1至1之間，數值趨近於-1顯示該文本平均屬於負面情感，趨近於1則顯示該文本平均屬於正面情感，若趨近於0的數值則代表情感表現不明顯，為中性情感文本。


| 特徵欄位名稱 | 說明定義 | 資料型態 |
| -------- | -------- | -------- |
| ave_sentiment | 情感極性分數（介於-1至1。） | 浮點數 |
		
> 使用檔案：sentimentr.R
* 事先安裝套件
```r
install.packages("sentimentr")

# 導入所需套件
library(sentimentr)
```
1. 導入資料集
    ```r    
    path = 'Dataset\\'
    setwd(path)
    script <- read.csv("(simple)(clean)Data.csv", stringsAsFactors=FALSE)
    script$index = row.names(script)
    ```
3. element_id 
    新增一欄位elemet_id，存放評論編號(索引值)，從1開始。
    切記script$後面放的是編號欄位名(ex.ID)
    ```r
    script_len <- length(script$id)
    script$element_id <- seq(from=1,to=script_len,by=1)
    ```
5. get_sentences() 可拆分句子
    將原始文本欄位導入get_sentences，去拆分句子
    切記script$後面放的是原始評論欄位名(ex.comment_text)
    ```r
    comment_text <- get_sentences(script$comment_text)
    ```
7. sentiment_by() 可以顯示該欄位評論中每個句子的平均情緒分數
    ```r
    comment_text_sentiment <- sentiment_by(comment_text)
    ```
9. 合併兩個數據框，tidy_sen將會根據element_id顯示其meanSentiment
    ```r
    comment_sentiment_merge <- merge(script, comment_text_sentiment)
    ```
11. 遺失值補0
    ```r
    comment_sentiment_merge[is.na(comment_sentiment_merge)] <- 0
    ```

11. 刪除數據集中原本沒有的element_id與word_count欄位
    ```r
    comment_sentiment_merge <- subset(comment_sentiment_merge, select = c(-element_id, -word_count))
    ```

11. 最後存成CSV檔
    用paste連接路徑名(path_out)與檔案名，row.names =F 表示不儲存列編號
    ```r
    write.csv(comment_sentiment_merge, file=paste(path,'(sentimentr)(simple)(clean)Data.csv', sep=''), row.names=F)
    ```

* **輸出檔案**
    **(sentimentr)(simple)(clean)Data.csv**
---


## 八種情感類別
使用**R語言中的tidytext套件**來針對文本內容進行八種情緒的辨別，tidytext根據加拿大國家研究委員會（NRC）所發布的情感詞典，將所有單詞分成八個情緒類別：**憤怒（Anger）、期待（Anticipation）、厭惡（Disgust）、恐懼（Fear）、喜悦（Joy）、悲傷（Sadness）、驚訝（Surprise）、信任（Trust）**。
tidytext會計算出文本內容出現的八類情緒單詞數量，主要以高於平均數量的情緒類別作為該文本較為顯著的情緒標籤，意即一篇文本可能出現兩種以上的情緒標籤，因此這八類情緒標籤會各別以單一欄位儲存內容，內容為數值0或1，來代表無或有此類情緒。


| 特徵欄位名稱 | 說明定義 | 資料型態 |
| -------- | -------- | -------- |
| anger	| 憤怒標籤（0為無，1為含有該情緒）	|整數|
| anticipation	|期待標籤（0為無，1為含有該情緒）	|整數|
| disgust	| 厭惡標籤（0為無，1為含有該情緒）	|整數|
| fear	|恐 懼標籤（0為無，1為含有該情緒）	|整數 |
| joy	| 喜悅標籤（0為無，1為含有該情緒）	|整數|
| sadness	| 悲傷標籤（0為無，1為含有該情緒）	|整數|
| surprise	| 驚訝標籤（0為無，1為含有該情緒）	|整數|
| trust	| 信任標籤（0為無，1為含有該情緒）	|整數|

> 使用檔案：tidy_text_sentiment.R
> 使用資料集：(sentimentr)(simple)(clean)Data.csv
* 事先安裝套件
```r
install.packages("dplyr")
install.packages("tidytext")
install.packages("tidyr")
install.packages("textdata")

# 導入所需套件
library(dplyr)
library(tidytext)
library(tidyr)
library(textdata)
```
1. 導入資料集
    ```r    
    path = 'Dataset\\'
    setwd(path)
    script <- read.csv("(simple)(clean)Data.csv", stringsAsFactors=FALSE)
    script$index = row.names(script)
    ```
3. tidy_test 
    依據經拼字檢查的文字欄位(clear_text_spell)，將每個句子拆成一個個單詞
    ```r
    tidy_test <- script %>% 
                unnest_tokens(word, clear_text_spell)
    ```
5. tidy_sen
    使用inner_join調用NRC情緒詞典，會顯示出每評論中每個有情緒單詞的情緒標籤
    另外把情緒數量寫入tidy_sen，並去除正負向，只留下八種情緒
    ```r
    tidy_sen <- tidy_test %>% 
              inner_join(get_sentiments("nrc")) %>%
              count(id, sentiment) %>%
              filter(sentiment != "negative" & sentiment != "positive") %>%
              arrange(id)
    ```
7. tidy_sen_comment 計算每條評論的平均情緒數
    ```r
    tidy_sen_comment <- group_by(tidy_sen, id) %>%
                        summarise(nSentiment= n(), meanSentiment=mean(n))
    ```
9. 合併兩個數據框，tidy_sen將會根據索引值顯示其meanSentiment
    ```r
    tidy_sen_comment_merge <- merge(tidy_sen, tidy_sen_comment)
    ```
11. tidy_sen_comment_emotion_Label 建立新數據框存放顯著情緒標籤
    創建新欄位Sign_sen，指定為meanSentiment與n於get_sign_emotion的回傳結果
    ```r
    tidy_sen_comment_emotion_Label <- tidy_sen_comment_merge
    tidy_sen_comment_emotion_Label$Sign_sen <- ifelse(tidy_sen_comment_merge$n > tidy_sen_comment_merge$meanSentiment, 1, 0)
    ```
1. 儲存每條評論的顯著情緒標籤
    ```r
    tidy_sign_sen_comment <- group_by(tidy_sen_comment_emotion_Label, id) %>%
    filter(Sign_sen==1) %>%
    summarise(Sign_sen_str= list(sentiment))
    ```
    
1. 複製成新數據框存放八個情緒標籤
    ```r
    tidy_eight_sen_label <- tidy_sign_sen_comment
    ```
    
1. 使用ifelse與搭配grepl判斷情緒標籤是否存在於顯著情緒字串中，傳回0或1
    ```r
    tidy_eight_sen_label$anger <- ifelse(grepl('anger', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$anticipation <- ifelse(grepl('anticipation',  tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$disgust <- ifelse(grepl('disgust', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$fear <- ifelse(grepl('fear', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$joy <- ifelse(grepl('joy', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$sadness <- ifelse(grepl('sadness', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$surprise <- ifelse(grepl('surprise', tidy_eight_sen_label$Sign_sen_str),1,0)
    tidy_eight_sen_label$trust <- ifelse(grepl('trust', tidy_eight_sen_label$Sign_sen_str),1,0)
    ```
    
1. 與原始資料結合成新的數據框
    By=結合要依據的欄位(就是索引值ID或唯一值)，可使用all=T保留沒對應到的資料列
    ```r
    data_eight_sen_merge <- merge(script, tidy_eight_sen_label, by='id', all=T)
    ```
    
1. 遺失值補0
    ```r
    data_eight_sen_merge[is.na(data_eight_sen_merge)] <- 0
    ```
    
1. 刪除數據集中多餘欄位，最後存成CSV檔
    ```r
    data_eight_sen_merge <- subset(data_eight_sen_merge, select = -Sign_sen_str)
    write.csv(data_eight_sen_merge, file=paste(path,'(tidytext)(sentimentr)(simple)(clean)Data.csv', sep=''), row.names=F)
    ```


* **輸出檔案**
    **(tidytext)(sentimentr)(simple)(clean)Data.csv**

---

### 下一階段：[合併特徵類別](https://hackmd.io/@yizhenchen/BJlLt-wLt)
