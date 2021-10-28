# 安裝所需套件
install.packages("dplyr")
install.packages("tidytext")
install.packages("tidyr")
install.packages("textdata")

# 導入所需套件
library(dplyr)
library(tidytext)
library(tidyr)
library(textdata)

# 導入資料集
path = 'Dataset\\'
setwd(path)
script <- read.csv("(sentimentr)(simple)(clean)Data.csv", stringsAsFactors=FALSE)

# 如果沒有索引值或唯一值，可以用預設的index(在R叫row names)作為索引欄位
# script$index = row.names(script)

# 依據經拼字檢查的文字(clear_text_spell)欄位，將每個句子拆成一個個單詞
# 注意每個資料集的文本欄位名
tidy_test <- script %>%
unnest_tokens(word, clear_text_spell)
# View(tidy_test)

# inner_join調用NRC情緒詞典，會顯示出每評論中每個有情緒單詞的情緒標籤
# 另外把情緒數量寫入tidy_sen，並去除正負向，只留下八種情緒
tidy_sen <- tidy_test %>% 
  inner_join(get_sentiments("nrc")) %>%
  count(id, sentiment) %>%
  filter(sentiment != "negative" & sentiment != "positive") %>%
  arrange(id)
# View(tidy_sen)

# 計算每條評論的平均情緒數
tidy_sen_comment <- group_by(tidy_sen, id) %>%
summarise(nSentiment= n(), meanSentiment=mean(n))
# View(tidy_sen_comment)

# 合併兩個數據框，tidy_sen將會根據索引值顯示其meanSentiment
tidy_sen_comment_merge <- merge(tidy_sen, tidy_sen_comment)
# View(tidy_sen_comment_merge)

# 建立新數據框tidy_sen_comment_emotion_Label存放顯著情緒標籤
# 以$創建新欄位Sign_sen，指定為meanSentiment與n於get_sign_emotion的回傳結果
tidy_sen_comment_emotion_Label <- tidy_sen_comment_merge
tidy_sen_comment_emotion_Label$Sign_sen <- ifelse(tidy_sen_comment_merge$n > tidy_sen_comment_merge$meanSentiment, 1, 0)
# View(tidy_sen_comment_emotion_Label)

# 儲存每條評論的顯著情緒標籤
tidy_sign_sen_comment <- group_by(tidy_sen_comment_emotion_Label, id) %>%
filter(Sign_sen==1) %>%
summarise(Sign_sen_str= list(sentiment))
# View(tidy_sign_sen_comment)

# 複製成新數據框tidy_eight_sen_label存放八個情緒標籤
tidy_eight_sen_label <- tidy_sign_sen_comment

# 使用ifelse與搭配grepl判斷情緒標籤是否存在於顯著情緒字串中，傳回0或1
tidy_eight_sen_label$anger <- ifelse(grepl('anger', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$anticipation <- ifelse(grepl('anticipation', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$disgust <- ifelse(grepl('disgust', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$fear <- ifelse(grepl('fear', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$joy <- ifelse(grepl('joy', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$sadness <- ifelse(grepl('sadness', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$surprise <- ifelse(grepl('surprise', tidy_eight_sen_label$Sign_sen_str),1,0)
tidy_eight_sen_label$trust <- ifelse(grepl('trust', tidy_eight_sen_label$Sign_sen_str),1,0)
# View(tidy_eight_sen_label)

# 與原始資料結合成新的數據框，By=結合要依據的欄位(就是索引值ID或唯一值)，可使用all=T保留沒對應到的資料列
data_eight_sen_merge <- merge(script, tidy_eight_sen_label, by='id', all=T)
# View(data_eight_sen_merge)

# 遺失值補0
data_eight_sen_merge[is.na(data_eight_sen_merge)] <- 0
View(data_eight_sen_merge)

# 刪除數據集中多餘欄位
data_eight_sen_merge <- subset(data_eight_sen_merge, select = -Sign_sen_str)
# # 欄位Sign_sen_str資料型態為list會影響無法存成csv，要轉換成字串(character)型態
# # data_eight_sen_merge$Sign_sen_str <- as.character(data_eight_sen_merge$Sign_sen_str)

# 最後存成CSV檔，用paste連接路徑名與檔案名，row.names =F 表示不儲存列編號
write.csv(data_eight_sen_merge, file=paste(path,'(tidytext)(sentimentr)(simple)(clean)Data.csv', sep=''), row.names=F)
