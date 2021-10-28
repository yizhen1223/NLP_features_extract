# 安裝所需套件
install.packages("sentimentr")

# 導入所需套件
library(sentimentr)

# 導入資料集
path = 'Dataset\\'
setwd(path)
script <- read.csv("(simple)(clean)Data.csv", stringsAsFactors=FALSE)
# 如果沒有索引值或唯一值，可以用預設的index(在R叫row names)作為索引欄位
# script$index = row.names(script)

# 新增一欄位elemet_id，存放評論編號(索引值)，從1開始
# 切記script$後面放的是編號欄位名(ex.ID)
script_len <- length(script$id)
script$element_id <- seq(from=1,to=script_len,by=1) ##1~20，中間相隔1

# 將原始文本欄位導入get_sentences，去拆分句子
# 切記script$後面放的是原始評論欄位名(ex.comment_text)
comment_text <- get_sentences(script$comment_text)

# 可以顯示每個句子的字數、情緒分數
# sentiment(comment_text)

# 主要用這個：sentiment_by()可以顯示該欄位評論中每個句子的平均情緒分數
comment_text_sentiment <- sentiment_by(comment_text)

# 合併兩個數據框，tidy_sen將會根據comment_counter顯示其meanSentiment
comment_sentiment_merge <- merge(script, comment_text_sentiment)
View(comment_sentiment_merge)

# 遺失值補0
comment_sentiment_merge[is.na(comment_sentiment_merge)] <- 0

# 刪除數據集中原本沒有的element_id與word_count欄位
# comment_sentiment_merge$element_id <- NULL
comment_sentiment_merge <- subset(comment_sentiment_merge, select = c(-element_id, -word_count))

# 最後存成CSV檔，用paste連接路徑名(path_out)與檔案名，row.names =F 表示不儲存列編號
write.csv(comment_sentiment_merge, file=paste(path,'(sentimentr)(simple)(clean)Data.csv', sep=''), row.names=F)


# sentiment_by() Returns a data.table with grouping variables plus:
# • element_id - The id number of the original vector passed to sentiment
# • sentence_id - The id number of the sentences within each element_id
# • word_count - 通過分組變量求和的字數
# • sd - Standard deviation (sd) 情緒/極性得分的標準偏差（sd）（按分組變量）
# • ave_sentiment - Sentiment/polarity score 情感/極性得分按分組變量平均
