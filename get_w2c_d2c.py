#!/usr/bin/env python
# coding: utf-8
from gensim.test.utils import get_tmpfile, datapath
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm 
import multiprocessing


# 讓UtilWordEmbedding可以正確被引用，在sys.path加入套件所在資料夾
import os, sys, re, string
sys.path.append('package')

from UtilWordEmbedding import MeanEmbeddingVectorizer
from UtilWordEmbedding import DocPreprocess
from UtilWordEmbedding import DocModel


# 用pandas讀取資料集檔案
folerName = 'Dataset\\'
fileName = '(tidytext)(sentimentr)(simple)(clean)Data.csv'
df = pd.read_csv(folerName + fileName, encoding='latin1')

# 先將要使用到的文本與標籤，轉換資料型態為字串
df['comment_text'] = df['comment_text'].astype(str)
df['harmful_label_binary'] = df['harmful_label_binary'].astype(str)



# 利用spacy載入相對應語言的模型，以便後續過濾停用詞
# 如果spacy.load出錯，改用import en_core_web_sm 並使用en_core_web_sm.load()
nlp = en_core_web_sm.load()
stop_words = spacy.lang.en.stop_words.STOP_WORDS
all_docs = DocPreprocess(nlp, stop_words, df['comment_text'], df['harmful_label_binary'])



workers = multiprocessing.cpu_count()
print('number of cpu: {}'.format(workers))
# assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise."



# 用gensim的Word2Vec函數來訓練詞向量
word_model = Word2Vec(all_docs.doc_words,
                      min_count=2,
                      size=50,
                      window=5,
                      workers=workers,
                      iter=100)



# 平均每個文檔的詞嵌入
# 將word2vec模型作引數，代入MeanEmbedding-Vectorizer
# 待類別實例化後(instantiated)，再使用class.transform，並代入預存文本的詞陣列(list of tokens for each document)，
# 它的output就是我們要的文本向量。
mean_vec_tr = MeanEmbeddingVectorizer(word_model)
doc_vec = mean_vec_tr.transform(all_docs.doc_words)



out_fileName = '(word_mean_d2c)'+fileName
np.savetxt(folerName+out_fileName, doc_vec, delimiter=',')



# 建立Doc2vec
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




# Instantiate a pv-dm model.
dm = DocModel(docs=all_docs.tagdocs, **dm_args)
dm.custom_train()


# Save doc2vec as feature dataframe.
dm_doc_vec_ls = []
for i in range(len(dm.model.docvecs)):
    dm_doc_vec_ls.append(dm.model.docvecs[i])


dm_doc_vec = pd.DataFrame(dm_doc_vec_ls)
out_fileName = '(dm_d2c)'+fileName
dm_doc_vec.to_csv(folerName+out_fileName, index=False, header=False)


# 建立標籤csv檔案
target_labels = all_docs.labels
out_fileName = '(target_labels)'+fileName
target_labels.to_csv(folerName+out_fileName, index=False, header=True)

