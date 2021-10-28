#!/usr/bin/env python
# coding: utf-8
# 此檔案將進行資料前處理，包含基本斷詞（移除標點符號與超連結、Hashtags），以及過濾停用詞與拼寫檢查
# 基本斷詞：clear_text包含（pbx_into_sp & mark_remove）、停用詞過濾：rmsw_function、拼寫檢查：spell_check
# 定義三個新欄位：
# (1) clear_comment（進行基本斷詞）
# (2) clear_comment_sw（進行基本斷詞、過濾停用詞）
# (3) clear_comment_spell（進行基本斷詞、過濾停用詞、拼寫檢查）

import pandas as pd
import os, sys, re, string, time
import pkg_resources
import nltk
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity


# 讀取csv
folerName = 'Dataset\\'
fileName = 'Data.csv'
df = pd.read_csv(folerName + fileName, encoding='latin1')



# 簡易斷詞(單詞之間以空格區隔，移除多餘空格與標點符號)
def pbx_into_sp(org_str) : 
    # Replace punctuation marks into spaces
    org_str = re.sub('\W+', ' ', org_str).replace("_", '').strip()
    return org_str




# 刪除標記(刪除@與#符號開頭到空白處)、刪除超連結(刪除至結尾)
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
#             print('removeAtSymbol = %s' % cleanText)
            
        if '#' in cleanText :
            hashIndex = cleanText.find('#')
            backStr = cleanText[hashIndex:]
            if backStr.find(' ') == -1 :
                cleanText = cleanText.replace('#','').strip()
                break
            spIndex = backStr.find(' ')
            hashStr = cleanText[hashIndex:hashIndex+spIndex+1]
            cleanText = cleanText.replace(hashStr,'').strip()
#             print('removeHashSymbol = %s' % cleanText)

        if 'http' in cleanText:
            httpIndex = cleanText.index('http')
            backStr = cleanText[httpIndex:]
            if backStr.find(' ') == -1 :
                cleanText = cleanText[:httpIndex].strip()  # 只保留前面
                break
            spIndex = backStr.find(' ')
            httpStr = cleanText[httpIndex:httpIndex+spIndex+1]
            cleanText = cleanText.replace(httpStr,'').strip()
#             cleanText = cleanText[:httpIndex].strip()
#             print('removeHTTP = %s' % cleanText)
#     print('cleanText = %s' % cleanText)
    return cleanText


# 評論文字前處理(清除標點符號與超連結，以空格區隔單詞)
def clear_text(comment) : 
    # Delete HTML & hashtags
    clear_comment = mark_remove(comment)
    
    # Replace punctuation marks into spaces
    clear_comment = pbx_into_sp(comment)
    
    return clear_comment



# 建立新欄位'clear_comment'存放清理好的評論
df['clear_text'] = df['comment_text'].apply(lambda x: clear_text(x))

# 進行停用詞過濾
# 下載NLTK的停用詞列表
nltk.download('punkt')
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')


# 定義函式，輸入文本與使用的停用詞列表進行過濾，回傳過濾後文本
def rmsw_function(text, stopword_list):
    return ' '.join([word for word in word_tokenize(text) if word not in stopword_list])




# 使用前面初步清理過的clear_comment來過濾停用詞
# 建立新欄位'clear_comment_sw'存放過濾停用詞的評論
df['clear_text_sw'] = df['clear_text'].apply(lambda x: rmsw_function(x, nltk_stopwords))


# 定義拼寫檢查函式，輸入文本回傳替換新單詞的文本
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
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


space = ' '
df['clear_text_spell'] = df['clear_text_sw'].apply(lambda x: spell_check(x))



# 儲存檔案
df.to_csv(folerName + '(clean)' + fileName, index=False)




