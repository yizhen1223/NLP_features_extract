#!/usr/bin/env python
# coding: utf-8
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs
import numpy as np
import pandas as pd
import torch
torch.cuda.current_device()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    folerName = 'Dataset\\'
    fileName = '(clean)Data'
    fileType = '.csv'
    df = pd.read_csv(folerName + fileName + fileType, encoding='latin1')
    df['clear_text_spell'] = df['clear_text_spell'].astype(str)
    # input must be a list even only one sentence
    sentence_list = df['clear_text_spell']

    model_args = ModelArgs(max_seq_length=156)
    # HNC 在 base-cased 表現上較好
    model = RepresentationModel('bert', 'bert-base-cased',
                                args=model_args, use_cuda=True)

    word_embeddings = model.encode_sentences(sentence_list, combine_strategy='mean')
    # print(word_embeddings)
    # print(len(word_embeddings[2]))

    np_word_embeddings = np.array(word_embeddings) 
    # print(np_word_embeddings)
    # print(word_embeddings.shape)

    ## (1)為bert_vec添增標題列
    bert_header_name=[]
    # range(1, 51):實際會從1-768
    for i in range(1, 769):
        bert_header_str = 'bert_%d' % (i)
        bert_header_name.append(bert_header_str)

    df_bert_vec = pd.DataFrame(np_word_embeddings, columns=bert_header_name)
    df_bert_vec_key = pd.concat([df['id'], df_bert_vec], axis=1)

    out_name = '(bert_vec)'+fileName
    df_bert_vec_key.to_csv(folerName + out_name + fileType ,index=False)





