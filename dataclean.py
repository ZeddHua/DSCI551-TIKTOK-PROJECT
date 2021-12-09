#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
def clean():
    #dealing with column explosion
    df = pd.read_csv('trending_sub.csv')
    df.rename(columns={'authorMeta.name':'authorMetaName', 'authorMeta.nickName':'authorMetaNickName', 'authorMeta.verified':'authorMetaVerified', 'musicMeta.musicName':'musicMetaMusicName', 'musicMeta.musicAuthor':'musicMetaMusicAuthor', 'musicMeta.musicOriginal':'musicMetaMusicOriginal', 'videoMeta.height':'videoMetaHeight', 'videoMeta.width':'videoMetaWidth', 'videoMeta.duration':'videoMetaDuration', 'hashtags.name':'hashtagsName', 'hashtags.title':'hashtagsTitle'}, inplace=True)

    #groupby & merge hashtagsName
    df.hashtagsName = df.hashtagsName.astype(str)   # NAN counts as float
    dff = df.groupby('createTime')['hashtagsName'].apply(lambda x:','.join(x)).reset_index()   #lambda x:x.str.cat(sep=',')   #use reset_index() to transform groupby object into dataframe
    df.drop(['hashtagsName'], axis=1, inplace=True)
    df = pd.merge(df, dff, on='createTime', how='left')

    #drop_duplicates
    df.drop_duplicates(subset=['createTime'], inplace=True)   # subset=[], keep='first'
    df.reset_index(drop=True, inplace=True)

    #dealing with null data
    df.drop(['hashtagsTitle', 'mentions'], axis=1, inplace=True)  # only leaving 'text' with 38 null data

    #droping needless columns
    df.drop(['authorMetaName','authorMetaNickName','musicMetaMusicName','musicMetaMusicAuthor'], axis=1, inplace=True)

    #feature extraction using machine learning algorthms
    # turn 'authorMetaVerified' into 1/0
    df.authorMetaVerified = df.authorMetaVerified.apply(lambda x: 1 if x==True else 0)

    # turn 'musicMetaMusicOriginal' into 1/0
    df.musicMetaMusicOriginal = df.musicMetaMusicOriginal.apply(lambda x: 1 if x==True else 0)

    # extract features from 'text'&'hashtagsName'
    # text_len
    df.text = df.text.astype(str)
    df['text_len'] = df.text.apply(len)
    
    return df

