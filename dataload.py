#!/usr/bin/env python
# coding: utf-8

# In[46]:


#load data into Firebase
import pandas as pd
import requests
import json

def load(data, **kwargs):
    if data == 'trending_sub.csv':
        df = pd.read_csv(data, encoding='utf-8', sep=',')
        df.rename(columns={'authorMeta.name':'authorMetaName', 'authorMeta.nickName':'authorMetaNickName', 'authorMeta.verified':'authorMetaVerified', 'musicMeta.musicName':'musicMetaMusicName', 'musicMeta.musicAuthor':'musicMetaMusicAuthor', 'musicMeta.musicOriginal':'musicMetaMusicOriginal', 'videoMeta.height':'videoMetaHeight', 'videoMeta.width':'videoMetaWidth', 'videoMeta.duration':'videoMetaDuration', 'hashtags.name':'hashtagsName', 'hashtags.title':'hashtagsTitle'}, inplace=True)
        df_json = df.to_json()
        myjson = json.loads(df_json)    
        newjson = json.dumps(myjson, ensure_ascii=False)   
        url = 'https://dsci551-tiktok-project-4167a-default-rtdb.firebaseio.com/trending_sub.json'
        req1 = requests.put(url, newjson.encode('utf-8'))
        
    elif data == 'trending_cleaned.csv':
        df = pd.read_csv(data, encoding='utf-8', sep=',')
        df_json = df.to_json()
        myjson = json.loads(df_json)    
        newjson = json.dumps(myjson, ensure_ascii=False)    
        url = 'https://dsci551-tiktok-project-4167a-default-rtdb.firebaseio.com/trending_cleaned.json'
        req1 = requests.put(url, newjson.encode('utf-8'))
    
    elif isinstance(data, int):
        dic = {}
        dic['prediction'] = data
        for (key, value) in kwargs.items():
            dic[key] = value
        myjson = json.dumps(dic)
        url = f'https://dsci551-tiktok-project-4167a-default-rtdb.firebaseio.com/prediction_results/{int(data)}.json'
        req1 = requests.put(url, myjson)
        
    elif data == 'features':
        dic = {}
        for (key, value) in kwargs.items():
            dic[key] = value
        myjson = json.dumps(dic)
        url = f'https://dsci551-tiktok-project-4167a-default-rtdb.firebaseio.com/features_extracted.json'
        req1 = requests.put(url, myjson)
    
    else:
        return 'Not applicable.'

