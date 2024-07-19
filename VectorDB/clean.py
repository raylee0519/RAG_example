# 전처리 과정 수행
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm.auto import tqdm
import re


def clean(text:str):
    # https://github.com/YongWookHa/kor-text-preprocess/blob/master/src/clean.py
    not_used = re.compile('[^ .?!/@$%~|0-9|ㄱ-ㅣ가-힣]+')
    dup_space = re.compile('[ \t]+')  # white space duplicate
    dup_stop = re.compile('[\.]+')  # full stop duplicate

    cleaned = not_used.sub('', text.strip()) 
    cleaned = dup_space.sub(' ', cleaned)
    cleaned = dup_stop.sub('.', cleaned) 

    return cleaned

df = pd.read_csv('./dataset/BalancedNewsCorpus_train.csv')
df['News'] = df['News'].apply(lambda text: text.replace('<p>', '\n').replace('</p>', '\n'))
df['News'] = df['News'].apply(clean)
df