import torch
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

DATASET = 'BalancedNewsCorpus'  
MODEL = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'  
COLLECTION_NAME = 'BalancedNewsCorpus_db'  # Collection name
DIMENSION = 768  # Embeddings size
MILVUS_HOST = 'IP HOST 입력' # 'localhost'
MILVUS_PORT = 'IP PORT 입력' # '8888'
INDEX_TYPE = "IVF_FLAT"


connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    
    
## 가지고 있는 샘플데이터의 dtype과 맞게, collection의 필드 정보들을 선언
EMBEDDING_FIELD_NAME ='News_embedding'
field2args = {
    'file_id': {'dtype': DataType.INT64, 'is_primary':True, 'auto_id':True},
    'filename': {'dtype': DataType.VARCHAR, 'max_length': 20},
    'date': {'dtype': DataType.INT64, 'max_length': 10},
    'NewsPaper': {'dtype': DataType.VARCHAR, 'max_length': 100},
    'Topic': {'dtype': DataType.VARCHAR, 'max_length': 100},
    'News': {'dtype': DataType.VARCHAR, 'max_length': 30000},
    EMBEDDING_FIELD_NAME: {'dtype': DataType.FLOAT_VECTOR, 'dim': DIMENSION},
}

# Milvus 연결을 설정합니다. 작업하기 전에 반드시 Milvus 서버에 연결해야 합니다.
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 스키마 정의
fields = [
    FieldSchema(name = field_name, **args)
    for field_name, args in field2args.items()
]

schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

# collection에 이미 인덱스 유무 파악
if collection.has_index():
    try:
        collection.drop_index()
    except:
        collection.release()
        collection.drop_index()

# index 정의하기
index_params = {
    'metric_type': 'L2',         # "L2", "IP", "COSINE" 
    'index_type': INDEX_TYPE,    # "IVF_FLAT","IVF_SQ8", "IVF_PQ", "HNSW", "ANNOY"
    'params': {"nlist": 1},
}

collection.create_index(field_name=EMBEDDING_FIELD_NAME, index_params=index_params)
collection.load()
collection