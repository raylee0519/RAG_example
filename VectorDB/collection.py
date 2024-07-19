import torch
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MODEL = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'  # 내가 사용하고자 하는 임베딩 모델
COLLECTION_NAME = 'BalancedNewsCorpus_db'  # 저장할 Vector Database의 Collection name
DIMENSION = 768  # 내가 사용하는 임베딩 모델의 dimension
LIMIT = 10  # topk와 같이 내가 검색했을 때 최대 몇개를 볼 것 인지
MILVUS_HOST = '000.000.00.00' # 설치 Host
MILVUS_PORT = '8888' # 설치 Port
INDEX_TYPE = "IVF_FLAT" # 다양한 Index 알고리즘들을 활용 가능 "HNSW" , "ANNOY" , "IVF_FLAT" , "IVF_SQ8" , "IVF_PQ"

# host와 port에 연결
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# 내가 collection 이름이 겹치는 경우에는 삭제하겠다는 뜻
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 직접 밀어넣을 필드 & 스키마 정의하기
fields = [
    FieldSchema(name='file_id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name='date', dtype=DataType.INT64, max_length=10),
    FieldSchema(name='NewsPaper', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='Topic', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='News', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='News_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 매우 중요한 인덱스 부분
index_params = {
    'metric_type':'L2',
    'index_type': INDEX_TYPE,
    'params':{"nlist":1536}
}

# colletion에서 색인 부분을 생성하는 부분을 명시하고 부르겠다는 뜻
collection.create_index(field_name="News_embedding", index_params=index_params)
collection.load()