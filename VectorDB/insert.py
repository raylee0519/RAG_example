from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from clean import df
from collections import collection, EMBEDDING_FIELD_NAME, embeddings

# 만약에 DataFrame과 VectorDB에 적재할 Column이 다른 경우에 이와 같은 방식을 고려해볼 수 있음
column2field = {
    'filename':'filename',
    'date':'date',
    'NewsPaper':'NewsPaper',
    'Topic':'Topic',
    'News':'News',
}

field2column = {v:k for k,v in column2field.items()}


class CustomDataset(Dataset):
    
    def __init__(self, df, embeddings, field2column, embedding_field):
        self.df = df
        self.embeddings = embeddings
        self.field2column = field2column
        self.embedding_field = embedding_field
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        items = {field: self.df.iloc[idx][column] for field, column in self.field2column.items()}
        items[self.embedding_field] = self.embeddings[idx].tolist()
        return items
    
dataset = CustomDataset(df, embeddings, field2column, EMBEDDING_FIELD_NAME)

# insert 수행하기
for data in tqdm(iter(dataset), total=len(dataset)):
    collection.insert(data)
    
collection.flush()
collection.num_entities # 9000, 삽입된 엔티티의 수를 나타낸다.