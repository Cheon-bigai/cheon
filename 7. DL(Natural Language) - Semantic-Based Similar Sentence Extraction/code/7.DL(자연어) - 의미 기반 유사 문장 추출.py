import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
from gensim.models import Word2Vec
import pickle
import logging
import math
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로드 함수
def load_labels(label_path, max_items=100):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if max_items > 0 and 'data' in labels and len(labels['data']) > max_items:
        labels['data'] = labels['data'][:max_items]
    
    return labels

def load_documents(data_dir, doc_ids=None):
    documents = {}
    
    if doc_ids:
        for doc_id in doc_ids:
            file_paths = [f"{doc_id}", f"{doc_id}.txt", f"{doc_id}.json"]
            for path in file_paths:
                try:
                    with open(os.path.join(data_dir, path), 'r', encoding='utf-8') as f:
                        documents[doc_id] = f.read()
                        break
                except:
                    pass
    else:
        for filename in os.listdir(data_dir):
            doc_id = filename.split('.')[0]
            try:
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    documents[doc_id] = f.read()
            except:
                pass
    
    return documents

# 텍스트 전처리 함수
def preprocess_text(text, tokenizer):
    text = text.lower()
    tokens = tokenizer.morphs(text)
    return tokens

# 커스텀 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, text_data, embeddings, max_len=256):
        self.text_data = text_data
        self.embeddings = embeddings
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text_data)
        
    def __getitem__(self, idx):
        tokens = self.text_data[idx]['tokens'][:self.max_len]
        
        # 토큰을 임베딩으로 변환
        token_vectors = []
        for token in tokens:
            if token in self.embeddings:
                token_vectors.append(self.embeddings[token])
            else:
                token_vectors.append(np.random.randn(self.embeddings.vector_size))
        
        # 패딩 또는 자르기
        if len(token_vectors) < self.max_len:
            padding = [np.zeros(self.embeddings.vector_size) for _ in range(self.max_len - len(token_vectors))]
            token_vectors.extend(padding)
        else:
            token_vectors = token_vectors[:self.max_len]
        
        return {
            'input': torch.tensor(token_vectors, dtype=torch.float32),
            'id': self.text_data[idx]['id']
        }

# CNN 모델 정의
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters=128, filter_sizes=(3, 4, 5), dropout=0.5, output_dim=256):
        super(TextCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, filter_size)
            for filter_size in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # 입력 형태: (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, sequence_length)
        
        conved = [self.activation(conv(x)) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        cat = torch.cat(pooled, dim=1)
        dropout = self.dropout(cat)
        output = self.fc(dropout)
        
        return output

# BiLSTM 모델 정의
class TextBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, dropout=0.5, output_dim=256):
        super(TextBiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        dropout = self.dropout(hidden)
        output = self.fc(dropout)
        
        return output

# 데이터 준비 함수
def prepare_data(train_data_dir, train_label_path, max_items=100):
    train_labels = load_labels(train_label_path, max_items)
    doc_ids = [item['book_id'] for item in train_labels['data']]
    train_docs = load_documents(train_data_dir, doc_ids)
    
    # 형태소 분석기 초기화
    try:
        mecab = Mecab()
    except:
        try:
            from konlpy.tag import Okt
            okt = Okt()
            mecab = type('', (), {})()
            mecab.morphs = lambda x: okt.morphs(x)
        except:
            mecab = type('', (), {})()
            mecab.morphs = lambda x: x.split()
    
    # 데이터셋 구성
    dataset = []
    for item in train_labels['data']:
        book_id = item['book_id']
        if book_id in train_docs:
            tokens = preprocess_text(train_docs[book_id], mecab)
            dataset.append({
                'id': book_id,
                'tokens': tokens,
                'text': ' '.join(tokens),
                'raw_text': train_docs[book_id],
                'category': item['category'],
                'keywords': item['keyword']
            })
    
    return dataset

# 법률 검색 모델 클래스
class LegalSearchModel:
    def __init__(self, model_type='cnn', embedding_dim=100, hidden_dim=128, output_dim=256):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.document_vectors = None
        self.documents = None
        self.word_embeddings = None
        self.model = None
        
    def train_word_embeddings(self, dataset, embedding_dim=100, window=5, min_count=2, epochs=5):
        all_tokens = [doc['tokens'] for doc in dataset]
        word2vec = Word2Vec(
            sentences=all_tokens,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
        return word2vec.wv
    
    def fit(self, dataset):
        self.documents = dataset
        
        # Word2Vec 임베딩 학습
        self.word_embeddings = self.train_word_embeddings(dataset, embedding_dim=self.embedding_dim)
        
        # 모델 초기화
        if self.model_type == 'cnn':
            self.model = TextCNN(
                embedding_dim=self.embedding_dim,
                output_dim=self.output_dim
            ).to(device)
        elif self.model_type == 'bilstm':
            self.model = TextBiLSTM(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            ).to(device)
        
        # 데이터셋 준비
        text_dataset = TextDataset(dataset, self.word_embeddings)
        dataloader = DataLoader(text_dataset, batch_size=16, shuffle=True)
        
        # 손실 함수 및 옵티마이저
        criterion = nn.TripletMarginLoss(margin=0.5)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 모델 훈련
        self.model.train()
        num_epochs = 5
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch['input'].to(device)
                
                # 트리플렛 학습을 위한 미니배치 확인
                if inputs.size(0) < 3:
                    continue
                
                # 앵커, 포지티브, 네거티브 샘플 인덱스 생성
                anchor_idx = list(range(inputs.size(0)))
                positive_idx = [(i + 1) % inputs.size(0) for i in range(inputs.size(0))]
                negative_idx = [(i + 2) % inputs.size(0) for i in range(inputs.size(0))]
                
                # 미니배치에서 트리플렛 샘플 추출
                anchor = inputs[anchor_idx]
                positive = inputs[positive_idx]
                negative = inputs[negative_idx]
                
                # 순방향 전파 및 손실 계산
                optimizer.zero_grad()
                anchor_out = self.model(anchor)
                positive_out = self.model(positive)
                negative_out = self.model(negative)
                
                loss = criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 문서 임베딩 생성
        self.model.eval()
        self.document_vectors = []
        
        with torch.no_grad():
            for doc in dataset:
                tokens = doc['tokens'][:256]
                token_vectors = []
                
                for token in tokens:
                    if token in self.word_embeddings:
                        token_vectors.append(self.word_embeddings[token])
                    else:
                        token_vectors.append(np.random.randn(self.embedding_dim))
                
                # 패딩
                if len(token_vectors) < 256:
                    padding = [np.zeros(self.embedding_dim) for _ in range(256 - len(token_vectors))]
                    token_vectors.extend(padding)
                else:
                    token_vectors = token_vectors[:256]
                
                input_tensor = torch.tensor(token_vectors, dtype=torch.float32).unsqueeze(0).to(device)
                embedding = self.model(input_tensor).cpu().numpy()[0]
                self.document_vectors.append(embedding)
        
        self.document_vectors = np.array(self.document_vectors)
        logger.info(f"Created document embeddings with shape: {self.document_vectors.shape}")
    
    def preprocess_query(self, query):
        try:
            mecab = Mecab()
        except:
            try:
                from konlpy.tag import Okt
                okt = Okt()
                mecab = type('', (), {})()
                mecab.morphs = lambda x: okt.morphs(x)
            except:
                mecab = type('', (), {})()
                mecab.morphs = lambda x: x.split()
        
        return preprocess_text(query, mecab)
    
    def search(self, query, top_k=5):
        # 쿼리 전처리
        query_tokens = self.preprocess_query(query)
        
        # 쿼리 토큰을 임베딩으로 변환
        query_vectors = []
        for token in query_tokens:
            if token in self.word_embeddings:
                query_vectors.append(self.word_embeddings[token])
            else:
                query_vectors.append(np.random.randn(self.embedding_dim))
        
        # 패딩 또는 자르기
        if len(query_vectors) < 256:
            padding = [np.zeros(self.embedding_dim) for _ in range(256 - len(query_vectors))]
            query_vectors.extend(padding)
        else:
            query_vectors = query_vectors[:256]
        
        # 모델을 통한 쿼리 임베딩 생성
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(query_vectors, dtype=torch.float32).unsqueeze(0).to(device)
            query_embedding = self.model(input_tensor).cpu().numpy()[0]
        
        # 코사인 유사도 계산
        similarities = np.array([
            cosine_similarity(query_embedding.reshape(1, -1), doc_vector.reshape(1, -1))[0][0]
            for doc_vector in self.document_vectors
        ])
        
        # 상위 k개 결과 반환
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                'id': self.documents[idx]['id'],
                'category': self.documents[idx]['category'],
                'keywords': self.documents[idx]['keywords'],
                'similarity': similarities[idx],
                'text_snippet': self.documents[idx]['raw_text'][:200] + '...'
            })
        
        return results
    
    def save(self, filepath):
        # 단어 임베딩 저장
        word_embedding_path = filepath + '_word_embeddings.pkl'
        with open(word_embedding_path, 'wb') as f:
            pickle.dump(self.word_embeddings, f)
        
        # 모델 상태 저장
        model_state_path = filepath + '_model_state.pt'
        torch.save(self.model.state_dict(), model_state_path)
        
        # 기타 데이터 저장
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'document_vectors': self.document_vectors,
                'documents': self.documents,
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 단어 임베딩 로드
        word_embedding_path = filepath + '_word_embeddings.pkl'
        with open(word_embedding_path, 'rb') as f:
            word_embeddings = pickle.load(f)
        
        # 인스턴스 생성
        instance = cls(
            model_type=data['model_type'],
            embedding_dim=data['embedding_dim'],
            hidden_dim=data['hidden_dim'],
            output_dim=data['output_dim']
        )
        
        # 데이터 설정
        instance.document_vectors = data['document_vectors']
        instance.documents = data['documents']
        instance.word_embeddings = word_embeddings
        
        # 모델 초기화
        if instance.model_type == 'cnn':
            instance.model = TextCNN(
                embedding_dim=instance.embedding_dim,
                output_dim=instance.output_dim
            )
        elif instance.model_type == 'bilstm':
            instance.model = TextBiLSTM(
                embedding_dim=instance.embedding_dim,
                hidden_dim=instance.hidden_dim,
                output_dim=instance.output_dim
            )
        
        # 모델 상태 로드
        model_state_path = filepath + '_model_state.pt'
        instance.model.load_state_dict(torch.load(model_state_path))
        instance.model.to(device)
        instance.model.eval()
        
        return instance

# 메인 함수
def main():
    # 데이터 경로 설정
    train_data_dir = r'C:\Users\KDT-13\Desktop\KDT7\0.Project\TP11_LLM\data\train_data'
    train_label_path = r'C:\Users\KDT-13\Desktop\KDT7\0.Project\TP11_LLM\data\train_label\Training_legal.json'
    
    # 데이터 크기 제한
    MAX_ITEMS = 100
    
    # 데이터 준비
    train_dataset = prepare_data(train_data_dir, train_label_path, MAX_ITEMS)
    
    # 모델 초기화 및 학습 (CNN 또는 BiLSTM 선택)
    model_type = 'bilstm'  # 'cnn' 또는 'bilstm'
    search_model = LegalSearchModel(model_type=model_type)
    search_model.fit(train_dataset)
    
    # 모델 저장
    model_path = f"legal_search_model_{model_type}.pkl"
    search_model.save(model_path)
    
    # 테스트 쿼리로 검색 테스트
    test_queries = [
        "간주취득세 소유권",
        "지방세법 제105조",
        "합의해제 효력"
    ]
    
    print("검색 결과:")
    for query in test_queries:
        results = search_model.search(query, top_k=3)
        print(f"\n쿼리: '{query}'")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['id']} - {result['category']} (유사도: {result['similarity']:.4f})")
            print(f"   키워드: {', '.join(result['keywords'])}")
            print(f"   미리보기: {result['text_snippet'][:100]}...")

if __name__ == "__main__":
    main()