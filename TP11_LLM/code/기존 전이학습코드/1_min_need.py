import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import logging
import math
from collections import defaultdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('small_test.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 시간 측정 시작
start_time = time.time()

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 데이터 로드 함수 - 일부 파일만 로드하도록 수정
def load_labels(label_path, max_items=100):
    logger.info(f"Loading labels from {label_path} (max {max_items} items)")
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # 데이터 개수 제한
    if max_items > 0 and 'data' in labels and len(labels['data']) > max_items:
        logger.info(f"Limiting from {len(labels['data'])} to {max_items} items")
        labels['data'] = labels['data'][:max_items]
    
    logger.info(f"Loaded {len(labels['data'])} label items")
    return labels

def load_documents(data_dir, doc_ids=None):
    logger.info(f"Loading documents from {data_dir}")
    documents = {}
    
    if doc_ids:
        logger.info(f"Limiting to {len(doc_ids)} specific document IDs")
        # 특정 ID만 로드
        for doc_id in tqdm(doc_ids, desc="Reading documents"):
            # 가능한 파일 확장자 시도
            possible_filenames = [
                f"{doc_id}",       # 확장자 없음
                f"{doc_id}.txt",   # .txt 확장자
                f"{doc_id}.json"   # .json 확장자
            ]
            
            file_found = False
            for filename in possible_filenames:
                try:
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        documents[doc_id] = f.read()
                        file_found = True
                        break  # 파일을 찾았으면 루프 중단
                except Exception as e:
                    pass  # 해당 확장자로 없으면 다음 확장자 시도
            
            # 모든 확장자 시도 후에도 파일을 찾지 못한 경우
            if not file_found:
                logger.error(f"Could not find file for document ID {doc_id} with any of the tried extensions")
    else:
        # 전체 파일 로드
        for filename in tqdm(os.listdir(data_dir), desc="Reading documents"):
            doc_id = filename.split('.')[0]  # 확장자 제거
            try:
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    documents[doc_id] = f.read()
            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

# 데이터 전처리 함수
def preprocess_text(text, mecab):
    """한국어 텍스트 전처리 및 형태소 분석"""
    # 기본 전처리 (특수문자 제거, 소문자 변환 등)
    text = text.lower()
    # 형태소 분석
    tokens = mecab.morphs(text)
    # 불용어 제거 (필요시 불용어 리스트 추가)
    # stopwords = ['은', '는', '이', '가', '을', '를', '에', '의', '으로', '로']
    # tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens)

# 메인 데이터 준비 함수
def prepare_data(train_data_dir, train_label_path, max_items=100):
    # 데이터 로드 - 제한된 수의 항목만
    train_labels = load_labels(train_label_path, max_items)
    
    # 필요한 문서 ID만 추출
    doc_ids = [item['book_id'] for item in train_labels['data']]
    train_docs = load_documents(train_data_dir, doc_ids)
    
    # Mecab 초기화
    try:
        # 먼저 Mecab 시도
        mecab_dic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mecab-ko-dic')
        mecab = Mecab(dicpath=mecab_dic_path)
        logger.info("Initialized Mecab tokenizer")
    except Exception as e:
        logger.error(f"Error initializing Mecab: {e}")
        try:
            # Mecab 실패 시 Okt(구 Twitter) 형태소 분석기 시도
            from konlpy.tag import Okt
            okt = Okt()
            mecab = type('', (), {})()
            mecab.morphs = lambda x: okt.morphs(x)
            logger.info("Using Okt tokenizer instead of Mecab")
        except Exception as e2:
            logger.error(f"Error initializing Okt: {e2}")
            # 모든 형태소 분석기 실패 시 간단한 토큰화로 대체
            logger.info("Falling back to simple tokenization")
            mecab = type('', (), {})()
            mecab.morphs = lambda x: x.split()
    
    # 데이터와 라벨 매핑
    dataset = []
    skipped = 0
    
    for item in tqdm(train_labels['data'], desc="Mapping data and labels"):
        book_id = item['book_id']
        if book_id in train_docs:
            # 텍스트 전처리
            processed_text = preprocess_text(train_docs[book_id], mecab)
            dataset.append({
                'id': book_id,
                'text': processed_text,
                'raw_text': train_docs[book_id],
                'category': item['category'],
                'keywords': item['keyword'],
                'publication_date': item.get('publication_ymd', '')
            })
        else:
            skipped += 1
    
    logger.info(f"Created dataset with {len(dataset)} items, skipped {skipped} items")
    return dataset

# 키워드 검색 모델 클래스
class LegalSearchModel:
    def __init__(self, model_type='bert'):
        self.model_type = model_type
        self.document_vectors = None
        self.documents = None
        self.tokenizer = None
        self.model = None
    
    def fit(self, dataset):
        """모델 학습"""
        logger.info(f"Training {self.model_type} model...")
        self.documents = dataset
        
        # BERT 기반 임베딩
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.model = AutoModel.from_pretrained("klue/bert-base").to(device)
        
        # 문서 임베딩 생성
        self.document_vectors = []
        
        for doc in tqdm(dataset, desc="Creating BERT embeddings"):
            text = doc['text']
            # 긴 텍스트의 경우 앞부분만 사용 (BERT 토큰 제한)
            tokens = self.tokenizer(text[:512], return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                # [CLS] 토큰 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                self.document_vectors.append(embedding[0])
                
        self.document_vectors = np.array(self.document_vectors)
        logger.info(f"Created BERT embeddings with shape: {self.document_vectors.shape}")
    
    def search(self, query, top_k=5):
        """키워드로 관련 문서 검색"""
        # 쿼리 임베딩
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            query_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 코사인 유사도 계산
        similarities = np.array([
            cosine_similarity(query_vector, doc_vector.reshape(1, -1))[0][0]
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
                'text_snippet': self.documents[idx]['raw_text'][:200] + '...',  # 텍스트 일부만 표시
                'rank': len(results) + 1  # 랭킹 정보 추가
            })
        
        return results
    
    def evaluate_search(self, query, relevant_doc_id, top_k=10):
        """검색 결과에 대한 평가 지표 계산"""
        results = self.search(query, top_k=top_k)
        
        # 결과가 없는 경우
        if not results:
            return {
                'precision': 0,
                'recall': 0,
                'mrr': 0, 
                'ndcg': 0
            }
        
        # 정답 문서가 결과에 포함되어 있는지 확인
        position = -1
        for i, result in enumerate(results):
            if result['id'] == relevant_doc_id:
                position = i
                break
        
        # 지표 계산
        precision = 1.0/top_k if position != -1 else 0
        recall = 1.0 if position != -1 else 0  # 정답은 1개라고 가정
        mrr = 1.0/(position+1) if position != -1 else 0
        
        # nDCG 계산
        dcg = 0
        idcg = 1.0  # 이상적인 경우 정답이 첫 번째
        
        # DCG 계산
        for i, result in enumerate(results):
            relevance = 1 if result['id'] == relevant_doc_id else 0
            dcg += relevance / math.log2(i + 2)  # i+2 사용 이유는 log2(1)=0 피하기 위해
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    def save(self, filepath):
        """모델 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'document_vectors': self.document_vectors,
                'documents': self.documents,
            }, f)
        
        # BERT 모델 별도 저장
        model_dir = filepath + '_bert_model'
        os.makedirs(model_dir, exist_ok=True)
        self.tokenizer.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """모델 로드"""
        instance = cls()
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        instance.model_type = data['model_type']
        instance.document_vectors = data['document_vectors']
        instance.documents = data['documents']
        
        # BERT 모델 로드
        model_dir = filepath + '_bert_model'
        if os.path.exists(model_dir):
            instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            instance.model = AutoModel.from_pretrained(model_dir)
        
        logger.info(f"Model loaded from {filepath}")
        return instance

# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model is not None:
                    model.load_state_dict(self.best_model)
        return self.early_stop

# 성능 평가 함수
def evaluate_model(model, test_dataset, top_k_values=[1, 3, 5, 10]):
    """모델 성능 평가
    
    - Precision@k: 상위 k개 문서 중 정답 문서의 비율
    - Recall@k: 정답 문서 중 상위 k개에 포함된 비율
    - MRR (Mean Reciprocal Rank): 정답 문서가 나타난 순위의 역수 평균
    - nDCG (Normalized Discounted Cumulative Gain): 문서 순위를 고려한 점수
    """
    # 테스트 쿼리 생성
    test_queries = []
    
    # 각 문서의 키워드를 조합하여 쿼리 생성
    for doc in test_dataset:
        if len(doc['keywords']) >= 2:
            query_text = ' '.join(doc['keywords'][:2])  # 처음 2개 키워드로 쿼리 생성
            test_queries.append({
                'query': query_text,
                'relevant_doc_id': doc['id'],
                'doc': doc
            })
    
    max_k = max(top_k_values)
    
    # 평가 지표 초기화
    metrics = {
        f'precision@{k}': 0 for k in top_k_values
    }
    metrics.update({
        f'recall@{k}': 0 for k in top_k_values
    })
    metrics['mrr'] = 0  # Mean Reciprocal Rank
    metrics['ndcg@5'] = 0  # Normalized DCG at 5
    metrics['ndcg@10'] = 0  # Normalized DCG at 10
    
    # 유효한 쿼리 개수
    valid_queries = 0
    
    # 각 쿼리에 대해 평가
    logger.info(f"Evaluating on {len(test_queries)} test queries...")
    for query_data in tqdm(test_queries, desc="Evaluating queries"):
        query = query_data['query']
        relevant_doc_id = query_data['relevant_doc_id']
        
        # 쿼리 실행
        results = model.search(query, top_k=max_k)
        
        if not results:  # 결과가 없는 경우 스킵
            continue
        
        valid_queries += 1
        
        # 랭킹에서 정답 문서의 위치 찾기
        position = -1
        for i, result in enumerate(results):
            if result['id'] == relevant_doc_id:
                position = i
                break
        
        # Precision@k & Recall@k 계산
        for k in top_k_values:
            if k <= len(results):
                found = relevant_doc_id in [r['id'] for r in results[:k]]
                metrics[f'precision@{k}'] += (1 if found else 0)
                metrics[f'recall@{k}'] += (1 if found else 0)  # 정답은 1개라고 가정
        
        # MRR 계산
        if position != -1:  # 정답이 결과에 포함되어 있으면
            metrics['mrr'] += 1.0 / (position + 1)
        
        # nDCG 계산
        # 이상적인 랭킹: 정답 문서가 가장 상위에 있을 때
        idcg5 = 1.0  # 이상적인 DCG@5
        idcg10 = 1.0  # 이상적인 DCG@10
        
        # 실제 DCG 계산
        dcg5 = 0.0
        dcg10 = 0.0
        
        # 순위에 따른 점수 계산
        for i, result in enumerate(results[:10]):
            relevance = 1 if result['id'] == relevant_doc_id else 0
            # log_2(i+2)를 사용하는 이유는 1부터 순위가 시작하기 때문
            if i < 5:
                dcg5 += relevance / math.log2(i + 2)
            dcg10 += relevance / math.log2(i + 2)
        
        # nDCG 계산
        if idcg5 > 0:
            metrics['ndcg@5'] += dcg5 / idcg5
        if idcg10 > 0:
            metrics['ndcg@10'] += dcg10 / idcg10
    
    # 평균 계산
    if valid_queries > 0:
        for key in metrics:
            metrics[key] /= valid_queries
    
    return metrics

# 메인 함수
def main():
    # 데이터 경로 설정
    train_data_dir = '../TP11_LLM/data/train_data'
    train_label_path = r'C:\Users\KDT-13\Desktop\KDT7\0.Project\TP11_LLM\data\train_label\Training_legal.json'
    test_data_dir = '../TP11_LLM/data/test_data'
    test_label_path = r'C:\Users\KDT-13\Desktop\KDT7\0.Project\TP11_LLM\data\test_label\Validation_legal.json'
    
    # 테스트용 - 데이터 크기 제한 (100개만 사용)
    MAX_ITEMS = 100
    
    # 데이터 준비 - 제한된 크기로
    logger.info(f"Preparing training data (limited to {MAX_ITEMS} items)...")
    train_dataset = prepare_data(train_data_dir, train_label_path, MAX_ITEMS)
    
    # 모델 선택 - BERT 고정
    model_type = 'bert'
    
    # 모델 학습
    search_model = LegalSearchModel(model_type=model_type)
    search_model.fit(train_dataset)
    
    # 모델 저장
    model_path = f"legal_search_model_{model_type}_small.pkl"
    search_model.save(model_path)
    
    # 테스트 쿼리로 검색 테스트
    test_queries = [
        "간주취득세 소유권",
        "지방세법 제105조",
        "합의해제 효력"
    ]
    
    logger.info("Testing queries:")
    for query in test_queries:
        results = search_model.search(query, top_k=5)
        logger.info(f"\nResults for query: '{query}'")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['id']} - {result['category']} (유사도: {result['similarity']:.4f})")
            logger.info(f"   키워드: {', '.join(result['keywords'])}")
            logger.info(f"   미리보기: {result['text_snippet'][:100]}...")
    
    # 검증 데이터 준비 및 성능 평가
    logger.info("\nEvaluating model performance...")
    validation_dataset = prepare_data(test_data_dir, test_label_path, MAX_ITEMS)
    eval_metrics = evaluate_model(search_model, validation_dataset)
    
    # 평가 지표 출력
    logger.info("\nPerformance Metrics:")
    for metric, value in eval_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()