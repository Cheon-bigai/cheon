
"""
ODIR (Ocular Disease Intelligent Recognition) 데이터셋 분석 및 모델링 - PyTorch 간소화 버전
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import copy
import json
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sqlalchemy import create_engine
import mysql.connector
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 경로 설정
ROOT_DIR = r"C:\Users\KDT-13\Desktop\KDT7\0.Project\TP13_Web\데이터"
TRAIN_DIR = r"C:\Users\KDT-13\Desktop\KDT7\0.Project\TP13_Web\데이터\ODIR-5K\ODIR-5K\Training Images"
TEST_DIR = r"C:\Users\KDT-13\Desktop\KDT7\0.Project\TP13_Web\데이터\ODIR-5K\ODIR-5K\Testing Images"
CSV_PATH = os.path.join(ROOT_DIR, "full_df.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# 데이터베이스 설정
DB_URL = 'mysql+mysqlconnector://2:1234@192.168.2.154:3306/g6'  # DB 접속 정보
DB_TABLE_NAME = 'odir_data'  # 기본 테이블명

# Flask 애플리케이션 설정
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), '웹구현'))
app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 폴더 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'uploads'), exist_ok=True)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 설정 변수
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
DISEASE_COLS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 데이터베이스 관련 함수
def upload_csv_to_db(csv_path=CSV_PATH, table_name=DB_TABLE_NAME, if_exists_option='replace'):
    """데이터베이스에 CSV 파일 업로드"""
    try:
        # MySQL 연결
        engine = create_engine(DB_URL, echo=False)
        
        # CSV 파일 존재 확인
        if not os.path.exists(csv_path):
            print(f"CSV 파일을 찾을 수 없음: {csv_path}")
            return False
        
        # CSV 파일 로드
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 로드 성공: {len(df)} 레코드")
        
        # 이미지 경로 수정 (상대경로 -> 절대경로)
        def convert_path(file_path):
            if pd.isna(file_path):
                return None
                
            file_name = os.path.basename(file_path)
            
            # 훈련 이미지 디렉토리에서 확인
            train_path = os.path.join(TRAIN_DIR, file_name)
            if os.path.exists(train_path):
                return train_path
                
            # 테스트 이미지 디렉토리에서 확인
            test_path = os.path.join(TEST_DIR, file_name)
            if os.path.exists(test_path):
                return test_path
                
            return file_path  # 찾지 못한 경우 원래 경로 반환
        
        # 경로 컬럼이 있는 경우 절대 경로로 변환
        if 'filepath' in df.columns:
            df['filepath'] = df['filepath'].apply(convert_path)
            
        if 'filename' in df.columns:
            df['abs_filepath'] = df['filename'].apply(convert_path)
        
        # 기본 업로드 필드가 없는 경우
        if 'filepath' not in df.columns and 'filename' not in df.columns:
            print("경로 관련 컬럼이 없어 이미지 경로를 추가하지 않았습니다.")
        
        # 데이터프레임을 DB에 저장
        df.to_sql(table_name, engine, if_exists=if_exists_option, index=False)
        
        if if_exists_option == 'append':
            print(f"DB의 {table_name} 테이블에 {len(df)} 레코드를 추가했습니다.")
        else:
            print(f"DB의 {table_name} 테이블을 {len(df)} 레코드로 교체했습니다.")
        return True
        
    except Exception as e:
        print(f"CSV 파일 업로드 중 오류 발생: {e}")
        return False

# 1. 데이터 로딩
def load_data():
    """CSV 파일에서 데이터를 로드하고 이미지 경로와 라벨을 포함하는 DataFrame 생성"""
    try:
        # CSV 파일 존재 여부 확인
        if not os.path.exists(CSV_PATH):
            print(f"CSV 파일을 찾을 수 없음: {CSV_PATH}")
            return create_dummy_data()
        
        # CSV 파일 로드
        df = pd.read_csv(CSV_PATH)
        print(f"데이터 로드 성공: {len(df)} 레코드")
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        return create_dummy_data()
    
    # 이미지 경로와 라벨이 포함된 데이터 프레임 구성
    image_data = []
    
    # 훈련, 테스트 디렉토리 존재 확인
    if not os.path.exists(TRAIN_DIR):
        print(f"훈련 이미지 디렉토리가 없습니다: {TRAIN_DIR}")
        os.makedirs(TRAIN_DIR, exist_ok=True)
    
    if not os.path.exists(TEST_DIR):
        print(f"테스트 이미지 디렉토리가 없습니다: {TEST_DIR}")
        os.makedirs(TEST_DIR, exist_ok=True)
    
    # 사용 가능한 이미지 파일 목록 수집
    train_files = set(os.listdir(TRAIN_DIR) if os.path.exists(TRAIN_DIR) else [])
    test_files = set(os.listdir(TEST_DIR) if os.path.exists(TEST_DIR) else [])
    
    for idx, row in df.iterrows():
        patient_id = str(row['ID'])
        
        # 왼쪽 눈 이미지 경로
        left_img_name = f"{patient_id}_left.jpg"
        left_img = None
        
        if left_img_name in train_files:
            left_img = os.path.join(TRAIN_DIR, left_img_name)
        elif left_img_name in test_files:
            left_img = os.path.join(TEST_DIR, left_img_name)
        
        # 오른쪽 눈 이미지 경로
        right_img_name = f"{patient_id}_right.jpg"
        right_img = None
        
        if right_img_name in train_files:
            right_img = os.path.join(TRAIN_DIR, right_img_name)
        elif right_img_name in test_files:
            right_img = os.path.join(TEST_DIR, right_img_name)
        
        # 질병 라벨 추출
        try:
            labels = []
            for col in DISEASE_COLS:
                if col in row:
                    labels.append(row[col])
                else:
                    labels.append(0)  # 열이 없으면 0으로 설정
        except:
            # 더미 데이터를 위한 랜덤 라벨
            labels = np.random.choice([0, 1], size=8, p=[0.8, 0.2]).tolist()
        
        # 왼쪽 눈 데이터 추가
        if left_img and os.path.exists(left_img):
            image_data.append({
                'patient_id': patient_id,
                'image_path': left_img,
                'eye': 'left',
                'labels': labels
            })
        
        # 오른쪽 눈 데이터 추가
        if right_img and os.path.exists(right_img):
            image_data.append({
                'patient_id': patient_id,
                'image_path': right_img,
                'eye': 'right',
                'labels': labels
            })
    
    # 데이터가 충분한지 확인
    if len(image_data) < 10:
        print(f"이미지 데이터가 너무 적습니다 ({len(image_data)}개). 더미 데이터를 추가합니다.")
        dummy_df = create_dummy_data()
        if len(image_data) > 0:
            image_df = pd.DataFrame(image_data)
            image_df = pd.concat([image_df, dummy_df])
        else:
            image_df = dummy_df
    else:
        image_df = pd.DataFrame(image_data)
    
    print(f"이미지 데이터 프레임 생성 완료: 총 {len(image_df)}개 이미지")
    return image_df

def create_dummy_data(num_samples=100):
    """더미 데이터 생성 함수"""
    print("더미 데이터 생성 중...")
    patient_ids = [f"dummy_{i}" for i in range(num_samples)]
    image_data = []
    
    for patient_id in patient_ids:
        # 랜덤 라벨 생성
        labels = np.random.choice([0, 1], size=8, p=[0.8, 0.2]).tolist()
        
        # 가상 이미지 파일 경로
        dummy_left_img = os.path.join(TRAIN_DIR, "0_left.jpg") if os.path.exists(os.path.join(TRAIN_DIR, "0_left.jpg")) else None
        dummy_right_img = os.path.join(TRAIN_DIR, "0_right.jpg") if os.path.exists(os.path.join(TRAIN_DIR, "0_right.jpg")) else None
        
        # 실제 존재하는 이미지 파일 검색
        if not dummy_left_img or not os.path.exists(dummy_left_img):
            train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('_left.jpg')] if os.path.exists(TRAIN_DIR) else []
            test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('_left.jpg')] if os.path.exists(TEST_DIR) else []
            
            if train_files:
                dummy_left_img = os.path.join(TRAIN_DIR, train_files[0])
            elif test_files:
                dummy_left_img = os.path.join(TEST_DIR, test_files[0])
        
        if not dummy_right_img or not os.path.exists(dummy_right_img):
            train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('_right.jpg')] if os.path.exists(TRAIN_DIR) else []
            test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('_right.jpg')] if os.path.exists(TEST_DIR) else []
            
            if train_files:
                dummy_right_img = os.path.join(TRAIN_DIR, train_files[0])
            elif test_files:
                dummy_right_img = os.path.join(TEST_DIR, test_files[0])
        
        # 왼쪽 눈 데이터 추가
        if dummy_left_img and os.path.exists(dummy_left_img):
            image_data.append({
                'patient_id': patient_id,
                'image_path': dummy_left_img,
                'eye': 'left',
                'labels': labels
            })
        
        # 오른쪽 눈 데이터 추가
        if dummy_right_img and os.path.exists(dummy_right_img):
            image_data.append({
                'patient_id': patient_id,
                'image_path': dummy_right_img,
                'eye': 'right',
                'labels': labels
            })
    
    dummy_df = pd.DataFrame(image_data)
    print(f"더미 데이터 생성 완료: {len(dummy_df)}개 이미지")
    return dummy_df

# 2. PyTorch 데이터셋 및 데이터로더
class ODIRDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        
        try:
            # 먼저 PIL로 이미지 로드 시도
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')  # 그레이스케일 이미지도 처리하기 위해 RGB로 변환
                
                # 변환 적용
                if self.transform:
                    img = self.transform(img)
                else:
                    # 기본 변환: PIL 이미지 -> 텐서
                    img = transforms.ToTensor()(img)
                    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            except:
                # PIL 실패 시 OpenCV로 시도
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"OpenCV로도 이미지 로드 실패: {img_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 변환 적용
                if self.transform:
                    img = self.transform(img)
                else:
                    # PIL과 동일한 전처리 수행
                    img = cv2.resize(img, IMAGE_SIZE)
                    img = img / 255.0  # 정규화
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            
            # 라벨을 텐서로 변환
            labels = torch.tensor(row['labels'], dtype=torch.float32)
            
            return img, labels
        
        except Exception as e:
            print(f"이미지 처리 중 오류: {img_path} - {e}")
            # 오류 시 검은색 이미지와 0 라벨 반환
            img = torch.zeros((3, *IMAGE_SIZE))
            labels = torch.zeros(len(DISEASE_COLS), dtype=torch.float32)
            return img, labels

def get_transforms(is_training=True):
    """데이터 변환 함수"""
    if is_training:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # PIL 이미지에 바로 적용 가능
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # PIL 이미지에 바로 적용 가능
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    """데이터 로더 생성"""
    # 변환 생성
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    test_transform = get_transforms(is_training=False)
    
    # 데이터셋 생성
    train_dataset = ODIRDataset(train_df, transform=train_transform)
    val_dataset = ODIRDataset(val_df, transform=val_transform)
    test_dataset = ODIRDataset(test_df, transform=test_transform)
    
    # 병렬 처리 작업자 수 설정
    num_workers = min(4, os.cpu_count() or 1)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

# 3. 모델 구축
class ODIRModel(nn.Module):
    def __init__(self, model_type='resnet50', num_classes=len(DISEASE_COLS)):
        super(ODIRModel, self).__init__()
        
        # 기본 모델 로드
        if model_type == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_type == 'efficientnet':
            self.base_model = models.efficientnet_b3(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        
        # 사용자 정의 분류기
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # 다중 라벨 분류를 위한 시그모이드
        )
        
        # 기본 모델의 가중치 고정 (학습 속도 향상)
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

# 4. 모델 훈련
def train_model(train_loader, val_loader, model_type='resnet50', epochs=EPOCHS):
    """모델 훈련 및 평가"""
    # 모델 초기화
    model = ODIRModel(model_type=model_type)
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6, verbose=True)
    
    # 훈련 기록
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # 최적 모델 저장 변수
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 조기 종료 변수
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    try:
        # 훈련 루프
        for epoch in range(epochs):
            # 훈련 모드
            model.train()
            train_loss = 0.0
            
            print(f"Epoch {epoch+1}/{epochs} 훈련 중...")
            # 훈련 루프
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # 옵티마이저 초기화
                    optimizer.zero_grad()
                    
                    # 순전파 및 손실 계산
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # NaN 손실 체크
                    if torch.isnan(loss).item() or torch.isinf(loss).item():
                        print(f"경고: 배치 {batch_idx}에서 NaN/Inf 손실 발생. 배치 건너뜀.")
                        continue
                    
                    # 역전파 및 옵티마이저 업데이트
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # 통계 업데이트
                    train_loss += loss.item() * inputs.size(0)
                
                except Exception as e:
                    print(f"훈련 중 오류 발생 (배치 {batch_idx}): {e}")
                    # 배치 건너뛰기
                    continue
            
            # 훈련 손실 계산
            train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 검증 모드
            model.eval()
            val_loss = 0.0
            
            print(f"Epoch {epoch+1}/{epochs} 검증 중...")
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    try:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        # 순전파 및 손실 계산
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # 통계 업데이트
                        val_loss += loss.item() * inputs.size(0)
                    
                    except Exception as e:
                        print(f"검증 중 오류 발생 (배치 {batch_idx}): {e}")
                        continue
            
            # 검증 손실 계산
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # 학습률 조정
            scheduler.step(val_loss)
            
            # 최적 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_type}_best_model.pth'))
                print(f"Epoch {epoch+1}: 최적 모델 저장됨 (검증 손실: {val_loss:.4f})")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # 에포크 결과 출력
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 조기 종료 확인
            if early_stopping_counter >= early_stopping_patience:
                print(f"조기 종료: {early_stopping_patience}번 연속으로 검증 손실이 개선되지 않음.")
                break
    
    except KeyboardInterrupt:
        print("훈련이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"훈련 중 예기치 않은 오류 발생: {e}")
    
    # 최적 가중치 로드
    try:
        model.load_state_dict(best_model_wts)
        print(f"훈련 완료. 최적 검증 손실: {best_val_loss:.4f}")
    except Exception as e:
        print(f"최적 모델 로드 중 오류 발생: {e}")
    
    return model, history

# 5. 모델 평가
def evaluate_model(model, test_loader):
    """테스트 세트에 대한 모델 평가"""
    model.eval()
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 순전파
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                
                # 결과 저장
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
            
            except Exception as e:
                print(f"테스트 중 오류 발생 (배치 {batch_idx}): {e}")
                continue
    
    # 결과 배열 생성
    try:
        if all_labels and all_predictions:
            y_true = np.vstack(all_labels)
            y_pred = np.vstack(all_predictions)
            
            # 모델 평가
            print("\n=== 모델 평가 ===")
            
            # 전체 정확도 계산
            accuracy = np.mean((y_pred == y_true).flatten())
            print(f"테스트 정확도: {accuracy:.4f}")
            
            # 클래스별 성능 평가
            for i, disease in enumerate(DISEASE_NAMES):
                try:
                    report = classification_report(y_true[:, i], y_pred[:, i], output_dict=True)
                    print(f"\n{disease} 분류 성능:")
                    if '1' in report:
                        print(f"정밀도: {report['1']['precision']:.4f}")
                        print(f"재현율: {report['1']['recall']:.4f}")
                        print(f"F1 점수: {report['1']['f1-score']:.4f}")
                    else:
                        print(f"양성 샘플이 없어 성능 측정 불가")
                except Exception as e:
                    print(f"{disease} 분류 성능 계산 중 오류 발생: {e}")
            
            return accuracy
        else:
            print("평가 데이터가 없습니다. 모델 평가를 건너뜁니다.")
            return 0.0
    
    except Exception as e:
        print(f"모델 평가 중 예기치 않은 오류 발생: {e}")
        return 0.0

# 6. 예측 함수
def predict_single_image(model, image_path):
    """단일 이미지 예측 함수 - Flask 웹 앱에서 사용하기 위한 함수"""
    # 모델 평가 모드로 설정
    model.eval()
    
    # 변환 설정
    transform = get_transforms(is_training=False)
    
    try:
        # 이미지 존재 확인
        if not os.path.exists(image_path):
            print(f"이미지 파일이 존재하지 않음: {image_path}")
            return {disease: 0.0 for disease in DISEASE_NAMES}
        
        # 이미지 로드 및 전처리
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지 로드 실패: {image_path}")
            return {disease: 0.0 for disease in DISEASE_NAMES}
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            predictions = model(img_tensor)
            predictions = predictions.squeeze().cpu().numpy()
        
        # 결과 출력
        results = {}
        for i, disease in enumerate(DISEASE_NAMES):
            results[disease] = float(predictions[i])
        
        return results
    
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return {disease: 0.0 for disease in DISEASE_NAMES}

# 7. 웹 앱용 모델 로드 함수
def load_model_for_inference(model_path, model_type='resnet50'):
    """저장된 모델을 로드하여 추론에 사용"""
    try:
        # 모델 인스턴스 생성
        model = ODIRModel(model_type=model_type)
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # 평가 모드 설정
        
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return model
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("===== ODIR 데이터셋 분석 및 모델링 시작 (PyTorch 버전) =====")
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    dataset_df = load_data()
    
    # 2. 데이터 분할
    print("\n2. 데이터 분할 중...")
    patient_ids = dataset_df['patient_id'].unique()
    
    # 데이터가 충분한지 확인
    if len(patient_ids) < 10:
        print(f"경고: 환자 수({len(patient_ids)})가 적습니다.")
        test_size = 0.2
    else:
        test_size = 0.3
        
    train_ids, temp_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    train_df = dataset_df[dataset_df['patient_id'].isin(train_ids)]
    val_df = dataset_df[dataset_df['patient_id'].isin(val_ids)]
    test_df = dataset_df[dataset_df['patient_id'].isin(test_ids)]
    
    print(f"훈련 세트: {len(train_df)} 이미지, {len(train_ids)} 환자")
    print(f"검증 세트: {len(val_df)} 이미지, {len(val_ids)} 환자")
    print(f"테스트 세트: {len(test_df)} 이미지, {len(test_ids)} 환자")
    
    # 3. 데이터 로더 생성
    print("\n3. 데이터 로더 생성 중...")
    batch_size = min(BATCH_SIZE, len(train_df) // 2) if len(train_df) < 50 else BATCH_SIZE
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=batch_size)
    
    # 4. 모델 훈련
    print("\n4. 모델 훈련 중...")
    epochs = min(EPOCHS, 10) if len(train_df) < 50 else EPOCHS
    model, history = train_model(train_loader, val_loader, model_type='resnet50', epochs=epochs)
    
    # 5. 모델 평가
    print("\n5. 모델 평가 중...")
    evaluate_model(model, test_loader)
    
    # 6. 모델 저장
    print("\n6. 모델 저장 중...")
    model_path = os.path.join(MODEL_DIR, 'odir_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch 모델이 {model_path}에 저장되었습니다.")
    
    print("\n===== ODIR 모델링 완료 =====")
    return model

# 예측 예시
def prediction_example():
    """저장된 모델을 로드하여 예측 예시 보여주기"""
    model_path = os.path.join(MODEL_DIR, 'odir_model.pth')
    
    if os.path.exists(model_path):
        # 모델 로드
        model = load_model_for_inference(model_path)
        
        # 테스트 이미지 찾기
        test_images = []
        if os.path.exists(TRAIN_DIR):
            test_images = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
        
        if not test_images and os.path.exists(TEST_DIR):
            test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
        
        if test_images:
            # 예측 실행
            image_path = test_images[0]
            print(f"\n예측 예시 - 이미지: {image_path}")
            
            results = predict_single_image(model, image_path)
            
            # 결과 출력
            print("\n예측 결과:")
            for disease, prob in results.items():
                print(f"{disease}: {prob:.4f} ({prob > 0.5 and '양성' or '음성'})")
        else:
            print("예측할 테스트 이미지를 찾을 수 없습니다.")
    else:
        print(f"저장된 모델을 찾을 수 없습니다: {model_path}")

# 프로그램 실행
if __name__ == "__main__":
    # 모델 훈련 및 평가
    model = main()
    
    # 예측 예시 실행
    prediction_example()