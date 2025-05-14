import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import random

# 모델 불러오기  python basic_ver4.py --mode train --load_pretrained




# 재현성을 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 데이터 경로 설정
IMG_ROOT = r'C:\Users\cheon\Desktop\vscode\Project\DL\data\upimage'

# 이미지 증강 함수 (256x256 크기에 최적화)
def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            # 원본 크기가 256x256이므로 리사이즈 필요 없음
            # transforms.Resize((256, 256)),
            
            # 랜덤 크롭 (새의 다양한 부분에 집중)
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            
            # 다양한 증강 기법
            transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
            transforms.RandomRotation(15),  # 약간의 회전
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변형 강화
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 약간의 이동과 스케일링
            
            # 정규화
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
    else:  # 테스트 변환 (증강 없음)
        return transforms.Compose([
            # 원본 크기가 256x256이므로 리사이즈 필요 없음
            # transforms.Resize((256, 256)),
            
            # 중앙 크롭 선택 (또는 전체 이미지 사용)
            # transforms.CenterCrop(224),
            
            # 정규화
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 데이터 로딩 함수
def load_data():
    # 변수 초기화
    train_dataset = None
    test_dataset = None
    num_classes = 0
    class_names = []
    
    try:
        # ImageFolder로 데이터셋 로드
        train_dataset = ImageFolder(root=IMG_ROOT, transform=get_transforms('train'))
        test_dataset = ImageFolder(root=IMG_ROOT, transform=get_transforms('test'))
        
        # 데이터 분리
        indices = list(range(len(train_dataset)))
        labels = [train_dataset[i][1] for i in indices]
        
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
        
        # Subset 생성
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
        num_classes = len(train_dataset.dataset.classes)
        class_names = train_dataset.dataset.classes
        
    except Exception as e:
        print(f"ImageFolder 로드 중 오류 발생: {e}")
        # CSV 파일을 사용하는 방식으로 변경
        import pandas as pd
        from PIL import Image
        import os
        from torch.utils.data import Dataset
        
        class BirdDataset(Dataset):
            def __init__(self, csv_file, img_dir, transform=None):
                self.data_info = pd.read_csv(csv_file, header=None, names=['img_path', 'upscale_img_path', 'label'])
                self.img_dir = img_dir
                self.transform = transform
                
                self.label_set = sorted(list(set(self.data_info['label'])))
                self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
                self.data_info['label_idx'] = self.data_info['label'].apply(lambda x: self.label_to_idx[x])
                
                self.classes = self.label_set
            
            def __len__(self):
                return len(self.data_info)
            
            def __getitem__(self, idx):
                img_name = os.path.join(self.img_dir, os.path.basename(self.data_info.iloc[idx, 1]))  # upscale_img_path 사용
                image = Image.open(img_name).convert('RGB')
                label = self.data_info.iloc[idx, 3]  # label_idx 사용
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
        
        # CSV 파일 경로 설정 (기존 코드와 일치시키세요)
        csv_file = r'C:\Users\cheon\Desktop\vscode\Project\DL\data\filtered_dataset.csv'
        
        # 데이터셋 생성
        full_dataset = BirdDataset(csv_file=csv_file, img_dir=IMG_ROOT, transform=get_transforms('train'))
        full_dataset_test = BirdDataset(csv_file=csv_file, img_dir=IMG_ROOT, transform=get_transforms('test'))
        
        # 데이터 분리
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.data_info.iloc[i]['label_idx'] for i in indices]
        
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
        
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset_test, test_indices)
        
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
    
    if train_dataset is None or test_dataset is None:
        raise ValueError("데이터셋을 로드하는 데 실패했습니다. 경로 및 파일을 확인하세요.")
        
    return train_dataset, test_dataset, num_classes, class_names

# 앙상블을 위한 모델 클래스 정의
class BirdEnsemble(nn.Module):
    def __init__(self, num_classes, models_list):
        super(BirdEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        
    def forward(self, x):
        # 각 모델의 출력을 수집
        outputs = [model(x) for model in self.models]
        # 평균 출력 계산
        return torch.mean(torch.stack(outputs), dim=0)

# 전이 학습 모델 생성 함수
def create_transfer_model(model_name, num_classes, pretrained=True):
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    else:
        raise ValueError(f"지원되지 않는 모델 이름: {model_name}")
        
    return model

# 모델 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 시간 형식화 함수
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# 학습 함수
def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        batch_start_time = time.time()
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        # 진행 상황 출력 (10배치마다)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
            print(f'에폭 {epoch_num+1}/{num_epochs} | 배치 {batch_idx+1}/{len(data_loader)} | '
                  f'손실: {loss.item():.4f} | 배치 처리 시간: {batch_time:.2f}초')
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    
    return running_loss / len(data_loader), 100. * correct / total, epoch_time

# 평가 함수
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    eval_start_time = time.time()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    
    # 클래스별 정확도 계산 (선택 사항)
    class_correct = {}
    class_total = {}
    
    for pred, label in zip(all_preds, all_labels):
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_accuracy = {label: 100 * class_correct[label] / class_total[label] for label in class_total}
    
    return running_loss / len(data_loader), 100. * correct / total, eval_time, class_accuracy

# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 카운터: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# 메인 훈련 함수
def train_model(model_name, num_epochs=40, batch_size=16, learning_rate=0.001, weight_decay=0.0005, load_pretrained=False, pretrained_path=None):
    # 데이터 로드
    train_dataset, test_dataset, num_classes, class_names = load_data()
    print(f"클래스 수: {num_classes}")
    print(f"클래스 이름: {class_names}")
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 모델 생성
    model = create_transfer_model(model_name, num_classes)
    
    # 기존 학습된 파라미터 불러오기
    if load_pretrained and pretrained_path:
        try:
            print(f"기존 학습된 모델 파라미터를 불러옵니다: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("모델 파라미터 불러오기 성공!")
        except Exception as e:
            print(f"모델 파라미터 불러오기 실패: {e}")
            print("새로운 모델로 학습을 시작합니다.")
    
    model = model.to(device)

    
    # 모델 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"모델 파라미터 수: {total_params:,}")
    
    # 손실 함수
    loss_fn = nn.CrossEntropyLoss()
    
    # Stochastic Weight Averaging 적용 (선택 사항)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # swa_model = torch.optim.swa_utils.AveragedModel(model)
    # swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0001)
    
    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 조기 종료
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 결과 저장용 변수
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    lr_history = []
    
    # 최고 모델 저장 경로
    best_model_path = f'best_{model_name}_bird_model.pth'
    best_acc = 0.0
    
    # 훈련 시작
    total_start_time = time.time()
    print(f"모델 훈련 시작 ({model_name})...")
    
    for epoch in range(num_epochs):
        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 훈련
        train_loss, train_acc, train_time = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, num_epochs)
        
        # 평가
        test_loss, test_acc, test_time, class_accuracy = evaluate(model, test_loader, loss_fn, device)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(test_loss)
        
        # 결과 기록
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 결과 출력
        print("\n" + "-"*70)
        print(f'에폭 {epoch+1}/{num_epochs} 결과:')
        print(f'학습률: {current_lr:.6f}')
        print(f'훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.2f}%')
        print(f'테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_acc:.2f}%')
        print(f'훈련 시간: {format_time(train_time)} ({train_time:.2f}초)')
        print(f'평가 시간: {format_time(test_time)} ({test_time:.2f}초)')
        
        # 클래스별 정확도 출력 (선택 사항)
        print("\n클래스별 정확도:")
        for label in sorted(class_accuracy.keys()):
            print(f"  클래스 {label} ({class_names[label] if label < len(class_names) else 'Unknown'}): {class_accuracy[label]:.2f}%")
        
        # 현재 모델이 이전에 저장된 최고 모델보다 성능이 좋으면 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'\n새로운 최고 성능 모델이 저장되었습니다! 정확도: {best_acc:.2f}%')
        
        # 경과 시간 계산 및 출력
        elapsed_time = time.time() - total_start_time
        estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
        remaining_time = estimated_total - elapsed_time
        
        print(f'\n총 경과 시간: {format_time(elapsed_time)} ({elapsed_time:.2f}초)')
        print(f'예상 남은 시간: {format_time(remaining_time)} ({remaining_time:.2f}초)')
        print("-"*70 + "\n")
        
        # 조기 종료 확인
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("조기 종료됨! 테스트 손실이 10 에폭 동안 개선되지 않았습니다.")
            break
    
    # 최종 시간 계산
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    
    print(f'훈련 완료!')
    print(f'총 훈련 시간: {format_time(total_training_time)} ({total_training_time:.2f}초)')
    print(f'최종 최고 성능 모델 정확도: {best_acc:.2f}%')
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_results.png')
    plt.show()
    
    # 훈련 결과 요약 출력
    print("\n" + "="*50)
    print(f"{model_name} 훈련 결과 요약:")
    print("="*50)
    print(f"에폭 수: {epoch+1} (최대 {num_epochs})")
    print(f"배치 크기: {batch_size}")
    print(f"초기 학습률: {learning_rate}")
    print(f"최종 학습률: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"최종 훈련 손실: {train_losses[-1]:.4f}")
    print(f"최종 훈련 정확도: {train_accs[-1]:.2f}%")
    print(f"최종 테스트 손실: {test_losses[-1]:.4f}")
    print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%")
    print(f"최고 테스트 정확도: {best_acc:.2f}%")
    print(f"총 훈련 시간: {format_time(total_training_time)}")
    print(f"평균 에폭 시간: {format_time(total_training_time/(epoch+1))}")
    print("="*50)
    
    # 모델 결과 반환
    return model, best_acc, best_model_path

# 앙상블 모델 학습 함수
def train_ensemble():
    # 데이터 로드
    train_dataset, test_dataset, num_classes, class_names = load_data()
    
    # 앙상블을 위한 3개의 다른 모델 훈련
    print("\n1번째 모델 훈련 (ResNet50)...")
    model1, acc1, path1 = train_model('resnet50', num_epochs=40, batch_size=16, learning_rate=0.001)
    
    print("\n2번째 모델 훈련 (EfficientNet-B0)...")
    model2, acc2, path2 = train_model('efficientnet_b0', num_epochs=40, batch_size=16, learning_rate=0.001)
    
    print("\n3번째 모델 훈련 (MobileNetV2)...")
    model3, acc3, path3 = train_model('mobilenet_v2', num_epochs=40, batch_size=16, learning_rate=0.001)
    
    # 앙상블 모델 생성
    print("\n앙상블 모델 생성 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 최고 성능 모델 불러오기
    model1 = create_transfer_model('resnet50', num_classes)
    model1.load_state_dict(torch.load(path1))
    model1 = model1.to(device)
    
    model2 = create_transfer_model('efficientnet_b0', num_classes)
    model2.load_state_dict(torch.load(path2))
    model2 = model2.to(device)
    
    model3 = create_transfer_model('mobilenet_v2', num_classes)
    model3.load_state_dict(torch.load(path3))
    model3 = model3.to(device)
    
    # 앙상블 모델
    ensemble = BirdEnsemble(num_classes, [model1, model2, model3]).to(device)
    
    # 앙상블 모델 평가
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    
    ensemble_loss, ensemble_acc, eval_time, class_accuracy = evaluate(ensemble, test_loader, loss_fn, device)
    
    print("\n" + "="*50)
    print("앙상블 모델 성능:")
    print("="*50)
    print(f"앙상블 테스트 손실: {ensemble_loss:.4f}")
    print(f"앙상블 테스트 정확도: {ensemble_acc:.2f}%")
    print(f"개별 모델 정확도: ResNet50 {acc1:.2f}%, EfficientNet-B0 {acc2:.2f}%, MobileNetV2 {acc3:.2f}%")
    print("="*50)
    
    # 앙상블 모델 저장
    torch.save(ensemble.state_dict(), 'bird_ensemble_model.pth')
    print("앙상블 모델이 'bird_ensemble_model.pth'로 저장되었습니다.")
    
    # 클래스별 정확도 출력
    print("\n앙상블 모델 클래스별 정확도:")
    for label in sorted(class_accuracy.keys()):
        print(f"  클래스 {label} ({class_names[label] if label < len(class_names) else 'Unknown'}): {class_accuracy[label]:.2f}%")
    
    return ensemble, ensemble_acc

# 5-fold 교차 검증 함수 (선택 사항)
def cross_validation(model_name, num_epochs=20, batch_size=16, learning_rate=0.001, n_splits=5):
    # 데이터 로드
    train_dataset, _, num_classes, class_names = load_data()
    
    # K-fold 교차 검증
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 데이터셋에서 레이블 추출
    if isinstance(train_dataset, Subset):
        labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        labels = [sample[1] for sample in train_dataset]
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        # 폴드별 데이터 분할
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # 모델 생성
        model = create_transfer_model(model_name, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 손실 함수와 옵티마이저
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # 결과 저장용 변수
        best_val_acc = 0
        fold_train_losses, fold_val_losses = [], []
        fold_train_accs, fold_val_accs = [], []
        
        # 훈련 루프
        for epoch in range(num_epochs):
            # 훈련
            train_loss, train_acc, _ = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, num_epochs)
            
            # 평가
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 결과 기록
            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            
            # 결과 출력
            print(f'에폭 {epoch+1}/{num_epochs} - 훈련 손실: {train_loss:.4f}, 정확도: {train_acc:.2f}% | '
                  f'검증 손실: {val_loss:.4f}, 정확도: {val_acc:.2f}%')
            
            # 최고 성능 기록
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        fold_results.append(best_val_acc)
        print(f"Fold {fold+1} 최고 검증 정확도: {best_val_acc:.2f}%")
        
        # 폴드별 학습 곡선 시각화
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fold_train_losses, label='Train Loss')
        plt.plot(fold_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(fold_train_accs, label='Train Accuracy')
        plt.plot(fold_val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title(f'Fold {fold+1} Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(f'fold{fold+1}_results.png')
        plt.show()
    
    # K-fold 결과 요약
    print("\n" + "="*50)
    print(f"{n_splits}-Fold 교차 검증 결과 ({model_name}):")
    print("="*50)
    for fold, acc in enumerate(fold_results):
        print(f"Fold {fold+1} 정확도: {acc:.2f}%")
    print(f"평균 정확도: {sum(fold_results)/len(fold_results):.2f}%")
    print(f"표준 편차: {np.std(fold_results):.2f}%")
    print("="*50)
    
    return np.mean(fold_results)

# 예측 함수 (실제 사용 시 필요)
def predict_image(model, image_path, transform, class_names, device):
    from PIL import Image
    
    # 이미지 로드 및 변환
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 예측
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    
    # 결과 반환
    return {
        'predicted_class': class_names[predicted_idx],
        'confidence': probabilities[predicted_idx].item() * 100,
        'all_probabilities': {class_names[i]: prob.item() * 100 for i, prob in enumerate(probabilities)}
    }

# 메인 실행 부분
if __name__ == "__main__":
    import os
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='새 이미지 분류 모델 학습')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'web'], 
                        help='실행 모드 (train, evaluate, web)')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet_b0', 'mobilenet_v2'], 
                        help='사용할 모델 아키텍처')
    parser.add_argument('--epochs', type=int, default=40, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--load_pretrained', action='store_true', help='기존 학습된 모델 불러오기')
    parser.add_argument('--pretrained_path', type=str, default=None, help='기존 학습된 모델 경로')
    
    args = parser.parse_args()
    
    # 기존 학습된 모델 경로 자동 감지
    if args.load_pretrained and args.pretrained_path is None:
        possible_paths = [
            f'best_{args.model}_bird_model.pth',  # 모델별 경로
            'best_bird_model.pth',                # 일반 경로
            'bird_model_web.pt'                   # 웹 배포용 모델
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.pretrained_path = path
                print(f"기존 모델 파일을 발견했습니다: {path}")
                break
    
    if args.mode == 'train':
        # 모델 학습
        model, best_acc, best_model_path = train_model(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            load_pretrained=args.load_pretrained,
            pretrained_path=args.pretrained_path
        )
        
    elif args.mode == 'evaluate':
        # 모델 평가만 수행
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset, test_dataset, num_classes, class_names = load_data()
        
        model = create_transfer_model(args.model, num_classes)
        
        # 모델 파라미터 불러오기
        if args.pretrained_path:
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        else:
            print("평가할 모델 경로를 지정해야 합니다.")
            exit(1)
        
        model = model.to(device)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        loss_fn = nn.CrossEntropyLoss()
        
        test_loss, test_acc, eval_time, class_accuracy = evaluate(model, test_loader, loss_fn, device)
        
        print("\n" + "="*50)
        print(f"모델 평가 결과 ({args.model}):")
        print("="*50)
        print(f"테스트 손실: {test_loss:.4f}")
        print(f"테스트 정확도: {test_acc:.2f}%")
        
        print("\n클래스별 정확도:")
        for label in sorted(class_accuracy.keys()):
            print(f"  클래스 {label} ({class_names[label] if label < len(class_names) else 'Unknown'}): {class_accuracy[label]:.2f}%")
        
    elif args.mode == 'web':
        # 웹 배포용 모델 생성
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset, test_dataset, num_classes, class_names = load_data()
        
        # 모델 생성
        model = create_transfer_model(args.model, num_classes)
        
        # 모델 파라미터 불러오기
        if args.pretrained_path:
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        else:
            print("웹 배포용 모델 생성을 위한 기존 모델 경로를 지정해야 합니다.")
            exit(1)
        
        model.eval()
        
        # 클래스 이름 저장
        import json
        with open('bird_class_names.json', 'w') as f:
            json.dump(class_names, f)
        
        # 모델 변환 및 저장 (256x256 크기 사용)
        example_input = torch.rand(1, 3, 256, 256)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save('bird_model_web.pt')
        
        print(f"모델이 웹 배포용으로 'bird_model_web.pt'에 저장되었습니다.")
        print(f"클래스 이름이 'bird_class_names.json'에 저장되었습니다.")
        
    else:
        print(f"지원되지 않는 모드: {args.mode}")