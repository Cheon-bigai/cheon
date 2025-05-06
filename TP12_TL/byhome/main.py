import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchmetrics import JaccardIndex, Precision
from tqdm import tqdm

# 시작 시간 측정
start_time = time.time()

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 시드 값 설정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 경로 설정 - 절대 경로 사용
BASE_DIR = "C:/Users/cheon/Desktop/vscode/Project/TL/TP12_TL/data"
train_csv_path = os.path.join(BASE_DIR, "train.csv")
test_csv_path = os.path.join(BASE_DIR, "test.csv")

# CSV 파일 로드
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 이미지 크기 설정
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# PyTorch 데이터셋 클래스 정의
class SegmentationDataset(Dataset):
    def __init__(self, df, is_train=True, transform=None, mask_transform=None):
        self.df = df
        self.is_train = is_train
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 입력 이미지 로딩
        img_path = os.path.join(BASE_DIR, row['input_image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # 훈련 데이터일 경우 마스크 이미지도 로딩
        if self.is_train:
            mask_path = os.path.join(BASE_DIR, row['gt_image_path'])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
            mask = (mask > 128).astype(np.uint8)
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            return img, mask
        else:
            return img

# U-Net with ResNet50 모델 정의
class UNetWithResNet50Encoder(nn.Module):
    def __init__(self, n_classes=1):
        super(UNetWithResNet50Encoder, self).__init__()
        
        # ResNet50을 인코더로 사용
        resnet = models.resnet50(pretrained=True)
        
        # ResNet50의 계층들을 인코더로 사용
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 채널
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1  # 256 채널
        self.encoder3 = resnet.layer2  # 512 채널
        self.encoder4 = resnet.layer3  # 1024 채널
        self.encoder5 = resnet.layer4  # 2048 채널
        
        # 디코더 정의
        self.decoder5 = self._decoder_block(2048, 1024)
        self.decoder4 = self._decoder_block(1024 + 1024, 512)
        self.decoder3 = self._decoder_block(512 + 512, 256)
        self.decoder2 = self._decoder_block(256 + 256, 64)
        self.decoder1 = self._decoder_block(64 + 64, 32)
        
        # 최종 출력 레이어
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 인코더 단계
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # 디코더 단계와 스킵 연결
        d5 = self.decoder5(e5)
        d5 = nn.functional.interpolate(d5, size=e4.size()[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, e4], dim=1)
        
        d4 = self.decoder4(d5)
        d4 = nn.functional.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = nn.functional.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = nn.functional.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.decoder1(d2)
        d1 = nn.functional.interpolate(d1, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        out = self.final_conv(d1)
        out = self.final_activation(out)
        
        return out

# 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 손실 함수 정의
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, reduction='mean')
        
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        smooth = 1e-5
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_score
        
        # Combined loss
        return bce_loss + dice_loss

# 훈련 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_precisions, val_precisions = [], []
    
    # IoU와 Precision 메트릭 초기화
    iou_metric = JaccardIndex(task="binary", num_classes=2).to(device)
    precision_metric = Precision(task="binary").to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 훈련 단계
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_precision = 0.0
        
        with tqdm(train_loader, unit="batch") as t:
            for inputs, targets in t:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파 + 역전파 + 최적화
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 통계
                running_loss += loss.item() * inputs.size(0)
                pred_binary = (outputs > 0.5).float()
                running_iou += iou_metric(pred_binary, targets.int()).item() * inputs.size(0)
                running_precision += precision_metric(pred_binary, targets.int()).item() * inputs.size(0)
                
                t.set_postfix(loss=loss.item(), 
                             iou=iou_metric(pred_binary, targets.int()).item(),
                             precision=precision_metric(pred_binary, targets.int()).item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        epoch_precision = running_precision / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_ious.append(epoch_iou)
        train_precisions.append(epoch_precision)
        
        # 검증 단계
        model.eval()
        val_running_loss = 0.0
        val_running_iou = 0.0
        val_running_precision = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as t:
                for inputs, targets in t:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # 통계
                    val_running_loss += loss.item() * inputs.size(0)
                    pred_binary = (outputs > 0.5).float()
                    val_running_iou += iou_metric(pred_binary, targets.int()).item() * inputs.size(0)
                    val_running_precision += precision_metric(pred_binary, targets.int()).item() * inputs.size(0)
                    
                    t.set_postfix(val_loss=loss.item(), 
                                val_iou=iou_metric(pred_binary, targets.int()).item(),
                                val_precision=precision_metric(pred_binary, targets.int()).item())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_iou = val_running_iou / len(val_loader.dataset)
        val_epoch_precision = val_running_precision / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_ious.append(val_epoch_iou)
        val_precisions.append(val_epoch_precision)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_epoch_loss)
        
        print(f'Train Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, Precision: {epoch_precision:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, IoU: {val_epoch_iou:.4f}, Precision: {val_epoch_precision:.4f}')
        
        # 모델 저장 (검증 손실이 개선된 경우)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved at Epoch {epoch+1}')
        
        # 주기적인 모델 저장
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_epoch_loss,
            }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 훈련 결과 반환
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_ious,
        'val_iou': val_ious,
        'train_precision': train_precisions,
        'val_precision': val_precisions
    }
    
    return model, history

# 훈련 결과 시각화 함수
def plot_training_history(history, save_path):
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'])
    plt.plot(history['val_iou'])
    plt.title('Model IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_precision'])
    plt.plot(history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 메인 함수
def main(max_samples=None, epochs=100):
    # 모델 저장 경로
    model_save_dir = "C:/Users/cheon/Desktop/vscode/Project/TL/TP12_TL/models/pytorch_advanced"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 데이터 샘플 수 제한 (사용자 지정)
    if max_samples is not None and max_samples > 0:
        print(f"\n데이터 샘플 수를 {max_samples}개로 제한합니다.")
        train_sample_df = train_df.head(max_samples)
    else:
        train_sample_df = train_df
    
    print(f"훈련 데이터 크기: {len(train_sample_df)}")
    
    # 훈련 및 검증 데이터 분할
    train_data, val_data = train_test_split(train_sample_df, test_size=0.2, random_state=SEED)
    print(f"훈련 세트: {len(train_data)}개, 검증 세트: {len(val_data)}개")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SegmentationDataset(train_data, is_train=True, transform=train_transform)
    val_dataset = SegmentationDataset(val_data, is_train=True, transform=val_transform)
    
    # 데이터 로더 생성
    BATCH_SIZE = 8
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 생성
    model = UNetWithResNet50Encoder().to(device)
    
    # 손실 함수, 옵티마이저, 학습률 스케줄러 정의
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2, 
        patience=5,
        min_lr=1e-6, 
        verbose=True
    )
    
    # 모델 훈련
    print(f"\n{epochs} 에폭 훈련 시작...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs,
        device=device,
        save_dir=model_save_dir
    )
    
    # 모델 미세 조정(Fine-tuning)
    print("\n모델 미세 조정(Fine-tuning) 시작...")
    
    # 인코더 레이어 해제
    for param in model.parameters():
        param.requires_grad = True
    
    # 더 작은 학습률로 다시 컴파일
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2, 
        patience=5,
        min_lr=1e-7, 
        verbose=True
    )
    
    # 미세 조정 훈련
    FINE_TUNE_EPOCHS = 50
    model, fine_tune_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=FINE_TUNE_EPOCHS,
        device=device,
        save_dir=model_save_dir
    )
    
    # 학습 기록 합치기
    combined_history = {
        'train_loss': history['train_loss'] + fine_tune_history['train_loss'],
        'val_loss': history['val_loss'] + fine_tune_history['val_loss'],
        'train_iou': history['train_iou'] + fine_tune_history['train_iou'],
        'val_iou': history['val_iou'] + fine_tune_history['val_iou'],
        'train_precision': history['train_precision'] + fine_tune_history['train_precision'],
        'val_precision': history['val_precision'] + fine_tune_history['val_precision']
    }
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(model_save_dir, 'model-final.pth'))
    
    # 훈련 결과 시각화
    plot_training_history(combined_history, os.path.join(model_save_dir, 'training-history.png'))
    
    # 종료 시간 측정 및 출력
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n총 훈련 시간: {training_time:.2f}초 ({training_time/60:.2f}분, {training_time/3600:.2f}시간)")
    
    print("학습 완료!")

# 실행 코드
if __name__ == '__main__':
    # 사용자 입력 받기
    sample_input = input("사용할 데이터 샘플 수를 입력하세요 (전체 데이터 사용은 0 입력): ")
    max_samples = int(sample_input) if sample_input.isdigit() else 0
    
    epoch_input = input("훈련할 에폭 수를 입력하세요 (기본값 100): ")
    epochs = int(epoch_input) if epoch_input.isdigit() else 100
    
    # 메인 함수 실행
    main(max_samples=max_samples, epochs=epochs)