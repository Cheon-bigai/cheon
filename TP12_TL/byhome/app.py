import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import os
import glob

# 페이지 설정
st.set_page_config(
    page_title="이미지 세그멘테이션 및 색상 복원 도구",
    page_icon="🔍",
    layout="wide"
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 크기 설정
IMG_HEIGHT = 256
IMG_WIDTH = 256

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

# 이미지 전처리 함수
def preprocess_image(img):
    # RGB 이미지인지 확인
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA 이미지인 경우
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # 이미지 크기 조정
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # 원본 이미지 저장
    original_img = img.copy()
    
    # 흑백 이미지 생성
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 전처리 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 텐서로 변환
    tensor_img = transform(img)
    tensor_img = tensor_img.unsqueeze(0).to(device)
    
    return tensor_img, original_img, gray_img

# 예측 결과 처리 함수
def postprocess_prediction(output_tensor, original_img):
    # 예측 결과를 numpy 배열로 변환
    output_np = output_tensor.squeeze().cpu().detach().numpy()
    
    # 이진화 (0.5 임계값)
    binary_mask = (output_np > 0.5).astype(np.uint8) * 255
    
    # 마스크를 3채널로 확장
    mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # 원본 이미지와 마스크 결합 (색상 복원된 이미지)
    # 마스크 영역만 원본 이미지 색상 유지, 나머지는 흑백으로
    gray_3ch = cv2.cvtColor(cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    
    # 마스크를 이용하여 색상 복원 이미지 생성
    mask_bool = (binary_mask > 0)
    mask_bool_3ch = np.stack([mask_bool, mask_bool, mask_bool], axis=2)
    
    restored_img = np.where(mask_bool_3ch, original_img, gray_3ch)
    
    return binary_mask, restored_img

# 세그멘테이션 시각화 함수
def visualize_segmentation(img, mask, alpha=0.5):
    # 이미지와 마스크가 같은 크기인지 확인
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    
    # 마스크 시각화를 위한 색상 설정 (빨간색)
    mask_color = np.zeros_like(img)
    mask_color[mask > 0] = [255, 0, 0]  # BGR 형식에서 빨간색
    
    # 이미지와 마스크 합성
    return cv2.addWeighted(img, 1, mask_color, alpha, 0)

# 메트릭 계산 함수
def calculate_metrics(true_mask, pred_mask):
    # 이진화
    true_mask = (true_mask > 128).astype(np.uint8)
    pred_mask = (pred_mask > 128).astype(np.uint8)
    
    # IoU (Intersection over Union) 계산
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    # 정밀도 (Precision) 계산
    tp = intersection
    fp = pred_mask.sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 재현율 (Recall) 계산
    tp = intersection
    fn = true_mask.sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 점수 계산
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# 메트릭 시각화 함수
def visualize_metrics(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 메트릭 이름과 값
    names = list(metrics.keys())
    values = list(metrics.values())
    
    # 수평 바 차트 생성
    bars = ax.barh(names, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # 바 위에 값 표시
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # 차트 설정
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Score')
    ax.set_title('Segmentation Metrics')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    return fig

# 샘플 이미지 리스트 추가 (실제 경로로 수정 필요)
def get_sample_images(base_dir):
    sample_data = []
    
    # 경로 지정 (실제 디렉토리 구조에 맞게 수정)
    input_dir = os.path.join(base_dir, r"C:\Users\KDT-13\Desktop\KDT7\0.Project\TP12_TL\data\train_input")
    
    # 이미지 파일 검색
    if os.path.exists(input_dir):
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        
        # 최대 10개 샘플만 가져오기
        for i, img_path in enumerate(image_files[:10]):
            sample_name = os.path.basename(img_path)
            sample_data.append({
                "id": i,
                "name": f"샘플 이미지 {i+1}: {sample_name}",
                "input_path": img_path,
                # GT 이미지 경로 추정 (실제 경로에 맞게 수정)
                "gt_path": img_path.replace("input_images", "gt_images")
            })
    else:
        # 디렉토리가 없으면 더미 데이터 생성
        for i in range(5):
            sample_data.append({
                "id": i,
                "name": f"샘플 이미지 {i+1}",
                "input_path": None,
                "gt_path": None
            })
    
    return sample_data

# 샘플 이미지 로드
def load_sample_image(sample):
    if sample["input_path"] and os.path.exists(sample["input_path"]):
        # 실제 이미지 로드
        color_img = cv2.imread(sample["input_path"])
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # GT 마스크 로드 (있는 경우)
        if sample["gt_path"] and os.path.exists(sample["gt_path"]):
            true_mask = cv2.imread(sample["gt_path"], cv2.IMREAD_GRAYSCALE)
        else:
            # GT 마스크가 없으면 더미 마스크 생성
            true_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
            radius = min(IMG_WIDTH, IMG_HEIGHT) // 3
            cv2.circle(true_mask, (center_x, center_y), radius, 255, -1)
    else:
        # 이미지가 없으면 더미 이미지 생성
        color_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        color_img[:, :, 0] = np.random.randint(0, 255)
        color_img[:, :, 1] = np.random.randint(0, 255)
        color_img[:, :, 2] = np.random.randint(0, 255)
        
        # 더미 마스크 생성
        true_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
        radius = min(IMG_WIDTH, IMG_HEIGHT) // 3
        cv2.circle(true_mask, (center_x, center_y), radius, 255, -1)
    
    return color_img, true_mask

# 모델 로드 함수
@st.cache_resource
def load_model(model_path=None):
    model = UNetWithResNet50Encoder().to(device)
    
    try:
        if model_path and os.path.exists(model_path):
            # 저장된 모델 파일 로드
            checkpoint = torch.load(model_path, map_location=device)
            
            # 모델 가중치 로드
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("모델 가중치를 성공적으로 로드했습니다!")
            else:
                model.load_state_dict(checkpoint)
                print("모델 가중치를 성공적으로 로드했습니다!")
        else:
            print("모델 파일을 찾을 수 없어 초기화된 모델을 사용합니다.")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        st.error(f"모델 로드 중 오류 발생: {e}")
    
    model.eval()
    return model

# 메인 Streamlit 앱
def main():
    st.title("🖼️ 이미지 세그멘테이션 및 색상 복원 웹 애플리케이션")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.title("설정")
    
    # 기본 디렉토리 설정
    base_dir = st.sidebar.text_input(
        "기본 디렉토리 경로",
        value="C:/Users/KDT-13/Desktop/KDT7/0.Project/TP12_TL"
    )
    
    # 모델 경로 설정
    model_path = st.sidebar.text_input(
        "모델 파일 경로",
        value=os.path.join(base_dir, r"C:\Users\KDT-13\Desktop\KDT7\0.Project\TP12_TL\byhome\best_model.pth")
    )
    
    # 모델 로드
    with st.spinner("모델을 로드하는 중..."):
        model = load_model(model_path)
    
    # 사이드바 - 이미지 업로드 또는 샘플 선택
    st.sidebar.title("입력 이미지 선택")
    input_option = st.sidebar.radio(
        "이미지 소스 선택:",
        ["이미지 업로드", "샘플 이미지 사용"]
    )
    
    # 메인 영역
    col1, col2, col3 = st.columns(3)
    
    if input_option == "이미지 업로드":
        uploaded_file = st.sidebar.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 업로드된 이미지 로드
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 진행 상황 표시
            with st.spinner("이미지 처리 중..."):
                # 이미지 전처리
                tensor_img, color_img, gray_img = preprocess_image(image)
                
                # 모델 예측
                with torch.no_grad():
                    output = model(tensor_img)
                
                # 예측 결과 후처리
                pred_mask, restored_img = postprocess_prediction(output, color_img)
                
                # 결과 표시
                with col1:
                    st.subheader("1. 흑백 입력 이미지")
                    st.image(gray_img, use_container_width=True)
                
                with col2:
                    st.subheader("2. 모델 예측 결과")
                    st.image(pred_mask, use_container_width=True)
                
                with col3:
                    st.subheader("3. 색상 복원 이미지")
                    st.image(restored_img, use_container_width=True)
                
                # 여기서는 실제 마스크가 없으므로 임의의 마스크로 메트릭 계산
                dummy_true_mask = np.zeros_like(gray_img)
                center_x, center_y = IMG_WIDTH // 2, IMG_HEIGHT // 2
                radius = min(IMG_WIDTH, IMG_HEIGHT) // 3
                cv2.circle(dummy_true_mask, (center_x, center_y), radius, 255, -1)
                
                metrics = calculate_metrics(dummy_true_mask, pred_mask)
                
                # 메트릭 표시
                st.markdown("---")
                st.subheader("🔍 평가 지표")
                
                # 메트릭 차트
                metric_chart = visualize_metrics(metrics)
                st.pyplot(metric_chart)
                
                # 텍스트로 메트릭 표시
                st.markdown("### 상세 수치")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.info(f"IoU: {metrics['IoU']:.4f}")
                    st.info(f"Precision: {metrics['Precision']:.4f}")
                with col_m2:
                    st.info(f"Recall: {metrics['Recall']:.4f}")
                    st.info(f"F1 Score: {metrics['F1 Score']:.4f}")
    
    else:  # 샘플 이미지 사용
        sample_images = get_sample_images(base_dir)
        
        if not sample_images:
            st.warning("샘플 이미지를 찾을 수 없습니다. 기본 디렉토리 경로를 확인하세요.")
            return
            
        selected_sample = st.sidebar.selectbox(
            "샘플 이미지를 선택하세요:",
            options=sample_images,
            format_func=lambda x: x["name"]
        )
        
        if selected_sample:
            # 진행 상황 표시
            with st.spinner("샘플 이미지 처리 중..."):
                # 샘플 이미지 로드
                color_img, true_mask = load_sample_image(selected_sample)
                
                # 이미지 전처리
                tensor_img, _, gray_img = preprocess_image(color_img)
                
                # 모델 예측
                with torch.no_grad():
                    output = model(tensor_img)
                
                # 예측 결과 후처리
                pred_mask, restored_img = postprocess_prediction(output, color_img)
                
                # 결과 표시
                with col1:
                    st.subheader("1. 흑백 입력 이미지")
                    st.image(gray_img, use_container_width=True)
                
                with col2:
                    st.subheader("2. 모델 예측 결과")
                    st.image(pred_mask, use_container_width=True)
                
                with col3:
                    st.subheader("3. 색상 복원 이미지")
                    st.image(restored_img, use_container_width=True)
                
                # 메트릭 계산
                metrics = calculate_metrics(true_mask, pred_mask)
                
                # 메트릭 표시
                st.markdown("---")
                st.subheader("🔍 평가 지표")
                
                # 메트릭 차트
                metric_chart = visualize_metrics(metrics)
                st.pyplot(metric_chart)
                
                # 텍스트로 메트릭 표시
                st.markdown("### 상세 수치")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.info(f"IoU: {metrics['IoU']:.4f}")
                    st.info(f"Precision: {metrics['Precision']:.4f}")
                with col_m2:
                    st.info(f"Recall: {metrics['Recall']:.4f}")
                    st.info(f"F1 Score: {metrics['F1 Score']:.4f}")
    
    # 페이지 하단 정보
    st.markdown("---")
    st.markdown("### 모델 정보")
    st.markdown("이 웹 애플리케이션은 U-Net with ResNet50 아키텍처를 기반으로 한 세그멘테이션 모델을 사용합니다.")
    st.markdown("세그멘테이션 모델은 이미지에서 중요한 영역을 분리하고, 그 영역에 대해 원본 색상을 복원합니다.")

if __name__ == "__main__":
    main()