# 새 이미지 분류 프로젝트

## 프로젝트 개요
이 프로젝트는 딥러닝을 활용하여 새 종류를 분류하는 시스템을 구현합니다. 기존 전통적인 머신러닝 접근법으로는 25%의 낮은 정확도를 보였으나, 딥러닝을 적용하여 60%까지 향상시켰고, 최종적으로 ResNet 구조를 도입하여 96.68%의 높은 정확도를 달성했습니다.

## 주요 특징
- ResNet 구조의 도입으로 기존 CNN 대비 22.5%p 성능 향상
- 초기 4종 800개 데이터에서 25종 12,000개 데이터셋으로 확장
- 웹 애플리케이션 형태로 배포하여 사용자 접근성 확보
- 다양한 데이터 증강 기술을 적용한 학습 파이프라인

## 시스템 요구사항
- Python 3.8+
- PyTorch 1.8+
- Flask
- PIL (Pillow)
- NumPy
- Matplotlib
- torchvision

## 설치 방법
```bash
# 저장소 클론
git clone https://github.com/yourusername/bird-classification.git
cd bird-classification

# 필요 패키지 설치
pip install -r requirements.txt
```

## 프로젝트 구조
```
bird-classification/
├── 6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py  # 모델 훈련 코드
├── app.py                # Flask 웹 애플리케이션
├── static/               
│   ├── css/              
│   │   └── style.css     # 웹 스타일시트
│   └── js/               
│       └── main.js       # 웹 인터페이스 JavaScript
├── templates/           
│   └── index.html        # 웹 UI 템플릿
├── bird_model_web.pt     # 배포용 모델 (생성됨)
└── bird_class_names.json # 클래스 이름 목록 (생성됨)
```

## 사용법

### 모델 훈련
```bash
# 기본 훈련 (ResNet50 사용)
python "6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py" --mode train --model resnet50

# 다른 모델 아키텍처로 훈련
python "6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py" --mode train --model efficientnet_b0

# 기존 모델 이어서 훈련
python "6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py" --mode train --model resnet50 --load_pretrained --pretrained_path "best_resnet50_bird_model.pth"
```

### 모델 평가
```bash
python "6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py" --mode evaluate --model resnet50 --pretrained_path "best_resnet50_bird_model.pth"
```

### 웹 배포 모델 생성
```bash
python "6.DL - 파라미터 조절 모델 구조 혁신 (새 이미지 분류ver 2).py" --mode web --model resnet50 --pretrained_path "best_resnet50_bird_model.pth"
```

### 웹 서버 실행
```bash
python app.py
```
실행 후 `http://localhost:5000/`으로 접속하여 사용할 수 있습니다.

## 기술적 설명

### 기존 CNN vs ResNet 구조 비교

#### 일반 CNN의 한계
- 깊은 레이어를 쌓을수록 그래디언트 소실 문제 발생
- 훈련 복잡성 증가로 최적화 어려움
- 과적합 위험 증가

#### ResNet의 핵심 원리: 잔차 학습
- 일반 CNN: `H(x) = f(x)`
- ResNet: `H(x) = f(x) + x`
- 스킵 연결을 통해 입력 정보를 보존하며 잔차만 학습
- 그래디언트 흐름 개선으로 깊은 네트워크도 효과적 훈련 가능

### 데이터 증강 기법
- 랜덤 크롭을 통한 새의 다양한 부분 학습
- 좌우 반전으로 데이터 다양성 증가
- 랜덤 회전, 색상 변형, 아핀 변환 적용

### 성능 향상 요약
- 초기 CNN: 60% 정확도
- ResNet 도입: 82.5% 정확도
- 데이터셋 확장 후 최종: 96.68% 정확도

## 웹 인터페이스
웹 애플리케이션은 사용자가 새 이미지를 업로드하면 학습된 모델을 통해 새의 종류를 예측하고 각 클래스별 확률을 표시합니다:
1. 이미지 업로드 인터페이스
2. 이미지 미리보기 기능
3. 예측 결과 표시 (클래스명 및 신뢰도)
4. 모든 클래스에 대한 확률 막대 그래프

## 프로젝트 회고

1. **파라미터 조절의 한계**  
   단순한 하이퍼파라미터 조정만으로는 성능 향상에 명확한 한계가 있었습니다.

2. **구조적 혁신의 효과**  
   ResNet의 잔차 학습 패러다임 도입으로 22.5%p의 정확도 향상을 달성했습니다.

3. **전문성 개발의 가치**  
   "공통점을 공유하고 차이점에 집중하는" ResNet의 접근 방식이 새 분류 문제와 완벽하게 부합했습니다.

## 참고 문헌
He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv:1512.03385.
