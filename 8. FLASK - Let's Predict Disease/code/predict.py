"""
ODIR 안구질환 이미지 예측 모듈
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import traceback
import sys

# 코드 폴더 추가 (odir_analysis.py 모듈 임포트를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 설정 값
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

def predict_image(model, image_path, device="cpu"):
    """안구 이미지 분석 함수 - 모든 예외 처리 및 디버깅 기능 포함"""
    from odir_analysis import get_transforms
    
    # 로그 출력
    print(f"[예측] 이미지 분석 시작: {image_path}")
    
    try:
        # 모델 평가 모드 설정
        model.eval()
        
        # 이미지 존재 확인
        if not os.path.exists(image_path):
            print(f"[오류] 이미지 파일이 존재하지 않음: {image_path}")
            return {disease: 0.0 for disease in DISEASE_NAMES}
        
        # 이미지 변환 설정
        transform = get_transforms(is_training=False)
        
        # 이미지 로드
        try:
            # PIL 방식으로 이미지 로드 시도
            img = Image.open(image_path).convert('RGB')
            print(f"[예측] PIL 이미지 로드 성공 (크기: {img.size})")
        except Exception as e:
            print(f"[예측] PIL 로드 실패, OpenCV로 시도: {e}")
            # OpenCV 방식으로 시도
            img = cv2.imread(image_path)
            if img is None:
                print(f"[오류] OpenCV 이미지 로드 실패")
                return {disease: 0.0 for disease in DISEASE_NAMES}
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            print(f"[예측] OpenCV 이미지 로드 성공 (크기: {img.size})")
        
        # 이미지 변환 및 모델 입력 형식으로 변환
        try:
            img_tensor = transform(img).unsqueeze(0)
            print(f"[예측] 이미지 변환 성공 (형태: {img_tensor.shape})")
            
            # 모델 장치로 이동 (CPU/GPU)
            img_tensor = img_tensor.to(device)
        except Exception as e:
            print(f"[오류] 이미지 변환 실패: {e}")
            traceback.print_exc()
            return {disease: 0.0 for disease in DISEASE_NAMES}
        
        # 모델 예측 수행
        try:
            with torch.no_grad():
                predictions = model(img_tensor)
                predictions = predictions.squeeze().cpu().numpy()
                print(f"[예측] 예측 성공: {predictions}")
                
                # 확률값이 모두 0인지 확인
                if np.all(predictions < 0.001):
                    print("[경고] 모든 예측값이 0에 가깝습니다. 모델 문제 가능성 있음")
        except Exception as e:
            print(f"[오류] 모델 예측 실패: {e}")
            traceback.print_exc()
            return {disease: 0.0 for disease in DISEASE_NAMES}
        
        # 결과 출력
        results = {}
        for i, disease in enumerate(DISEASE_NAMES):
            results[disease] = float(predictions[i])
            print(f"[예측] {disease}: {results[disease]:.4f}")
        
        return results
    
    except Exception as e:
        print(f"[오류] 예측 중 예기치 않은 오류 발생: {e}")
        traceback.print_exc()
        return {disease: 0.0 for disease in DISEASE_NAMES}
