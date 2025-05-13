import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from sklearn.ensemble import VotingClassifier  # 앙상블 모델링을 위한 라이브러리
import koreanize_matplotlib
from tqdm import tqdm  # 진행 상황 표시용
import seaborn as sns
from scipy import stats


# 기본 경로 설정
BASE_DIR = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP8_2_CV\data'
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 이미 생성된 파일 경로
FEATURES_PATH = {
    'train_features': os.path.join(OUTPUT_DIR, "X_train.npy"),
    'test_features': os.path.join(OUTPUT_DIR, "X_test.npy"),
    'svm_model': os.path.join(OUTPUT_DIR, "SVM_model.pkl"),
    'rf_model': os.path.join(OUTPUT_DIR, "RandomForest_model.pkl"),
    'knn_model': os.path.join(OUTPUT_DIR, "KNN_model.pkl"),
    'ensemble_model': os.path.join(OUTPUT_DIR, "Ensemble_model.pkl"),  # 앙상블 모델 경로 추가
    'submission': os.path.join(OUTPUT_DIR, "submission.csv")
}

# 최대 사용할 샘플 수 (속도 향상을 위해)
MAX_SAMPLES_PER_CLASS = 1000  # 각 클래스당 최대 샘플 수

#------------------------------------------------------------------
# 오차율 계산 및 시각화 함수 (새로 추가)
def calculate_error_rate(y_true, y_pred, class_names, output_dir, title_prefix=""):
    """
    오차율을 계산하고 시각화하는 함수
    
    Parameters:
    y_true (numpy.ndarray): 실제 레이블
    y_pred (numpy.ndarray): 예측 레이블
    class_names (list): 클래스 이름 목록
    output_dir (str): 결과를 저장할 디렉토리
    title_prefix (str): 그래프 제목 앞에 추가할 접두사 (예: "훈련 데이터:" 또는 "테스트 데이터:")
    
    Returns:
    pandas.DataFrame: 클래스별 오차율이 포함된 데이터프레임
    """
    print(f"\n{title_prefix} 오차율 계산 및 시각화")
    
    # 정확도 계산
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    print(f"전체 오차율: {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 클래스별 오차율 계산
    class_error_rates = {}
    for i, class_name in enumerate(class_names):
        # 해당 클래스의 실제 샘플 수
        total_samples = np.sum(y_true == class_name)
        # 해당 클래스를 다른 클래스로 잘못 예측한 경우(오분류)
        misclassified = total_samples - cm[i, i]
        # 오차율 계산
        if total_samples > 0:
            error_rate = misclassified / total_samples
            class_error_rates[class_name] = error_rate
    
    # 데이터프레임으로 변환
    error_df = pd.DataFrame({
        'class': list(class_error_rates.keys()),
        'error_rate': list(class_error_rates.values()),
        'error_percentage': [rate * 100 for rate in class_error_rates.values()]
    })
    
    # 오차율 기준으로 정렬
    error_df = error_df.sort_values('error_rate', ascending=False).reset_index(drop=True)
    
    # 상위 10개 클래스 오차율 시각화
    plt.figure(figsize=(12, 8))
    top_errors = error_df.head(10)
    ax = sns.barplot(x='class', y='error_percentage', data=top_errors, palette='Reds')
    
    # 막대 위에 오차율 표시
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    fontsize=10, fontweight='bold')
    
    plt.title(f'{title_prefix} 오차율이 높은 상위 10개 클래스', fontsize=14)
    plt.xlabel('클래스', fontsize=12)
    plt.ylabel('오차율 (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(":", "").strip()}_top_error_rates.png'))
    plt.show()
    
    # 오차율 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(error_df['error_percentage'], bins=10, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(error_df['error_percentage']), color='red', linestyle='--', 
                label=f'평균 오차율: {np.mean(error_df["error_percentage"]):.2f}%')
    plt.title(f'{title_prefix} 클래스별 오차율 분포', fontsize=14)
    plt.xlabel('오차율 (%)', fontsize=12)
    plt.ylabel('클래스 수', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(":", "").strip()}_error_rate_distribution.png'))
    plt.show()
    
    # 히트맵으로 혼동 행렬 시각화 (전체가 아닌 오차율이 높은 상위 클래스만)
    top_error_classes = top_errors['class'].values
    top_indices = [list(class_names).index(cls) for cls in top_error_classes]
    
    # 상위 오차율 클래스에 대한 부분 혼동 행렬 추출
    cm_subset = cm[np.ix_(top_indices, top_indices)]
    cm_subset_normalized = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset_normalized, annot=True, fmt='.2f', cmap='Reds', 
              xticklabels=top_error_classes, yticklabels=top_error_classes)
    plt.title(f'{title_prefix} 오차율이 높은 클래스의 혼동 행렬', fontsize=14)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(":", "").strip()}_top_error_confusion_matrix.png'))
    plt.show()
    
    # 오차율과 샘플 수의 관계
    y_true_series = pd.Series(y_true)
    sample_counts = y_true_series.value_counts()
    
    # 데이터프레임에 샘플 수 추가
    error_df['sample_count'] = error_df['class'].map(sample_counts)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(error_df['sample_count'], error_df['error_percentage'], alpha=0.7, 
                c=error_df['error_percentage'], cmap='coolwarm', s=50)
    
    # 회귀선 추가
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        error_df['sample_count'], error_df['error_percentage'])
    
    x = np.array(error_df['sample_count'])
    y = slope * x + intercept
    plt.plot(x, y, 'r--', label=f'회귀선 (r={r_value:.2f})')
    
    # 상관관계 출력
    correlation = error_df['sample_count'].corr(error_df['error_percentage'])
    print(f"샘플 수와 오차율의 상관관계: {correlation:.4f}")
    
    plt.title(f'{title_prefix} 샘플 수와 오차율의 관계', fontsize=14)
    plt.xlabel('샘플 수', fontsize=12)
    plt.ylabel('오차율 (%)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(":", "").strip()}_sample_count_vs_error.png'))
    plt.show()
    
    # 가장 많이 혼동하는 클래스 쌍 분석
    confusion_pairs = []
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                # 클래스 i가 클래스 j로 잘못 예측된 비율
                confusion_rate = cm[i, j] / np.sum(y_true == class_names[i])
                confusion_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'confusion_count': cm[i, j],
                    'confusion_rate': confusion_rate,
                    'confusion_percentage': confusion_rate * 100
                })
    
    # 혼동 비율이 높은 상위 10개 클래스 쌍
    if confusion_pairs:
        confusion_df = pd.DataFrame(confusion_pairs)
        top_confusion_pairs = confusion_df.sort_values('confusion_rate', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='true_class', y='confusion_percentage', hue='predicted_class', 
                         data=top_confusion_pairs, palette='viridis')
        
        plt.title(f'{title_prefix} 가장 많이 혼동하는 상위 10개 클래스 쌍', fontsize=14)
        plt.xlabel('실제 클래스', fontsize=12)
        plt.ylabel('혼동 비율 (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='예측 클래스', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(":", "").strip()}_top_confusion_pairs.png'))
        plt.show()
        
        print("\n가장 많이 혼동하는 클래스 쌍 (상위 10개):")
        for _, row in top_confusion_pairs.iterrows():
            print(f"실제: {row['true_class']}, 예측: {row['predicted_class']}, " +
                  f"혼동 비율: {row['confusion_percentage']:.2f}%, 개수: {row['confusion_count']}")
    
    return error_df

#------------------------------------------------------------------
# CSV 파일 로드 및 균형 잡힌 샘플링 수행
def load_and_sample_data():
    print("데이터 로드 중...")
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    
    # 클래스 이름 가져오기
    class_names = sorted(train_df['label'].unique())
    
    print(f"원본 훈련 샘플: {len(train_df)}")
    print(f"테스트 샘플: {len(test_df)}")
    print(f"고유 새 종류: {len(class_names)}개")
    
    # 균형 잡힌 샘플링 수행 (각 클래스당 MAX_SAMPLES_PER_CLASS개 샘플 사용)
    sampled_train_df = pd.DataFrame()
    for label in class_names:
        class_samples = train_df[train_df['label'] == label]
        if len(class_samples) > MAX_SAMPLES_PER_CLASS:
            class_samples = class_samples.sample(MAX_SAMPLES_PER_CLASS, random_state=42)
        sampled_train_df = pd.concat([sampled_train_df, class_samples])
    
    # 샘플링 결과 확인
    print(f"샘플링 후 훈련 샘플: {len(sampled_train_df)}")
    class_distribution = sampled_train_df['label'].value_counts()
    print("클래스 분포:")
    print(class_distribution)
    
    return sampled_train_df, test_df, class_names

#------------------------------------------------------------------
# 이미지에서 특징 추출 함수
def extract_features_from_image(img_path, is_train=True, upscale_img_path=None):
    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    try:
        # 바운딩 박스 설정
        if is_train and upscale_img_path:
            # 고해상도 이미지가 있는 경우 (훈련 데이터)
            upscale_img = cv2.imread(upscale_img_path)
            if upscale_img is None:
                raise Exception("고해상도 이미지를 읽을 수 없음")
            
            # 간소화된, 빠른 바운딩 박스 추출
            h, w = upscale_img.shape[:2]
            center_margin = 0.2
            x = int(w * center_margin)
            y = int(h * center_margin)
            width = int(w * (1 - 2 * center_margin))
            height = int(h * (1 - 2 * center_margin))
            
            upscale_bbox = (x, y, width, height)
            
            # 저해상도 이미지를 위한 바운딩 박스 크기 조정
            scale_x = img.shape[1] / upscale_img.shape[1]
            scale_y = img.shape[0] / upscale_img.shape[0]
            
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            bbox = (x_scaled, y_scaled, w_scaled, h_scaled)
        else:
            # 테스트 이미지 또는 고해상도 이미지가 없는 경우
            h, w = img.shape[:2]
            center_margin = 0.1
            x = int(w * center_margin)
            y = int(h * center_margin) 
            width = int(w * (1 - 2 * center_margin))
            height = int(h * (1 - 2 * center_margin))
            
            bbox = (x, y, width, height)
        
        # HSV 특징 추출 (마스킹 없이 빠르게 수행)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h_bins, s_bins, v_bins = 10, 10, 10
        
        h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
        
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        hsv_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        
        # HOG 특징 추출
        # 바운딩 박스로 이미지 잘라내기
        x, y, w, h = bbox
        cropped_img = img[y:y+h, x:x+w]
        
        # 크기 조정
        resized_img = cv2.resize(cropped_img, (64, 64))
        
        # 그레이스케일로 변환
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # HOG 특징 추출 - 빠른 설정으로 변경
        fd, _ = hog(
            gray, 
            orientations=8,  # 9에서 8로 감소
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm='L2-Hys'
        )
        
        # 특징 결합
        combined_features = np.concatenate([hsv_features, fd])
        return combined_features
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        return None

#------------------------------------------------------------------
# 특징 추출 (이미 추출된 특징이 있으면 로드)
def extract_or_load_features(train_df, test_df):
    # 훈련 특징 추출 또는 로드
    if os.path.exists(FEATURES_PATH['train_features']):
        print(f"기존 훈련 특징 파일 로드 중: {FEATURES_PATH['train_features']}")
        X_train = np.load(FEATURES_PATH['train_features'])
        y_train = train_df['label'].values
        
        # 기존 특징 파일의 크기가 현재 데이터셋과 다른 경우 다시 추출
        if len(X_train) != len(train_df):
            print("데이터셋 크기가 변경되어 특징을 다시 추출합니다.")
            X_train, y_train = extract_train_features(train_df)
        else:
            print(f"로드된 훈련 특징 형태: {X_train.shape}")
    else:
        X_train, y_train = extract_train_features(train_df)
    
    # 테스트 특징 추출 또는 로드
    if os.path.exists(FEATURES_PATH['test_features']):
        print(f"기존 테스트 특징 파일 로드 중: {FEATURES_PATH['test_features']}")
        X_test = np.load(FEATURES_PATH['test_features'])
        test_ids = test_df['id'].values
        
        # 기존 특징 파일의 크기가 현재 데이터셋과 다른 경우 다시 추출
        if len(X_test) != len(test_df):
            print("테스트 데이터셋 크기가 변경되어 특징을 다시 추출합니다.")
            X_test, test_ids = extract_test_features(test_df)
        else:
            print(f"로드된 테스트 특징 형태: {X_test.shape}")
    else:
        X_test, test_ids = extract_test_features(test_df)
    
    return X_train, y_train, X_test, test_ids

#------------------------------------------------------------------
# 훈련 데이터 특징 추출
def extract_train_features(train_df):
    print("훈련 이미지에서 특징 추출 중...")
    
    features_list = []
    labels = []
    successful_extractions = 0
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        # 이미지 경로 가져오기
        img_path = os.path.join(BASE_DIR, row['img_path'][2:])
        upscale_img_path = os.path.join(BASE_DIR, row['upscale_img_path'][2:])
        
        # 특징 추출
        features = extract_features_from_image(img_path, True, upscale_img_path)
        
        if features is not None:
            features_list.append(features)
            labels.append(row['label'])
            successful_extractions += 1
    
    print(f"총 {len(train_df)}개 중 {successful_extractions}개 이미지 처리 완료")
    
    # 리스트를 NumPy 배열로 변환
    if features_list:
        X_train = np.array(features_list)
        y_train = np.array(labels)
        
        print(f"추출된 특징 형태: {X_train.shape}")
        
        # 특징 저장
        np.save(FEATURES_PATH['train_features'], X_train)
        print(f"특징 저장 완료: {FEATURES_PATH['train_features']}")
        
        return X_train, y_train
    else:
        raise Exception("추출된 특징이 없습니다!")

#------------------------------------------------------------------
# 테스트 데이터 특징 추출
def extract_test_features(test_df):
    print("테스트 이미지에서 특징 추출 중...")
    
    test_features_list = []
    test_ids = []
    successful_extractions = 0
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # 이미지 경로 가져오기
        img_path = os.path.join(BASE_DIR, row['img_path'][2:])
        
        # 특징 추출
        features = extract_features_from_image(img_path, False)
        
        if features is not None:
            test_features_list.append(features)
            test_ids.append(row['id'])
            successful_extractions += 1
    
    print(f"총 {len(test_df)}개 중 {successful_extractions}개 테스트 이미지 처리 완료")
    
    # 리스트를 NumPy 배열로 변환
    if test_features_list:
        X_test = np.array(test_features_list)
        test_ids = np.array(test_ids)
        
        print(f"추출된 테스트 특징 형태: {X_test.shape}")
        
        # 특징 저장
        np.save(FEATURES_PATH['test_features'], X_test)
        print(f"테스트 특징 저장 완료: {FEATURES_PATH['test_features']}")
        
        return X_test, test_ids
    else:
        raise Exception("추출된 테스트 특징이 없습니다!")

#------------------------------------------------------------------
# 모델 훈련 및 평가 (이미 훈련된 모델이 있으면 로드)
def train_or_load_models(X_train, y_train, class_names):
    # 훈련 및 검증 세트로 분할
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"훈련 세트 크기: {X_train_subset.shape}")
    print(f"검증 세트 크기: {X_val.shape}")
    
    # 모델 정의
    models = {
        'SVM': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, C=10, gamma='scale', kernel='rbf'))
            ]),
            'path': FEATURES_PATH['svm_model']
        },
        'RandomForest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),  # n_jobs=-1로 병렬 처리
            'path': FEATURES_PATH['rf_model']
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),  # n_jobs=-1로 병렬 처리
            'path': FEATURES_PATH['knn_model']
        }
    }
    
    # 결과 저장 사전
    results = {}
    trained_models = {}  # 앙상블에 사용할 훈련된 모델 저장
    
    # 각 모델 훈련 또는 로드 및 평가
    for name, model_info in models.items():
        # 이미 훈련된 모델이 있는지 확인
        if os.path.exists(model_info['path']):
            print(f"기존 {name} 모델 로드 중: {model_info['path']}")
            model = joblib.load(model_info['path'])
        else:
            print(f"\n{name} 모델 훈련 중...")
            model = model_info['model']
            model.fit(X_train_subset, y_train_subset)
            
            # 모델 저장
            joblib.dump(model, model_info['path'])
            print(f"{name} 모델 저장 완료: {model_info['path']}")
        
        # 훈련된 모델 저장
        trained_models[name] = model
        
        # 검증 세트에서 예측
        y_pred = model.predict(X_val)
        
        # 정확도 계산
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"{name} 검증 정확도: {accuracy:.4f}")
        
        # 결과 저장
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        }
    
    # 모든 모델 훈련 후 앙상블 모델 생성
    print("\n앙상블 모델 생성 중...")
    if os.path.exists(FEATURES_PATH['ensemble_model']):
        print(f"기존 앙상블 모델 로드 중: {FEATURES_PATH['ensemble_model']}")
        ensemble_model = joblib.load(FEATURES_PATH['ensemble_model'])
    else:
        print("새로운 앙상블 모델 훈련 중...")
        # 투표 기반 앙상블 모델 생성
        estimators = [(name, model) for name, model in trained_models.items()]
        ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        ensemble_model.fit(X_train_subset, y_train_subset)
        
        # 모델 저장
        joblib.dump(ensemble_model, FEATURES_PATH['ensemble_model'])
        print(f"앙상블 모델 저장 완료: {FEATURES_PATH['ensemble_model']}")
    
    # 검증 세트에서 앙상블 모델 평가
    y_ensemble_pred = ensemble_model.predict(X_val)
    ensemble_accuracy = accuracy_score(y_val, y_ensemble_pred)
    print(f"앙상블 모델 검증 정확도: {ensemble_accuracy:.4f}")
    
    # 앙상블 모델 결과 추가
    results['Ensemble'] = {
        'model': ensemble_model,
        'accuracy': ensemble_accuracy,
        'report': classification_report(y_val, y_ensemble_pred, target_names=class_names, output_dict=True)
    }
    
    # 가장 좋은 모델 찾기
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n최고 모델: {best_model_name}, 정확도: {best_accuracy:.4f}")
    
    # 검증 세트에 대한 오차율 계산 (새로 추가)
    val_error_df = calculate_error_rate(y_val, y_ensemble_pred, class_names, OUTPUT_DIR, "검증 데이터:")
    
    return results, best_model_name, best_model, val_error_df



#------------------------------------------------------------------
# 결과 시각화 및 제출 파일 생성
def visualize_and_predict(results, best_model_name, best_model, X_test, test_ids):
    # 모델 비교 시각화
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # 막대 위에 정확도 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.4f}',
            ha='center',
            fontweight='bold'
        )
    
    plt.title('모델별 검증 정확도 비교')
    plt.xlabel('모델')
    plt.ylabel('정확도')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # 최고 모델 하이라이트
    for i, name in enumerate(model_names):
        if name == best_model_name:
            bars[i].set_color('#f39c12')
            plt.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                bars[i].get_height() + 0.05,
                '최고 모델',
                ha='center',
                fontweight='bold'
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
    plt.show()
    
    # 제출 파일이 이미 있는지 확인
    if os.path.exists(FEATURES_PATH['submission']):
        print(f"기존 제출 파일 발견: {FEATURES_PATH['submission']}")
        submission_df = pd.read_csv(FEATURES_PATH['submission'])
        print(f"제출 파일 형태: {submission_df.shape}")
        
        # 이미 있는 제출 파일 예측 분포 시각화
        pred_counts = submission_df['label'].value_counts()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(pred_counts.index, pred_counts.values, color='#3498db')
        
        plt.title('테스트 데이터 예측 분포 (기존 제출 파일)')
        plt.xlabel('새 종류')
        plt.ylabel('빈도')
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars, pred_counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'test_predictions_distribution.png'))
        plt.show()
    else:
        print("\n최고 모델로 테스트 데이터 예측 중...")
        
        # 테스트 데이터 예측
        test_predictions = best_model.predict(X_test)
        
        # 예측 결과를 데이터프레임으로 변환
        submission_df = pd.DataFrame({
            'id': test_ids,
            'label': test_predictions
        })
        
        # 결과 저장
        submission_df.to_csv(FEATURES_PATH['submission'], index=False)
        print(f"제출 파일 저장 완료: {FEATURES_PATH['submission']}")
        
        # 테스트 데이터의 예측 분포 시각화
        pred_counts = pd.Series(test_predictions).value_counts()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(pred_counts.index, pred_counts.values, color='#3498db')
        
        plt.title('테스트 데이터 예측 분포 (새로운 예측)')
        plt.xlabel('새 종류')
        plt.ylabel('빈도')
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars, pred_counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'test_predictions_distribution.png'))
        plt.show()
        
        
#-------------------------------------------------------------------------------------------------------        
def compare_actual_vs_predicted(base_dir, output_dir):
    """
    Compare the actual distribution of bird species in the training data
    with the predicted distribution in the submission file
    
    Parameters:
    base_dir (str): Base directory containing train.csv and test.csv
    output_dir (str): Directory to save the comparison visualization
    """
    print("실제 분포와 예측 분포 비교 중...")
    
    # 파일 경로 설정
    train_path = os.path.join(base_dir, "train.csv")
    submission_path = os.path.join(output_dir, "submission.csv")
    
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    
    # 실제 분포 계산
    actual_counts = train_df['label'].value_counts().sort_index()
    
    # 제출 파일이 있는지 확인
    if os.path.exists(submission_path):
        submission_df = pd.read_csv(submission_path)
        predicted_counts = submission_df['label'].value_counts().sort_index()
        
        # 모든 클래스 가져오기
        all_classes = sorted(set(actual_counts.index) | set(predicted_counts.index))
        
        # 비교를 위한 데이터프레임 생성
        comparison_df = pd.DataFrame(index=all_classes)
        comparison_df['actual'] = actual_counts
        comparison_df['predicted'] = predicted_counts
        
        # NaN 값을 0으로 채우기
        comparison_df = comparison_df.fillna(0)
        
        # 시각화 설정
        plt.figure(figsize=(14, 10))
        
        # 막대 그래프 너비
        bar_width = 0.35
        
        # 위치 설정
        r1 = np.arange(len(all_classes))
        r2 = [x + bar_width for x in r1]
        
        # 막대 그래프 그리기
        plt.bar(r1, comparison_df['actual'], color='#3498db', width=bar_width, label='실제 분포')
        plt.bar(r2, comparison_df['predicted'], color='#e74c3c', width=bar_width, label='예측 분포')
        
        # 차트 설정
        plt.xlabel('새 종류', fontsize=12)
        plt.ylabel('개수', fontsize=12)
        plt.title('실제 분포 vs 예측 분포 비교', fontsize=14)
        plt.xticks([r + bar_width/2 for r in range(len(all_classes))], all_classes, rotation=90)
        plt.legend()
        
        # 그리드 추가
        plt.grid(axis='y', alpha=0.3)
        
        # 여백 조정
        plt.tight_layout()
        
        # 저장
        plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_comparison.png'))
        plt.show()
        
        # 요약 통계 출력
        print("\n분포 비교 요약:")
        print(f"총 실제 샘플: {comparison_df['actual'].sum()}")
        print(f"총 예측 샘플: {comparison_df['predicted'].sum()}")
        print(f"클래스 개수: {len(all_classes)}")
        
        # 클래스별 차이 계산
        comparison_df['difference'] = comparison_df['predicted'] - comparison_df['actual']
        comparison_df['percentage_diff'] = (comparison_df['difference'] / comparison_df['actual'] * 100).round(2)
        
        # 가장 큰 차이가 나는 클래스 찾기
        max_diff_class = comparison_df.loc[comparison_df['difference'].abs().idxmax()]
        print(f"\n가장 큰 차이가 나는 클래스: {max_diff_class.name}")
        print(f"  실제: {max_diff_class['actual']}")
        print(f"  예측: {max_diff_class['predicted']}")
        print(f"  차이: {max_diff_class['difference']}")
        print(f"  비율 차이: {max_diff_class['percentage_diff']}%")
        
        # 추가 시각화: 비율 차이 
        plt.figure(figsize=(14, 10))
        sns.barplot(x=comparison_df.index, y=comparison_df['percentage_diff'], palette='coolwarm')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('클래스별 실제 대비 예측 비율 차이 (%)', fontsize=14)
        plt.xlabel('새 종류', fontsize=12)
        plt.ylabel('비율 차이 (%)', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'percentage_difference.png'))
        plt.show()
        
        return comparison_df
    else:
        print(f"제출 파일을 찾을 수 없습니다: {submission_path}")
        
        # 실제 분포만 시각화
        plt.figure(figsize=(14, 8))
        ax = actual_counts.plot(kind='bar', color='#3498db')
        plt.title('훈련 데이터의 실제 분포 (예측 없음)', fontsize=14)
        plt.xlabel('새 종류', fontsize=12)
        plt.ylabel('개수', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        
        # 막대 위에 숫자 표시
        for i, v in enumerate(actual_counts):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'actual_distribution.png'))
        plt.show()
        
        return None
    
    
    #---------------------------------------------------------------------------------------------------
# 훈련 데이터에 대한 평가 및 오차율 분석
def train_and_evaluate_on_same_data(X_train, y_train, best_model, class_names, output_dir):
    """
    훈련 데이터에 대해 모델을 평가하고 실제 분포와 예측 분포를 비교합니다.
    오차율 분석 기능 추가.
    
    Parameters:
    X_train (numpy.ndarray): 훈련 데이터의 특징
    y_train (numpy.ndarray): 훈련 데이터의 실제 레이블
    best_model: 훈련된 모델
    class_names (list): 클래스 이름 목록
    output_dir (str): 결과를 저장할 디렉토리
    
    Returns:
    tuple: (comparison_df, error_df) - 비교 데이터프레임과 오차율 데이터프레임
    """
    print("\n훈련 데이터로 모델 평가 중...")
    
    # 훈련 데이터에 대한 예측 수행
    y_pred = best_model.predict(X_train)
    
    # 정확도 계산
    accuracy = accuracy_score(y_train, y_pred)
    print(f"훈련 데이터 정확도: {accuracy:.4f}")
    
    # 분류 보고서 생성
    report = classification_report(y_train, y_pred, target_names=class_names)
    print("\n분류 보고서:")
    print(report)
    
    # 실제 분포와 예측 분포 계산
    actual_counts = pd.Series(y_train).value_counts().sort_index()
    predicted_counts = pd.Series(y_pred).value_counts().sort_index()
    
    # 모든 클래스 가져오기
    all_classes = sorted(set(actual_counts.index) | set(predicted_counts.index))
    
    # 비교를 위한 데이터프레임 생성
    comparison_df = pd.DataFrame(index=all_classes)
    comparison_df['actual'] = actual_counts
    comparison_df['predicted'] = predicted_counts
    
    # NaN 값을 0으로 채우기
    comparison_df = comparison_df.fillna(0)
    
    # 클래스별 차이 계산
    comparison_df['difference'] = comparison_df['predicted'] - comparison_df['actual']
    comparison_df['percentage_diff'] = (comparison_df['difference'] / comparison_df['actual'] * 100).round(2)
    
    # 시각화
    plt.figure(figsize=(14, 10))
    
    # 막대 그래프 너비
    bar_width = 0.35
    
    # 위치 설정
    r1 = np.arange(len(all_classes))
    r2 = [x + bar_width for x in r1]
    
    # 막대 그래프 그리기
    plt.bar(r1, comparison_df['actual'], color='#3498db', width=bar_width, label='실제 분포')
    plt.bar(r2, comparison_df['predicted'], color='#e74c3c', width=bar_width, label='예측 분포')
    
    # 차트 설정
    plt.xlabel('새 종류', fontsize=12)
    plt.ylabel('개수', fontsize=12)
    plt.title('훈련 데이터: 실제 분포 vs 예측 분포 비교', fontsize=14)
    plt.xticks([r + bar_width/2 for r in range(len(all_classes))], all_classes, rotation=90)
    plt.legend()
    
    # 그리드 추가
    plt.grid(axis='y', alpha=0.3)
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장
    plt.savefig(os.path.join(output_dir, 'train_actual_vs_predicted_comparison.png'))
    plt.show()
    
    # 요약 통계 출력
    print("\n분포 비교 요약:")
    print(f"총 실제 샘플: {comparison_df['actual'].sum()}")
    print(f"총 예측 샘플: {comparison_df['predicted'].sum()}")
    print(f"클래스 개수: {len(all_classes)}")
    
    # 가장 큰 차이가 나는 클래스 찾기
    max_diff_class = comparison_df.loc[comparison_df['difference'].abs().idxmax()]
    print(f"\n가장 큰 차이가 나는 클래스: {max_diff_class.name}")
    print(f"  실제: {max_diff_class['actual']}")
    print(f"  예측: {max_diff_class['predicted']}")
    print(f"  차이: {max_diff_class['difference']}")
    print(f"  비율 차이: {max_diff_class['percentage_diff']}%")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(y_train, y_pred)
    
    # 혼동 행렬을 정규화하여 비율로 표시
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 히트맵으로 시각화
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('훈련 데이터 정규화된 혼동 행렬', fontsize=14)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_confusion_matrix.png'))
    plt.show()
    
    # 백분율 차이 시각화
    plt.figure(figsize=(14, 10))
    sns.barplot(x=comparison_df.index, y=comparison_df['percentage_diff'], palette='coolwarm')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('클래스별 실제 대비 예측 비율 차이 (%)', fontsize=14)
    plt.xlabel('새 종류', fontsize=12)
    plt.ylabel('비율 차이 (%)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_percentage_difference.png'))
    plt.show()
    
    # 오차율 분석 (새로 추가)
    error_df = calculate_error_rate(y_train, y_pred, class_names, output_dir, "훈련 데이터:")
    
    return comparison_df, error_df

#------------------------------------------------------------------
# 메인 함수
def main():
    try:
        # 1. 데이터 로드 및 샘플링
        train_df, test_df, class_names = load_and_sample_data()
        
        # 2. 특징 추출 또는 로드
        X_train, y_train, X_test, test_ids = extract_or_load_features(train_df, test_df)
        
        # 3. 모델 훈련 또는 로드 및 평가 (오차율 분석 추가)
        results, best_model_name, best_model, val_error_df = train_or_load_models(X_train, y_train, class_names)
        
        # 4. 결과 시각화 및 제출 파일 생성
        visualize_and_predict(results, best_model_name, best_model, X_test, test_ids)
        
        # 5. 훈련 데이터에 대한 예측과 실제 분포 비교 (오차율 분석 포함)
        train_comparison_df, train_error_df = train_and_evaluate_on_same_data(
            X_train, y_train, best_model, class_names, OUTPUT_DIR
        )
        
        # 6. 오차율 분석 결과 요약
        print("\n==== 오차율 분석 요약 ====")
        print(f"검증 세트 전체 오차율: {(1 - results[best_model_name]['accuracy']) * 100:.2f}%")
        print(f"훈련 세트 전체 오차율: {(1 - accuracy_score(y_train, best_model.predict(X_train))) * 100:.2f}%")
        
        # 오차율이 가장 높은 클래스 출력
        print("\n오차율이 가장 높은 클래스 (훈련 데이터):")
        for i, row in train_error_df.head(5).iterrows():
            print(f"{i+1}. {row['class']}: {row['error_percentage']:.2f}%")
        
        print("\n오차율이 가장 높은 클래스 (검증 데이터):")
        for i, row in val_error_df.head(5).iterrows():
            print(f"{i+1}. {row['class']}: {row['error_percentage']:.2f}%")
        
        print("\n분석 완료!")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

# 스크립트 실행
if __name__ == "__main__":
    main()