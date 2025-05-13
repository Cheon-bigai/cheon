import pandas as pd
import numpy as np
import os
import shutil

# 1. CSV 파일 경로
csv_file_path = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP9\데이터\filtered_dataset.csv'  # 실제 CSV 파일 경로로 변경해주세요

# 2. 이미지 폴더 경로
original_img_dir = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP8_2_CV\data\test'  # 원본 이미지 폴더
upscale_img_dir = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP8_2_CV\data\upscale_train'  # 업스케일 이미지 폴더

# 3. 결과물을 저장할 폴더 경로
output_train_dir = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP9\image'
output_upscale_dir = r'C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP9\upimage'
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_upscale_dir, exist_ok=True)

# 4. CSV 파일 읽기 (헤더가 없는 경우)
df = pd.read_csv(csv_file_path, header=None, names=['img_path', 'upscale_img_path', 'label'])

# 5. 전체 데이터 수와 클래스별 비율 확인
total_samples = len(df)
print(f"전체 데이터 수: {total_samples}")

class_counts = df['label'].value_counts()
print("\n클래스 별 데이터 수:")
print(class_counts)

class_ratios = class_counts / total_samples
print("\n클래스 별 비율:")
print(class_ratios)

# 6. 목표 샘플 수
target_samples = 840

# 7. 클래스 별로 비율에 맞춰 샘플링할 데이터 수 계산
target_counts = (class_ratios * target_samples).round().astype(int)

# 비율에 맞춰 계산한 후 합계가 정확히 5000이 되지 않을 수 있으므로 조정
diff = target_samples - target_counts.sum()
if diff != 0:
    # 가장 데이터가 많은 클래스에서 조정
    most_common_class = class_counts.index[0]
    target_counts[most_common_class] += diff

print("\n목표 샘플링 수:")
print(target_counts)

# 8. 각 클래스별로 비율에 맞게 샘플링
sampled_df = pd.DataFrame(columns=df.columns)

for class_name, count in target_counts.items():
    class_data = df[df['label'] == class_name]
    
    # 만약 특정 클래스의 데이터 수가 목표 수보다 적을 경우, 모두 포함
    if len(class_data) <= count:
        sampled_class_data = class_data
        print(f"클래스 '{class_name}'의 데이터는 {len(class_data)}개로, 목표 수 {count}보다 적어 모두 포함합니다.")
    else:
        # 무작위 샘플링
        sampled_class_data = class_data.sample(n=count, random_state=42)
        print(f"클래스 '{class_name}'에서 {count}개 샘플링 (원래 {len(class_data)}개)")
    
    sampled_df = pd.concat([sampled_df, sampled_class_data])

# 9. 결과 확인
print(f"\n샘플링 후 데이터 수: {len(sampled_df)}")
print("\n샘플링 후 클래스 별 데이터 수:")
print(sampled_df['label'].value_counts())

# 10. 파일명만 추출하기
def extract_filename(path):
    return os.path.basename(path)

sampled_df['original_filename'] = sampled_df['img_path'].apply(extract_filename)
sampled_df['upscale_filename'] = sampled_df['upscale_img_path'].apply(extract_filename)

# 11. 샘플링된 파일명 목록 생성
sampled_original_files = list(sampled_df['original_filename'])
sampled_upscale_files = list(sampled_df['upscale_filename'])

# 12. 이미지 파일 복사
copied_original_count = 0
copied_upscale_count = 0

# 원본 이미지 복사
for filename in sampled_original_files:
    src_path = os.path.join(original_img_dir, filename)
    dst_path = os.path.join(output_train_dir, filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_original_count += 1
    else:
        print(f"원본 파일을 찾을 수 없음: {src_path}")

# 업스케일 이미지 복사
for filename in sampled_upscale_files:
    src_path = os.path.join(upscale_img_dir, filename)
    dst_path = os.path.join(output_upscale_dir, filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_upscale_count += 1
    else:
        print(f"업스케일 파일을 찾을 수 없음: {src_path}")

print(f"\n복사된 원본 이미지 파일 수: {copied_original_count}/{len(sampled_original_files)}")
print(f"복사된 업스케일 이미지 파일 수: {copied_upscale_count}/{len(sampled_upscale_files)}")

# 13. 새로운 CSV 파일 생성
output_csv_path = 'sampled_dataset.csv'
sampled_df[['img_path', 'upscale_img_path', 'label']].to_csv(output_csv_path, index=False, header=False)
print(f"\n샘플링된 데이터셋이 '{output_csv_path}'에 저장되었습니다.")