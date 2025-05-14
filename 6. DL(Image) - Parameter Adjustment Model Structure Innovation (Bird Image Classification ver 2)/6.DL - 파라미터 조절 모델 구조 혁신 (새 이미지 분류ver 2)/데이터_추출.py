import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# CSV 파일 경로
csv_file_path = r"C:\Users\KDT-13\Desktop\KDT 7\0.Project\TP8_2_CV\data\train.csv"  # 실제 CSV 파일 경로로 변경해주세요

# CSV 파일 읽기 (헤더가 없는 경우)
df = pd.read_csv(csv_file_path, header=None, names=['img_path', 'upscale_img_path', 'label'])

# 전체 데이터 수 확인
total_samples = len(df)
print(f"전체 데이터 수: {total_samples}")

# 클래스(레이블) 별 데이터 수 확인
class_counts = df['label'].value_counts()
print("\n클래스 별 데이터 수:")
print(class_counts)

# 클래스 별 비율 계산
class_ratios = class_counts / total_samples
print("\n클래스 별 비율:")
print(class_ratios)

# 목표 샘플 수
target_samples = 5000

# 클래스 별로 비율에 맞춰 샘플링할 데이터 수 계산
target_counts = (class_ratios * target_samples).round().astype(int)

# 비율에 맞춰 계산한 후 합계가 정확히 5000이 되지 않을 수 있으므로 조정
diff = target_samples - target_counts.sum()
if diff != 0:
    # 가장 데이터가 많은 클래스에서 조정
    most_common_class = class_counts.index[0]
    target_counts[most_common_class] += diff

print("\n목표 샘플링 수:")
print(target_counts)

# 결과를 저장할 데이터프레임
sampled_df = pd.DataFrame(columns=df.columns)

# 각 클래스별로 비율에 맞게 샘플링
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

# 결과 확인
print(f"\n샘플링 후 데이터 수: {len(sampled_df)}")
print("\n샘플링 후 클래스 별 데이터 수:")
print(sampled_df['label'].value_counts())

# 결과를 CSV 파일로 저장
output_csv_path = 'sampled_dataset.csv'
sampled_df.to_csv(output_csv_path, index=False, header=False)
print(f"\n결과가 '{output_csv_path}'에 저장되었습니다.")