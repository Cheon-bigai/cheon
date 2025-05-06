import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

file = r"C:\Users\KDT-13\Desktop\KDT 7\project\TP3_public\전국_평균_기온_94_24.xlsx"
df = pd.read_excel(file)

df.columns = ['날짜', '지점', '평균기온', '최저기온', '최고기온']
df = df.drop(labels=['지점'], axis=1)
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
df['년'] = pd.to_datetime(df['날짜']).dt.year # 연 컬럼 뽑기
df['월'] = pd.to_datetime(df['날짜']).dt.month # 월 컬럼 만들기

df_new = df[df['날짜'].dt.month > 6]

df_new['온도_충족'] = ((df_new['평균기온'] >= 5) & (df_new['평균기온'] < 20)).astype(int)

monthly_counts = df_new.groupby(['년', '월'])['온도_충족'].sum().reset_index()

threshold = 15
monthly_counts['가을_여부'] = monthly_counts['온도_충족'] >= threshold

df_new = df_new.merge(monthly_counts[['년', '월', '가을_여부']], on=['년', '월'], how='left')

df_new = df_new[df_new['가을_여부']]
# 5~19도이며 6,7,8,12월 포함


df_old=df[df['날짜'].dt.month.isin([9,10,11])] # 1-2) 가을 기준 해당 달에 따른 전처리 = df_old
#df_old=df_old[(df_old['평균기온']>=5) &  (df_old['평균기온']<19) ] #1-3) 봄 기준 해당 달 + 온도 해당 = df_old


'''
df['년']=pd.to_datetime(df['날짜']).dt.year
df['월']=pd.to_datetime(df['날짜']).dt.month
#결과 = df.groupby('년')['월'].agg(['min', 'max'])
결과 = df.groupby('년')['월'].nuniqu()
'''

#### df_new 전처리


df_new_result = df_new[df_new['온도_충족'] == 1].groupby('년')['월'].value_counts().to_frame() # 온도충족이 1에 해당하는 것들만
df_new_result = df_new_result.sort_index()


### df_old 전처리


df_old_result = df_old.groupby('년')['월'].value_counts().to_frame() # 년에 해당하는 월의 일들을 카운트하고 데이트프레임으로 만듬
df_old_result = df_old_result.sort_index()  # df_old_result는 7,8,9월 중에 5~19도에 해당하는 두개의 조건을 적용한 개수


#### 연도별 월별 카운트 합
df_old_Cyear = df_old_result.groupby(level=0)['count'].sum().to_frame() # 연도별에 따른 월별 카운트를 합침 즉 1994년도에 대한 총 일수가 나옴
df_new_Cyear = df_new_result.groupby(level=0)['count'].sum().to_frame()



## 1번 그래프 : 그래프 그리기 (old와 new 조건 해당 카운트 개수)
plt.figure(figsize=(8, 5))

# 전통적 계절 그래프
plt.plot(df_old_Cyear, marker='o', linestyle='--', color='blue', label='전통적 계절 (일)')

# 기온기반 계절 그래프
plt.plot(df_new_Cyear, marker='s', linestyle='-', color='red', label='기온기반 계절 (일)')

# 그래프 제목 및 축 레이블 설정
plt.xlabel("연도", fontsize=12)
plt.ylabel("일수 차이", fontsize=12)
plt.title("전통적 계절 - 기온기반 계절 일수 차이", fontsize=14, fontweight="bold")

# x축 눈금 설정
plt.xticks(range(1994, 2025, 3), fontsize=10)
plt.yticks(fontsize=10)

# 범례 설정 (위치 조정 및 스타일 추가)
plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(8, 5))
plt.plot(df_old_Cyear, marker='o', label='전통적 계절',linestyle='--')
plt.plot(df_new_Cyear, marker='s', label='기온기반 계절',linestyle='-')
plt.xticks(range(1994, 2025, 3))
plt.title(" 전통적 계절  / 기온기반 계절 일수 차이 ")
plt.legend(loc="upper right", bbox_to_anchor=(0.99, 0.93))
plt.tight_layout()
plt.show()
'''

## 2번 그래프 그래프 그리기 (old - new의 추세)


# 데이터 비교 (전통적 계절 vs 기온기반 계절)
df_compare = df_old_Cyear.join(df_new_Cyear, lsuffix='_old', rsuffix='_new', how='inner')

# 차이 계산
df_compare['차이'] = df_compare['count_old'] - df_compare['count_new']

# 연도 배열
years = df_compare.index.astype(int)

# 추세선 (1차 다항 회귀)
z = np.polyfit(years, df_compare['차이'], 1)
p = np.poly1d(z)
slope = z[0]  # 기울기 값 저장

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 차이 그래프
sns.lineplot(x=years, y=df_compare['차이'], marker="o", label="전통적 계절 - 기온기반 계절 일 수 차이",  linewidth=2)

# 추세선
plt.plot(years, p(years), linestyle="--", color="red", linewidth=2, label="추세선")

# 기울기 수치를 그래프 오른쪽 상단에 표시
x_max = years.max() + 1  # x축 최댓값
y_max = p(x_max)  # 해당 x값에서의 y 예측값
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='red', fontweight="bold", verticalalignment='bottom')

# 제목 및 축 라벨
plt.xlabel("연도", fontsize=12)
plt.ylabel("가을 일수 차이 (전통적 계절 - 기온기반 계절)", fontsize=12)
plt.title("전통적 계절과 기온기반 계절의 일 수 차이 추세", fontsize=14, fontweight="bold")

# x축 눈금 회전 및 크기 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# 범례 설정 (위치 조정 및 스타일 추가)
plt.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

'''
df_compare = df_old_Cyear.join(df_new_Cyear, lsuffix='_old', rsuffix='_new', how='inner')
# 차이 계산
df_compare['차이'] = df_compare['count_old'] - df_compare['count_new']
# 연도 배열
years = df_compare.index.astype(int)
# 추세선 (1차 다항 회귀)
z = np.polyfit(years, df_compare['차이'], 1)
p = np.poly1d(z)
# 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(x=years, y=df_compare['차이'], marker="o", label="전통적 계절 - 기온기반 계절 일 수 차이", color="b")
plt.plot(years, p(years), linestyle="--", color="r", label="추세선")
plt.xticks(rotation=45)

slope = z[0]

# 기울기 수치를 그래프 오른쪽 상단에 표시
x_max = years.max() + 2  # x축 최댓값
y_max = p(x_max)  # 해당 x값에서의 y 예측값
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='red', verticalalignment='bottom')

plt.xlabel("연도")
plt.ylabel("가을 일수 차이 (df_old - df_new)")
plt.title("전통적 계절의 일 수 - 기온기반 계절의 일 수 차이 추세 ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

## 2-1 연도별 기온기반 계절 (가을) 최저기온 추세선

df_new['연도'] = df_new['날짜'].dt.year
df_min_temp_by_year = df_new.groupby('연도')['최저기온'].min()


plt.figure(figsize=(12, 6))

# 연도별 최저기온 플롯
plt.plot(df_min_temp_by_year.index, df_min_temp_by_year.values, marker="o", linestyle="-", label="연도별 최저기온", color="red")

# 추세선 추가
z = np.polyfit(df_min_temp_by_year.index, df_min_temp_by_year.values, 1)
p = np.poly1d(z)
plt.plot(df_min_temp_by_year.index, p(df_min_temp_by_year.index), color='blue', linestyle='--', label="추세선")

# 그래프 꾸미기
plt.xlabel("연도", fontsize=12)
plt.ylabel("최저기온 (°C)", fontsize=12)
plt.title("기온기반 계절 (가을) 최저기온 추세", fontsize=14, fontweight="bold")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)  # 범례 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
'''
plt.figure(figsize=(12,6))
plt.plot(df_min_temp_by_year.index, df_min_temp_by_year.values, marker="o")
z = np.polyfit(df_min_temp_by_year.index, df_min_temp_by_year.values, 1)
p = np.poly1d(z)
plt.plot(df_min_temp_by_year.index, p(df_min_temp_by_year.index), color='red', linestyle='--')
plt.title("기온기반 계절 (가을) 최저기온 추세")
plt.grid()
plt.tight_layout()
plt.show()
'''



'''chpater2'''



## 3번 그래프: 연 계절 별 일교차


# 가을 평균 일교차 계산
df_dif_month = df.copy()
df_dif_month['일교차'] = df['최고기온'] - df['최저기온']

df_dif_year_mean = df_dif_month[df_dif_month['월'].isin([9, 10, 11])].groupby('년')['일교차'].mean().to_frame()
df_dif_year_mean = df_dif_year_mean.rename(columns={'일교차': '가을'})

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 가을 일교차 그래프
plt.plot(df_dif_year_mean.index, df_dif_year_mean['가을'], marker='o', linestyle='-', color='blue', linewidth=2, label='가을')

# 추세선 추가
z = np.polyfit(df_dif_year_mean.index, df_dif_year_mean['가을'], 1)
p = np.poly1d(z)
plt.plot(df_dif_year_mean.index, p(df_dif_year_mean.index), color='red', linestyle='--', linewidth=2, label='추세선')

# 기울기 수치 표시
slope = z[0]
x_max = df_dif_year_mean.index.max() + 1
y_max = p(x_max)
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='red', fontweight="bold", verticalalignment='bottom')

# 제목 및 축 라벨
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 일교차 (°C)', fontsize=12)
plt.title('전통적 계절(가을)의 평균 일교차 변화', fontsize=14, fontweight="bold")

# x축 눈금 조정
plt.xticks(range(1994, 2025, 3), fontsize=10)
plt.yticks(fontsize=10)

# 범례 스타일 조정
plt.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

'''
df_dif_month = df.copy() 
df_dif_month['일교차']=df['최고기온']-df['최저기온']


df_dif_year_mean = df_dif_month[df_dif_month['월'].isin([9, 10, 11])].groupby('년')['일교차'].mean().to_frame() # 년도별 월 일교차 평균 만들기
#df_dif_year_mean['여름'] = df_dif_month[df_dif_month['월'].isin([6, 7, 8])].groupby('년')['일교차'].mean() # 계절별(여름) 일교차 평균 구함
#df_dif_year_mean['겨울'] = df_dif_month[df_dif_month['월'].isin([12, 1, 2])].groupby('년')['일교차'].mean() #  계절별(겨울) 일교차 평균 구함
df_dif_year_mean = df_dif_year_mean.rename(columns ={'일교차':'가을'}) # 컬림 이름 재설정

plt.figure(figsize=(8, 5))
plt.plot(df_dif_year_mean.index, df_dif_year_mean['가을'], marker='o', label='가을', linestyle='-', color='blue')
#plt.plot(df_dif_year_mean.index, df_dif_year_mean['여름'], marker='s', label='여름', linestyle='--', color='blue')
#plt.plot(df_dif_year_mean.index, df_dif_year_mean['겨울'], marker='^', label='겨울', linestyle='-.', color='green')

# 추세선 그리기
z = np.polyfit(df_dif_year_mean.index, df_dif_year_mean['가을'], 1)
p = np.poly1d(z)
plt.plot(df_dif_year_mean.index, p(df_dif_year_mean.index), color='red', linestyle='--')

slope = z[0]

# 기울기 수치를 그래프 오른쪽 상단에 표시
x_max = df_dif_year_mean.index.max() + 2  # x축 최댓값
y_max = p(x_max)  # 해당 x값에서의 y 예측값
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='red', verticalalignment='bottom')

plt.xlabel('연도')
plt.ylabel('평균 일교차')
plt.title('전통적 계절에 따른 가을의 평균 일교차')
plt.xticks(range(1994, 2025, 3))
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
'''

## 4번 그래프 : 조건의 일교차


# 기온기반 계절의 가을 평균 일교차
df_dif = df_new.copy()
df_dif['일교차'] = df['최고기온'] - df['최저기온']
df__select_option_mean = df_dif.groupby('년')['일교차'].mean()

# 그래프 설정
plt.figure(figsize=(10, 6))

# 가을 평균 일교차 그래프
plt.plot(df__select_option_mean.index, df__select_option_mean.values, marker='o', linestyle='-', 
         color='red', linewidth=2, markersize=6, label='가을')

# 추세선 추가
z = np.polyfit(df__select_option_mean.index, df__select_option_mean.values, 1)
p = np.poly1d(z)
plt.plot(df__select_option_mean.index, p(df__select_option_mean.index), 
         color='blue', linestyle='--', linewidth=2, label='추세선')

# 기울기 수치 표시
slope = z[0]
x_max = df__select_option_mean.index.max() + 1
y_max = p(x_max)
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='blue', fontweight="bold", verticalalignment='bottom')

# 제목 및 축 라벨
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 일교차 (°C)', fontsize=12)
plt.title('기온기반 계절(가을)에 따른 평균 일교차 변화', fontsize=14, fontweight="bold")

# x축 눈금 조정
plt.xticks(range(1994, 2025, 3), fontsize=10)
plt.yticks(fontsize=10)

# 범례 스타일 조정
plt.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()



'''
df_dif = df_new.copy()
df_dif['일교차'] = df['최고기온']-df['최저기온']
df__select_option_mean = df_dif.groupby('년')['일교차'].mean()

plt.figure(figsize=(8, 5))
plt.xlabel('연도')
plt.ylabel('평균 일교차')
plt.title('기온기반 계절에 따른 가을 평균 일교차')
plt.plot(df__select_option_mean.index, df__select_option_mean.values, marker='o', label='가을', linestyle='-',color='red')


z = np.polyfit(df__select_option_mean.index, df__select_option_mean.values, 1)
p = np.poly1d(z)
plt.plot(df__select_option_mean.index, p(df__select_option_mean.index), color='blue', linestyle='--')


slope = z[0]

# 기울기 수치를 그래프 오른쪽 상단에 표시
x_max = df__select_option_mean.index.max() + 2  # x축 최댓값
y_max = p(x_max)  # 해당 x값에서의 y 예측값
plt.text(x_max, y_max, f'기울기: {slope:.4f}', fontsize=12, color='blue', verticalalignment='bottom')


plt.xticks(range(1994, 2025, 3))
plt.tight_layout()
plt.legend()
plt.grid(True)

plt.show()
'''


### 5번 그래프 : 계절별 일교차 (여름, 겨울, 가을의 평균 일교차 변화 추세)

df_dif_year_mean1 = df_dif_month[df_dif_month['월'].isin([6, 7, 8])].groupby('년')['일교차'].mean().to_frame()
df_dif_year_mean1['겨울'] = df_dif_month[df_dif_month['월'].isin([12, 1, 2])].groupby('년')['일교차'].mean()
df_dif_year_mean1['가을'] = df_dif_month[df_dif_month['월'].isin([9, 10, 11])].groupby('년')['일교차'].mean()
df_dif_year_mean1 = df_dif_year_mean1.rename(columns={'일교차': '여름'})

# 그래프 설정
plt.figure(figsize=(10, 6))

# 여름, 겨울, 가을 일교차 그래프
plt.plot(df_dif_year_mean1.index, df_dif_year_mean1['가을'], marker='o', linestyle='-', 
         color='purple', linewidth=2, markersize=6, label='가을')
plt.plot(df_dif_year_mean1.index, df_dif_year_mean1['겨울'], marker='^', linestyle='-', 
         color='green', linewidth=2, markersize=6, label='겨울')
plt.plot(df_dif_year_mean1.index, df_dif_year_mean1['여름'], marker='s', linestyle='-', 
         color='blue', linewidth=2, markersize=6, label='여름')



# 가을의 추세선
z_fall = np.polyfit(df_dif_year_mean1.index, df_dif_year_mean1['가을'], 1)
p_fall = np.poly1d(z_fall)
plt.plot(df_dif_year_mean1.index, p_fall(df_dif_year_mean1.index), 
         color='purple', linestyle='--', linewidth=2, label='가을 추세선')


# 겨울의 추세선
z_winter = np.polyfit(df_dif_year_mean1.index, df_dif_year_mean1['겨울'], 1)
p_winter = np.poly1d(z_winter)
plt.plot(df_dif_year_mean1.index, p_winter(df_dif_year_mean1.index), 
         color='orange', linestyle='--', linewidth=2, label='겨울 추세선')

# 여름의 추세선
z_summer = np.polyfit(df_dif_year_mean1.index, df_dif_year_mean1['여름'], 1)
p_summer = np.poly1d(z_summer)
plt.plot(df_dif_year_mean1.index, p_summer(df_dif_year_mean1.index), 
         color='red', linestyle='--', linewidth=2, label='여름 추세선')



# 기울기 수치 표시
slope_summer = z_summer[0]
slope_winter = z_winter[0]
slope_fall = z_fall[0]

x_max = df_dif_year_mean1.index.max() + 1
y_max_summer = p_summer(x_max)
y_max_winter = p_winter(x_max)
y_max_fall = p_fall(x_max)

plt.text(x_max, y_max_summer, f'여름 기울기: {slope_summer:.4f}', fontsize=12, color='red', fontweight="bold", verticalalignment='bottom')
plt.text(x_max, y_max_winter, f'겨울 기울기: {slope_winter:.4f}', fontsize=12, color='orange', fontweight="bold", verticalalignment='bottom')
plt.text(x_max, y_max_fall + 0.3, f'가을 기울기: {slope_fall:.4f}', fontsize=12, color='purple', fontweight="bold", verticalalignment='bottom')

# 제목 및 축 라벨
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 일교차 (°C)', fontsize=12)
plt.title('연도에 따른 전통적 계절별 평균 일교차 변화', fontsize=14, fontweight="bold")

# x축 눈금 조정
plt.xticks(range(1994, 2025, 3), fontsize=10)
plt.yticks(fontsize=10)

# 범례 스타일 조정
plt.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()




'''
df_dif_year_mean1 = df_dif_month[df_dif_month['월'].isin([6, 7, 8])].groupby('년')['일교차'].mean().to_frame() # 계절별(여름) 일교차 평균 구함
df_dif_year_mean1['겨울'] = df_dif_month[df_dif_month['월'].isin([12, 1, 2])].groupby('년')['일교차'].mean() 
df_dif_year_mean1 = df_dif_year_mean1.rename(columns ={'일교차':'여름'})

plt.figure(figsize=(8, 5))
plt.plot(df_dif_year_mean1.index, df_dif_year_mean1['겨울'], marker='^', label='겨울',  color='green')
plt.plot(df_dif_year_mean1.index, df_dif_year_mean1['여름'], marker='s', label='여름',  color='blue')


z = np.polyfit(df_dif_year_mean1.index, df_dif_year_mean1['여름'], 1)
p = np.poly1d(z)
plt.plot(df_dif_year_mean1.index, p(df_dif_year_mean1.index), color='red', linestyle='--')

z = np.polyfit(df_dif_year_mean1.index, df_dif_year_mean1['겨울'], 1)
p = np.poly1d(z)
plt.plot(df_dif_year_mean1.index, p(df_dif_year_mean1.index), color='orange', linestyle='--')


plt.xlabel('연도')
plt.ylabel('평균 일교차')
plt.title('연도에 따른 전통적 계절별 평균 일교차 추세')
plt.xticks(range(1994, 2025, 3))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''


'''  chapter 3  '''


############# +1 기존 뺀 일수의 그래프에 대한 결정계수


# 기존 "전통적 계절 - 기온기반 계절 일 수 차이"에 대한 1차 회귀선
z_original = np.polyfit(years, df_compare['차이'], 1)
p_original = np.poly1d(z_original)

# 기울기 및 결정계수(R²) 계산
slope_original = z_original[0]
r2_original = r2_score(df_compare['차이'], p_original(years))

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 기존 데이터의 차이 그래프
sns.lineplot(x=years, y=df_compare['차이'], marker="o", label="전통적 계절 - 기온기반 계절 일 수 차이", linewidth=2)

# 기존 데이터의 추세선
plt.plot(years, p_original(years), linestyle="--", color="red", linewidth=2, label="추세선")

# 기울기 및 결정계수(R²) 값 표시 (색상 다르게 적용)
x_max = years.max() + 1  # x축 최댓값
y_max = p_original(x_max)  # 해당 x값에서의 y 예측값

plt.text(x_max, y_max, f'기울기: {slope_original:.4f}', 
         fontsize=12, color='purple', fontweight="bold", verticalalignment='bottom')

plt.text(x_max, y_max - (y_max * 0.05), f'R²: {r2_original:.4f}', 
         fontsize=12, color='black', fontweight="bold", verticalalignment='top')

# 제목 및 축 라벨
plt.xlabel("연도", fontsize=12)
plt.ylabel("가을 일수 차이 (전통적 계절 - 기온기반 계절)", fontsize=12)
plt.title("전통적 계절과 기온기반 계절의 일 수 차이 및 결정계수", fontsize=14, fontweight="bold")

# x축 눈금 회전 및 크기 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# 범례 설정
plt.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# 기울기 출력
print(f"기울기: {slope_original:.4f}")


############# +2 이동평균 5년치에 대한 수정계수값


from sklearn.metrics import r2_score

# 5년 이동평균 계산
df_compare['5년 이동평균'] = df_compare['차이'].rolling(window=5, min_periods=1).mean()

# 연도 배열
years = df_compare.index.astype(int)

# 5년 이동평균에 대한 추세선 계산
z_moving_avg = np.polyfit(years, df_compare['5년 이동평균'], 1)
p_moving_avg = np.poly1d(z_moving_avg)

# 수정계수(R^2) 계산
r2 = r2_score(df_compare['5년 이동평균'], p_moving_avg(years))

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 5년 이동평균 그래프
sns.lineplot(x=years, y=df_compare['5년 이동평균'], marker="s", linestyle="-", color="blue", linewidth=2, label="5년 이동평균")

# 5년 이동평균에 대한 추세선
plt.plot(years, p_moving_avg(years), linestyle="--", color="red", linewidth=2, label="5년 이동평균 추세선")

# 기울기 및 R^2 값 표시
slope_moving_avg = z_moving_avg[0]
x_max = years.max() + 1  # x축 최댓값
y_max_moving_avg = p_moving_avg(x_max)  # 해당 x값에서의 y 예측값

plt.text(x_max, y_max_moving_avg, f'기울기: {slope_moving_avg:.4f}', fontsize=12, color='red', fontweight="bold", verticalalignment='bottom')
plt.text(x_max, y_max_moving_avg - 2, f'R²: {r2:.4f}', fontsize=12, color='black', fontweight="bold", verticalalignment='top')

# 제목 및 축 라벨
plt.xlabel("연도", fontsize=12)
plt.ylabel("가을 일수 차이 (전통적 계절 - 기온기반 계절)", fontsize=12)
plt.title("전통적 계절과 기온기반 계절(5년 이동평균)의 일 수 차이 및 결정계수", fontsize=14, fontweight="bold")

# x축 눈금 회전 및 크기 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# 범례 설정
plt.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


'''
from sklearn.metrics import r2_score

# 기존 "전통적 계절 - 기온기반 계절 일 수 차이"에 대한 1차 회귀선
z_original = np.polyfit(years, df_compare['차이'], 1)
p_original = np.poly1d(z_original)

# 기존 데이터의 결정계수(R^2) 계산
r2_original = r2_score(df_compare['차이'], p_original(years))

# 그래프 그리기
plt.figure(figsize=(12, 6))

# 기존 데이터의 차이 그래프
sns.lineplot(x=years, y=df_compare['차이'], marker="o", label="전통적 계절 - 기온기반 계절 일 수 차이", linewidth=2)

# 기존 데이터의 추세선
plt.plot(years, p_original(years), linestyle="--", color="red", linewidth=2, label="추세선")

# 결정계수(R²) 값 표시
x_max = years.max() + 1  # x축 최댓값
y_max = p_original(x_max)  # 해당 x값에서의 y 예측값
plt.text(x_max, y_max, f'R²: {r2_original:.4f}', fontsize=12, color='red', fontweight="bold", verticalalignment='bottom')

# 제목 및 축 라벨
plt.xlabel("연도", fontsize=12)
plt.ylabel("가을 일수 차이 (전통적 계절 - 기온기반 계절)", fontsize=12)
plt.title("전통적 계절과 기온기반 계절의 일 수 차이 및 결정계수", fontsize=14, fontweight="bold")



# x축 눈금 회전 및 크기 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# 범례 설정
plt.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)

# 격자 추가
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
'''

