import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# 데이터셋 경로

'''데이터전처리'''
dataset_path = r"C:\Users\KDT-13\.cache\kagglehub\datasets\mahmoudelhemaly\students-grading-dataset\versions\2"

# CSV 파일 읽기
csv_file = os.path.join(dataset_path, "Students_Grading_Dataset.csv")
df = pd.read_csv(csv_file)

df = df.drop(df.columns[[1,2,3]], axis=1)

df.columns=['ID','성별','나이','전공','참석률',
            '중간고사','기말고사','과제','퀴즈','참여점수',
            '프로젝트','종합점수','등급','1주 공부시간','과외활동','인터넷가능여부',
                '부모교육수준','가족소득수준','스트레스레벨','수면시간']


df['부모교육수준'] = df['부모교육수준'].fillna('No')
df['참석률'] = df['참석률'].fillna(df['참석률'].mean())
df['과제'] = df['과제'].fillna(df['과제'].mean())


'''부모교육수준에 따른 가족 소득수준 알아보기'''


parent_edu_and_family_income = df.groupby(['부모교육수준','가족소득수준']).size().unstack()

order = ["No", "High School", "Bachelor's", "Master's", "PhD"]
proportions = parent_edu_and_family_income.div(parent_edu_and_family_income.sum(axis=1), axis=0)
proportions = proportions.loc[order]

fig, axes = plt.subplots(1, len(proportions), figsize=(15, 6))
colors = ['blue', 'orange', 'green']
for i in range(len(proportions)):
    proportions.iloc[i].plot(kind='pie',ax=axes[i],
                             autopct='%1.1f%%',
                             colors=colors,startangle=90)
    axes[i].set_ylabel('')
    axes[i].set_title(proportions.index[i])
plt.suptitle('부모교육수준별 가족소득수준 비율')
plt.tight_layout()
plt.show()


'''가족소득에 따른 스트레스 레벨과 수면시간'''


income_colors = {"Low": "gold", "Medium": "forestgreen", "High": "royalblue"}
order = ["Low", "Medium", "High"]

# 그래프 생성
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 가족소득수준별 스트레스레벨 분포
sns.boxplot(x="가족소득수준", y="스트레스레벨", data=df, ax=axes[0], order=order, palette=income_colors)
axes[0].set_title("가족소득수준별 스트레스 레벨 분포", fontsize=14)
axes[0].set_xlabel("가족소득수준", fontsize=12)
axes[0].set_ylabel("스트레스레벨", fontsize=12)

# 가족소득수준별 수면시간 분포
sns.boxplot(x="가족소득수준", y="수면시간", data=df, ax=axes[1], order=order, palette=income_colors)
axes[1].set_title("가족소득수준별 수면시간 분포", fontsize=14)
axes[1].set_xlabel("가족소득수준", fontsize=12)
axes[1].set_ylabel("수면시간", fontsize=12)

# 평균선 및 평균값 표시
for i, var in enumerate(["스트레스레벨", "수면시간"]):
    means = df.groupby("가족소득수준")[var].mean()
    for j, level in enumerate(order):
        mean_value = means[level]

        axes[i].text(j, mean_value + 0.3, f"{mean_value:.2f}", 
                     horizontalalignment='center', 
                     color='black', 
                     fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


'''인터넷가능여부에 따른 참석률과 1주 공부시간 수면시간'''

# 색상 설정
internet_colors = {"Yes": "royalblue", "No": "tomato"}

# 인터넷 가능 여부별 박스플롯
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col in zip(axes, ["참석률", "1주 공부시간", "수면시간"]):
    sns.boxplot(x="인터넷가능여부", y=col, data=df, ax=ax, palette=internet_colors)

    # 각 그룹별 평균 계산
    means = df.groupby("인터넷가능여부")[col].mean()

    # 평균값 그래프에 추가
    for i, (category, mean_value) in enumerate(means.items()):
        ax.annotate(f'{mean_value:.1f}', (i, mean_value), ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

    ax.set_title(f"인터넷가능 여부별 {col}", fontsize=14)
    ax.set_xlabel("인터넷가능여부", fontsize=12)
    ax.set_ylabel(col, fontsize=12)

plt.tight_layout()
plt.show()



'''가족 소득 수준에 따른 과외활동'''


# 데이터 비율 계산
cross_tab = pd.crosstab(df['가족소득수준'], df['과외활동'])
proportions = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

# 색상 설정
tutor_colors = {"Yes": "dodgerblue", "No": "tomato"}

# 파이 차트 그리기
fig, axes = plt.subplots(1, len(proportions), figsize=(12, 5))

for i, income_level in enumerate(proportions.index):  
    axes[i].pie(
        proportions.loc[income_level], 
        labels=proportions.columns, 
        autopct='%1.1f%%', 
        colors=[tutor_colors[label] for label in proportions.columns],
        startangle=90
    )
    axes[i].set_title(f"{income_level} 소득층", fontsize=12)

plt.suptitle("가족 소득 수준별 과외활동 비율", fontsize=14)
plt.tight_layout()
plt.show()


'''부모교육수준에 따른 종합점수 분석'''


# 평균 비교
df.groupby("부모교육수준")["종합점수"].mean()

# ANOVA 검정
anova_result = stats.f_oneway(
    df[df["부모교육수준"] == "No"]["종합점수"],
    df[df["부모교육수준"] == "High School"]["종합점수"],
    df[df["부모교육수준"] == "Bachelor's"]["종합점수"],
    df[df["부모교육수준"] == "Master's"]["종합점수"],
    df[df["부모교육수준"] == "PhD"]["종합점수"]
)
print("ANOVA p-value:", anova_result.pvalue)

# ANOVA p-value: 0.8639853431825331


'''가족소득수준에 따른 종합점수분석'''


# 평균 비교
df.groupby("가족소득수준")["종합점수"].mean()

# ANOVA 검정
anova_result = stats.f_oneway(
    df[df["가족소득수준"] == "Low"]["종합점수"],
    df[df["가족소득수준"] == "Medium"]["종합점수"],
    df[df["가족소득수준"] == "High"]["종합점수"]
)
print("ANOVA p-value:", anova_result.pvalue)

# ANOVA p-value: 0.11634387252012994


'''여러요소에 따른 종합점수'''


# 2. 범주형 변수를 숫자로 변환
df["부모교육수준"] = df["부모교육수준"].astype("category")
df["가족소득수준"] = df["가족소득수준"].astype("category")
df["인터넷가능여부"] = df["인터넷가능여부"].astype("category")

# 3. 종합점수 숫자 변환
df["종합점수"] = pd.to_numeric(df["종합점수"], errors="coerce")

# 4. 원-핫 인코딩 (더미 변수 생성)
df_encoded = pd.get_dummies(df, columns=["부모교육수준", "가족소득수준", "인터넷가능여부"], drop_first=True)

# 독립변수: 부모교육수준, 가족소득수준, 인터넷가능여부 관련 더미 변수만 선택
X = df_encoded[[
    "부모교육수준_High School", 
    "부모교육수준_Master's", 
    "부모교육수준_No", 
    "부모교육수준_PhD", 
    "가족소득수준_Low", 
    "가족소득수준_Medium", 
    "인터넷가능여부_Yes"
]]

# 종속변수: 종합점수
y = df_encoded["종합점수"]

# 상수항 추가
X = sm.add_constant(X)
X = X.astype(int)


# 회귀모델 적합
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())

'''
1 R-squared (설명력) : 0.001 => 설명력 없음 
2 P>|t| (유의확률) : 0.05보다 작을 때 유의미
                  : 가족소득수준 = 0.037 (유의미)
                  
3 Coefficiet (종속변수 증가) :
4 F-statistic = (모델 전체 유의성) : 10 이상이면 높은 편
                                        0.9364 => 전체적으로 유의미하지 않음
  Prob =(F-statistic 에 대한 유의확률 (= p-value)) = 0.47
'''


''' OLS회귀 시각화'''
predictions = model.predict(X)
conf_int = model.get_prediction(X).conf_int()  # 신뢰구간

# 2. 그래프 그리기
plt.figure(figsize=(10, 6))

# 실제 종합점수 vs 예측값
plt.scatter(y, predictions, alpha=0.5, label="실제 vs 예측")

# 신뢰구간 그리기
plt.fill_between(y, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label="95% 신뢰구간")

# 대각선 (완벽한 예측일 경우)
x_vals = np.linspace(min(y), max(y), 100)
plt.plot(x_vals, x_vals, 'r--', label="y = ŷ (완벽한 예측)")

# 그래프 설정
plt.xlabel("실제 종합점수")
plt.ylabel("예측 종합점수")
plt.title("OLS 회귀 결과: 실제 vs 예측")
plt.legend()
plt.show()




'''비선형모델'''


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# 100 개의 결정 트리

# 모델 학습
rf_model.fit(X, y)

# 예측값 계산
y_pred_rf = rf_model.predict(X)

# MSE 계산
mse_rf = mean_squared_error(y, y_pred_rf)
print(f"랜덤 포레스트 회귀 MSE: {mse_rf:.4f}")




'''전공별 부모교육수준 '''

cross_tab1 = pd.crosstab(df['부모교육수준'], df['전공'])

proportions1 = cross_tab1.div(cross_tab1.sum(axis=1),axis=0) * 100
fig, axes = plt.subplots(1, len(proportions1), figsize=(15, 5))
fig.suptitle("부모교육수준별 전공", fontsize=16)
for i, major in enumerate(proportions1.index):  
    axes[i].pie(proportions1.loc[major], labels=proportions1.columns, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f"{major}")

plt.tight_layout()

plt.show()

'''전공별 가족소득수준'''

cross_tab2 = pd.crosstab(df['가족소득수준'], df['전공'])
proportions2 = cross_tab2.div(cross_tab2.sum(axis=1),axis=0) * 100
fig, axes = plt.subplots(1, len(proportions2), figsize=(15, 5))
fig.suptitle("가족소득수준별 전공", fontsize=16)

for i, major in enumerate(proportions2.index):  
    axes[i].pie(proportions2.loc[major], labels=proportions2.columns, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f"{major}")

plt.tight_layout()

plt.show()

'''회귀계수 그래프'''

# 회귀 계수 저장
coefficients = {
    "부모교육수준_No": -0.2214,
    "부모교육수준_High School": -0.4395,
    "부모교육수준_Master's": -0.6655,
    "부모교육수준_PhD": 0.0137,
    "가족소득수준_Low": 1.1485,
    "가족소득수준_Medium": 0.8076,
    "인터넷가능여부_Yes": -0.6524
}

# 데이터 변환
features = list(coefficients.keys())
values = list(coefficients.values())

# 그래프 그리기
plt.figure(figsize=(10, 5))
sns.barplot(x=values, y=features, palette="coolwarm")

# 제목 및 축 레이블
plt.axvline(0, color="gray", linestyle="--")  # 기준선
plt.xlabel("회귀 계수 (Coef)")
plt.ylabel("변수")
plt.title("회귀 계수 시각화")
plt.show()


''' p-value 그래프'''


variables = [
    "부모교육수준_High School", "부모교육수준_Master's", "부모교육수준_No", 
    "부모교육수준_PhD", "가족소득수준_Low", "가족소득수준_Medium", "인터넷가능여부_Yes"
]
p_values = [0.541, 0.357, 0.716, 0.985, 0.037, 0.143, 0.331]

# 막대그래프 생성
plt.figure(figsize=(10, 5))
plt.barh(variables, p_values, color=['gray' if p > 0.05 else 'red' for p in p_values])

# p-value 값 표시
for i, v in enumerate(p_values):
    plt.text(v + 0.02, i, f"{v:.3f}", va='center', fontsize=10)

# 기준선 (유의수준 0.05)
plt.axvline(x=0.05, color='blue', linestyle='dashed', label='Significance Threshold (0.05)')

# 그래프 제목 및 레이블 설정
plt.xlabel("p-value")
plt.ylabel("변수")
plt.title("각 변수의 p-value 시각화")
plt.legend()
plt.xlim(0, 1)

# 그래프 출력
plt.show()


'''분산 시각화'''

df.groupby('부모교육수준')['종합점수'].std() 
df.groupby('가족소득수준')['종합점수'].std()






