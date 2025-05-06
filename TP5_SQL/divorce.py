import pymysql
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# 한글 폰트 설정 (Windows 사용자)
plt.rc('font', family='Malgun Gothic')  # Windows 사용자는 Malgun Gothic
plt.rcParams['axes.unicode_minus'] = False

""" 
user_id
청하 ch
우성 ws
정혜 jh
정욱 jw

pass 1234
"""
#%%
conn = pymysql.connect(host='172.20.135.53', user='ws',
                        password='1234',
                        db='SQL_P3', charset='utf8')
cur = conn.cursor(pymysql.cursors.DictCursor)

def fetch_sql_to_df(cursor, table_name, df_name):

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    columns = [col[0] for col in cursor.description]
    
    df = pd.DataFrame(rows, columns=columns)
    print(f"\n[{df_name}]")  # 데이터프레임 이름 출력
    print(tabulate(df.head(), headers='keys', tablefmt='pretty'))
    
    return df

n1_region_divorce_df = fetch_sql_to_df(cur, "1_년도에_따른_전국이혼건수", "년도별 전국 이혼 건수")
n2_region_cause_divorce_df = fetch_sql_to_df(cur, "2_년도에_따른_시도별이혼사유", "지역별 이혼 사유")
n3_sexage_divorce_df = fetch_sql_to_df(cur,'3_성별_나이_따른_이혼율','성별 나이에 따른 이혼율')
n4_sexage_cause_divorce_df = fetch_sql_to_df(cur,'4_성별_나이_따른_이혼사유','성별 나이에 따른 이혼 사유')
n5_income_first_married_df = fetch_sql_to_df(cur,'5_초혼부부_소득','초혼부부 소득')
n6_consumer_Price_df = fetch_sql_to_df(cur,'6_소비자물가지수','소비자 물가 지수')
n7_total_income_solo_df = fetch_sql_to_df(cur,'7_국민총소득','1인 국민 총소득')
n8_remarry_count_df = fetch_sql_to_df(cur, '8_총혼인_중에_재혼건_수', '총 혼인 중 재혼 건 수')
n9_age_remarry_df = fetch_sql_to_df(cur,'9_재혼연령별','연령별 재혼')
#%%
file=r"C:\Users\KDT-13\Desktop\KDT 7\project\TP4_SQL\data\n0_혼인건수.csv"
n0_marry_count = pd.read_csv(file)
n0_marry_count = n0_marry_count.drop(0)
n0_marry_count = n0_marry_count.astype('int')

n0_n1_merge = n0_marry_count.merge(n1_region_divorce_df, on='시점', how='inner')  # 공통된 시점만 포함


#%%
#1. 결혼 건수 대비 이혼 건수 그래프
n0_n1_merge["결혼대비 이혼 건수"] = n0_n1_merge["전국_y"] / n0_n1_merge["전국_x"]
plt.figure(figsize=(10, 5))
# 그래프 그리기
plt.plot(n0_n1_merge['시점'], n0_n1_merge['결혼대비 이혼 건수'], marker='o', linestyle='-', color='b', label="결혼 대비 이혼 건수")
# 제목 및 라벨 설정
plt.title("연도별 결혼 대비 이혼 건수", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("비율", fontsize=12)
# X축 눈금 설정 (1년 단위)
plt.xticks(n0_n1_merge['시점'], rotation=45)
# 그리드 추가
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 범례 추가
plt.legend()
# 그래프 표시
plt.show()

#%%
# 계 잘라내기
n2_region_cause_divorce_df = n2_region_cause_divorce_df.iloc[18:,:]


#%%
# 2.전국 '성격차이'와 '기타' 제외
filtered_df = n2_region_cause_divorce_df[~n2_region_cause_divorce_df["항목"].isin(["성격차이", "기타"])]

# 2. 전국 이혼사유 (성격차이, 기타 제외)
all_region_pivot_df = filtered_df.pivot(index="시점", columns="항목", values="전국")

# 연도별 합계 구해서 비율 변환
all_region_pivot_df_percentage = all_region_pivot_df.div(all_region_pivot_df.sum(axis=1), axis=0) * 100

# 연도 리스트 가져오기 (정렬 후 3년 간격으로 선택, 2017 포함)
years = sorted(all_region_pivot_df_percentage.index.unique())
xticks = [year for year in years if (year - years[0]) % 3 == 0 or year == 2017]  # 3년 간격 또는 2017 포함

# 그래프 그리기
plt.figure(figsize=(10, 6))
ax = all_region_pivot_df_percentage.plot(kind="area", stacked=True, alpha=0.7, colormap="Set2", figsize=(10, 6))

# 그래프 설정
ax.set_title("[전국] 이혼 사유별 비율 변화 (2000~2017, '성격차이' & '기타' 제외)", fontsize=14)
ax.set_xlabel("년도", fontsize=12)
ax.set_ylabel("비율 (%)", fontsize=12)
ax.legend(title="이혼 사유", loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_xticks(xticks)  # X축 눈금 설정

plt.show()


#%%

# 지역별 이혼 원인
def plot_divorce_trend_by_region(df, regions):
    """
    특정 지역들의 이혼 사유별 비율 변화를 subplot으로 시각화하는 함수

    Parameters:
        df (DataFrame): '시점'이 index, '항목'이 columns, 지역별 이혼 사유 데이터가 values로 구성된 데이터프레임
        regions (list): 분석할 지역 리스트
    """
    # '성격차이'와 '기타' 제외
    df = df[~df["항목"].isin(["성격차이", "기타"])]

    num_regions = len(regions)
    fig, axes = plt.subplots(num_regions, 1, figsize=(10, 6 * num_regions), sharex=True)

    if num_regions == 1:
        axes = [axes]

    # 시점 값 가져오기 (년도 리스트)
    years = df["시점"].unique()
    years = sorted(years)  # 정렬
    xticks = years[::3]  # 3년 간격으로 선택

    for ax, region in zip(axes, regions):
        region_pivot_df = df.pivot(index="시점", columns="항목", values=region)
        region_percentage = region_pivot_df.div(region_pivot_df.sum(axis=1), axis=0) * 100

        region_percentage.plot(kind="area", stacked=True, alpha=0.7, colormap="Set2", ax=ax)
        
        ax.set_title(f"{region} 이혼 사유별 비율 변화 (2000~2017, '성격차이' & '기타' 제외)", fontsize=14)
        ax.set_xlabel("년도", fontsize=12)
        ax.set_ylabel("비율 (%)", fontsize=12)
        ax.set_xticks(xticks)  # 3년 간격으로 x축 설정
        ax.grid(True, linestyle="--", alpha=0.7)  # x축, y축 모두 그리드 추가
        ax.legend(title="이혼 사유", loc="upper right")  # 범례를 우측 상단으로 이동

    plt.tight_layout()
    plt.show()

# 사용할 지역 리스트
regions = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시"]

# 함수 실행 (성격차이 & 기타 제외된 데이터 사용)
plot_divorce_trend_by_region(n2_region_cause_divorce_df, regions)


#%%
# 결혼 생활 계획서

data = {
    "전혀 그렇지 않다 (%)": [12.3, 12.9, 11.7],
    "별로 그렇지 않다 (%)": [28.2, 29.1, 27.2],
    "보통이다 (%)": [31.4, 31.8, 31.1],
    "대체로 그렇다 (%)": [25.6, 24.1, 27.1],
    "매우 그렇다 (%)": [2.4, 2.0, 2.9]
}

# 데이터프레임 생성
contract_df = pd.DataFrame(data)

labels = contract_df.columns  # 항목명
sizes = contract_df.iloc[0]  # 첫 번째 행 데이터

# 파이 차트 생성
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct="%.1f%%", startangle=140, colors=plt.cm.Set2.colors)
plt.title("2023년 결혼생활 계약서 필요성 응답 비율")
plt.show()


#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Chapter 2
# 성별에 따른 평균 이혼률
#*"남성 이혼률"**은 남성 인구 대비 이혼한 남성 수, **"여성 이혼률"**은 여성 인구 대비 이혼한 여성 수
n3_sexage_divorce_df = n3_sexage_divorce_df.drop(0)
n3_sexage_divorce_df['시점'] = n3_sexage_divorce_df['시점'].astype('int')
n3_sexage_divorce_df['남편(해당연령 천명당 건)'] = n3_sexage_divorce_df['남편(해당연령 천명당 건)'].astype('float')
n3_sexage_divorce_df['아내(해당연령 천명당 건)'] = n3_sexage_divorce_df['아내(해당연령 천명당 건)'].astype('float')

grouped_df = n3_sexage_divorce_df.groupby("시점")[["남편(해당연령 천명당 건)", "아내(해당연령 천명당 건)"]].mean()
plt.figure(figsize=(10, 5))
grouped_df.plot(kind="bar", figsize=(10, 5), alpha=0.8)

# 그래프 설정
plt.title("성 별 평균 이혼률 (연령별)", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("이혼 건수 (천명당)", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(["남편", "아내"], title="성별")

# 그래프 출력
plt.show()


#%%
#남성 연령병 이혼율 -꺽은선그래프
# 데이터프레임 생성 (가정: n3_sexage_divorce_df가 이미 있음)
man_pivot_df = n3_sexage_divorce_df.pivot(index="시점", columns="연령별", values="남편(해당연령 천명당 건)")

# 그래프 그리기
plt.figure(figsize=(12, 15))
for age_group in man_pivot_df.columns:
    plt.plot(man_pivot_df.index, man_pivot_df[age_group], marker="o", label=age_group)

# 그래프 설정
plt.title("시점별 연령대에 따른 남편 이혼율 변화", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("이혼 건수 (천명당)", fontsize=12)
plt.xticks(man_pivot_df.index, rotation=45)
plt.legend(title="연령별", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.5)

# 그래프 출력
plt.show()


#%%
#여성 연령별 이혼률 - 꺾은선그래프
# 데이터프레임 생성 (가정: n3_sexage_divorce_df가 이미 있음)
wo_pivot_df = n3_sexage_divorce_df.pivot(index="시점", columns="연령별", values="아내(해당연령 천명당 건)")

# 그래프 그리기
plt.figure(figsize=(12, 15))
for age_group in wo_pivot_df.columns:
    plt.plot(wo_pivot_df.index, wo_pivot_df[age_group], marker="o", label=age_group)

# 그래프 설정
plt.title("시점별 연령대에 따른 아내 이혼율 변화", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("이혼 건수 (천명당)", fontsize=12)
plt.xticks(wo_pivot_df.index, rotation=45)
plt.legend(title="연령별", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.5)

# 그래프 출력
plt.show()


#%%
# 남성 연령별 이혼률 - 히트맵
pivot_heatmap = n3_sexage_divorce_df.pivot(index="연령별", columns="시점", values="남편(해당연령 천명당 건)")

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)

# 그래프 설정
plt.title("시점별 연령대 남편 이혼율 히트맵", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("연령별", fontsize=12)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()


age_groups = ["50 - 54세", "55 - 59세", "60 - 64세", "65 - 69세", "70 - 74세", "75세 이상"]
n3_filtered = n3_sexage_divorce_df[n3_sexage_divorce_df["연령별"].isin(age_groups)]

# 피벗 테이블 생성
pivot_heatmap = n3_filtered.pivot(index="연령별", columns="시점", values="남편(해당연령 천명당 건)")

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)

# 그래프 설정
plt.title("시점별 연령대 남편 이혼율 히트맵", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("연령별", fontsize=12)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()
#%%
# 여성 연령별 이혼률 -히트맵
pivot_heatmap2 = n3_sexage_divorce_df.pivot(index="연령별", columns="시점", values="아내(해당연령 천명당 건)")

# 값의 최소, 최대 구하기
vmin = pivot_heatmap2.min().min()  # 최소값
vmax = pivot_heatmap2.max().max()  # 최대값
center_value = pivot_heatmap2.mean().mean()  # 평균값 중심 설정

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap2, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)

# 그래프 설정
plt.title("시점별 연령대 아내 이혼율 히트맵", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("연령별", fontsize=12)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()

#%%
#50대이상여자 히트맵 마지막쓸것
'''
# 50세 이상만 필터링
pivot_heatmap_50up = pivot_heatmap2.loc[pivot_heatmap2.index >= "50세"]

# 값의 최소, 최대 구하기
vmin = pivot_heatmap_50up.min().min()  # 최소값
vmax = pivot_heatmap_50up.max().max()  # 최대값
center_value = pivot_heatmap_50up.mean().mean()  # 평균값 중심 설정

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap_50up, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, 
            vmin=vmin, vmax=vmax, center=center_value)

# 그래프 설정
plt.title("시점별 50세 이상 아내 이혼율 히트맵", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("연령별", fontsize=12)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()
'''
#%%
#n4 전처리
file3=r"C:\Users\KDT-13\Desktop\KDT 7\project\TP4_SQL\data\4.성별,나이 따른 이혼사유.csv"
n4_sexage_cause_divorce_df=pd.read_csv(file3)

n4_sexage_cause_divorce_df.columns = n4_sexage_cause_divorce_df.iloc[0]
n4_sexage_cause_divorce_df = n4_sexage_cause_divorce_df.drop(range(0,19), axis=0)
n4_sexage_cause_divorce_df['시점'] = n4_sexage_cause_divorce_df['시점'].astype('int')
#n4_sexage_cause_divorce_df['경제문제'] = n4_sexage_cause_divorce_df['경제문제'].astype('int')

n4_sexage_cause_divorce_select_df = n4_sexage_cause_divorce_df[['연령별','시점','경제문제']]

n4_sexage_cause_divorce_select_df.columns.values[2] = '경제문제(남자)'
n4_sexage_cause_divorce_select_df.columns.values[3] = '경제문제(여자)'

#%%
'''
age_group_df = n4_sexage_cause_divorce_select_df[n4_sexage_cause_divorce_select_df['연령별'].isin(['15 - 19세', '20 - 24세'])].groupby('시점', as_index=False)['경제문제(남자)'].sum()

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(age_group_df['시점'], age_group_df['경제문제(남자)'], marker='o', linestyle='-')

# 그래프 설정
plt.title("15 - 24세 경제문제(남자)", fontsize=14)
plt.xlabel("시점", fontsize=12)
plt.ylabel("경제문제(남자)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)

# 그래프 출력
plt.show()
'''
#%%
'''
#남성 경제문제로 인한 연령별 
age_groups = ["15 - 19세", "20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세", 
              "40 - 44세", "45 - 49세", "50 - 54세", "55 - 59세", "60 - 64세", 
              "65 - 69세", "70 - 74세", "75세 이상"]

grouped_df = n4_sexage_cause_divorce_select_df[n4_sexage_cause_divorce_select_df['연령별'].isin(age_groups)].groupby(['시점', '연령별'], as_index=False)['경제문제(남자)'].sum()

# 그래프 그리기
plt.figure(figsize=(12, 6))

for age in age_groups:
    subset = grouped_df[grouped_df['연령별'] == age]
    plt.plot(subset['시점'], subset['경제문제(남자)'], marker='o', linestyle='-', label=age)

# 그래프 설정
plt.title("연령별 경제문제(남자)", fontsize=14)
plt.xlabel("시점", fontsize=12)
plt.ylabel("경제문제(남자)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="연령별", loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)

# 그래프 출력
plt.show()
'''
#%%
'''
#여성 경제문제로 인한 연령별 
age_groups = ["15 - 19세", "20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세", 
              "40 - 44세", "45 - 49세", "50 - 54세", "55 - 59세", "60 - 64세", 
              "65 - 69세", "70 - 74세", "75세 이상"]

grouped_df = n4_sexage_cause_divorce_select_df[n4_sexage_cause_divorce_select_df['연령별'].isin(age_groups)].groupby(['시점', '연령별'], as_index=False)['경제문제(여자)'].sum()

# 그래프 그리기
plt.figure(figsize=(12, 10))

for age in age_groups:
    subset = grouped_df[grouped_df['연령별'] == age]
    plt.plot(subset['시점'], subset['경제문제(여자)'], marker='o', linestyle='-', label=age)

# 그래프 설정
plt.title("연령별 경제문제(여자)", fontsize=14)
plt.xlabel("시점", fontsize=12)
plt.ylabel("경제문제(여자)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="연령별", loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)

# 그래프 출력
plt.show()
'''

#%%

#남성연령별이혼사유변화파이그래프
df_2000 = n4_sexage_cause_divorce_df[
    (n4_sexage_cause_divorce_df['연령별'].isin(["45 - 49세", "50 - 54세", "55 - 59세"])) &
    (n4_sexage_cause_divorce_df['시점'] == 2000)
]
df_2017 = n4_sexage_cause_divorce_df[
    (n4_sexage_cause_divorce_df['연령별'].isin(["45 - 49세", "50 - 54세", "55 - 59세"])) &
    (n4_sexage_cause_divorce_df['시점'] == 2017)
]

# '계', '성격차이', '기타' 컬럼 제외하고 이혼사유별 합산
exclude_columns = ["계", "성격차이", "기타"]
reason_2000 = df_2000.drop(columns=exclude_columns, errors="ignore").iloc[:, 3:9].sum()
reason_2017 = df_2017.drop(columns=exclude_columns, errors="ignore").iloc[:, 3:9].sum()

# 파이그래프 생성
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2000년도 파이그래프
axes[0].pie(reason_2000, labels=reason_2000.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[0].set_title("2000년 (35-59세 남성 이혼사유)")

# 2017년도 파이그래프
axes[1].pie(reason_2017, labels=reason_2017.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].set_title("2017년 (35-59세 남성 이혼사유)")

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2000년도 파이그래프
axes[0].pie(
    reason_2000, 
    labels=reason_2000.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors, 
    pctdistance=0.85,  # 퍼센트 위치 조정
    labeldistance=1.1   # 라벨 위치 조정
)
axes[0].set_title("2000년 (45-59세 남성 이혼사유)")

# 2017년도 파이그래프
axes[1].pie(
    reason_2017, 
    labels=reason_2017.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors, 
    pctdistance=0.85,  
    labeldistance=1.1   
)
axes[1].set_title("2017년 (45-59세 남성 이혼사유)")

# 레이아웃 조정
plt.tight_layout()
plt.show()

#%%
# 제외할 컬럼 리스트
'''
exclude_columns = ["계", "성격차이", "기타"]

# 남성 이혼 사유 (3~9번째 컬럼 선택 후 제외할 컬럼 삭제)
reason_2000_male = df_2000.iloc[:, 3:9].drop(labels=exclude_columns, errors="ignore", axis=1).sum()
reason_2017_male = df_2017.iloc[:, 3:9].drop(labels=exclude_columns, errors="ignore", axis=1).sum()

# 여성 이혼 사유 (11~17번째 컬럼 선택 후 제외할 컬럼 삭제)
reason_2000_female = df_2000.iloc[:, 10:16].drop(labels=exclude_columns, errors="ignore", axis=1).sum()
reason_2017_female = df_2017.iloc[:, 10:16].drop(labels=exclude_columns, errors="ignore", axis=1).sum()

# 남성 파이그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(reason_2000_male, labels=reason_2000_male.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[0].set_title("2000년 (45-59세 남성 이혼사유)")

axes[1].pie(reason_2017_male, labels=reason_2017_male.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].set_title("2017년 (45-59세 남성 이혼사유)")

plt.show()

# 여성 파이그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(reason_2000_female, labels=reason_2000_female.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[0].set_title("2000년 (45-59세 여성 이혼사유)")

axes[1].pie(reason_2017_female, labels=reason_2017_female.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].set_title("2017년 (45-59세 여성 이혼사유)")

plt.show()
'''

import matplotlib.pyplot as plt

exclude_columns = ["계", "성격차이", "기타"]

# 남성 이혼 사유 (3~9번째 컬럼 선택 후 제외할 컬럼 삭제)
reason_2000_male = df_2000.iloc[:, 3:9].drop(labels=exclude_columns, errors="ignore", axis=1).sum()
reason_2017_male = df_2017.iloc[:, 3:9].drop(labels=exclude_columns, errors="ignore", axis=1).sum()

# 여성 이혼 사유 (11~17번째 컬럼 선택 후 제외할 컬럼 삭제)
reason_2000_female = df_2000.iloc[:, 10:16].drop(labels=exclude_columns, errors="ignore", axis=1).sum()
reason_2017_female = df_2017.iloc[:, 10:16].drop(labels=exclude_columns, errors="ignore", axis=1).sum()

# 파이그래프 설정 (글자 간격 조정)
label_distance = 1.2  # 라벨 위치
pct_distance = 0.85  # 퍼센트 위치

# 남성 파이그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(
    reason_2000_male, 
    labels=reason_2000_male.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors,
    labeldistance=label_distance,  
    pctdistance=pct_distance  
)
axes[0].set_title("2000년 (45-59세 남성 이혼사유)")

axes[1].pie(
    reason_2017_male, 
    labels=reason_2017_male.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors,
    labeldistance=label_distance,  
    pctdistance=pct_distance  
)
axes[1].set_title("2017년 (45-59세 남성 이혼사유)")

plt.show()

# 여성 파이그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(
    reason_2000_female, 
    labels=reason_2000_female.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors,
    labeldistance=label_distance,  
    pctdistance=pct_distance  
)
axes[0].set_title("2000년 (45-59세 여성 이혼사유)")

axes[1].pie(
    reason_2017_female, 
    labels=reason_2017_female.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors,
    labeldistance=label_distance,  
    pctdistance=pct_distance  
)
axes[1].set_title("2017년 (45-59세 여성 이혼사유)")

plt.show()




#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#chapter 3
n5_income_first_married_df_copy = n5_income_first_married_df.iloc[10:64,1:]
n5_income_first_married_df_copy[['시점','혼인연차별']] = n5_income_first_married_df_copy[['시점','혼인연차별']].astype('float')

#%%
#초혼 소득수준 그래프
# 그래프 그리기
plt.figure(figsize=(12, 4))
sns.lineplot(data=n5_income_first_married_df_copy, x='시점', y='혼인연차별', hue='소득구간별(2)', marker='o')

# 제목 및 축 라벨 설정
plt.title('초혼 소득구간 비중')
plt.xlabel('시점')
plt.ylabel('소득구간')
plt.legend(title='소득구간', loc='center right', fontsize=8)

plt.show()
#%%
#소비자 물가지수
plt.figure(figsize=(10, 6))

# 그래프 선 스타일 설정
plt.plot(n6_consumer_Price_df['시점'], n6_consumer_Price_df['전국'], marker='o', linestyle='-', color='darkred', linewidth=2, markersize=6)

# 제목 및 축 라벨 설정
plt.title('소비자물가지수 변화', fontsize=14, fontweight='bold')
plt.xlabel('시점', fontsize=12)
plt.ylabel('소비자물가지수', fontsize=12)

# 눈금 표시 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 그래프 테두리 스타일 조정
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()
#%%
#1인당 국민 총소득
plt.figure(figsize=(10, 6))

# 그래프 스타일 설정
plt.plot(n7_total_income_solo_df['시점'], 
         n7_total_income_solo_df['1인당 국민총소득(명목 원화표시) (만원)'], 
         marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)

# 제목 및 축 라벨 설정
plt.title('1인당 국민총소득 변화 (명목 원화표시)', fontsize=14, fontweight='bold')
plt.xlabel('시점', fontsize=12)
plt.ylabel('1인당 국민총소득 (만원)', fontsize=12)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.3f}'))
# 눈금 조정
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 그래프 테두리 스타일 조정
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()
#%%
#2인가구

data = {
    '시점': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    '1인가구': [1624831, 1652931, 1672105, 1707008, 1757194, 1827831, 1944812, 2077892],
    '2인가구': [2766603, 2814449, 2847097, 2906528, 2991980, 3088079, 3260085, 3456155]
}

two_df = pd.DataFrame(data)
print(two_df)
#%%
#2인가구 인당 소득 그래프
two_df['2인가구'] = two_df['2인가구'] / 10000  

plt.figure(figsize=(8, 5))  # 그래프 크기 설정
plt.plot(two_df['시점'], two_df['2인가구'], marker='o', linestyle='-', color='b', label='2인가구')

# 제목 및 축 레이블
plt.title('2인가구 인당 소득 추이 (단위: 만원)')
plt.xlabel('시점')
plt.ylabel('2인가구 소득 (만원)')

# 눈금 및 격자 설정
plt.xticks(two_df['시점'])
plt.grid(True, linestyle='--', alpha=0.6)

# 범례 위치 설정
plt.legend(loc='best')

plt.show()

#%%
#재혼율
import pandas as pd

data = {
    "시점": list(range(1990, 2024)),  # 1990년부터 2023년까지
    "재혼비율": [10.7, 10.6, 11.2, 11.9, 12.5, 13.6, 14.0, 14.7, 16.0, 17.6, 18.0, 20.3, 
               21.0, 22.3, 24.5, 25.4, 22.4, 22.4, 23.8, 23.5, 21.9, 21.4, 21.4, 20.8, 
               21.6, 21.3, 21.3, 21.9, 22.2, 22.8, 21.6, 22.3, 22.3, 22.3]
}

remarry_per_df = pd.DataFrame(data)
print(remarry_per_df)
#%%
#재혼율 그래프
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
plt.plot(remarry_per_df['시점'], remarry_per_df['재혼비율'], marker='o', linestyle='-', color='b', label='재혼비율')  

plt.xlabel('시점', fontsize=12)  # x축 레이블
plt.ylabel('재혼비율 (%)', fontsize=12)  # y축 레이블
plt.title('연도별 재혼(혼인) 비율 변화', fontsize=14, fontweight='bold')  # 제목

plt.xticks(rotation=45)  # x축 눈금 기울이기
plt.grid(True, linestyle='--', alpha=0.7)  # 격자 표시
plt.legend(loc='upper right', fontsize=10)  # 범례 추가

plt.show()

#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#chapter 4
#연령별재혼건수
file=r"C:\Users\KDT-13\Desktop\KDT 7\project\TP4_SQL\data\시도_재혼연령별_혼인_20250217011733.csv"
remarry_per_df=pd.read_csv(file)

remarry_per_df.columns=remarry_per_df.iloc[0]
remarry_per_df=remarry_per_df.drop(0)
remarry_per_df[['시점',	'50 - 54세', '55 - 59세','60 - 64세','65 - 69세','70 - 74세','75세 이상']] = remarry_per_df[['시점',	'50 - 54세', '55 - 59세','60 - 64세','65 - 69세','70 - 74세','75세 이상']].astype('int')

remarry_per_man_df = remarry_per_df.iloc[:24,1:]
remarry_per_women_df = remarry_per_df.iloc[24:,1:]
#%%
#연령별 혼인건수
file2=r"C:\Users\KDT-13\Desktop\KDT 7\project\TP4_SQL\data\시도_연령_5세_별_혼인_20250217013943.csv"
remerry_count_df=pd.read_csv(file2)

#remerry_count_df.columns=remerry_count_df.iloc[0]
remerry_count_df=remerry_count_df.drop(0)

remerry_count_df =remerry_count_df.drop(remerry_count_df.columns[0], axis=1)

remerry_count_df=remerry_count_df.astype('int')
#%%
remarry_per_man_df['합계'] = remarry_per_man_df[['50 - 54세', '55 - 59세', '60 - 64세', '65 - 69세', '70 - 74세', '75세 이상']].sum(axis=1)

remerry_count_df['50세 이상 재혼률'] = remarry_per_man_df['합계']/remerry_count_df['전국']*100

#%%
#50세 이상 재혼률
plt.figure(figsize=(10, 5))  # 그래프 크기 조정
plt.plot(remerry_count_df['시점'], remerry_count_df['50세 이상 재혼률'], marker='o', linestyle='-', color='b', label='50세 이상 재혼률')

plt.xlabel('시점', fontsize=12)  # x축 라벨
plt.ylabel('재혼률 (%)', fontsize=12)  # y축 라벨
plt.title('50세 이상 재혼률 변화', fontsize=14, fontweight='bold')  # 제목
plt.grid(True, linestyle='--', alpha=0.5)  # 격자 추가
plt.xticks(rotation=45)  # x축 눈금 기울이기
plt.legend(loc='upper right')  # 범례 추가

plt.show()




'''
pivot_heatmap2 = n3_sexage_divorce_df.pivot(index="연령별", columns="시점", values="아내(해당연령 천명당 건)")

# 값의 최소, 최대 구하기
vmin = pivot_heatmap2.min().min()  # 최소값
vmax = pivot_heatmap2.max().max()  # 최대값
center_value = pivot_heatmap2.mean().mean()  # 평균값 중심 설정

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_heatmap2, annot=True, fmt=".1f", cmap="RdBu_r", linewidths=0.5, 
            vmin=vmin, vmax=vmax, center=center_value)

# 그래프 설정
plt.title("시점별 연령대 아내 이혼율 히트맵", fontsize=14)
plt.xlabel("년도", fontsize=12)
plt.ylabel("연령별", fontsize=12)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()

'''
#%%
#초혼 년도별 소득

data = {
    "시점": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "소득 평균(만원)": [4950, 5131, 5370, 5639, 5867, 6175, 6400, 6790, 7265],
    "소득 중앙값(만원)": [4455, 4639, 4823, 5164, 5504, 5639, 5639, 6246, 6768]
}

df = pd.DataFrame(data)

# 그래프 그리기
plt.figure(figsize=(10, 5))

plt.plot(df["시점"], df["소득 평균(만원)"], marker='o', linestyle='-', label="소득 평균", color='b')
plt.plot(df["시점"], df["소득 중앙값(만원)"], marker='s', linestyle='--', label="소득 중앙값", color='r')

plt.xlabel("시점", fontsize=12)
plt.ylabel("소득 (만원)", fontsize=12)
plt.title("혼인 1년차 소득 평균 및 중앙값 변화", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(df["시점"])  # x축 눈금 설정

plt.show()

#%%
#맞벌이 부부 소득
data = {
    "시점": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "소득 평균 (만원)": [6734, 6898, 7199, 7364, 7582, 7709, 8040, 8433, 8972],
    "소득 중앙값 (만원)": [6070, 6210, 6446, 6641, 6811, 6918, 7164, 7506, 7985]
}

df = pd.DataFrame(data)

# 출력


# 그래프 크기 설정
plt.figure(figsize=(10, 5))

# 소득 평균 그래프
plt.plot(df["시점"], df["소득 평균 (만원)"], marker='o', linestyle='-', color='b', label="소득 평균")

# 소득 중앙값 그래프
plt.plot(df["시점"], df["소득 중앙값 (만원)"], marker='s', linestyle='--', color='r', label="소득 중앙값")

# 그래프 제목과 라벨 설정
plt.xlabel("시점", fontsize=12)
plt.ylabel("소득 (만원)", fontsize=12)
plt.title("맞벌이 가구 소득 평균 및 중앙값 변화", fontsize=14, fontweight="bold")

# 격자 추가
plt.grid(True, linestyle='--', alpha=0.5)

# x축 눈금 기울이기
plt.xticks(df["시점"])

# 범례 추가
plt.legend(loc="upper left")

# 그래프 출력
plt.show()