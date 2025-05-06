import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import re
import numpy as np
import time
# ========================================================
#   DataFrame1=채용공고.csv
#   DataFrame2=기업분석.csv
# ========================================================
# 인덱스 기준으로 병합
# --------------------------------------------------------
def merge_DF(DataFrame1:pd.DataFrame,DataFrame2:pd.DataFrame):
    DataFrame=pd.concat([DataFrame1,DataFrame2],axis=1)
    # 회사명, 채용공고사항, 링크,지역,경력,학력,...
    DataFrame=DataFrame.drop('기업명',axis=1)
    return DataFrame
# ========================================================
#   str=str type(ex 10조 10억 1만원원)
# ========================================================
# 채용률 or 영업이익 등 like '10억' str타입으로 되어있는 
# Type int타입으로 형변환(단위: 10,000) 
# --------------------------------------------------------  
def type_int_cast(str):
    str=str.replace(',','')
    if '-' in str:
        try:
            a=re.search(r'\d+(?=조)',str).group()
            a=int(a)
        except:
            a=0
        try:
            b=re.search(r'\d+(?=억)',str).group()
            b=int(b)
        except:
            b=0

        try:
            c=re.search(r'\d+(?=만원)',str).group()
            c=int(c)
        except:
            c=0
        try:    
            d=re.search(r'\d+(?=원)',str).group()
            d=0
        except:
            d=0
        str=(a*10**8+b*10**4+c+d)*-1
        return str
    elif str=='0':
        str=0
        return str
    else:
        try:
            a=re.search(r'\d+(?=조)',str).group()
            a=int(a)
        except:
            a=0
        try:
            b=re.search(r'\d+(?=억)',str).group()
            b=int(b)
        except:
            b=0

        try:
            c=re.search(r'\d+(?=만원)',str).group()
            c=int(c)
        except:
            c=0
        try:    
            d=re.search(r'\d+(?=원)',str).group()
            d=0
        except:
            d=0
        str=(a*10**8+b*10**4+c+d)
        return str
# ========================================================
#   DataFrame1=채용공고.csv
#   DataFrame2=기업분석.csv
# ========================================================
# 함수사용 
# - merge_DF()
# - type_int_cast()
# ========================================================
# - ex) 대구전체, 경기전체 -> 대구 전체, 경기 전체
# - 신입,경력무관 fliter
# - 연봉관련 column int type casting
# --------------------------------------------------------  
def merge_type_cast(DataFrame1:pd.DataFrame,DataFrame2:pd.DataFrame):
    DataFrame=merge_DF(DataFrame1,DataFrame2)
    # # 지역 앞에서 하나 또는 두개까지만 긁어오기
    try:
        DataFrame['지역']=DataFrame['지역'].apply(lambda x:re.match(r'^.+(?=전체)',x).group()+' 전체' if '전체' in x else x)
    except: pass
    DataFrame.fillna('0',inplace=True)
    # # # 학력 신입과 경력무관만 뽑기
    DataFrame=DataFrame[DataFrame['경력'].apply(lambda x:bool(re.match(r'.*신입.*|.*무관.*',x)))]
    # # 기업정보 형변환
    list1=['매출액(2020)','매출액(2021)','매출액(2022)','매출액(2023)','영업이익(2020)','영업이익(2021)','영업이익(2022)','영업이익(2023)']
    for x in list1:
        DataFrame[x]=DataFrame[x].apply(type_int_cast)
    # 연봉정보 형변환
    list2=['평균연봉','최저연봉','최고연봉']
    for x in list2:
        DataFrame[x]=DataFrame[x].str.replace(',','').apply(int)   
    return DataFrame
# ========================================================
#   DataFrame1=채용공고.csv
#   DataFrame2=기업분석.csv
#   rank_number=Top (숫자)
#   column_name=1.정렬기준column, 2.정렬기준column, ... n.정렬기준column
#     * type(column_name)=tuple
# ========================================================
# 함수사용 
# - merge_type_cast()
# --- merge_DF()
# --- type_int_cast()
# ========================================================
# - ex) 대구전체, 경기전체 -> 대구 전체, 경기 전체
# - 신입,경력무관 fliter
# - 연봉관련 column int type casting
# --------------------------------------------------------  
def top_rank_sort(DataFrame1:pd.DataFrame,DataFrame2:pd.DataFrame,rank_number:int,*column_name):
    DataFrame=merge_type_cast(DataFrame1,DataFrame2)
    DataFrame=DataFrame.drop_duplicates(subset=['회사명'])
    DataFrame=DataFrame.sort_values(by=list(column_name),ascending=False,axis=0)
    DataFrame.reset_index(drop=True,inplace=True)
    return DataFrame.iloc[:rank_number]
# ========================================================
#   *DataFrame
#   hline=x축에 평행하게 표현하고 싶은 라인(y값)
#     * 기본값=평균연봉 
#   sort=[정렬기준,정렬기준,...,정렬기준]
#     * 데이터프레임수와 동일하게 반드시 입력
# ========================================================
# 정렬기준으로 subplots에 해당 값을 bar로 채움
# - 정렬기준이 평균연봉이 값은 오른쪽 y축 값 없음
# - 나머지는 왼쪽 y축은 정렬기준 값, 오른쪽 y축은 평균연봉 값
# --------------------------------------------------------      
def vizual_DF_bar(*DataFrame:pd.DataFrame,hline=None,sort:list=list()):
    number=len(DataFrame)
    number1=len(sort)
    if number==number1:
        if hline==None:
            DF=DataFrame[0]
            hline=int(DF['평균연봉'].sum()/DF['평균연봉'].shape[0])
        else:
            hline=hline
        # -----------------------------------------------------------------------
        if number<=3:
            fig,axes=plt.subplots(1,number,figsize=(15,7),constrained_layout=True)
        elif number>=4 and number<=6:
            fig,axes=plt.subplots(2,number-3,figsize=(15,7),constrained_layout=True) 
        else:
            print('그래프 표현범위를 벗어났습니다!')
        # ------------------------------------------------------------------------
        plt.suptitle('[각 요소별 Top 기업순위]',fontweight='bold',fontsize=15)
        color=['#483D8B','#8B0000','#556B2F','#2F4F4F','#B22222','#191970']
        for x,y,num in zip(axes.flatten(),sort,range(number)):
            xlable=pd.Series(range(0,(DataFrame[num]['회사명'].shape[0]*2),2))
            DF=DataFrame[num]
            if y=='평균연봉':
                x.bar(xlable,DF.loc[:,'평균연봉'],color=color[num],label='평균연봉')
                x.set_title(y,pad=10)
                x.set_ylabel('만원')
                x.axhline(hline,color='r',linestyle='--',label='전체평균연봉')
                x.set_ylim(0,max(DF.loc[:,'평균연봉'])*1.3)
            else:
                x.bar(xlable,DF.loc[:,y],color=color[num],label=y)
                axes1=x.twinx()
                axes1.plot(xlable,DF.loc[:,'평균연봉'],'o-',c='gray',label='평균연봉')
                x.set_title(y,pad=10)
                x.set_ylim(0,max(DF.loc[:,y])*1.3)
                axes1.axhline(hline,color='r',linestyle='--',label='전체평균연봉')
                axes1.legend(loc=1)
                axes1.set_ylim(0,max(DF.loc[:,'평균연봉'])*1.3)
            x.set_xticks(range(0,(DF['회사명'].shape[0]*2),2),DF['회사명'])
            x.set_xticklabels(labels=DF['회사명'],rotation=45)
            x.legend(loc=2)
        for Data,axes,value,height in zip(DataFrame,axes.flatten(),sort,[300,300000,30000]):
            str=Data['사원수']
            x=range(0,Data.shape[0]*2,2)
            y=Data.loc[:,value]   
            for a,b,c in zip(str,x,y):
                axes.text(x=b,y=c+height,s=a,ha='center',va='center',color='k',fontweight='bold')
        plt.show()    
    else: print('sort 변수에 잘못된 값의 수를 입력하였습니다.')
# ========================================================
#   DataFrame: merge_type_cast()를 거친 DataFrame
# ========================================================
# 지역수로 파이차트를 그리고 상위 5개 이외 값은 null처리
# --------------------------------------------------------     
def vizual_DF_pie(DataFrame:pd.DataFrame):
    DataFrame['지역']=DataFrame['지역'].apply(lambda x:x.split(' ')[0])
    ratio=DataFrame['지역'].value_counts()
    label=ratio.index
    
    label=list(label)
    print(label[3:])
    if len(label) >=5:
        label[5]=''*(len(label)-5)
    else:ratio=ratio.values
    # # 색깔 정하기
    # # 중심에서 0.1씩 뛰우기
    explodes=[0.1]*len(ratio)
    # # pie 그리기
    # # 도넛 모양 만들기
    wedgeprops={'width':0.7,'edgecolor':'w','linewidth':1}
    plt.figure(figsize=(7,7))
    plt.pie(ratio,labels=label,explode=explodes,autopct='%1.1f%%',shadow=True,wedgeprops=wedgeprops)
    plt.legend(loc=1,bbox_to_anchor=(1.3,1))
    plt.title('지역분포도')
    plt.show()
    # time.sleep(5)


# ===========================입력===============================
# DataFrame1=pd.read_csv('./DATA/defense_it.csv',index_col=0)
# DataFrame2=pd.read_csv('./DATA/방산_기업분석.csv')
# ===============================================================
# 지역 filter '대구' 수정
# DF=DF[DF['지역'].apply(lambda x:bool(re.search(r'^대구.*',x)))]
# print(DF)
# -------------------------------------------------------------
# 연봉 filter 추가
# DF=DF[DF['평균연봉']<=4500]
# --------------------------------------------------------------
# DF=top_rank_sort(DataFrame1,DataFrame2,10,'평균연봉')
# DF1=top_rank_sort(DataFrame1,DataFrame2,10,'매출액(2023)')
# DF2=top_rank_sort(DataFrame1,DataFrame2,10,'영업이익(2023)')

# vizual_DF_bar(DF,DF1,DF2,sort=['평균연봉','매출액(2023)','영업이익(2023)'])
# ========================================================
# hline양식
# hline=int(DF['평균연봉'].sum()/DF['평균연봉'].shape[0])
# ========================================================
# vizual_DF_pie(DF1)


