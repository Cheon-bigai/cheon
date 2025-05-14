import pandas as pd
import numpy as np
import re
DF=pd.read_csv('./DATA\/sunslope_ver4.csv')
cunDF=pd.read_csv('./DATA/construction_difficulty.csv')
dis=pd.read_csv('./DATA/social_distance.csv')
# =======================================================================
# 전처리
DF=DF.rename(columns={'경사도_점수(1-10점)':'경사도'})
# DF['경사도']=DF['경사도'].replace(dict(zip(range(1,11),range(0,10))))
DF.columns=['도로명', '시작위도', '시작경도', '중간위도', '중간경도', '끝위도', '끝경도',
       '연속 일조량', '총 일조량', '경사도']
DF=pd.concat([DF,dis.loc[:,['거리']]],axis=1)
compile=re.compile(r'^Achasan-ro\s+(\d+)-gil\(\d+\)$')
DF.replace(compile,r'아차산로(\1)길',regex=True,inplace=True)

# 미가로 처리
for x in range(16):
    DF.loc[DF.loc[:,'도로명']==f'미가로({x})','도로명']=f'자양로18길({x})'
# 자양나들목 삭제
for x in range(2):
    idx=DF.loc[DF.loc[:,'도로명']==f'자양나들목({x})'].index
    DF.drop(index=idx,inplace=True)
# 강동대로 삭제
idx=DF.loc[DF.loc[:,'도로명']==f'강동대로(0)'].index
DF.drop(index=idx,inplace=True)
# 자양중앙나들목 변경
DF.loc[DF.loc[:,'도로명']==f'자양중앙나들목(0)','도로명']=f'뚝섬로52길(20)'
# 경사도 데이터에는 동부간선로 -> 동부간선도로로
cunDF.replace({'동부간선로':'동부간선도로'},inplace=True)
# 낙천정지하도 삭제
idx=DF.loc[DF.loc[:,'도로명']=='낙천정지하도(0)'].index
DF.drop(index=idx,inplace=True)
# 교차로 삭제
cond=DF.도로명.apply(lambda x: True if '교차로' in x else False)
idx=DF.loc[cond].index
DF.drop(index=idx,inplace=True)
#=========================================================================
# 조회
# print(DF.loc[DF.loc[:,'도로명']=='낙천정지하도(0)'])
# ========================================================================
DF.loc[:,'공사난이도']=np.nan
DF.loc[:,'세부 난이도 등급']=np.nan
# print(cunDF.loc[:,'난이도등급'].unique())
# ========================================================================
# 도로명 포함 데이터 프레임반환
def find(string):
    cond=DF.도로명.apply(lambda x: True if string in x else False)
    print(DF.loc[cond])
# ========================================================================
# 병합(공사난이도 + 세부 난이도 + 일조량 + 경사도)
for idx,road in enumerate(cunDF.도로명.unique()):
    boolean=DF.도로명.apply(lambda x: True if road in x else False)
    DF.loc[boolean,'공사난이도']=cunDF.loc[idx,'총점']
    DF.loc[boolean,'세부 난이도 등급']=cunDF.loc[idx,'난이도등급']
DF.loc[:,'공사난이도']=DF.loc[:,'공사난이도']+DF.loc[:,'세부 난이도 등급'].replace({'상':0.9,'중':0.6,'하':0.0})
DF.drop(columns=['세부 난이도 등급'],inplace=True)
# ========================================================================
# 인구 데이터 병합
people=pd.read_csv('./DATA/people.csv')
people.dropna(inplace=True)
# ['정류소명', '승객수_집계', 'label', 'X', 'Y', '가장가까운도로']
values=people.loc[:,'승객수_집계'].values
total,bins=pd.cut(values,bins=[0,1000,2000,4000,5000,6000,8000,10000,30000,70000,200000],labels=range(10),retbins=True)
a=pd.DataFrame(total)
new=pd.concat([people,a],axis=1).drop(columns=['정류소명', '승객수_집계', 'label', 'X', 'Y'])
new.columns=['도로명','인구점수']
DF.loc[:,'인구점수']=0
for x,y in enumerate(new.도로명):
    DF.loc[DF.도로명==y,'인구점수']=new.loc[x,'인구점수']
# ========================================================================
# 정규화
for x in ['연속 일조량', '총 일조량','경사도', '거리', '공사난이도','인구점수']:
    value=DF.loc[:,x]
    max=DF.loc[:,x].max()
    DF.loc[:,x]=value/max
# ========================================================================
# category별 ratio설정
def category_ratio(cat=None):
    if cat=='accident':
        scoreDF=pd.DataFrame()
        accident=['아차산로(105)','교차로(580)','광나루로(26)','광나루로(30)','군자로(39)','교차로(57)','천호대로(12)',"['워커힐로', '영화사로'](0)",'용마산로5길(8)','강변북로(1)']
        total_info=accident
        for x in total_info:
            a=DF.loc[DF.loc[:,'도로명']==x]
            scoreDF=pd.concat([scoreDF,a])
        values=scoreDF.iloc[:,7:].sum(axis=0)
        tot=scoreDF.iloc[:,7:].sum(axis=0).sum()
        return values/tot
    if cat=='warm':
        scoreDF=pd.DataFrame()
        warm_line=np.load('./DATA/warm_line.npy')
        total_info=warm_line
        for x in total_info:
            a=DF.loc[DF.loc[:,'도로명']==x]
            scoreDF=pd.concat([scoreDF,a])
        values=scoreDF.iloc[:,7:].sum(axis=0)
        tot=scoreDF.iloc[:,7:].sum(axis=0).sum()
        return values/tot
    if cat==None:
        scoreDF=pd.DataFrame()
        accident=['아차산로(105)','교차로(580)','광나루로(26)','광나루로(30)','군자로(39)','교차로(57)','천호대로(12)',"['워커힐로', '영화사로'](0)",'용마산로5길(8)','강변북로(1)']
        warm_line=np.load('./DATA/warm_line.npy')
        accident.extend(warm_line)
        total_info=accident
        for x in total_info:
            a=DF.loc[DF.loc[:,'도로명']==x]
            scoreDF=pd.concat([scoreDF,a])
        values=scoreDF.iloc[:,7:].sum(axis=0)
        tot=scoreDF.iloc[:,7:].sum(axis=0).sum()
        print(values)
        return values/tot
# =======================================================================
# 예시 데이터: 병합 점수
'''
연속 일조량    2.222222
총 일조량     2.444444
경사도       2.33333333
거리        5.908774
공사난이도     3.911111
인구점수      0.777778
'''
# =======================================================================
def ratio_reflect(dataframe,ratio):
    reflect=dataframe.iloc[:,7:]*ratio
    dataframe=pd.concat([dataframe.loc[:,['도로명', '시작위도', '시작경도', '중간위도', '중간경도', '끝위도', '끝경도']],reflect.sum(axis=1)],axis=1)
    dataframe.rename(columns={0:'총점수'},inplace=True)
    rankDF=dataframe.sort_values(by='총점수',ascending=False)
    rankDF.reset_index(drop=True,inplace=True)
    return rankDF
# ======================================================================
def dataframe2csv(DF:pd.DataFrame,name):
    DF.to_csv(f'final_score_({name}).csv')
# ======================================================================
ratio=category_ratio()
print(ratio)
rank=ratio_reflect(DF,ratio)
# dataframe2csv(rank,'ac')    