# 하나의 좌표에 전체 좌표에 거리값을 계산
# axis=1 합치기
# cut으로 0부터 10까지 라벨링하기
# 처음 하나만들고 전부 해당 값에 더하기로 간다
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

DF=pd.read_csv('./DATA/sun_location(0-2650-2700-3100-3200-5600-5700-6600-6850-).csv')
DF.columns=['도로명','시작위도','시작경도','연속 일조량','총 일조량','중간위도','중간경도','연속 일조량','총 일조량','끝위도','끝경도','연속 일조량','총 일조량']
# 3개 같은 위도 경도 제거
DF=DF.drop_duplicates(subset=['시작위도', '시작경도', '중간위도', '중간경도','끝위도', '끝경도'])
DF.reset_index(drop=True,inplace=True)
DF=DF[['도로명','시작위도','시작경도','중간위도','중간경도','끝위도','끝경도']]
count_road_listDF=pd.DataFrame()
for x in DF['도로명'].unique():
    a=DF.loc[DF['도로명']==x,:]
    b=np.array(list(map(lambda x:'('+str(x)+')',range(a.shape[0]))))
    a['도로명']=(a['도로명']+b)
    count_road_listDF=pd.concat([count_road_listDF,a],axis=0)
count_road_listDF.reset_index(drop=True,inplace=True)
# ======================================================================================
# 복지시설 데이터 input: pd.DataFrame형식으로
# location=pd.DataFrame({'위도':[37.55009,37.5355857],'경도':[127.0876695,127.0745919]})
socialDF=pd.read_csv('./DATA/social.csv')
socialDF.dropna(inplace=True)
location=socialDF[['위도','경도']].applymap(lambda x:float(x))
location.info()
# ====================================================================================
total_location_distance_sum_list=[]
for idx in tqdm.tqdm(range(location.shape[0])):
    x=location.iloc[idx].values[0]
    y=location.iloc[idx].values[1]
    one_location_distance_sum_list=[]
    for test_idx in range(count_road_listDF.shape[0]):
        x_loc=count_road_listDF.iloc[test_idx].values[1::2]
        y_loc=count_road_listDF.iloc[test_idx].values[2::2]
        distance_sum=0
        for x1,y1 in zip(x_loc,y_loc):
            distance_sum+=np.sqrt((x1-x)**2 + (y1-y)**2)
        one_location_distance_sum_list.append(distance_sum.item())
    label,bin=pd.cut(pd.Series(one_location_distance_sum_list),bins=10,labels=range(9,-1,-1),retbins=True)   
    total_location_distance_sum_list.append(label.tolist())
distance_scoreDF=pd.DataFrame(total_location_distance_sum_list).T.sum(axis=1)
total_distanceDF=pd.concat([count_road_listDF,distance_scoreDF],axis=1)
total_distanceDF.rename(columns={0:'거리'},inplace=True)
total_distanceDF.to_csv('./DATA/social_distance.csv',index=False)
def position_check(address):
    address
    print(DF.loc[DF['도로명'] in address,['시작위도','시작경도','중간위도','중간경도','끝위도','끝경도']])
