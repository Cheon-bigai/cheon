import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DF=pd.read_csv('./DATA/sun_location(0-2650-2700-3100-3200-5600-5700-6600-6850-).csv')
DF.columns=['도로명','시작위도','시작경도','연속 일조량','총 일조량','중간위도','중간경도','연속 일조량','총 일조량','끝위도','끝경도','연속 일조량','총 일조량']
DF['총 일조량']=DF['총 일조량'].applymap(lambda x:x.translate(str.maketrans({'시':'.','분':'','간':''})))
DF['연속 일조량']=DF['연속 일조량'].applymap(lambda x:x.translate(str.maketrans({'시':'.','분':'','간':''})))
DF['총 일조량']=DF['총 일조량'].applymap(lambda x:int(x.split('.')[0])*60+int(x.split('.')[1]))
DF['연속 일조량']=DF['연속 일조량'].applymap(lambda x:int(x.split('.')[0])*60+int(x.split('.')[1]))
# 3개 같은 위도 경도 제거
DF=DF.drop_duplicates(subset=['시작위도', '시작경도', '중간위도', '중간경도','끝위도', '끝경도'])
DF.reset_index(drop=True,inplace=True)
a=DF['연속 일조량'].sum(axis=1)/3
b=DF['총 일조량'].sum(axis=1)/3
newDF=pd.concat([DF[['도로명','시작위도', '시작경도', '중간위도', '중간경도','끝위도', '끝경도']],a.apply(lambda x:round(x,2)),b.apply(lambda x:round(x,2))],axis=1)
newDF.rename(columns={0:'연속 일조량',1:'총 일조량'},inplace=True)
count_road_listDF=pd.DataFrame()
for x in newDF['도로명'].unique():
    a=newDF.loc[newDF['도로명']==x,:]
    b=np.array(list(map(lambda x:'('+str(x)+')',range(a.shape[0]))))
    a['도로명']=(a['도로명']+b)
    count_road_listDF=pd.concat([count_road_listDF,a],axis=0)
count_road_listDF.reset_index(drop=True,inplace=True)
print(count_road_listDF)
def position_check(address):
    address
    print(DF.loc[DF['도로명'] in address,['시작위도','시작경도','중간위도','중간경도','끝위도','끝경도']])
total,bin_t=pd.cut(count_road_listDF['총 일조량'],bins=10,labels=range(0,10),retbins=True)
count_road_listDF['총 일조량']=total
max,bin_m=pd.cut(count_road_listDF['연속 일조량'],bins=10,labels=range(0,10),retbins=True)
count_road_listDF['연속 일조량']=max
print(count_road_listDF)
count_road_listDF.to_csv('count_road_info.csv',index=False)
# print(bin_m)
# plt.plot(max)
# plt.plot(pd.Series(total).astype('int')+10,alpha=0.5)
# plt.show()