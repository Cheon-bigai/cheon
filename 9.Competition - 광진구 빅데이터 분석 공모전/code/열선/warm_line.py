import pandas as pd
import numpy as np
# 거리 제일 가까운 지점 찾기
warmDF=pd.read_csv('./DATA/warm_line(1).csv')
DF=pd.read_csv('./DATA/sunslope_ver4.csv')
len9=[]
len11=[]
len13=[]
for x in range(warmDF.shape[0]):
    length=len(warmDF.iloc[x].dropna())
    if length==9:
        len9.append(warmDF.iloc[x].dropna().values[5:])
    if length==11:
        len11.append(warmDF.iloc[x].dropna().values[5:])
    if length==13:
        len13.append(warmDF.iloc[x].dropna().values[5:])
DF11=pd.DataFrame(len11)
DF9=pd.DataFrame(len9)
DF13=pd.DataFrame(len13)
DF11.drop(index=2,inplace=True)
DF11.reset_index(drop=True,inplace=True)
DF9.drop(index=[7,12],inplace=True)
DF9.reset_index(drop=True,inplace=True)
DF13.drop(index=1,inplace=True)
DF13.reset_index(drop=True,inplace=True)
name11=['영화사로13길','자양로50','긴고랑로41길']
name9=['용마산로22길','용마산로33길','용마산로28길','용마산로34길','용마산로32길',
       '용마산로22길','영화사로11길','능동로55길','용마산로36길(7)','능동로48길',
       '자양로44길','자양로13다길','뚝섬로59길','자양번영로8길','능동로32길','아차산로78길(19)',
       '아차산로73길','천호대로137길(9)','긴고랑로39길(9)','능동로17길(1)','용마산로8가길','자양로(51)','동일로56길(27)']
name13=['영화사로16길','아차산로78길(10)','용마산로30길(7)']
DF=(DF.set_index('도로명',drop=True))
total_warn_list=[]
for df,tag in zip([DF9,DF11,DF13],[name9,name11,name13]):
    address_list=[]
    for x,addr in zip(range(df.shape[0]),tag):
        dist_lat_long_list=[]
        for lat,long in zip(df.iloc[x,::2],df.iloc[x,1::2]):
            dist_list=[]
            for y in range(DF.shape[0]):
                dist=np.sum(np.sqrt(np.power(DF.iloc[y,:6:2].values-lat,2)+np.power(DF.iloc[y,1:6:2].values-long,2)))
                dist_list.append(dist)
            dist_lat_long_list.append(dist_list)
        a=pd.DataFrame(np.sum(np.array(dist_lat_long_list),axis=0),index=DF.index)
        b=DF.iloc[:,:6]
        c=pd.concat([b,a],axis=1)
        # print(c.sort_values(0).iloc[range(15)])
        # print(list(map(lambda x:x.split('(')[0],c.sort_values(0).iloc[range(15)].values)))
        # print('#'*100)
        for ad in addr:
            address=list(map(lambda x: x if ad in x and len(x.split(',')) !=2 else 0, c.sort_values(0).iloc[range(5)].index))
            address=[x for x in address if x !=0]
            address_list.extend(address) if len(address) !=0 else 1
            # print(address)         
    total_warn_list.append(address_list)
print([y for x in total_warn_list for y in x])
total=np.unique([y for x in total_warn_list for y in x])
# np.save('warm_line.npy',total)