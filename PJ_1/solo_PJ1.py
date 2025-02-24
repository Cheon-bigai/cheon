import pandas as pd
import matplotlib.pyplot as plt


file_inout=r"C:\Users\KDP-516\Desktop\KDT 7\3_VISUAL\DATA\총입수.xlsx"
file_population=r"C:\Users\KDP-516\Desktop\KDT 7\3_VISUAL\DATA\지역별인구수.xlsx"
file_pure=r"C:\Users\KDP-516\Desktop\KDT 7\3_VISUAL\DATA\순이동자수.xlsx"
#-------------------------------------
df_inout=pd.read_excel(file_inout)
df_population=pd.read_excel(file_population)
df_pure=pd.read_excel(file_pure)


df_inout.set_index(['전출지별','전입지별'],inplace=True)
df_population.set_index(['행정구역(시군구)별'],inplace=True)
df_pure.set_index(['전출지별','전입지별'],inplace=True)


df_inout=df_inout.iloc[1:,2:]
df_inout=df_inout.replace('-',0)
df_inout=df_inout.fillna(0)
df_inout=df_inout.astype('int')


df_pure=df_pure.iloc[1:,2:]
df_pure=df_pure.replace('-',0)
df_pure=df_pure.fillna(0)
df_pure=df_pure.astype('int')

df_population=df_population.iloc[1:,:]
df_population=df_population.astype('int')
#--------------------------------------------

df_population=df_population.T
df_population = df_population.rename_axis("연도", axis="index")

df_inout=df_inout.T
df_pure=df_pure.T
#-------------------------------------------------
df_inout_cut=df_inout.iloc[:,1:18]

df_inout_cut=df_inout_cut.reset_index()
#------------------------------------------------
df_inout_cut.set_index(['index'],inplace=True)
even_columns = df_inout_cut.iloc[:, 1::2]
