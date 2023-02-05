import sys
sys.path.append(r'D:\Python')

from DS_Basic_Module import *


# Library 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



# 주로쓰는 항목 -------------------------
data0 = pd.read_clipboard(sep='\t')
dc = DataColumns()
cols = dc.cols_dict['format0']
cols = dc.cols_dict['format1']
cols = dc.cols_dict['format1_detail']
cols = dc.cols_dict['format2']
cols = dc.cols_dict['format2_detail']

data0[cols].to_clipboard(index=False)



# Read_Clipboard ---------------------------------------------------------------------------------
read_clipboard(copy=True, cut_string=7, return_type='list')
read_clipboard(copy=True, cut_string=7, return_type='str')
read_clipboard(copy=True, cut_string=7, return_type='series')
read_clipboard(copy=True, cut_string=7)

read_clipboard(copy=True, cut_string=9)
read_clipboard(copy=True, cut_string=10)
read_clipboard(copy=True)
read_clipboard(copy=True, return_type='list')
read_clipboard(copy=True, return_type='str')
read_clipboard(copy=True, return_type='series')



# Micro Plot ---------------------------------------------------------------------------------
from DS_Basic_Module import micro_plot, img_to_clipboard

df_sql = pd.read_clipboard(sep='\t')
df_sql
# df_sql = DB.result.copy()
df_sql['ZONE'] = pd.Categorical(df_sql['ZONE'], ['CT','LS','HS','SS','SCS','RCS','OAS','RHS','FCS','GA_Furnace', 'GA_IH', 'GI_ACE','SPM_EL','SPM_RF', 'SPM_RF_ST1','SPM_RF_ST2'], ordered=True)
df_sql.sort_values(['MTL_NO','ZONE','LEN_POS'], axis=0, inplace=True)
# df_sql.sort_values('LEN_POS', inplace=True)
df_sql_index = ', '.join(list(df_sql['MTL_NO'].value_counts().index))
print(df_sql_index); pyperclip.copy(str(df_sql_index))
# CQNC156, CQNB173, CQNB174

img_to_clipboard( micro_plot(df_sql.query("MTL_NO == 'CQNB173'")
                             ,line=dict(HS=785, SS=785, SCS=650, RCS=490), fill=10) )
# ---------------------------------------------------------------------------------------------


# 비교분석 (ttest, graph) ---------------------------------------------------------------------------
from DS_Basic_Module import ttest_each, distbox, violin_box_plot, hist_compare

data0 = pd.read_clipboard(sep='\t')

# t-test
result = ttest_each(data=data0, x='YP', group='열연공장')
result['plot'].iloc[0]

ttest_each(data=data0, x='YP', group='규격약호')

# distbox, violin_plot
distbox(data=data0, on='YP', group='열연공장')

violin_box_plot(data=data0, x='열연공장', y='YP')

# compare_hist 
col = 'YP'
data_g1 = data0.query("열연공장 == '광) 3열연'")[col]
data_g2 = data0.query("열연공장 == '광) 4열연'")[col]

hist_compare(data_g1, data_g2, label=['3열연', '4열연'], bins=30)
# ---------------------------------------------------------------------------------------------


# 공정능력 Graph --------------------------------------------------------------------------------
from DS_Basic_Module import Capability

data0 = pd.read_clipboard(sep='\t')


col = 'YP'
cpk_object = Capability(lsl=600, usl=740)

# cpk_object = Capability()
# cpk_object = Capability(lean=True)
# cpk_object.lsl = 600
# cpk_object.usl = 740


# Analysis
cpk_object(data0[col])
# cpk_object.performance_analysis(data0[col])
cpk_object(data0[col], display=True)


# Plot
fig = cpk_object.plot(data0[col], bins=30)
fig
img_to_clipboard(fig)


# 추가기능
cpk_object(data0[col])
cpk_object.cpk

cpk_object.usl_reject_prob


# 능력 검토
cpk_object.capability_analysis(data0[col])
cpk_object.capability_analysis(data0[col], cpk=[0.3, 0.5, 0.7, 1.0])


# pandas 적용
data0[col].agg(cpk_object)
# ---------------------------------------------------------------------------------------------






# Modeling 분석 -------------------------------------------------------------------------------
from DS_Basic_Module import FittedModel

data0 = pd.read_clipboard(sep='\t')

y_col = 'YP'
x_col = 'SS_POS'
FittedModel(data0[x_col], data0[y_col]).plot()
FittedModel(data0[x_col], data0[y_col]).fitted_data

fm =FittedModel(data0[x_col], data0[y_col])
fm.linear       # Linear 정보
fm.linear.formula       # 선형식
fm.metrics      # 모델성능
fm.plot()       # graph
fm.predict(data0[x_col])      # graph



from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
FittedModel(data0[x_col], data0[y_col], model=RF).plot()
# ---------------------------------------------------------------------------------------------





# 빈도분석 -------------------------------------------------------------------------------------
from DS_Basic_Module import Mode

data0 = pd.read_clipboard(sep='\t')

data0[['주문두께','주문폭']]   # 숫자형 Data → 통계값 추출가능

data0.groupby('규격약호')['주문두께'].mean()
data0.groupby('규격약호')['주문두께'].agg(['mean','std'])
data0.groupby('규격약호')[['주문두께','주문폭']].agg(['mean','std'])

data0['고객사_국가']   # 문자형 Data → ??? (빈도수 추출)
# data0.groupby('규격약호')['고객사_국가'].mean()

md = Mode(seq='all', return_type='list')
data0.groupby('규격약호')['고객사_국가'].agg(md)
data0.groupby('규격약호')['고객사_국가'].agg(md).apply(len)

md3 = Mode(seq=3, return_type='list')    # 갯수 점유비 기반 상위 3개고객사 
data0.groupby('규격약호')['고객사_국가'].agg(md3)

md3_str = Mode(seq=3, return_type=', ')    # 출력 type을 string으로 (구분자를 입력)
data0.groupby('규격약호')['고객사_국가'].agg(md3_str)

md3_str = Mode(seq=3, return_format='{i} ({c}, {round(p*100,1)}%)', return_type=', ')    # 출력 type을 string으로 (구분자를 입력)
data0.groupby('규격약호')['고객사_국가'].agg(md3_str)


# ---------------------------------------------------------------------------------------------




# 특별시험 -------------------------------------------------------------------------------------
from DS_Basic_Module import extract_coils

# 코일 추출 (Crawling)
ecs = extract_coils(html_from_clipboard=True)
# ecs = pd.read_clipboard()
ecs['parent_coil_no'] = ecs.coil_no.apply(lambda x: x[:7])
ecs_group = ecs.groupby(['parent_coil_no'])[['loc', 'date']].agg({'loc':'sum', 'date':'min'})
ecs_group.to_clipboard()


# 특별시험 결과정리
data9 = pd.read_clipboard(sep='\t')

# Tensile
special_test(data=data9, reverse=False)[['YP','TS','EL']].to_clipboard()

# her
special_test(data=data9, mode='HER', reverse=False).to_clipboard()
# ---------------------------------------------------------------------------------------------





# 생산이력 Size Plot ---------------------------------------------------------------------------
from DS_Basic_Module import cummax_summary

data_size = pd.read_clipboard(sep='\t')
cs_object = cummax_summary(data=data_size,x='주문폭',group='주문두께')

cs_object['data_agg'].to_clipboard()
# cs_object.keys()

fig = plt.figure()
for gi, gv in data_size.groupby("소둔공장"):
    plt.scatter(gv['주문두께'], gv['주문폭'], label=gi, alpha=0.3, color='skyblue')
plt.legend(loc='upper right')
plt.plot(cs_object['data_group_melt']['주문두께'], cs_object['data_group_melt']['value'])
plt.xlabel('주문두께')
plt.ylabel('주문폭')
plt.show()


img_to_clipboard(fig)
# ---------------------------------------------------------------------------------------------

# 
# hr_reduction_plot

# 공정능력 Group
# Group Capability ===========================================================
data0 = pd.read_clipboard(sep='\t')



# CapabilityGroup *** ---------------------------------
from DS_Basic_Module import CapabilityGroup
# CapabilityGroup?

criteria_df = pd.read_clipboard()
# criteria_df = pd.DataFrame({'열연공장': ['광) 3열연', '광) 4열연'],
#                 'YP': ['700~900', '650~850'],
#                 'TS': ['980~1100', '950~1050']})

criteria_df2 = criteria_df.set_index('열연공장')


cpk_group = CapabilityGroup()

# cpk_group.analysis?
cpk_group.analysis(data=data0, criteria=criteria_df2)

# cpk_group.analysis(data=data0, criteria=criteria_df2, hist_kwargs={'bins':30})
cpk_group.result
cpk_group.result['cpk_plot']
cpk_group.result['cpk_plot'][0]



# CapabilityData *** ---------------------------------
from DS_Basic_Module import CapabilityData
# CapabilityData?
cpk_data = CapabilityData()

# cpk_data.analysis?
cpk_data.analysis(data=data0, criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['12~']}, hist_kwargs={'bins':30})
cpk_data.result
cpk_data.result['cpk_plot']
cpk_data.result['cpk_plot'][0]

# manual range
cpk_data.analysis(data=data0, group='열연공장', criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['12~']})
cpk_data.analysis(data=data0, group='열연공장', criteria={'YP': ['700~900', '700~850'], 'TS':['980~'], 'EL':['12~']})
cpk_data.analysis(data=data0, group=['열연공장','규격약호'], criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['12~']})

# select column
cpk_data.analysis(data=data0, group='열연공장', criteria_column= {'YP':'YP_보증범위', 'TS':'TS_보증범위', 'EL':'EL_보증범위'})



