# from six.moves import cPickle
# origin_file_path = r'D:\WorkForPython\DB\Data_Tabular'
# df =cPickle.load(open(f"{origin_file_path}/230612 19-22년 980DP 전체 생산이력.pkl", 'rb'), encoding='utf-8-sig')

# df = df[df['재질시험_대표구분'].apply(lambda x: x in ['BOT대표 - POS', 'TOP대표 - POS'])]
# df = df[df['소둔작업완료일시'].apply(lambda x: not pd.isna(x))]
# df = df[df['출강목표N'].apply(lambda x: x in ['C070250H3SP601', 'C070250H3SP604', 'C070230H3SP601', 'C100280L5XX001'])]
# df.to_csv(f'{origin_file_path}/SampleData_980DP_semifull.csv', encoding='utf-8-sig', index=False)

# df00 = pd.read_csv(f"{origin_file_path}/SampleData_980DP_semifull.csv", encoding='utf-8-sig')
# df[columns].to_csv(f'{origin_file_path}/SampleData_980DP.csv', encoding='utf-8-sig', index=False)

import sys
sys.path.append(r'D:\WorkForPython\00_DataAnalysis_Basic')
sys.path.append(r'D:\WorkForPython\DS_Library')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DS_Basic_Module import search, DF_Summary, Outlier, PredictDL
import missingno as msno


file_path = r'D:\WorkForPython\DB\Data_Education'
df00 = pd.read_csv(f"{file_path}/SampleData_980DP_Analysis.csv", encoding='utf-8-sig')
df00.shape



# 【 Feature Selection 】 ###################################################
columns = []
columns_order = ['강종_소구분', '품종명', '소둔작업완료일시', '규격약호', '출강목표N', '재질시험_대표구분']
columns_size =  ['주문두께', '소둔_폭']
columns_target = ['YP','TS', 'EL', 'BMB']
columns_criteria = ['YP_보증범위', 'TS_보증범위', 'EL_보증범위']
columns_criteria_cond = ['인장_방향', '인장_폭방향', '인장_호수', 'BMB_방향', 'BMB_폭방향']
columns_craim = ['소둔_HS목표온도', '소둔_SS목표온도', '소둔_SCS목표온도', '소둔_RCS목표온도', 
           '소둔_RHS목표온도', '소둔_OAS목표온도', '소둔_SPM_EL목표']
columns_cr = ['LS_POS', 'HS_POS', 'SS_POS', 'SCS_POS', 'RCS_POS', 'RHS_POS','OAS_POS', 'FCS_POS', 'SPM_RollForce_ST1', 'SPM_RollForce_ST2']
columns_hr = ['SRT', 'FDT', 'CT']
columns_alloy = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적', 'SolAl_실적', 'TotAl_실적', 'Cu_실적', 'Nb_실적',
                'B_실적', 'Ni_실적', 'Cr_실적', 'Mo_실적', 'Ti_실적', 'V_실적', 'Sn_실적', 'Ca_실적', 'Sb_실적', 'N_실적']
columns_fac = ['PCM공장', '소둔공장', '열연공장', '제강공장']

columns = columns_order + columns_size + columns_target + columns_criteria + columns_criteria_cond \
        + columns_craim + columns_cr + columns_hr + columns_alloy + columns_fac

df00[columns].head(3).T

# ###################################################
# df01 = df00[columns]
# df02_YP = df01.copy()
# df02_YP = df02_YP[df02_YP['출강목표N'].isin(['C070250H3SP601', 'C070250H3SP604'])]      # 출강목표 : C070230H3SP601
# df02_YP = df02_YP[df02_YP['품종명'] == 'CR']     # 품종 : CR
# df02 = df02_YP.copy()
# df02 = df02[[c for c in df02.columns if c !='RHS_POS']]
# df02 = df02.dropna()
# df03 = df02.copy()
# outlier = Outlier(df03, sigma=4, method=['sigma', 'of_box'])
# df03 = outlier.fit_transform()
# df03 = df02.loc[df03.index]
# df04 = df03.copy()

# 【 EDA 】 ###################################################
pd.set_option('display.max_rows', 100)
# pd.reset_option('display.max_rows')
df01 = df00[columns]
df01_summary = DF_Summary(df01)
df01_summary.summary_plot()
# df01_summary.summary.to_clipboard()



# 【 모델링 대상 필터링 】 ###################################################
# Categorical Value
categorical_columns = [k for k,v in df01.dtypes.items() if v == object]
categorical_columns


# 고YS 980DP
df02_YP = df01.copy()
df02_YP = df02_YP[df02_YP['출강목표N'].isin(['C070250H3SP601', 'C070250H3SP604'])]      # 출강목표 : C070230H3SP601
DF_Summary(df02_YP[categorical_columns], n_samples=500)

df02_YP = df02_YP[df02_YP['품종명'] == 'CR']     # 품종 : CR
# df02_YP = df02_YP[df02_YP['소둔공장'] == '2CAL']     # 소둔공장 : 2CAL


# # 저CEQ 980DP
# df02_CEQ = df01.copy()
# df02_CEQ = df02_CEQ[df02_CEQ['출강목표N'] == 'C070230H3SP601']      # 출강목표 : C070230H3SP601
# DF_Summary(df02_CEQ[categorical_columns], n_samples=500)

# df02_CEQ = df02_CEQ[df02_CEQ['소둔공장'] == '5CGL']     # 소둔공장 : 5CGL
# df02_CEQ = df02_CEQ[df02_CEQ['품종명'] == 'GA']     # 품종 : GA

# 고연신 980DP
df02_EL = df01.copy()
df02_EL = df02_EL[df02_EL['출강목표N'].isin(['C100280L5XX001'])]      # 출강목표 : C100280L5XX001
DF_Summary(df02_EL[categorical_columns], n_samples=500)

df02_EL = df02_EL[df02_EL['품종명'] == 'CR']     # 품종 : CR
# df02_EL = df02_EL[df02_EL['소둔공장'] == '2CAL']     # 소둔공장 : 2CAL



# 【 결측치 처리 】 ###################################################
df02_bef = df02_YP.copy()
# df02_bef = df02_CEQ.copy()
# df02_bef = df02_EL.copy()


df02 = df02_bef.copy()

DF_Summary(df02)
msno.matrix(df02, labels=list(df02.columns), label_rotation=90)

# RHS column 제거
df02 = df02[[c for c in df02.columns if c not in ['RHS_POS']]]


# 결측치 row 제거
df02 = df02.dropna()

# 결측치 처리 후 data상태 확인
df02_summary = DF_Summary(df02)
# df02_summary.summary_plot()
msno.matrix(df02, labels=list(df02.columns), label_rotation=90)

print(f"{df02_bef.shape} → {df02.shape}")




# 【 이상치 처리 】 ###################################################
df03 = df02.drop(['BMB','SPM_RollForce_ST1','SPM_RollForce_ST2'], axis=1)
outlier = Outlier(df03, sigma=5, method=['sigma', 'of_box'])
# outlier.outlier_plot()
df03 = outlier.fit_transform()
# outlier.outlier_plot(df03)

# apply
df03 = df02.loc[df03.index]

print(f"{df02.shape} → {df03.shape}")


# df03.to_csv(f"{file_path}/SampleData_980DP_YS_Modeling.csv", index=False, encoding='utf-8-sig')
# df03.to_csv(f"{file_path}/SampleData_980DP_EL_Modeling.csv", index=False, encoding='utf-8-sig')









# 【 비교분석 】 ###################################################
# 고YS_980DP CR재 소둔공장에 따른 재질차이 분석
from DS_Basic_Module import ttest_each, distbox, group_plots

df04 = df03.copy()
df04['소둔공장'].value_counts()

# search(df04, 'SS')
distbox(df04, on='YP', group=['소둔공장','열연공장'])
distbox(df04, on='YP', group='소둔공장')
distbox(df04, on='SS_POS', group='소둔공장')
distbox(df04, on='YP', group='소둔_SS목표온도')
distbox(df04, on='SS_POS')

distbox(df04, on='YP', group=['소둔공장', '소둔_SS목표온도'])



# 1 group
group_plots(df04, x='YP', group='소둔공장', 
        box_plot=True, violin_plot=False, norm_dist=False, hist_plot=False)
group_plots(df04, x='YP', group='소둔공장', 
        box_plot=False, violin_plot=False, norm_dist=True, hist_plot=True)
group_plots(df04, x='SS_POS', group='소둔공장', 
        box_plot=False, violin_plot=True, norm_dist=False, hist_plot=True)

# multiple groups
group_plots(df04, x='YP', group=['소둔공장', '소둔_SS목표온도'],
        box_plot=False, violin_plot=True, norm_dist=False, hist_plot=True, xlabel_rotation=45)
group_plots(df04, x='YP', group='소둔_SS목표온도',
        box_plot=False, violin_plot=True, norm_dist=False, hist_plot=True, xlabel_rotation=45)

# ttest
ttest_result1 = ttest_each(df04, x='YP', group='소둔공장')
ttest_result1
ttest_result1['plot'].iloc[0]

ttest_result2 = ttest_each(df04, x='YP', group='소둔_SS목표온도')
ttest_result2
ttest_result2['plot'].iloc[0]

ttest_result3 = ttest_each(df04, x='YP', group=['소둔공장', '열연공장'])
ttest_result3
ttest_result3['plot'].iloc[0]



# 【 공정능력통합분석 】 ###################################################
# CapabilityData *** ---------------------------------
from DS_Basic_Module import CapabilityData
# CapabilityData?
cpk_data = CapabilityData()

# cpk_data.analysis?
cpk_data.analysis(data=df04, criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['8~']}, hist_kwargs={'bins':30})
cpk_data.result
cpk_data.result['cpk_plot']
cpk_data.result['cpk_plot'][2]

# manual range
cpk_data.analysis(data=df04, group='소둔공장', criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['12~']})
cpk_data.analysis(data=df04, group='소둔공장', criteria={'YP': ['700~900', '700~850'], 'TS':['980~'], 'EL':['12~']})
cpk_data.analysis(data=df04, group=['소둔공장','열연공장'], criteria={'YP': ['700~900'], 'TS':['980~'], 'EL':['12~']})

# select column
cpk_data.analysis(data=df04, group='소둔공장', criteria_column= {'YP':'YP_보증범위', 'TS':'TS_보증범위'})



