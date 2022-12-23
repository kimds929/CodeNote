import os
import math
import pandas as pd
import numpy as np
import missingno as msno
# import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
# matplotlib.pyplot 한글폰트 적용
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import seaborn as sns
# import plotnine
from IPython.display import clear_output
# clear_output(wait=True)
# import scipy.stats as stats

from DS_GroupAnalysis import *
from DS_OLS import *
from DS_Plot import *
from DS_Image import *
# NA = np.nan



# 불량재 정보 Summary -----------------------------------------------------------------------------------------------
df_reject = pd.read_clipboard()

df_reject = df_reject[df_reject['재질시험_대표구분']== 'BOT대표']
df_reject = df_reject[df_reject['재질시험_대표구분'].isna() == False]

df_reject['주문Size'] = df_reject.apply(lambda x: str(format(x['주문두께'],'.2f')) + 't × ' + str(format(x['주문폭'], ',d') + 'w'), axis=1)

x_list = ['MainKey_번호','제품사내보증번호', '고객품질요구기준번호', '냉연제조표준번호',
            '품종명', '강종_중구분', '규격약호', '출강목표', '냉연코일번호','주문두께', '주문폭', '주문Size', 'Mode변경재선정',
            '인장_방향', '인장_호수', '초_TS',
            '소둔작업완료일', '소둔_공장공정', '소둔_SS목표온도', '소둔_SPM목표',
            '소둔_LineSpeed', 'SS', 'SPM_EL', '냉연정정(RCL/MCL)공정', '냉연정정_SPM횟수', '냉연정정_SPM_EL', '누적SPM_EL',
            # '소둔_검사특기사항',
            'Ra조도_하한', 'Ra조도_상한'
            ]  # 열연제조표준번호

df_reject[x_list].to_clipboard(index=False)


# 시편번호 중복 제거
# df['SPCMN_Seq'] = df.groupby(by=['시편번호'])['제품번호'].transform(lambda x: x.rank())
# df['SPCMN_Seq'] = df.groupby(by=['시편번호'])['제품번호'].transform(lambda x: pd.Series(x.index).rank())
    # [ rank(method = 'average') ]
        # average(평균): 그룹의 평균 순위 부여 (예: 두 명이 공동 1등이라면 둘 다 1.5등으로 처리)
        # min(최솟값): 그룹에서 가장 낮은 순위 부여 (예: 두 명이 공동 1등이라면 둘 다 1등으로 처리)
        # max(최댓값): 그룹에서 가장 낮은 순위 부여 (예: 두 명이 공동 1등이라면 둘 다 2등으로 처리)
        # first(첫 번째): 그룹에서 표시되는 순서대로 순위 부여 (예: 두 명이 공동 1등이라면 순서가 빠른 사람을 1등으로 처리)
        # dense(밀도): min과 동일함. 다만 순위는 항상 1씩 증가


# test재 정보공유 -----------------------------------------------------------------------------------------------
test_mtl_df = pd.read_clipboard()
test_mtl_df_target = test_mtl_df.copy()

test_mtl_df_target = test_mtl_df[test_mtl_df['ORDER번호'].isin(['01S4328185010'])]

# PGBISDS
test_cols1 = ['ORDER번호', '규격약호N', '출강목표', '출강목표N', '용도N', 'Slab번호', '열연코일번호', '냉연코일번호',
 '진행재료_중량', '주문두께', '주문폭', '현공정N', '냉연재료진도', 'O/T', '고객사코드', '고객사명']
test_mtl_df_target[test_cols1].to_clipboard(index=False)

# PKMAS
test_cols2 = ['대표ORDER','규격약호',  '출강목표', '출강목표N', '용도N', 'Slab번호', '열연코일번호', '냉연코일번호',
 '냉연코일Net실평중량', '주문두께', '주문폭', '저장위치코드', '소둔_공장공정', 'O/T', '고객사코드', '고객사명']
test_mtl_df_target[test_cols2].to_clipboard(index=False)


# 재질시험 대표재만 추출 -----------------------------------------------------------------------------------------------
df = pd.read_clipboard()  #Clipboard로 입력하기 ★★

df['재질시험_대표구분'].value_counts()
df['재질시험_대표구분'].isna().sum()
df[df['재질시험_대표구분']== 'BOT대표'].to_clipboard(index=False)      # copy to clipboard
# df.shape
# df[df['재질시험_대표구분'].isna() == False].shape
df[df['재질시험_대표구분'].isna() == False].to_clipboard(index=False)      # copy to clipboard
# df[df['재질시험_대표구분'].isna() == False].to_clipboard()



# Group Analysis ----------------------------------------------------------------------------------------------------------
df = pd.read_clipboard()  #Clipboard로 입력하기 ★★
print(df.shape)
df.info()
# df.describe().T
df.head().T

print(df['소둔_공장공정'].value_counts().sort_index())
# 【 NULL 값 ROW Drop 】 ------------------------------------
df2 = df.copy()
msno.matrix(df[['재질시험_대표구분', '인장_방향', 'Mode변경재선정', '초_TS', 'SS']]); plt.show();
msno.bar(df[['재질시험_대표구분', '인장_방향', 'Mode변경재선정', '초_TS', 'SS']]); plt.show();
print(f"인장방향 NA : {df2['인장_방향'].isna().sum()}")
print(df2['인장_방향'].value_counts())
print(f"\nSS NA : {df2['SS'].isna().sum()}")
print(f"\n초_재질실적 NA : {df2['초_TS'].isna().sum()}")
print('')
print(df2['Mode변경재선정'].value_counts())

df2 = df2[df2['재질시험_대표구분'].isna() == False]
df2 = df2[df2['인장_방향'].isna() == False]
df2 = df2[df2['SS'].isna() == False]
df2 = df2[df2['초_TS'].isna() == False]
df2 = df2[df2['Mode변경재선정'].isna() == True]
# df2 = df2[df2['CGL_SPM_EL'].isna() == False]
print('')
print(f'df: {df.shape} → df2: {df2.shape}')
df2.to_clipboard(index=False)

# 【 1차 필터링 】 ------------------------------------
df5 = df2.copy()
print(df5['재질시험_대표구분'].value_counts())
print('')
print(df5['품종명'].value_counts())
print('')
print(pd.crosstab(df5['인장_방향'], df5['인장_호수']))

# df5 = df5[df5['품종_계획공정'] != 'PosMAC1.5) ']
df5 = df5[df5['재질시험_대표구분'].isin(['BOT대표'])]
# df5 = df5[df5['재질시험_대표구분'].isin(['TOP대표'])]
df5 = df5[df5['품종명'].isin(['CR', 'EG'])]     # 냉연
df5 = df5[df5['품종명'].isin(['GI', 'GA'])]     # 도금
df5 = df5[df5['품종명'].isin(['CR', 'EG', 'GI', 'GA'])]     # 냉연, 도금

print(pd.crosstab(df5['인장_방향'], df5['인장_호수']))
df5 = df5[df5['인장_방향'].isin(['L'])]
df5 = df5[df5['인장_방향'].isin(['C'])]
df5 = df5[df5['인장_호수'].isin([4])]
df5 = df5[df5['인장_호수'].isin([5])]
df5 = df5[df5['인장_호수'].isin([6])]
df5 = df5[df5['인장_호수'].isin([4,5])]
print(f'df5: {df2.shape} → df5: {df5.shape}')
# df5.to_clipboard(index=False)




# 【 이상치 처리 】 ------------------------------------
df6 = df5.copy()
    # X variable
# x = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적']
x = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'SS','누적SPM_EL']
x = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'SS', 'SPM_EL']
x = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적', '누적SPM_EL']

# Histogram
x_obj = fun_Hist(data=df6, x=x, figsize=[5,3], xtick=45, alpha=1, norm=True, color='skyblue')

df6 = fun_Outlier_Remove(data=df6, on=x, method='quantile', criteria=0.01)   # Outlier 제거
df6 = fun_Outlier_Remove(data=df6, on=x, method='quantile', criteria=0.02)   # Outlier 제거

fun_Img_To_Clipboard(x_obj['C_실적'])
fun_Img_To_Clipboard(x_obj['Si_실적'])
fun_Img_To_Clipboard(x_obj['Mn_실적'])
fun_Img_To_Clipboard(x_obj['P_실적'])
fun_Img_To_Clipboard(x_obj['SS'])
fun_Img_To_Clipboard(x_obj['SPM_EL'])
fun_Img_To_Clipboard(x_obj['누적SPM_EL'])


# Group별 이상치 제거
outlier_remove_df = pd.DataFrame()
for i, v in df6.groupby('출강목표'):
    index_list = list(df6.groupby('출강목표').groups.keys())
    print( f"Process: {round((index_list.index(i)+1)/len(index_list)*100, 1)}%" )
    part_df = fun_Outlier_Remove(data=v, on=x, method='quantile', criteria=0.02) 
    outlier_remove_df = pd.concat([outlier_remove_df, part_df], axis=0)
    clear_output(wait=True)
df6 = outlier_remove_df.copy()

    # Y variable
y = ['초_YP', '초_TS', '초_EL', '초_Ra조도', '초_R_bar']
y_obj = fun_Hist(data=df6, x=y, figsize=[5,3], xtick=45, alpha=1, norm=True, color='mediumseagreen')

df6['초_TS'].value_counts().sort_index()
df6['초_R_bar'].value_counts().sort_index()

fun_Img_To_Clipboard(y_obj['초_TS'])
df6 = df6[df6['초_TS']<550]
# df6 = fun_Outlier_Remove(data=df6, on=['초_EL'], method='quantile', criteria=0.001)   # Outlier 제거



df_describe = df6[x].describe().to_frame().T
df_describe

    # Sigma Outlier
df_Sigma = pd.DataFrame()
df_Sigma['Mean'] = df_describe['mean']
df_Sigma['Std'] = df_describe['std']
sigma_criteria = list(np.arange(-3,-0.5,0.5)) + list(np.arange(1,3.5,0.5))
for s in sigma_criteria:
    if s>0 :
        nameS = '+' + str(s) + 'σ'
    else :
        nameS = str(s) + 'σ'
    df_Sigma[nameS] = df_describe['mean'] + s * df_describe['std']
df_Sigma

    # IQR Outlier
df_IQR = pd.DataFrame()
IQR = df_describe['75%']-df_describe['25%']
df_IQR['IQR'] = IQR
df_IQR['Extreme_L'] = df_describe['25%'] - 3*IQR
df_IQR['Outlier_L'] = df_describe['25%'] - 1.5*IQR
df_IQR['Q25'] = df_describe['25%']
df_IQR['Q50'] = df_describe['50%']
df_IQR['Q75'] = df_describe['75%']
df_IQR['Outlier_U'] = df_describe['75%'] + 1.5*IQR
df_IQR['Extreme_U'] = df_describe['75%'] + 3*IQR
df_IQR

    # Quentile Outlier
df_Quentile = pd.DataFrame()
df_Quentile['Min'] = df_describe['min']
quantile_criteria = [0.01, 0.02, 0.03, 0.05, 0.10, 0.90, 0.97, 0.98, 0.99]
for q in quantile_criteria:
    if int(q*100) < 10:
        nameQ = '0' + str(int(q*100))
    else:
        nameQ = str(int(q*100))
    df_Quentile['Q'+nameQ] = df6[var_x].quantile(q)
df_Quentile['Max'] = df_describe['max']
df_Quentile


df6 = fun_Outlier_Remove(data=df6, on=['SS'], method='quantile', criteria=0.01)   # Outlier 제거
df6 = fun_Outlier_Remove(data=df6, on=['SPM_EL'], method='quantile', criteria=0.02)   # Outlier 제거
# pd.cut(df6['CGL_SS'], range(720,900,20), right=False).unique().sort_values()

# df6['두께그룹'] = pd.cut(df6['두께'], [0.6, 1.0, 1.5, 2.5], right=False)       # [0.5, 0.6, 1.0, 1.5, 2.5]
# df6['SS_Group20'] = fun_Cut_Group(df=df6, columns='SS', interval=20)
# df6['SPM_EL_Group02'] = fun_Cut_Group(df=df6, columns='SPM_EL', interval=0.2)
print(f'df5: {df5.shape} → df6: {df6.shape}')
# df6.head()



# 【 2차 필터링 】 ------------------------------------
df8 = df6.copy()
if list(set(df8['냉연정정(RCL/MCL)공정']))[0] == 'RCL/MCL':
    df8['냉연정정(RCL/MCL)공정'] = df8.apply(lambda x: np.nan if np.isnan(x['냉연정정_최종작업일시']) else
                        'MCL' if '223' in x['통과공정N'] and str(x['통과공정N']).index('223')%3 == 0 else
                        '1-1RCL' if '121' in x['통과공정N'] and str(x['통과공정N']).index('121')%3 == 0 else
                        '1-2RCL' if '122' in x['통과공정N'] and str(x['통과공정N']).index('122')%3 == 0 else
                        '1-3RCL' if '123' in x['통과공정N'] and str(x['통과공정N']).index('123')%3 == 0 else
                        '2-1RCL' if '221' in x['통과공정N'] and str(x['통과공정N']).index('221')%3 == 0 else
                        '2-2RCL' if '222' in x['통과공정N'] and str(x['통과공정N']).index('222')%3 == 0 else
                        '3-1RCL' if '321' in x['통과공정N'] and str(x['통과공정N']).index('321')%3 == 0 else
                        '3-2RCL' if '322' in x['통과공정N'] and str(x['통과공정N']).index('322')%3 == 0 else
                        '4-1RCL' if '421' in x['통과공정N'] and str(x['통과공정N']).index('421')%3 == 0 else
                        '4-2RCL' if '422' in x['통과공정N'] and str(x['통과공정N']).index('422')%3 == 0 else
                        '4-3RCL' if '423' in x['통과공정N'] and str(x['통과공정N']).index('423')%3 == 0 else
                        '기타', axis=1)
df8['냉연정정(RCL/MCL)공정'].value_counts().sort_index().to_frame()
pd.crosstab(df8['냉연제조표준번호'], df8['소둔_공장공정'])
pd.crosstab(df8['두께그룹'], df8['소둔_공장공정'])

df8['소둔_공장공정'].value_counts().sort_index()
df8['냉연제조표준번호'].value_counts().sort_index()
df8['가공경화N측정구분'].value_counts().sort_index()
# df8['Ra조도_하한'].value_counts()
df8['Ra조도_하한구분'].value_counts()
df8['규격약호'].value_counts()
df8['인장_호수'].value_counts()

df8['규격약호'].value_counts()

df8 = df8[df8['소둔_공장공정'] == '4-2CAL']
df8 = df8[df8['냉연제조표준번호']=='CTTKC41']
df8 = df8[df8['냉연제조표준번호'].isin(['AGZKE01', 'GGZKE01', 'GGZKE08', 'QGZKE01'])]
df8 = df8[df8['규격약호']=='EN-DX56D']
df8['Ra조도_하한구분'] = df8.apply(lambda x: 'E7 (Ra ≥0.9)' if x['Ra조도_하한']>= 0.9 else 'E5 (Ra <0.9)', axis=1)
df8['Ra조도_보증범위'] = df8.apply(lambda x: str(x['Ra조도_하한']) + '~' + str(x['Ra조도_상한'])
                        if x['Ra조도_하한']>0 or x['Ra조도_상한']>0 else '', axis=1 )
df8 = df8[df8['Ra조도_보증범위'] != '']
df8 = df8[df8['규격약호']=='CSP3X-E']
df8 = df8[df8['가공경화N측정구분']==4]
df8 = df8[df8['인장_호수']==4]
df8 = df8[df8['냉연정정(RCL/MCL)공정'].isin(['1-1RCL'])==False]
df8 = df8[df8['냉연정정(RCL/MCL)공정'].isin(['1-1RCL','MCL'])==False]
# df8 = df8[df8['정정_최종작업완료일시'].isna()==True]

# df8.groupby('CGL_SPM_EL그룹').count().iloc[:,0]
print(f'df6: {df6.shape} → df8: {df8.shape}')


# 【 3차 필터링 】 ------------------------------------
df9 = df8.copy()
df9['두께그룹'].value_counts().sort_index()
df9['주문두께'].hist()
round(df9['주문두께'].describe(),2)
# df9 = df10.copy()
df9['품종명'].value_counts()
pd.crosstab(df9['인장_방향'], df9['인장_호수'])
pd.crosstab(df9['품종명'], df9['인장_방향'])

df9 = df9[df9['품종명'].isin(['CR', 'EG'])]     # 냉연
df9 = df9[df9['품종명'].isin(['GI', 'GA'])]     # 도금

df9 = df9[df9['인장_방향'].isin(['L'])]
df9 = df9[df9['인장_방향'].isin(['C'])]
df9 = df9[df9['인장_호수'].isin(['4'])]
df9 = df9[df9['인장_호수'].isin(['5'])]
df9 = df9[df9['인장_호수'].isin(['6'])]

df9 = df9[(0.6 <= df9['주문두께']) & (df9['주문두께'] < 1.5)]
df9 = df9[(1.0 <= df9['주문두께']) & (df9['주문두께'] < 2.5)]
df9 = df9[df9['두께그룹'] == '0.40 ~ 0.49']
df9 = df9[df9['두께그룹'] == '0.50 ~ 0.59']
df9 = df9[df9['두께그룹'] == '0.60 ~ 0.99']
df9 = df9[df9['두께그룹'] == '1.00 ~ 1.49']
df9 = df9[df9['두께그룹'] == '1.50 ~ 2.49']

df9 = df9[df9['Ra조도_하한']>= 0.9]
df9 = df9[df9['Ra조도_하한'].isin([0.9, 1.0])]
print(f'df8: {df8.shape} → df9: {df9.shape}')
# -----------------------------------------


#     # box-plot
y = '초_TS'
x = '두께그룹'
df_box =  df5
# df_box =  df_final

# df_box[[y, x]].boxplot(by=x)

df_box[x].value_counts().sort_index().to_frame()
box_data = fun_Group_Array(data=df_box, x=y, group=x)
OLS_Model = sm.OLS.from_formula(y + '~' + x, data=df_box).fit()
pValue = format(sm.stats.anova_lm(OLS_Model)['PR(>F)'][0], '0.3f')
plt.boxplot(box_data['value'], labels=box_data['index'])
plt.scatter(x=range(1,len(box_data['index'])+1), y=[i.mean() for i in box_data['value']], color='r')
plt.title(y + '~' +x + ' / p-value: ' +pValue)
plt.show()
[i.mean() for i in box_data['value']]

for idx in df_box[x].value_counts().index:
    sns.distplot(df_box[df_box[x]==idx][y], label=idx)
plt.legend()
plt.show()
# fun_Hist(data=df_box, x=y, group=x, group_type='identity', norm=True, density=True, alpha=0.5)

mtc_Plot = fun_Hist(data=df_final, x=['TS', 'YP', 'EL', 'HD_Avg'],
             spec={'TS':[285,400], 'YP':[200,305], 'EL':[32], 'HD_Avg':[]},
              norm=True, alpha=0.7, color='mediumseagreen', title='0.50~0.59t')

fun_Img_To_Clipboard(mtc_Plot['HD_Avg'])


df['주문두께'].hist()

# 【 Pivot Groupby 】 ------------------------------------
target_idx = []
target_idx += list(df_group.groups[('C', '0.60 ~ 0.99', 0.5, 1.5)] )
target_idx += list(df_group.groups[('C', '0.60 ~ 0.99', 0.6, 1.2)] )
target_idx += list(df_group.groups[('C', '0.60 ~ 0.99', 1.1, 1.7)] )
df_target = df_final.loc[target_idx]
df_target.shape

# df_final = df_target.copy()
# df_final.shape

# df_final = pd.read_clipboard()
df_final = df6
df_final = df8
df_final = df9
# df_final = df
# df_final.to_clipboard(index=False)
# df6['소둔_공장공정'].value_counts()


# df_final.columns.tolist()
var_group = ['두께그룹'] 
var_group = ['두께그룹', 'SS_Group20'] 
var_group = ['두께그룹','누적SPM_EL_Group02']
var_group = ['두께그룹','Ra조도_하한','Ra조도_상한']

    # '소둔_공정실적','인장_방향'
    # 출강목표, 인장_방향, 두께그룹, SPM_EL_Group02, 누적SPM_EL_Group02, SS_Group20, 소둔_표면마무리지정코드, 누적SPM횟수, Ra조도_하한구분
# var_group = ['CustomerType','Gender']
# var_group = ['소둔_공정실적', '출강목표', '인장_방향', '두께그룹']

# df_group = df_target.groupby(var_group)      # Grouped DataFrame (target)

df_group = df_final.groupby(var_group)      # Grouped DataFrame (final)
group_key = list(df_group.groups.keys())

      # Group Count ***
groupCount_df = df_group.count().iloc[:,0].rename('count')
groupCount_df = groupCount_df.to_frame()
groupCount_df.columns = [['Total']*len(groupCount_df.columns), groupCount_df.columns]
groupAnalysis_df = groupCount_df.copy()
groupAnalysis_df

# fun_OLS(data=df_final, y='EL', x=['SS','주문두께']).to_clipboard()


    # Group X_value ***
groupX_df = groupAnalysis_df.copy()
# groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='Mn_실적', count=False)   # + Mean/Std
# groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='주문폭', count=False)   # + Mean/Std
groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='소둔_LineSpeed', count=False)   # + Mean/Std
groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='SS', count=False)   # + Mean/Std
groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='누적SPM_EL', count=False)   # + Mean/Std
# groupX_df = fun_Concat_Group_MeanStd(base=groupX_df, groupby=df_group, on='SPM_EL', count=False)   # + Mean/Std
groupAnalysis_df = groupX_df.copy()
groupAnalysis_df
groupAnalysis_df.T


    # Group Y_value ***
Y_criteria = [120, 170]
# Y_criteria = [2.0,9.9]
Y_criteria = [0.22,0.99]
Y_Var = '초_가공경화_N지수'
# '초_YP', '초_TS', '초_EL', '초_YR', '초_R_bar', '초_R90', '초_가공경화_N지수', '초_가공경화_N90', '초_Ra조도', '초_WA조도', '초_PPC조도'
# Y_Var = ['YP', 'TS', 'EL', 'YR', 'R_bar', 'R0', 'R45','R90', '가공경화_N지수','HD_Avg']

Y_group_df = groupAnalysis_df.copy()
Y_group_df = fun_Concat_Group_MeanStd(base=Y_group_df, groupby=df_group, on=Y_Var)   # + Mean/Std
Y_group_df = fun_Concat_Group_Cpk_Range(base=Y_group_df, on=Y_Var, criteria=[0.7, 0.8, 0.9, 1.0], n_min=5 )   # + Cpk_Range
if type(Y_criteria[0]) == list:
    for Y_c in Y_criteria:
        Y_group_df = fun_Concat_Group_Cpk(base=Y_group_df, on=Y_Var, criteria=Y_c, n_min=5 )   # + Cpk
        Y_group_df = fun_Concat_Group_Reject(base=Y_group_df, groupby=df_group, on=Y_Var, criteria=Y_c) # + Reject
else : 
    Y_group_df = fun_Concat_Group_Cpk(base=Y_group_df, on=Y_Var, criteria=Y_criteria, n_min=5 )   # + Cpk
    Y_group_df = fun_Concat_Group_Reject(base=Y_group_df, groupby=df_group, on=Y_Var, criteria=Y_criteria) # + Reject
groupAnalysis_df = Y_group_df.copy()
groupAnalysis_df.T
groupAnalysis_df = fun_Drop_Var_Count(base=groupAnalysis_df, on=Y_Var)   # Drop Var-Count if it same Total-Count
groupAnalysis_df.T

groupAnalysis_df.drop(Y_Var, axis=1, inplace=True)

groupAnalysis_df.to_clipboard()
groupAnalysis_df.T.to_clipboard()


    # Group + Regression ***
# df_final.columns.tolist()
var_group = ['두께그룹']     
    # 출강목표, 인장_방향, 두께그룹, 누적SPM_EL_Group02, SS_Group20, 소둔_표면마무리지정코드, 누적SPM횟수,Ra조도_하한
# var_group = ['CustomerType','Gender']

df_group = df_final.groupby(var_group)      # Grouped DataFrame
group_key = list(df_group.groups.keys())

      # Group Count ***
groupCount_df = df_group.count().iloc[:,0].rename('count')
groupCount_df = groupCount_df.to_frame()
groupCount_df.columns = [['Total']*len(groupCount_df.columns), groupCount_df.columns]
groupCount_df

    # GroupOLS df Selection
groupOLS_Base = groupCount_df     # ***
# groupOLS_df = groupMeanStd_df
# groupOLS_df = groupCpk_df
# groupOLS_df = groupRejct_df
groupOLS_Base

groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_YP', x='누적SPM_EL', const=True)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_Ra조도', x='누적SPM_EL', const=True)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_가공경화_N지수', x='누적SPM_EL', const=True, random_state=0)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_가공경화_N90', x='누적SPM_EL', const=True, random_state=0)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_YR', x='SPM_EL', const=True)
# groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_YP', x='초_Ra조도', const=True)

groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_TS', x='SS', const=True)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_R_bar', x='SS', const=True, random_state=0)
groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_EL', x='SS', const=True)
# groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_TS', x='초_EL', const=True)
# groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y='초_TS', x='초_R_bar', const=True)

groupOLS_df
groupOLS_df.T

groupOLS_df.to_clipboard()

OLS_Result = fun_Group_OLS(groupby=df_group, y='초_YP', x='누적SPM_EL', const=True)
OLS_Result = fun_Group_OLS(groupby=df_group, y='초_Ra조도', x='누적SPM_EL', const=True)
OLS_Result = fun_Group_OLS(groupby=df_group, y='초_R_bar', x='SS', const=True)
OLS_Result = fun_Group_OLS(groupby=df_group, y='초_EL', x=['주문두께','SS'], const=True)
OLS_Result['result']
OLS_Result['result'].to_clipboard()
# OLS_Result[ ('AGZKE01', 'L', '0.60 ~ 0.99',  'E7')]['data'].to_clipboard()

# kFold
kFoldResult= fun_kFold_OLS(data=df_final.loc[df_group.groups[group_key[2]]], y='초_Ra조도', x='누적SPM_EL', kFoldN=5)
kFoldResult['result']

    # Group Summary ***
groupSummary_df = groupAnalysis_df
groupSummary_df = groupOLS_df
# groupSummary_df = groupSummary_df[groupSummary_df['count'].isna() == False]
groupSummary_df = groupSummary_df[groupSummary_df['Total']['count'] >= 5]
groupSummary_df = groupSummary_df[groupSummary_df.iloc[:,1] != '']

# groupSummary_df.sort_index(axis=1, inplace=True)

groupSummary_df.to_clipboard()
groupSummary_df.index.to_frame(index=False) # name=['a','b','c'] : Columns-Name


# OLS_Model = fun_OLS(data=df_final, y='초_TS', x=['SS'])
# OLS_Model.T
# fun_kFold_OLS(data=df_part_Object['0.40 ~ 0.49'], y='초_EL', x='SS', const=True)['result']



# Group Fitted_Plot
groupOLS_Fitted_Plot = fun_Group_OLS_Plot(df=df_final, y='초_YP', x='누적SPM_EL', group=var_group,
        figsize = [5,3],
        PointPlot=True, fitLine=True, histY=True, histX=True,
        specY=[140,180],   # specY = [1.5, 1.9]    # specY = {'0.60 ~ 0.99': [43,51]}
        # specY = {group_key[0]: [140,280], group_key[1]: [140,260], group_key[2]: [140,240,260],
        #         group_key[3]: [140, 240], group_key[4]: [140,240]},
        specX=[],
        spec_display='auto',
        lineX=[],
        lineY=[],
        xlim=False,
        ylim=False,
        )
# '초_YP', '초_TS', '초_EL', '초_YR', '초_R_bar', '초_R90', '초_가공경화_N지수', '초_가공경화_N90', '초_Ra조도', '초_WA조도', '초_PPC조도'
# 'SS', 'SPM_EL', '누적SPM_EL'


n = -1      # start***
n -=1; print(f"{n} : {group_key[n]}")
n +=1; print(f"{n} : {group_key[n]}")
fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot'][group_key[n]]['scatter'])
fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot'][group_key[n]]['histY'])
fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot'][group_key[n]]['histX'])


# fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot']['0.40 ~ 0.49'])
# fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot']['0.50 ~ 0.59'])
# fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot']['0.60 ~ 0.99'])
# fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot']['1.00 ~ 1.49'])
# fun_Img_To_Clipboard(groupOLS_Fitted_Plot['plot']['1.50 ~ 2.49'])


















    # Filtering 기준별 ---------------------------

# Clipboard Data to python ***
df_target = pd.read_clipboard()        # Base DataSet
df_filter = pd.read_clipboard()     # Condition DF
df_filter
# Filtering Criteria apply ***
result01 = fun_Filtering_DataFrame(data=df_target, criteria=df_filter, unique_condition=False)
print(result01['summary'])
print(result01['result']['criteria_index'].value_counts().sort_index())
# fun_Filtering_Column(data=result01['result'], on='criteria_index', criteria=0)

df_target1 = result01['result'].copy()
# df_target_final = result01['result'].copy()
df_target2 = df_target1[(~df_target1['냉연정정(RCL/MCL)공정'].isin(['1-1RCL', 'MCL'])) | (df_target1['냉연정정(RCL/MCL)공정'].isin(['1-1RCL', 'MCL']) & df_target1['냉연정정_SPM_EL'] > 0)]
df_group2 = df_target2.groupby('criteria_index')


# Remove Outlier by Filtering condition ***
df_target3 = pd.DataFrame()
n = 1
for i, v in df_group2:
        print(f'processing : {round(n/df_group2.ngroups*100,1)}%')
        v_part = fun_Outlier_Remove(data=v, on=['C_실적', 'SS', '누적SPM_EL'], criteria=0.01, method='quantile')
        df_target3 = pd.concat([df_target3, v_part], axis=0)
        n += 1
        clear_output(wait=True)

len(df_target3)
print(df_target3['criteria_index'].value_counts().sort_index())
df_target3.to_clipboard(index=False)


# group별 조업실적값
fun_Search(data=df_target2, on='사내보증')
df_target1.groupby(['criteria_index', '제품사내보증번호']).agg({'냉연코일번호':'count'}).to_clipboard()

fun_Search(data=df_target2, on='SPM')
df_group21 = df_group2.agg({'냉연코일번호':'count', '소둔_SPM목표':['min','max'], 'SPM_EL' : ['mean'], '누적SPM_EL': 'mean'})
df_group22 = df_target2.groupby(['criteria_index', '소둔_SPM목표']).agg({'냉연코일번호':'count','소둔_SPM목표':['min','max'], 'SPM_EL' : ['mean'], '누적SPM_EL': 'mean'})
df_group22


df_group21.to_clipboard()

df_target_final = df_target3.copy()

# .....
# YP = fun_Performance_Capability_Row(df=df_target_final, group=['criteria_index'], mtc_group=True, on=['YP'], cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=True)
# YP.to_clipboard()

# TN = fun_Performance_Capability_Row(df=df_target_final, group=['criteria_index'], mtc_group=True, on=['가공경화_N지수'], cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=True)
# TN.to_clipboard()



# Fitted_Plot ***
k = 17
y_val = '초_가공경화_N지수'             # 초_가공경화_N지수
# '초_YP', '초_TS', '초_EL', '초_YR', '초_R_bar', '초_R90', '초_가공경화_N지수', '초_가공경화_N90', '초_Ra조도', '초_WA조도', '초_PPC조도'
# 'SS', 'SPM_EL', '누적SPM_EL'
x_val = '누적SPM_EL'
# y_criteria_df = pd.read_clipboard()             # 보증기준 Table
# y_criteria = list(y_criteria_df.iloc[k,])
y_criteria = '가공경화N_하한'

df_target_plot = df_target_final[df_target_final['criteria_index'].apply(lambda x: str(k) in str(x).split(', '))]
df_target_plot[y_criteria].value_counts()


df_target_Fitted_Plot = fun_Group_OLS_Plot(df=df_target_plot, y=y_val, x=x_val,
        figsize = [4,2.3],
        PointPlot=True, fitLine=True, histY=True, histX=True,
        specY= y_criteria,
        specX=[],
        spec_display='auto',
        lineX=[],
        lineY=[],
        xlim=False,
        ylim=False,
        )
fun_Img_To_Clipboard(df_target_Fitted_Plot['plot']['total']['histY'])

pd.crosstab(df_target_plot['규격약호'], df_target_plot['출강목표'])
len(df_target_plot)
# print(df_target_Fitted_Plot['OLS'])
df_target_Fitted_Plot['OLS'].T.set_index(['total']).T.to_clipboard(index=False)          # OLS결과만 Clipboard로

fun_Img_To_Clipboard(df_target_Fitted_Plot['plot']['total']['scatter'])
fun_Img_To_Clipboard(df_target_Fitted_Plot['plot']['total']['histY'])
fun_Img_To_Clipboard(df_target_Fitted_Plot['plot']['total']['histX'])





















    # Group DataFrame ------------------------------------------------------
x = 'Ra조도'
spec_df = pd.DataFrame()
spec_dict ={}
for i,v in df_group:
    spec_dict[x + '_하한'] = v[x + '_하한'].max()
    spec_dict[x + '_상한'] = v[x + '_상한'].min()
    idx = []
    spec_df_part = pd.DataFrame([spec_dict], index =[idx+[c] for c in i])
    spec_df = pd.concat([spec_df, spec_df_part], axis=0)
spec_df.index.names = ['출강목표']
# '소둔공정', '출강목표','인장방향','두께그룹'

spec_df.to_clipboard()
spec_df

spec_df_part[x + '_mean'] = v[x].mean()
spec_df_part[x + '_std'] = v[x].std()




