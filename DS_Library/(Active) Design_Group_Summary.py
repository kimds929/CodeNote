import numpy as np
import pandas as pd
import missingno as msno

from IPython.display import clear_output

from DS_GroupAnalysis import *
from DS_OLS import *
from DS_Plot import *
from DS_Image import *

# DataSet
df_design = pd.read_clipboard()
df_design.shape
df_design.info()

# df_design = df2.copy()

df_design['소둔작업일'] = df_design['소둔작업완료일'].apply(lambda x: str(x)[0:6])
df_design['소둔작업일'] = df_design['소둔작업일'].astype(int)
df_design['소둔작업년월'] = df_design['소둔작업완료일'].apply(lambda x: x[0:4])
df_design['소둔_공정구분'] = df_design['소둔_공장공정'].apply(lambda x: (x.replace('CAL', '공정') if x in ['1CAL', '2CAL'] else 
                            ('3공정' if x == '3-2CAL' else 
                                ('4공정' if x == '4-1CAL' else
                                    ('5공정' if x == '4-2CAL' else
                                       '6공정' if x == '3-1CAL' else np.nan
                                    )))
                        ) if 'CAL' in x else x.replace('CGL', '공정') )
df_design['인장_호수그룹_길이'] = df_design['인장_호수'].apply(lambda x: '05 (80)' if x==5 else '04, 06 (50)' )
df_design['열연코일_폭_그룹'] = df_design['열연코일_폭'].apply(lambda x: '1500~' if x >=1500 else '~1499' )

# column_df = pd.read_clipboard()
# list(column_df.columns)


# Group Key ---------------------------------------------------------------------------------------------------
column_group01 = ['강종_중구분','출강목표','소둔_공정실적']      #,'냉연제조표준번호'
column_group02 = ['열연공장', '소둔_공정구분']      # 소둔_공장공정
# column_group2 = ['소둔_공정구분']

column_group11 = []

column_group21 = []

# column_group31 = ['설계_HS목표온도', '설계_SS목표온도', '설계_SCS목표온도', 
#                 '설계_RCS목표온도', '설계_RHS목표온도', '설계_OAS목표온도', '설계_FCS목표온도',
#                 '설계_DFF목표온도', '설계_IHS목표온도', '설계_RCS목표냉각속도', '설계_GA목표온도', 
#                 '설계_GA냉각목표온도', '설계_SS최소유지시간', '설계_SCS최소유지시간', '설계_SCS최대유지시간',
#                 '설계_OAS최소유지시간', '설계_SPM코드', '설계_SPM목표']
column_group31 = ['소둔_HS(PGL)목표온도', '소둔_HS목표온도', '소둔_SS목표온도', '소둔_SS목표유지시간',
                '소둔_SCS_CS1목표온도', '소둔_RCS_CS2목표온도', '소둔_RHS목표온도', '소둔_OAS목표온도',
                '소둔_OAS_목표유지시간', '소둔_FCS_CS3목표온도', '소둔_RCS목표냉각속도', '소둔_PGL목표냉각속도',
                '소둔_DFF목표온도', '소둔_GA_목표온도', '소둔_FSF(CGL)목표온도', '소둔_CCS(PGL)목표온도', '소둔_ES(PGL)목표온도',
                '소둔_SPM구분지시', '소둔_SPM목표']
# column_group71 = ['인장_방향']
# column_group72 = ['인장_호수그룹_길이']
# column_group91 = ['TS_하한', 'TS_상한']

# Calculator Key
column_Calc11 = ['C_실적', 'Nb_실적']
column_Calc21 = ['SRT', 'FDT_Tail', 'CT_Tail']
column_Calc31 = ['냉간압하율', '소둔_LineSpeed', 'PHS', 'PHS_유지시간', 'HS', 'HS_유지시간', 'SS', 'SS_유지시간', 'SCS', 'SCS_유지시간', 'RCS',
                'RHS분위기온도', 'OAS', 'OAS_유지시간', 'FCS', 'FCS_유지시간', 'RCS_Head냉각속도', 'RCS_Tail냉각속도', 'FCS_CS3',
                'FSF', 'DFF', 'DFF_유지시간', 'GA평균온도', 'GA_Center온도', 'GA_DS온도', 'GA_WS온도', 'GA로_분위기온도',
                'GA로출측Head평균온도', 'GA로출측Tail평균온도', 'SPM_EL', 'Zn_Pot온도']
column_Calc32 = ['HS_이슬점', 'SS_이슬점', 'SCS_이슬점',
                'RCS_이슬점1', 'RCS_이슬점2', 'RHS_이슬점', 'OAS_이슬점1', 'OAS_이슬점2', 'OAS_이슬점3', 'FCS_이슬점', 'IHS_이슬점',
                'IHS_이슬점2', 'TDS_이슬점', 'PHS_산소농도', 'HS_산소농도', 'SS_산소농도', 'SCS_산소농도', 'RCS_산소농도', 'RCS_산소농도2',
                'RCS_산소농도3', 'RCS_산소농도4', 'RHS_산소농도', 'OAS_산소농도1', 'OAS_산소농도2', 'OAS_산소농도3', 'FCS_산소농도', 'IHS_산소농도',
                'IHS_산소농도2', 'TDS_산소농도']
column_Calc41 = ['YP', 'TS', 'EL']


design_group = column_group01 + column_group02 + column_group11 + column_group21 + column_group31
msno.matrix(df_design[design_group])
# design_group = column_groups2
calc_group = column_Calc11 + column_Calc21 + column_Calc31 + column_Calc32 
calc_mechanical_group = column_Calc41


    # 결측치 0으로 치환
df_design[design_group] = df_design[design_group].fillna(0)     
msno.matrix(df_design[design_group])



# (1) KEY별 소둔공정에 따른 생산매수 ---------------------------------------------
df11_group = [c for c in design_group if c not in column_group02]
df_table11 = df_design.groupby(df11_group).agg({'주문두께':['min','max'], '소둔작업일':['min','max','count']})
df_table11.columns = ['두께Min', '두께Max', '소둔작업일Min', '소둔작업일Max','count']

# -------------------------------------------------------------------------------------------------------------------------------------------------
# group별 on_group에 따른 매수 산정
def fun_divide_group(df, group, on_group):
    '''
    # group별 on_group에 따른 매수 산정

    < Input >
    df : Base DataFrame
    group : Index_Group Variables
    on_group : Column Dividing Variable

    < Output >
    DataFrame : df.groupby( group + [on_group]).agg({'주문두께': 'count').unstack('on_group')
    '''
    group_by = group + [on_group]
    group_table1 = df.groupby(group_by).agg({'주문두께':'count'})
    group_table1.columns = ['count']
    group_table2 = group_table1.unstack([on_group])
    group_table2.columns = list(group_table2.columns.levels[1])
    return group_table2
# -------------------------------------------------------------------------------------------------------------------------------------------------

df_table12 = fun_divide_group(df=df_design, group=df11_group, on_group='소둔_공정구분')
df_table13 = fun_divide_group(df=df_design, group=df11_group, on_group='규격약호')

df_table17 = pd.concat([df_table11, df_table12, df_table13], axis=1)


    # 순서변경
df_table18 = df_table17.reset_index()
df18_sort_columns = ['강종_중구분', '출강목표', '소둔_공정실적'] + list(df_table11.columns) + list(df_table12.columns) + list(df_table13.columns) + column_group31
df_table19 = df_table18[df18_sort_columns]
df_table19 = df_table19.sort_values(['두께Min', '두께Max', '소둔작업일Min', '소둔작업일Max'], ascending=[True, True, False, False])
df_table19.set_index(column_group01, inplace=True)

df_table10 = df_table19.copy()
len(df_table10)
df_table10.to_clipboard()

df_table10.T



# (2) KEY별 적용 두께구간 및 소둔작업일 ---------------------------------------------
df_group21 = df_design.groupby(design_group)
df_table21 = df_group21.agg({'주문두께':['count','min','max'], '소둔작업일':['min','max']})
df_table21.columns = ['count', '두께Min', '두께Max', '소둔작업일Min', '소둔작업일Max']
len(df_table21)

df_table22 = df_table21.copy()
df_table22.sort_values(['두께Min', '두께Max'], ascending=True, inplace=True)

df_table20 = df_table22.copy()
len(df_table20)
df_table20.to_clipboard()



# KEY별 조업실적 평균값 ---------------------------------------------
# set([c if 'SRT' in c else np.nan for c in df_design.columns])

df_design31 = df_design.copy()
for c in calc_group:
    df_design31[c].replace(0, np.nan,inplace=True)
# df_design31[calc_group]
# df_design31[calc_group].to_clipboard()
df_table31 = df_design31.groupby(design_group)[calc_group].mean()
for tc in calc_group:
    if np.isnan(df_table31[tc].mean()) == False:
        dec_point = fun_Decimalpoint(df_table31[tc].mean())
        df_table31[tc] = round(df_table31[tc], dec_point)

df_table30 = df_table31.copy()
df_table30.T
# len(df_table30)
# df_table30.to_clipboard()


# 재질시험실적 평균 / 편차 ---------------------------------------------
df_design41 = df_design.copy()
df_table41 = df_design41.groupby(design_group)[calc_mechanical_group].agg(['mean','std'])
df_table41
for tc in calc_mechanical_group:
    if np.isnan(df_table41[tc]['mean'].mean()) == False:
        dec_point = fun_Decimalpoint(df_table41[tc]['mean'].mean())
        print(dec_point)
        df_table41[tc]['mean'] = round(df_table41[tc]['mean'], dec_point-1)
        df_table41[tc]['std'] = round(df_table41[tc]['std'], dec_point)
df_table40 = df_table41.copy()
df_table40.T
# df_table40.to_clipboard()

# fun_Concat_MultiColumnDF(df_left=df_table20, df_right=df_table40, fill='.', pre_fill=True).to_clipboard()

# KEY별 재질시험 보증기준에 따른 Cpk, 불량률 실적 ---------------------------------------------
df_design51 = df_design.copy()
df_table51_YP = fun_Performance_Capability_Col(df=df_design51, group=design_group, on='YP', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=False)
df_table51_TS = fun_Performance_Capability_Col(df=df_design51, group=design_group, on='TS', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=False)
df_table51_EL = fun_Performance_Capability_Col(df=df_design51, group=design_group, on='EL', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=True)


df_table52_Cpk = fun_Concat_MultiColumnDF(df_left=df_table51_YP.swaplevel(i=0,j=1, axis=1)['cpk'] ,
                                    df_right=df_table51_TS.swaplevel(i=0,j=1, axis=1)['cpk'], fill='.', pre_fill=False)
df_table52_Cpk = fun_Concat_MultiColumnDF(df_left=df_table52_Cpk, df_right=df_table51_EL.swaplevel(i=0,j=1, axis=1)['cpk'], fill='.', pre_fill=False)

df_table52_Reject = fun_Concat_MultiColumnDF(df_left=df_table51_YP.swaplevel(i=0,j=1, axis=1)['rejectRatio'] ,
                                    df_right=df_table51_TS.swaplevel(i=0,j=1, axis=1)['rejectRatio'], fill='.', pre_fill=False)
df_table52_Reject = fun_Concat_MultiColumnDF(df_left=df_table52_Reject, df_right=df_table51_EL.swaplevel(i=0,j=1, axis=1)['rejectRatio'], fill='.', pre_fill=False)

df_table50_Cpk = df_table52_Cpk.copy()
df_table50_Cpk.to_clipboard()

df_table50_Reject = df_table52_Reject.copy()
df_table50_Reject.to_clipboard()




# 전체 Table Concat ---------------------------------------------
df_table_fianl_20_30 = fun_Concat_MultiColumnDF(df_left=df_table20, df_right=df_table30, fill='.', pre_fill=True)
df_table_fianl_20_30_40 = fun_Concat_MultiColumnDF(df_left=df_table_fianl_20_30, df_right=df_table40, fill='.', pre_fill=True)
df_table_fianl_20_30_40_50Cpk = fun_Concat_MultiColumnDF(df_left=df_table_fianl_20_30_40, df_right=df_table50_Cpk, fill='.', pre_fill=True)
df_table_fianl_20_30_40_50CpkReject = fun_Concat_MultiColumnDF(df_left=df_table_fianl_20_30_40_50Cpk, df_right=df_table50_Reject, fill='.', pre_fill=True)

df_table_final = df_table_fianl_20_30_40_50CpkReject.copy()
df_table_final.to_clipboard()



# 보증기준별 규격약호 및 월별 실적 ------------------------------------------------------------------------------------------------------------
df_design81 = df_design.copy()

summary_group80 = ['소둔_공정실적','두께그룹']
df_table82_count = df_design81.groupby(summary_group80).count().iloc[:,0].to_frame()
df_table82_count.columns = ['count']

df_design81_YP = fun_Performance_Capability_Col(df=df_design81, group=summary_group80, on='YP', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=False)
df_design81_TS = fun_Performance_Capability_Col(df=df_design81, group=summary_group80, on='TS', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=False)
df_design81_EL = fun_Performance_Capability_Col(df=df_design81, group=summary_group80, on='EL', cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=True)

df_design82_mt1 = fun_Concat_MultiColumnDF(df_left=df_design81_YP, df_right=df_design81_TS, fill='.', pre_fill=False)
df_design82_mt2 = fun_Concat_MultiColumnDF(df_left=df_design82_mt1, df_right=df_design81_EL, fill='.', pre_fill=False)


df_table80 = fun_Concat_MultiColumnDF(df_left=df_table82_count, df_right=df_design82_mt2, fill='.', pre_fill=False)
# df_table80.to_clipboard()

df_table86 = fun_Performance_Capability_Row(df=df_design81, group=summary_group80, mtc_group=True, on=['YP','TS', 'EL'], cpk=True, rejectRatio=True, group_tensile_dir=True, group_tensile_cd=True)
df_table85 = df_table86.copy()
# df_table85.to_clipboard()



# 소둔공정 / 두께그룹 / 인장방향별 YP, TS -----------------------------------------------
df_design91 = df_design.copy()

summary_group91 = ['소둔_공정실적','인장_방향','두께그룹']
df_group91 = df_design91.groupby(summary_group91)

# YP, TS, EL 공통 보증범위
df_group91_mtc_criteria = df_group91.agg({'YP_하한':'max', 'YP_상한':'min', 'TS_하한':'max', 'TS_상한':'min', 'EL_하한':'max', 'EL_상한':'min'})
df_group91_mtc_criteria.to_clipboard()

# TS ~ YP Fitted Plot
group_key = list(df_group91.groups.keys())
specTS = {group_key[0]: [490, 600], group_key[1]: [490, 590], group_key[2]: [490, 590], group_key[3]: [490, 590],
         group_key[4]: [480],      group_key[5]: [480],      group_key[6]: [480],      group_key[7]: [480, 640],
         group_key[8]: [490, 590], group_key[9]: [490, 590], group_key[10]: [480],     group_key[11]: [480, 590],
         group_key[12]: [480, 590] 
        }

specYP = {group_key[0]: [420, 530], group_key[1]: [420, 520], group_key[2]: [420, 520], group_key[3]: [420, 520], 
         group_key[4]: [420, 520], group_key[5]: [420, 520], group_key[6]: [420, 520], group_key[7]: [420, 520], 
         group_key[8]: [420, 520], group_key[9]: [420, 520], group_key[10]: [420,520], group_key[11]: [420, 520],
         group_key[12]: [420, 520] 
        }

specEL = {group_key[0]: [16], group_key[1]: [18], group_key[2]: [18], group_key[3]: [18], 
         group_key[4]: [18], group_key[5]: [18], group_key[6]: [18], group_key[7]: [18], 
         group_key[8]: [17], group_key[9]: [17], group_key[10]: [18], group_key[11]: [18],
         group_key[12]: [18] 
        }

plot91_TS_YP = groupOLS_Fitted_Plot = fun_Group_OLS_Plot(df=df_design91, y='TS', x='YP',
                            figsize=[4,2.5],
                            group=summary_group91, histY=False, histX=False, specY=specTS, specX=specYP)

plot91_TS_EL = groupOLS_Fitted_Plot = fun_Group_OLS_Plot(df=df_design91, y='TS', x='EL',
                            figsize=[4,2.5],
                            group=summary_group91, histY=False, histX=False, specY=specTS, specX=specEL)

# Plot_Output = plot91_TS_YP.copy()
Plot_Output = plot91_TS_EL.copy()
Plot_Output['OLS'].to_clipboard()
Plot_Output['plot']


n = -1      # start***
n -=1; print(f"{n} : {group_key[n]}")
n +=1; print(f"{n} : {group_key[n]}"); 
fun_Img_To_Clipboard(Plot_Output['plot'][group_key[n]]['scatter'])
# fun_Img_To_Clipboard(Plot_Output['plot'][group_key[n]]['histY'])
# fun_Img_To_Clipboard(Plot_Output['plot'][group_key[n]]['histX'])











