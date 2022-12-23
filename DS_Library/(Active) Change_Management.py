# 변경관리 효과성 검증 ------------------------------


import scipy as sp

# change_df0 = pd.read_clipboard()
result_df = result01['result'].copy()
result01['summary'].to_clipboard()
# change_df0 = result_df[result_df['criteria_index'].isin(['0','1'])].copy()
change_df0 = result_df[result_df['criteria_index'].isin(['10'])].copy()
change_date = 200323

# fun_Search(data=change_df0, on='SPM')
# fun_Search(data=change_df0, on='정정')
# change_df0['소둔_SS목표온도']

# filtering_condition
# seperator_cols = {'규격약호':'', '주문폭':1400}
# seperator_cols = {'규격약호':'', '주문두께':1}
seperator_cols = {'규격약호':''}

evaluate_cols = ['YP']
evaluate_spec = [140, 180]
# change_criteria_cols = ['소둔_SPM목표']
change_cols = ['SPM_EL']

change_target = '변경관리_구분'

# ---------------------------------

# Data-Set 전처리
change_df1 = change_df0.copy()
change_df1['소둔완료일'] = change_df1['소둔작업완료일'].apply(lambda x: int(str(x)[0:6]))

change_df2 = change_df1.copy()
change_df2[change_target] = change_df2['소둔완료일'].apply(lambda x: '변경_후' if x >= change_date else '변경_전')

change_df3 = change_df2[change_df2['냉연정정(RCL/MCL)공정'].isin(['1-1RCL', 'MCL']) == False]
# change_df3 = change_df2[change_df2['냉연정정(RCL/MCL)공정'].isin(['1-1RCL', 'MCL']) == False  | (change_df2['냉연정정(RCL/MCL)공정'].isin(['1-1RCL', 'MCL']) & change_df2['냉연정정_SPM_EL'] > 0)]


# Data-Set apply
change_df_apply = change_df3.copy()


# 조건항목 Group항목으로 변경
group_list = []
for sc in seperator_cols:   
    if seperator_cols[sc]:
        change_df_apply[sc + '_group'] = change_df_apply[sc].apply(lambda x: str(seperator_cols[sc]) + ' 이상' if x >= seperator_cols[sc] else str(seperator_cols[sc]) + ' 미만')
        group_list.append(sc + '_group')
    else:
        group_list.append(sc)

group_df = change_df_apply.groupby(group_list)
group_change_df = change_df_apply.groupby([change_target] + group_list )


change_df_apply['규격약호'].value_counts()
# change_instruction = change_df_apply.groupby(['주문두께_group', '변경관리_구분','소둔_SS목표온도']).agg({'SS':'count'}).unstack(['소둔_SS목표온도'])
# change_instruction = change_df_apply.groupby(['주문두께_group', '변경관리_구분','소둔_SPM목표']).agg({'SPM_EL':'count'}).unstack(['소둔_SPM목표'])
change_instruction = change_df_apply.groupby([ '변경관리_구분','소둔_SPM목표']).agg({'SPM_EL':'count'}).unstack(['소둔_SPM목표'])
change_instruction
change_instruction.to_clipboard()


# Group result
# group_count_df = group_change_df.count().iloc[:,0].rename('count').to_frame()
# group_count_df.columns = [['Total']*len(group_count_df.columns), group_count_df.columns]
# group_count_df

# group_cols_df = group_count_df.copy()
# group_cols_df = fun_Concat_Group_MeanStd(base=group_cols_df, groupby=group_change_df, on='YP', count=True)   # + Mean/Std
# group_cols_df = fun_Concat_Group_Cpk(base=group_cols_df, on='YP', criteria=evaluate_spec, n_min=5 )   # + Cpk
# group_cols_df = fun_Concat_Group_Reject(base=group_cols_df, groupby=group_change_df, on='YP', criteria=evaluate_spec) # + Reject
# group_cols_df

# groupAnalysis_df = fun_Drop_Var_Count(base=groupAnalysis_df, on=Y_Var)   # Drop Var-Count if it same Total-Count
# groupAnalysis_df.drop(Y_Var, axis=1, inplace=True)


agg_obj={}
    # count
group_count_df = group_change_df.count().iloc[:,0].to_frame()
group_count_df.columns = ['count']
group_count = group_count_df.unstack([change_target]).swaplevel(i=0, j=1, axis=1)
group_count

    # column summary
for cc in change_cols:
    agg_obj[cc] = ['mean']

for ec in evaluate_cols:
    agg_obj[ec] = ['mean', 'std']

group_cols_df = group_change_df.agg(agg_obj).unstack([change_target]).swaplevel(i=0, j=2, axis=1).swaplevel(i=1, j=2, axis=1).sort_index(level=0, axis=1)
group_cols = group_cols_df.apply(lambda x: x.apply(lambda y: round(y, fun_Decimalpoint(x.mean())) ), axis=0)
group_cols

summary_df_count_cols = fun_Concat_MultiColumnDF(df_left=group_count, df_right=group_cols, fill='').sort_index(level=[0,1], ascending=[True, False], axis=1)

cpk_df = pd.DataFrame()
for chi in change_df_apply[change_target].value_counts().sort_index().index:        # 변경_전 / 변경_후
    for eci in evaluate_cols:   # Y값 항목
        cpk_part = summary_df_count_cols[chi].apply(lambda x: fun_Cpk(mean=x[eci]['mean'], std=x[eci]['std']
                , spec=evaluate_spec, count=x['count'][''], count_criteria=5, cpk_sign=False)
                , axis=1).to_frame()
        cpk_part.columns = [[chi], [eci], ['cpk ' + '~'.join([str(e) for e in evaluate_spec]) ]]
        cpk_df = pd.concat([cpk_df, cpk_part], axis=1)

cpk_df

summary_df_count_cols_cpk = fun_Concat_MultiColumnDF(df_left=summary_df_count_cols, df_right=cpk_df, fill='').sort_index(level=[0,1], ascending=[True, False], axis=1)
summary_df_count_cols_cpk



    # pvalue
pval_df = pd.DataFrame()
for i, v in  change_df_apply.groupby(group_list):
    for k in agg_obj.keys():
        # ch_bef = v[v[change_target]=='변경_전'][k].dropna()
        # ch_aft = v[v[change_target]=='변경_후'][k].dropna()
        # print(ch_bef.mean())
        # print( sp.stats.f_oneway(ch_bef, ch_aft).pvalue)
        pval = format(sp.stats.f_oneway(v[v[change_target]=='변경_전'][k].dropna(), v[v[change_target]=='변경_후'][k].dropna()).pvalue, '.2f')
        pval_df.loc[str(i), k] = pval
pval_df.columns = [['pvalue'] * len(pval_df.columns), list(pval_df.columns)]
pval_df.index = change_df_apply.groupby(group_list).count().index
pval_df = pval_df.apply(lambda x: x.apply(lambda y: format(float(y),'.2f')), axis=0).sort_index(level=1, ascending=False, axis=1)

# summary_df = fun_Concat_MultiColumnDF(df_left=group_count, df_right=group_cols, fill='')
# summary_df = fun_Concat_MultiColumnDF(df_left=summary_df, df_right=pval_df, fill='')



# Summary Concat
summary_df = fun_Concat_MultiColumnDF(df_left=summary_df_count_cols_cpk, df_right=pval_df, fill='')
summary_df
summary_df.to_clipboard()




# -----------------------------------
# fun_Concat_Group_MeanStd(base=group_count_df, groupby=group_change_df, on=['YP'])   # + Mean/Std
# fun_Concat_Group_Cpk_Range(base=Y_group_df, on=Y_Var, criteria=[0.7, 0.8, 0.9, 1.0], n_min=5 )   # + Cpk_Range
# fun_Concat_Group_Cpk(base=group_change_df, on=['YP'], criteria=[140,180], n_min=5 ) 
# -----------------------------------



# distplot + boxplot + avg_point
def fun_DistBox(data, on, group, figsize=[5,5], title=False, mean_line=False):
    # group = change_target
    # on = 'YP'
    # title = 'abc'
    normal_data = data.copy()
    box_colors = ['steelblue','orange']

    figs, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
    # distplot
    if title:
        axes[0].set_title(title)
    for vi in normal_data[group].value_counts().sort_index().index:
        try:
            sns.distplot(normal_data[normal_data[group] == vi][on], label=vi, ax=axes[0])
        except:
            pass
    axes[0].legend()
    # boxplot
    boxes = sns.boxplot(x=on, y=group, data=normal_data, orient='h', color='white', linewidth=1, ax=axes[1])
    for bi, box in enumerate(boxes.artists):
        box.set_edgecolor(box_colors[bi])
        for bj in range(6*bi,6*(bi+1)):    # iterate over whiskers and median lines
            boxes.lines[bj].set_color(box_colors[bi])
    # avg_point
    group_mean = normal_data.groupby(group)[on].mean()
    group_mean.sort_index(ascending=True, inplace=True)
    if mean_line:
        axes[0].axvline(x=group_mean.iloc[0], c=box_colors[0], alpha=1, linestyle='--')
        axes[0].axvline(x=group_mean.iloc[1], c=box_colors[1], alpha=1, linestyle='--')
    axes[1].scatter(x=group_mean, y=list(range(0,len(group_mean))), color=box_colors)
    plt.grid(alpha=0.1)

    return figs


group_df.count().iloc[:,0]
group_keys = list(group_df.groups.keys())
figure_df = pd.DataFrame()
for i,v in group_df:
    v = v.sort_values(change_target)
    if len(v) > 1:
        for ecc in evaluate_cols + change_cols:
            figure_df.loc[str(i), ecc] = fun_DistBox(data=v, on=ecc, group=change_target, figsize=[4,2.5], title=i, mean_line=False)
n = -1
n -=1; print(group_keys[n])
n +=1; print(group_keys[n])
fun_Img_To_Clipboard(figure_df.T[str(group_keys[n])][evaluate_cols[0]])
fun_Img_To_Clipboard(figure_df.T[str(group_keys[n])][change_cols[0]])
# fun_Img_To_Clipboard(figure_df.T[str(group_keys[n])][change_cols[1]])













