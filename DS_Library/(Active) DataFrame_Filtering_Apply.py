

df_target = pd.read_clipboard()        # Base DataSet
df_filter = pd.read_clipboard()     # Condition DF

result01 = fun_Filtering_DataFrame(data=df_target, criteria=df_filter, unique_condition=False)

result01['criteria_index'].value_counts()
result01['summary']
result01['result'].to_clipboard(index=False)
# result02 = pd.read_clipboard()
result02 = result01['result'].copy()

group2 = result02.groupby('criteria_index')

# seq_obj = pd.Series()
result_df = pd.DataFrame()
result_series = pd.Series()
for i in df_filter.index:
# for i in range(0, 10):
    part = result02[result02['criteria_index'].apply(lambda x: str(i) in str(x).split(', '))]
    if len(part):
        result_series_part = part['SS'].describe()
        result_series_part.name=i
        result_series_part['mean'] = int(result_series_part['mean'])
        result_series_part['std'] = round(result_series_part['std'],1)
        result_series_part = result_series_part.to_frame().T
        result_df = pd.concat([result_df, result_series_part], axis=0)
    #     part_result_YP = fun_Performance_Capability_Row(df=part, on='YP', first_test=True, cpk=True, rejectRatio=True)
    #     part_result_TS = fun_Performance_Capability_Row(df=part, on='TS', first_test=True, cpk=True, rejectRatio=True).drop(['Total'], axis=1)
    #     part_result = pd.concat([part_result_YP, part_result_TS], axis=1)
    #     part_result.index = [i]
    #     # seq_obj.loc[i] = len(part_result)
    #     result_df = pd.concat([result_df, part_result], axis=0)




# Histogram --------------
result_df.to_clipboard()

df_abc =pd.read_clipboard()

df_abc2 = df_abc[df_abc['count'] >= 5]


df_abc2['Cpk_group'] = fun_Cut_Group(df=df_abc2, columns='Min_Cpk', interval=0.1)
df_abc2['Reject_group'] = fun_Cut_Group(df=df_abc2, columns='Max_불량률', interval=0.5)

df_abc2['Min_Cpk'].mean
a = fun_Hist(data=df_abc2, x='Min_Cpk', norm=True, color='skyblue', lin)
b = fun_Hist(data=df_abc2, x='Max_불량률', norm=True, color='skyblue')


df_abc2['Cpk_group'].value_counts().sort_index().to_clipboard()

df_abc2['Reject_group'].value_counts().sort_index().to_clipboard()

fun_Img_To_Clipboard(b['Max_불량률'])

