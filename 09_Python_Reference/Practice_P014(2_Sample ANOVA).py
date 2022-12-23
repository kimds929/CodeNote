from itertools import combinations
import seaborn as sns

x_list = list(v_list.columns)
x_list = ['초_YP', '초_TS', 'C_실적', 'Mn_실적','소둔_LineSpeed', 'SS', '누적SPM_EL']
x_list = x_l
df.groupby('냉연번호생성년월')[x_list].agg(['mean','std']).to_clipboard()


fig_obj = {}
fig_obj['hist'] = {}
fig_obj['box'] = {}

for i in x_list:
    pVal = round(sp.stats.f_oneway(df[df['냉연번호생성년월']==2003][i], df[df['냉연번호생성년월']==2004][i]).pvalue, 3)         # Oneway ANOVA : F-value, p-value

    fig_obj['hist'][i] = plt.figure(figsize=[5,3])
    sns.distplot(df[df['냉연번호생성년월']==2003][i], label="'20. 3月")
    sns.distplot(df[df['냉연번호생성년월']==2004][i], label="'20. 4月")
    plt.title(i)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    fig_obj['box'][i] = plt.figure(figsize=[5,3])
    box_data = fun_Group_Array(data=df, x=i, group='냉연번호생성년월')
    plt.boxplot(box_data['value'], labels=box_data['index'])
    plt.scatter(x=range(1,len(box_data['index'])+1), y=[i.mean() for i in box_data['value']], color='r')
    plt.title(f"{i} / p-value: {format(pVal, '.3f')}")
    plt.show()

n = -1      # start***
n -=1; print(f"{n} : {x_list[n]}")
n +=1; print(f"{n} : {x_list[n]}")
fun_Img_Copy(fig_obj['hist'][x_list[n]])
fun_Img_Copy(fig_obj['box'][x_list[n]])


df_final[x].drop_duplicates()
a = df_final[df_final[x] == 'WU2007Y6PE2']['초_EL']
b = df_final[df_final[x] == 'WU2007Y6PE3']['초_EL']


sp.stats.f_oneway(a, b)