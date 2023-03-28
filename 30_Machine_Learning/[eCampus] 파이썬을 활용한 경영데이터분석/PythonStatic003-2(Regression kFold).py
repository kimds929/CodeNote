import pandas as pd
df = pd.read_csv('Database/supermarket_sales.csv')
df = pd.read_clipboard()  #Clipboard로 입력하기

df.describe()
df.info()
df.head()
df['City'].drop_duplicates().tolist()  # pandas Series to list




import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

# https://jjangjjong.tistory.com/9
Array1= [33, 16, 13, 38, 45, 24, 29, 17, 22, 13, 9, 8, 16, 32, 34, 5, 20, 3, 20, 20, 6, 33, 46, 27, 12, 29, 18, 33, 41, 3]
Array2= [50, 26, 45, 3, 1, 44, 45, 34, 49, 4, 17, 35, 5, 12, 16, 2, 21, 6, 28, 36, 22, 16, 5, 47, 30, 48, 37, 45, 38, 37]
Array3= [4, 38, 35, 49, 37, 8, 24, 26, 44, 26, 29, 4, 36, 35, 25, 48, 12, 22, 7, 5, 12, 18, 20, 31, 33, 19, 29, 33, 34, 23]
Array4= ['A', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'B', 'A']
Array5= ['A', 'B', 'C', 'B', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'A', 'C', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'A']
Array6= ['A', 'B', 'E', 'D', 'E', 'A', 'B', 'D', 'A', 'D', 'A', 'B', 'D', 'D', 'A', 'D', 'A', 'C', 'E', 'D', 'B', 'B', 'D', 'C', 'A', 'A', 'E', 'E', 'C', 'D']

# random_list = []
# for i in range(0,30):
#     if random.random()>=0.5:
#         random_list.append('A')
#     else:
#         random_list.append('B')
# random_list

df = pd.DataFrame([Array1, Array2, Array3, Array4, Array5, Array6]).T
df.columns = ['n1', 'n2', 'n3', 'g1', 'g2', 'g3']
df['n1'] = df['n1'].astype('int')
df['n2'] = df['n2'].astype('int')
df['n3'] = df['n3'].astype('int')

df.info()
df.head()

df['g1'].unique()
df['g2'].unique()

# 변수설정
var_group = ['g1','g2']
var_y = ['n1']
var_x = ['n2']

df_group = df.groupby(var_group)
dir(df.groupby(var_group))
df.groupby(var_group).all()
df_group.hist()
# df.groupby(var_group).groups
# df.groupby(var_group).get_group(('A','A'))
# df_group._group_selection    # ['n1', 'n2', 'n3', 'g3']
# df_group.mean()[['n1','n2']].add_prefix('mean_') # add_prefix : column명 접두어
# df_group.mean().add_suffix('mean_') # add_suffix : column명 접미어
# df_group.count().iloc[:,0].rename('count')    # Series Rename
# df_group.agg('mean')
# df_group.agg('std')
df_group.n1.plot()

for name, group in df_group:
    print(name)
    print(len(group))
    print(group)
    # print(type(group))

df_group.count()
df_group.count().iloc[0]
df_group.count().iloc[[0]]
df_group.count().loc[('A','A')]
df_group.count().loc['A'].loc['A']
df_group.count().loc[[('A','A')]]
df.loc[[0]]


# Multi-index (join) SingleIndex  &  Multi-Columns
count = df_group.count() #.add_prefix('count_')
index_name = count.index.names
count.columns = [['A']*len(count.columns), count.columns]   # Multi Columns
count.reset_index(inplace=True) # Index → columns
count.set_index('g1', inplace=True)
count

mean2 = df.groupby('g1').mean()#.add_prefix('mean_')
mean2.columns = [['B']*len(mean2.columns), mean2.columns]   # Multi Columns
# mean2.columns = [['A']*len(mean2.columns), mean2.columns]   # Multi Columns
mean2.reset_index(inplace=True)
mean2

merge = pd.merge(left=count, right=mean2, on='g1',how='outer', left_index=True, right_index=False)
merge
merge.set_index(index_name, inplace=True)
merge
merge.columns
merge['A']['count_n1']
merge[('A','count_n1')]

#하나의 컬럼에 aggregation을 여러개 적용할 수도 있습니다. 
# dictionary 형태로 key 값을 컬럼으로 value를 적용하고 싶은 집계함수를 사용하면 한 컬럼에 대해서
#  다양한 집계 함수를 적용하여 결과를 확인할 수 있습니다.
# df_phone.groupby(['month', 'item']).agg({'duration': [min, max, sum], 
#                                          'network_type': 'count',
#                                          'date': [min, 'first', 'nunique']})

# 계층순서변경 : df.swaplevl(0,1, axis=1)
merge.swaplevel(0,1, axis=1)
merge.swaplevel(0,1, axis=0)
merge.T

merge.stack(0)
merge.stack(1)

merge.unstack(0)
merge.unstack(1)






df_group.describe()
df_group.count().index.get_level_values  # indexList

df_group.get_group(('A','B'))       # 각각의 그룹에 접근
df_group.groups
df_group.ngroup(('A','B'))
df_group.ngroups


groupSummary_df = pd.concat([df_group.count().iloc[:,0].rename('count'),\
    df_group.agg('mean')[var_y].add_prefix('Ymean_'),df_group.agg('std')[var_y].add_prefix('Ystd_'),\
    df_group.agg('mean')[var_x].add_prefix('Xmean_'),df_group.agg('std')[var_x].add_prefix('Xstd_') ], axis=1)
groupSummary_df






# Regression -------------------------------------------------------------------
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import r2_score
from sklearn import datasets

# sklearn Dataset Load
def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df

    # iris Dataset

df = Fun_LoadData('iris')
df

var_y = 'petal length (cm)'
var_x = 'sepal length (cm)'


# https://datascienceschool.net/view-notebook/266d699d748847b3a3aa7b9805b846ae/     # 교차검증

    # 학습데이터, 테스트데이터 나누기
train_y, test_y, train_x1, test_x1 = train_test_split(df[var_y], df[var_x], test_size=0.3, random_state=1)

    # 그래프 그리기
plt.scatter(train_x1, train_y)
plt.xlabel(var_x)
plt.ylabel(var_y)

    # Regression
train_coef_x1 = sm.add_constant(train_x1)   # 상수항 결합
train_coef_x1
reg_model = sm.OLS(train_y, train_coef_x1)
reg_model_fit = reg_model.fit()    # regression model 생성

reg_model_fit.summary()     # regression 결과 보기
# reg_model_fit.summary().tables[0]
# reg_model_fit.summary().tables[1]
# reg_model_fit.summary().tables[2]
reg_model_fit.params        # coefficient parameter
reg_model_fit.bse           # standard Error
reg_model_fit.tvalues       # t-values
reg_model_fit.pvalues       # p-values
reg_model_fit.rsquared      # r-square
reg_model_fit.rsquared_adj  # r-square adj
reg_model_fit.df_model      # Degree Of Freedom 
reg_model_fit.aic           # AIC
reg_model_fit.bic           # BIC
reg_model_fit.resid         # 각데이터별 Residual


test_coef_x1 = sm.add_constant(test_x1)   # 상수항 결합
test_coef_x1
pred_y = reg_model_fit.predict(test_coef_x1)
r2_score(test_y, pred_y)

reg_model_fit.pvalues
reg_model_fit.pvalues[1]



# https://datascienceschool.net/view-notebook/266d699d748847b3a3aa7b9805b846ae/     # 교차검증

def fun_kFold_OLS(data, y, x, kFoldN, const=True):
    if type(x) == list:
        x_list = x
    else:
        x_list = [x]

    reg_kFold_result = {}
    cv = KFold(kFoldN, shuffle=True, random_state=1)
    reg_kFold_result_df = pd.DataFrame()
    for i, (idx_train, idx_test) in enumerate(cv.split(data)):
        reg_result = {}
        reg_result['idx'] = i
        df_train = data.iloc[idx_train]
        df_test = data.iloc[idx_test]
        df_train_y = df_train[y]
        df_train_x = df_train[x]
        df_test_y = df_test[y]
        df_test_x = df_test[x]
        if const :
            df_train_x = sm.add_constant(df_train_x)   # 상수항 결합
            df_test_x = sm.add_constant(df_test_x)   # 상수항 결합
        model = sm.OLS(df_train_y, df_train_x)
        model_fit = model.fit()
        reg_kFold_result['model_' + str(i)] = model_fit
        reg_kFold_result['summary_' + str(i)] = model_fit.summary()
        reg_result['nTrain'] = len(df_train)
        reg_result['r2_train'] = model_fit.rsquared
        pred = model_fit.predict(df_test_x)
        reg_result['r2_test'] = r2_score(df_test_y, pred)

        for xi in df_train_x.columns:
            reg_result['coef_' + xi] = model_fit.params[0]
            reg_result['pValue_' + xi] = round(model_fit.pvalues[0],3)
            
        reg_result_df = pd.DataFrame([reg_result])
        reg_result_df.set_index('idx', inplace=True)
        reg_kFold_result_df = pd.concat([reg_kFold_result_df, reg_result_df], axis=0)

    # 가중치 없이 전체 평균
    # reg_result_total = reg_kFold_result_df.mean().to_frame(name='Total').T
    # r2값에 따른 가중치 고려하여 평균
    reg_result_total = reg_kFold_result_df.apply(lambda x: x * reg_kFold_result_df['r2_test'] / reg_kFold_result_df['r2_test'].sum(), axis=0).sum().to_frame(name='Total').T
    reg_result_df = pd.concat([reg_kFold_result_df, reg_result_total], axis=0)
    reg_result_df['nTrain'].astype(str)
    reg_result_df['nTrain'].loc['Total'] = str(int(reg_result_df['nTrain'].loc['Total']))+ ' × ' + str(len(reg_result_df)-1)
    reg_kFold_result['result'] = reg_result_df
    return reg_kFold_result



kFold_N = []
MeanOfR2 = []
for k in range(2, int(np.floor(len(df)/30))+1):
    kFold_N.append(k)
    MeanOfR2.append(fun_kFold_OLS(data=df, y=var_y, x=var_x, kFoldN=k)['result'].loc['Total','r2_test'])

fun_kFold_OLS(data=df, y=var_y, x=var_x, kFoldN=2).loc[['Total']]

plt.figure(figsize=(6,4))
plt.plot(kFold_N, MeanOfR2, 'o-')
plt.show






# group별 Regression
    # supermarket_sales Data
df = pd.read_csv('./Database/supermarket_sales.csv')
df.head()
df.shape

df_group = df.groupby(['Gender','CustomerType'])
df_group.count()
var_y = 'Total'
var_x = ['Rating','Quantity']

    # iris Data
df = Fun_LoadData('iris')
df.head()
df.shape

df_group = df.groupby('Target')
df_group.count()
var_y = 'petal length (cm)'
var_x = 'sepal length (cm)'


# --- groupby kFold Regression -----
n=4
result_df = pd.DataFrame()
for i, v in df_group:
    # print(df)
    group_index = pd.DataFrame([i], columns = df_group.all().index.names)
    kFold_reg = fun_kFold_OLS(data=v, y=var_y, x=var_x, kFoldN=n)['result'].loc[['Total']]    
    kFold_reg.reset_index(drop=True, inplace=True)
    group_reg = pd.concat([group_index, kFold_reg], axis=1)
    result_df = pd.concat([result_df,group_reg], axis=0)
    # n+=1
    
result_df.set_index(df_group.all().index.names, inplace=True)
result_df
# ------------------------------------
fun_kFold_OLS(data=df, y=var_y, x=var_x, kFoldN=n)['result'].loc[['Total']]

fun_kFold_OLS(data=df, y='Total', x=['Rating', 'Quantity'], kFoldN=3,const=False)['result']




from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
formula = var_y + '~' + var_x
cross_reg_model = sm.OLS.from_formula(formula, data =df)
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(reg_model, df[var_x], df[var_y], cv=cv, scoring='r2') 


