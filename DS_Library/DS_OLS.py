# if __name__ == "__main__":
import numpy as np
import pandas as pd
from IPython.display import clear_output

import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import r2_score

def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df


# --- 【 Regression 】------------------------------------------------------------------------------------------
# fun_Train_OLS(data=df, y=var_y, x=var_x)['result']
# fun_TrainTest_OLS(data=df, y=var_y, x=var_x)['result']
# fun_kFold_OLS(data=df, y=var_y, x=var_x, kFoldN=5)['result'].loc[['Total']]
# fun_OLS(data=df[:200], y=var_y, x=var_x)
# fun_OLS(data=df[:200], y=var_y, x=var_x, random_state=1, kFoldN=3)


# Normal Regression --------
def fun_Train_OLS(data, y, x, const=True):
    OLS_result = {}   # Result Dictionary
    if type(x) == list:
        x_list = x.copy()
    else:
        x_list = [x].copy()

    # NULL Data Drop
    normal_data = data.copy()
    normal_data = normal_data[normal_data[y].isna() == False]
    for cx in x_list:
        normal_data = normal_data[normal_data[cx].isna() == False]

    reg_result = {}
    df_train = normal_data
    df_train_y = df_train[y]
    df_train_x = df_train[x]
    OLS_result['train_y'] = df_train_y
    OLS_result['train_x'] = df_train_x
    if const :
        df_train_x = sm.add_constant(df_train_x)   # 상수항 결합
    model = sm.OLS(df_train_y, df_train_x)
    model_fit = model.fit()
    OLS_result['model'] = model_fit
    OLS_result['summary'] = model_fit.summary()
    reg_result['nTrain'] = str(len(df_train)) + ' × 1'
    reg_result['r2_train'] = round(model_fit.rsquared, 3)
    reg_result['r2_test'] = ''
    pred_train = model_fit.predict(df_train_x)
    reg_result['rmse_train'] = rmse(df_train_y, pred_train)
    reg_result['rmse_test'] = ''
    for xi in df_train_x.columns:
        reg_result['coef_' + xi] = model_fit.params[xi]
    for xi in df_train_x.columns:
        reg_result['pValue_' + xi] = round(model_fit.pvalues[xi], 3)
    reg_result_df = pd.DataFrame([reg_result])
    OLS_result['result'] = reg_result_df
    return OLS_result


# Train-Test Regression --------
def fun_TrainTest_OLS(data, y, x, test_size=0.3, const=True, random_state=0):
    OLS_result = {}   # Result Dictionary
    if type(x) == list:
        x_list = x.copy()
    else:
        x_list = [x].copy()
    
    # NULL Data Drop
    normal_data = data.copy()
    normal_data = normal_data[normal_data[y].isna() == False]
    for cx in x_list:
        normal_data = normal_data[normal_data[cx].isna() == False]
    
    reg_result = {}
    df_train_y, df_test_y, df_train_x, df_test_x = train_test_split(normal_data[y], normal_data[x], test_size=test_size, random_state=random_state)
    OLS_result['train_y'] = df_train_y
    OLS_result['train_x'] = df_train_x
    OLS_result['test_y'] = df_test_y
    OLS_result['test_x'] = df_test_x
    if const :
        df_train_x = sm.add_constant(df_train_x)   # 상수항 결합
        df_test_x = sm.add_constant(df_test_x)   # 상수항 결합
    model = sm.OLS(df_train_y, df_train_x)
    model_fit = model.fit()
    OLS_result['model'] = model_fit
    OLS_result['summary'] = model_fit.summary()
    reg_result['nTrain'] = str(len(df_train_y)) + ' × 1'
    reg_result['r2_train'] = round(model_fit.rsquared, 3)
    pred_train = model_fit.predict(df_train_x)
    pred_test = model_fit.predict(df_test_x)
    reg_result['r2_test'] = round(r2_score(df_test_y, pred_test), 3)
    reg_result['rmse_train'] = rmse(df_train_y, pred_train)
    reg_result['rmse_test'] = rmse(df_test_y, pred_test)
    for xi in df_train_x.columns:
        reg_result['coef_' + xi] = model_fit.params[xi]
    for xi in df_train_x.columns:
        reg_result['pValue_' + xi] = round(model_fit.pvalues[xi], 3)
    reg_result_df = pd.DataFrame([reg_result])
    OLS_result['result'] = reg_result_df
    return OLS_result


# kFold Regression --------
def fun_kFold_OLS(data, y, x, kFoldN=5, const=True, random_state=0):
    OLS_result = {}   # Result Dictionary
    if type(x) == list:
        x_list = x.copy()
    else:
        x_list = [x].copy()
    # NULL Data Drop
    normal_data = data.copy()
    normal_data = normal_data[normal_data[y].isna() == False]
    for cx in x_list:
        normal_data = normal_data[normal_data[cx].isna() == False]

    cv = KFold(kFoldN, shuffle=True, random_state=random_state)    # kFold
    reg_kFold_result_df = pd.DataFrame()
    for i, (idx_train, idx_test) in enumerate(cv.split(normal_data)):
        OLS_result[i]={}
        reg_result = {}
        reg_result['idx'] = i
        df_train = normal_data.iloc[idx_train]
        df_test = normal_data.iloc[idx_test]
        df_train_y = df_train[y]
        df_train_x = df_train[x]
        df_test_y = df_test[y]
        df_test_x = df_test[x]
        OLS_result[i]['train_y'] = df_train_y
        OLS_result[i]['train_x'] = df_train_x
        OLS_result[i]['test_y'] = df_test_y
        OLS_result[i]['test_x'] = df_test_x
        if const :
            df_train_x = sm.add_constant(df_train_x)   # 상수항 결합
            df_test_x = sm.add_constant(df_test_x)   # 상수항 결합
        model = sm.OLS(df_train_y, df_train_x)
        model_fit = model.fit()
        OLS_result[i]['model'] = model_fit
        OLS_result[i]['summary'] = model_fit.summary()
        reg_result['nTrain'] = len(df_train)
        reg_result['r2_train'] = model_fit.rsquared
        pred_train = model_fit.predict(df_train_x)
        pred_test = model_fit.predict(df_test_x)
        reg_result['r2_test'] = r2_score(df_test_y, pred_test)
        reg_result['rmse_train'] = rmse(df_train_y, pred_train)
        reg_result['rmse_test'] = rmse(df_test_y, pred_test)
        for xi in df_train_x.columns:
            reg_result['coef_' + xi] = model_fit.params[xi]
        for xi in df_train_x.columns:
            reg_result['pValue_' + xi] = model_fit.pvalues[xi]
            
        reg_result_df = pd.DataFrame([reg_result])
        reg_result_df.set_index('idx', inplace=True)
        OLS_result[i]['result'] = reg_result_df
        reg_kFold_result_df = pd.concat([reg_kFold_result_df, reg_result_df], axis=0)

    # 가중치 없이 전체 평균
    # reg_result_total = reg_kFold_result_df.mean().to_frame(name='Total').T
    # r2값에 따른 가중치 고려하여 평균
    reg_kFold_result_df['weight'] = reg_kFold_result_df.apply(lambda x: x['r2_test']+x['r2_train'] if x['r2_test']+x['r2_train'] >0 else 0,axis=1)
    reg_result_total = reg_kFold_result_df.apply(lambda x: x * reg_kFold_result_df['weight'] / reg_kFold_result_df['weight'].sum(), axis=0).sum().to_frame(name='Total').T
    reg_result_df = pd.concat([reg_kFold_result_df, reg_result_total], axis=0)
    reg_result_df.drop('weight', axis=1, inplace=True)
    reg_result_df['nTrain'].astype(str)
    reg_result_df['nTrain'].loc['Total'] = str(int(reg_result_df['nTrain'].loc['Total']))+ ' × ' + str(len(reg_result_df)-1)
    # r2Square, pValue 자릿수 조절
    for c in reg_result_df.columns:
        if 'r2' in c or 'pValue' in c:
            reg_result_df[c] = round(reg_result_df[c],3)
    OLS_result['result'] = reg_result_df
    return OLS_result


# OLS Result Package ★★★-------------
def fun_OLS(data, y, x, const=True, min_n=30, random_state=0, kFoldN=0):
    if type(x) == list:
        x_list = x.copy()
    else:
        x_list = [x].copy()
    # NULL Data Drop
    normal_data = data.copy()
    normal_data = normal_data[normal_data[y].isna() == False]
    
    for cx in x_list:
        normal_data = normal_data[normal_data[cx].isna() == False]
    print(f"Data Length : {len(normal_data)}")
    
    if const:
        x_list.insert(0, 'const')

    OLS_result_df = pd.DataFrame()
    if len(normal_data) < min_n:
        empty_columns = ['nTrain', 'r2_train', 'r2_test','rmse_train','rmse_test']
        for cx in x_list:
            empty_columns.append('coef_'+cx)
        for cx in x_list:
            empty_columns.append('pValue_'+cx)
        OLS_result_df = pd.DataFrame([['']*len(empty_columns)], columns = empty_columns)
    elif len(normal_data) < 50:
        OLS_result_df = fun_Train_OLS(data=normal_data, y=y, x=x, const=const)['result']
    elif len(normal_data) < 100:
        OLS_result_df = fun_TrainTest_OLS(data=normal_data, y=y, x=x, const=const, random_state=random_state)['result']
    else:
        if kFoldN > 0:
            OLS_result_df = fun_kFold_OLS(data=normal_data, y=y, x=x, kFoldN=kFoldN, const=const, random_state=random_state)['result'].loc[['Total']]
        else:
            if len(normal_data) < 120:
                OLS_result_df = fun_kFold_OLS(data=normal_data, y=y, x=x, kFoldN=3, const=const, random_state=random_state)['result'].loc[['Total']]
            elif len(normal_data) < 150:
                OLS_result_df = fun_kFold_OLS(data=normal_data, y=y, x=x, kFoldN=4, const=const, random_state=random_state)['result'].loc[['Total']]
            elif len(normal_data) < 200:
                OLS_result_df = fun_kFold_OLS(data=normal_data, y=y, x=x, kFoldN=5, const=const, random_state=random_state)['result'].loc[['Total']]
            else:
                OLS_result_df = fun_kFold_OLS(data=normal_data, y=y, x=x, kFoldN=6, const=const, random_state=random_state)['result'].loc[['Total']]
    OLS_result_df.reset_index(drop=True, inplace=True)
    column_group_name = 'OLS) ' + str(y) + ' = ' + ' + '.join(x_list)
    OLS_result_df.columns = [[column_group_name]*len(OLS_result_df.columns), OLS_result_df.columns]
    return OLS_result_df


# DataFrame.groupby.Regression
def fun_Group_OLS(groupby, y, x, const=True, random_state=0, kFoldN=0):
    OLS_Object = {}
    OLS_df = pd.DataFrame()
    n=1
    for i, v in groupby:
        print(i)
        print( f"Process: {round(n/groupby.ngroups*100,1)}%" )
        OLS_Object[i] = {}
        OLS_Object[i]['data'] = v
        group_OLS = fun_OLS(data=v, y=y, x=x, const=const, random_state=random_state, kFoldN=kFoldN)
        group_index = pd.DataFrame([i], columns = groupby.all().index.names)
        group_OLS.index = group_index.set_index(groupby.all().index.names).index
        OLS_Object[i]['result'] = group_OLS
        OLS_df = pd.concat([OLS_df, group_OLS], axis=0)
        n+=1
        clear_output(wait=True)
    OLS_Object['result'] = OLS_df
    print('Complete!')
    return OLS_Object


# DataFrame.groupby.count + Regression
def fun_Concat_Group_OLS(base, groupby, y, x, const=True, random_state=0, kFoldN=0):
    OLS_df = fun_Group_OLS(groupby, y, x, const=const, random_state=random_state, kFoldN=kFoldN)['result']
    Result_df = base.copy()
    base_id_name = Result_df.index.names
    Result_df.reset_index(inplace=True)
    Result_df = pd.merge(left=Result_df, right=OLS_df, on=OLS_df.index.names, how='outer')
    Result_df.set_index(base_id_name, inplace=True)
    # Result_df = pd.concat([base, OLS_df], axis=1)
    return Result_df
