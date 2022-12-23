# if __name__ == "__main__":
import math
import numpy as np
import pandas as pd

from IPython.display import clear_output

import DS_OLS


# -------------------------------------------------------------------------------------
test_list = [['a',1,'A',20], ['b',5,'B',10], ['c',3,'B',30], ['d',9,'A',60], ['e',7,'B',40]]
test_df = pd.DataFrame(test_list, columns=['A','B','C','D'])


# -------------------------------------------------------------------------------------
# 특정 data(DataFrame, List)안에 해당 문자 또는 List안의 문자를 모두 포함하는 값을 반환
def fun_Search(data, on, method='in', case=True):
    result_list = [];
    on_list = on if type(on) == list else [on]

    for oc in on_list:
        target_list =[]
        # 입력 형식
        if type(data) == pd.core.frame.DataFrame:
            target_list = data.columns
        elif type(data) == list:
            target_list = data
        # 대소문자 구분
        target_list = target_list if case else [t.lower() for t in target_list]
        on = on if case else on.lower()
        # 찾을방법
        if method == 'in':
            result = list(set([c if on in c else np.nan for c in target_list]))
        elif method == 'equal':
            result = list(set([c if on == c else np.nan for c in target_list]))
        result.remove(np.nan)
    return result


# 어떤 값에 대하여 자동으로 소수점 자리수를 부여
def fun_Decimalpoint(value):
    point_log10 = np.floor(np.log10(abs(value)))
    point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
    return point


# 입력기준값을 기준으로 Min, Max값을 도출하는 함수
def fun_Auto_MinMax(mean, point=2):
    '''
    # 입력기준값을 기준으로 Min, Max값을 도출하는 함수
    < input >
    mean (int, float) : 입력기준값
    point (int) : 자릿수
    
    < output >
    result (list) : [Min, Max]
    '''
    sign = -1 if mean < 0 else 1
    MinMax = 10 ** math.floor(math.log10(abs(mean))+1) - 10 ** math.floor(math.log10(abs(mean))-point)
    result = [-1*MinMax, MinMax]
    return  result

# 평균, 편차, 갯수, Spec기준을 통해 Cpk 값 구하기
def fun_Cpk(mean, std, spec, count=0, count_criteria=5, cpk_sign=False):
    '''
    # 평균, 편차, 갯수, Spec기준을 통해 Cpk 값 구하기
    < input >
    mean (int, float) : 평균
    std (int, float) : 표준편차
    spec (list) : USL, LSL 기준을 담고있는 List
    count (int) : 데이터의 수
    count_criteria (int) : cpk계산을 위한 count의 최소 갯수 기준
    cpk_sign(boolean) : 평균이 USL에 치우쳐져있는지(+), LSL에 치우쳐져있는지(-) Cpk값의 부호로 구분

    < output >
    cpk (float) : cpk값
    '''
    count_bool = True
    if count:
        count_bool = True if count >= count_criteria else False

    if count_bool:
        if ~np.isnan(std) or std > 0:
            if len(spec) < 2:
                spec.append( 10 ** math.floor(math.log10(mean)+1) - 10 ** math.floor(math.log10(mean)-2) ) 
            if mean - spec[0] > spec[1] - mean:     # USL
                cpk = round((spec[1] - mean) / (3*std), 2)
                where = 'USL'
            else:    # LSL: mean - spec[0] > spec[1] - mean
                cpk = round((mean-spec[0]) / (3*std), 2)
                where = 'LSL'
            
            if cpk_sign:
                cpk = 0.01 if cpk < 0.01 else cpk
                if where == 'LSL':
                    cpk *= -1
            return cpk
        else:
            return np.nan
    else:
        return np.nan

# DataFrame Remove Outlier
def fun_Outlier_Remove(data, on, criteria=0.01, method='quantile'):
    '''
    # DataFrame Remove Outlier
    
    < input >
    data (DataFrame) : Target Data-Set
    on (String, List) : Outlier Filtering Target Column name(or names list)
    criteria (Numeric, float) : Outlier remove criteria
    method ('String') : Outlier remove method

    <output>
    result_df (DataFrame) : Outlier removed Data-Set
    '''
    if type(on) == list:
            on_list = on
    else:
        on_list = [on]
    result_df = data.copy()
    for co in on_list:
        if method == 'quantile':
            result_df = result_df[result_df[co] <= data[co].quantile(max([criteria, 1-criteria]))]
            result_df = result_df[result_df[co] >= data[co].quantile(min([criteria, 1-criteria]))]
    return result_df


# DataFrame transform number → Category
def fun_Cut_Group(df, columns, interval):
    return pd.cut(df[columns], np.arange(np.floor(df[columns].min()/interval)*interval,
        np.ceil(df[columns].max()/interval)*interval+interval, interval), right=False)




# --- 【 DataFrame 】------------------------------------------------------------------------------------------
# Column Level이 다른 두개의 DataFrame의 Column Level을 맞춰주는 함수
def fun_Transform_MultiColumns(df_target, df_base, fill='.', pre_fill=False):
    '''
    # Column Level이 다른 두개의 DataFrame의 Column Level을 맞춰주는 함수

    < input >
    df_target(DataFrame) : Column Level을 바꿀 DataFrame
    df_base(DataFrame) : Column Level의 Transform 기준이 되는 DataFrame
        ※ df_target Column Level < df_base Column Level
    fill(Str) : multi column변환에 따른 빈 column을 메울 값
    pre_fill(Boolean) : multi column변환에 따른 빈 column을 메울 값 (False: 뒤쪽, True: 앞쪽)

    < output >
    df_result(DataFrame) = transformed df_target (Column Level)
    '''
    df_target_columns_level = 1 if type(df_target.columns[0]) == str else (len(df_target.columns[0]) if type(df_target.columns[0]) == tuple else 0)
    df_base_columns_level = 1 if type(df_base.columns[0]) == str else (len(df_base.columns[0]) if type(df_base.columns[0]) == tuple else 0)
    df_result = df_target.copy()
    if df_target_columns_level < df_base_columns_level:
        multi_sep = [[fill] * len(df_result.columns)] * (df_base_columns_level - df_target_columns_level)
        if df_target_columns_level == 1:
            if pre_fill == True:     # 구분자 위치 앞쪽
                multi_sep.append(list(df_result.columns))
            else:     # 구분자 위치 뒤쪽
                multi_sep.insert(0, list(df_result.columns))
            multi_columns = multi_sep
        else:
            if pre_fill == True:     # 구분자 위치 앞쪽
                multi_columns = multi_sep + np.array(list(df_result.columns)).T.tolist()
            else:
                multi_columns = np.array(list(df_result.columns)).T.tolist() + multi_sep
        df_result.columns = multi_columns
    return df_result


# Column Level이 다른 두개의 DataFrame의 Column Level을 맞추어 axis=1 방향으로 concat하는 함수
def fun_Concat_MultiColumnDF (df_left, df_right, fill='.', pre_fill=False):
    '''
    # Column Level이 다른 두개의 DataFrame의 Column Level을 맞추어 axis=1 방향으로 concat하는 함수

    < input >
    df_left(DataFrame) : Left Concat DataFrame
    df_right(DataFrame) : Rightt Concat DataFrame
        ※ Two DataFrame have difference column level
    fill(Str) : multi column변환에 따른 빈 column을 메울 값
    pre_fill(Boolean) : multi column변환에 따른 빈 column을 메울 값 (False: 뒤쪽, True: 앞쪽)

    < output >
    df_result(DataFrame) = pd.concat([df_left, df_right], axis=1)
    '''
    df_result=pd.DataFrame()
    if df_left.index.names == df_right.index.names:
        # str type : pd.core.indexes.base.Index
        # tupe type : pd.core.indexes.multi.MultiIndex
        df_Left_columns_level = 1 if type(df_left.columns[0]) == str else (len(df_left.columns[0]) if type(df_left.columns[0]) == tuple else 0)
        df_Right_columns_level = 1 if type(df_right.columns[0]) == str else (len(df_right.columns[0]) if type(df_right.columns[0]) == tuple else 0)

        if df_Left_columns_level < df_Right_columns_level:
            df_left_result = fun_Transform_MultiColumns(df_target=df_left, df_base=df_right, fill=fill, pre_fill=pre_fill)    # (Function)
            df_right_result = df_right.copy()
        else:
            df_left_result = df_left.copy()
            df_right_result = fun_Transform_MultiColumns(df_target=df_right, df_base=df_left, fill=fill, pre_fill=pre_fill)    # (Function)
        df_result = pd.concat([df_left_result, df_right_result], axis=1)
    return df_result


# DataFrame내 특정 Column에 Filtering 기준을 적용
def fun_Filtering_Column(data, on, criteria, method='any'):
    '''
    # DataFrame내 특정 Column에 Filtering 기준을 적용
    < input >
    data (DataFrame) : 적용할 Data-set
    on (String) : Column명
    criteria (String, List) : Filtering할 기준
    method : Filtering 기준 
        ㄴ 'any' 하나라도 포함된경우 
        ㄴ 'all' 모두 포함하는경우

    < output >
    result (DataFrame) : Filtering된 결과 Data-set
    '''
    if type(criteria) == list:
        criteria_list = criteria
    else:
        criteria_list = [criteria]

    result = pd.DataFrame()
    if method == 'any':
        result = data[data[on].apply(lambda x: np.nan if type(x) != str and np.isnan(x) else x.split(', '))
            .apply(lambda x: any(list(map(lambda y: int(y) in criteria_list, x))) )]
    elif method == 'all':
        result = data[data[on].apply(lambda x: np.nan if type(x) != str and np.isnan(x) else x.split(', '))
            .apply(lambda x: all(list(map(lambda y: int(y) in criteria_list, x))) )]
    return result


# DataFrame에 대해 filtering table기준을 적용(여러 Filtering 기준을 동시에 적용)
def fun_Filtering_DataFrame(data, criteria, unique_condition=True):
    '''
    # DataFrame에 대해 filtering table기준을 적용(여러 Filtering 기준을 동시에 적용)
    < input >
    data (DataFrame) : Filtering 적용할 Data-Set
    criteria (DataFrame) : Filtering 기준을 가지고 있는 Table

    < output >
    result_df (Object)
        ㄴ result_df['summary'] (DataFrame) : Group별 Data 매칭갯수
        ㄴ result_df['groups'] (Object) : Gruop별 index값
        ㄴ result_df['filter'] (DataFrame) : Group filtering 된 결과 DataFrame
        ㄴ result_df['result'] (DataFrame) : Group 값을 표기한 Column('criteria_index')이 표기된 Original DataFrame
    '''
    if type(criteria) != pd.core.frame.DataFrame:
        raise("'criteria' must be DataFrame type")

    criteria_df = criteria.copy()
    if unique_condition:
        criteria_df.drop_duplicates(inplace=True)      # 중복 제거

    condition_obj = {}
    for cl in criteria_df.columns:
        if ('이상' in cl) or ('min' in cl.lower()): #  or ('하한' in cl)
            condition_obj[cl] = '>='
        elif ('초과' in cl):
            condition_obj[cl] = '>'
        elif ('이하' in cl) or ('max' in cl.lower()):   #  or ('상한' in cl)
            condition_obj[cl] = '<='
        elif ('미만' in cl):
            condition_obj[cl] = '<'
        else:
            condition_obj[cl] = 'in'

    n=1
    result_obj = {}
    result_df = data.copy()
    result_df['criteria_index'] = np.nan
    result_groups = {}
    for r in list(criteria_df.index):
        print( f'{r} proceeding: {round(n/len(criteria_df)*100,1)} %')
        # print(criteria_df.iloc[r,:].to_frame().T)
        result_df_part = data.copy()
        for c in criteria_df.columns:
            if type(criteria_df.loc[r,c]) != str and np.isnan(criteria_df.loc[r,c]):
                pass
            elif condition_obj[c] == 'in':
                if result_df_part[c].dtype == 'int64' and criteria_df[c].dtype == 'float64':
                   criteria_df[c] = criteria_df[c].apply(lambda x: np.nan if np.isnan(x) else str(int(x)))
                condition_in = [str.strip(ci) for ci in str(criteria_df.loc[r,c]).split(',')]  # 앞뒤공백제거
                result_df_part = result_df_part[result_df_part[c].isin(condition_in)]
            else:
                range_c = ''        # range_c 변수 정의
                if 'min' in c.lower():
                    range_c = c[0:c.lower().find('min')]
                elif 'max' in c.lower():
                    range_c = c[0:c.lower().find('max')]
                else:
                    range_c = str.strip(c.split('하한')[0].split('이상')[0].split('초과')[0].split('상한')[0].split('이하')[0].split('미만')[0])
                result_df_part = eval('result_df_part[result_df_part[range_c]' + condition_obj[c] + str(criteria_df.loc[r,c]) + ']')
        result_groups[r] = result_df_part.index
        # 'criteria_index' Column에 해당 filtering 기준들을 입력 ----
        result_df.loc[result_df_part.index, 'criteria_index'] = result_df.loc[result_df_part.index, 'criteria_index'].apply(lambda x: x.__add__([r]) if type(x) == list else [r])
        n += 1
        clear_output(wait=True)
    # 'criteria_index' Column내 List형태의 값들을 String 형태로 치환 ----
    result_df['criteria_index'] = result_df['criteria_index'].apply(lambda x: ', '.join([str(y) for y in x]) if type(x) == list else np.nan )

    # Summary ----
    result_summary = criteria_df.copy()
    result_summary['count'] = np.nan
    for k in result_groups:
        result_summary.loc[k,'count'] = len(result_groups[k])
    result_summary['count'] = result_summary['count'].astype(int)
    print('** criteria DataFrame에 따른 분류 결과 **')
    print(result_summary)
    print('----------------------------------------')

    # Result ----
    result_obj['condition_obj'] = condition_obj
    result_obj['summary'] = result_summary
    result_obj['groups'] = result_groups
    result_obj['filter'] = result_df[result_df['criteria_index'].isna()==False]
    result_obj['result'] = result_df.copy()
    
    return result_obj






# --- 【 Group Calculation 】------------------------------------------------------------------------------------------
# DataFrame.groupby + Mean/Std Group
def fun_Concat_Group_MeanStd(base, groupby, on, count=True, point='auto'):
    if type(on) == list:
        on_list = on
    else:
        on_list = [on]
    MeanStd_df = pd.DataFrame()
    for c in on_list:
        if point == 'auto':
            df_total_agg = groupby[c].agg(['count','mean'])
            df_total_mean = np.nansum([np.product(x) for x in df_total_agg.values]) / np.nansum([x[0] for x in df_total_agg.values if (math.isnan(x[1]) ==False or x[0]!=0) ])
            if np.isnan(df_total_mean) == False:
                point_c = fun_Decimalpoint(df_total_mean)   # 자릿수 구하는 함수
            else:
                point_c = 0
        else:
            point_c = point        
    
        if count:
            MeanStd_df_part = groupby[c].agg(['count','mean','std'])
        else:
            MeanStd_df_part = groupby[c].agg(['mean','std'])
        MeanStd_df_part['mean'] = round(MeanStd_df_part['mean'], point_c)
        MeanStd_df_part['std'] = round(MeanStd_df_part['std'], point_c)
        MeanStd_df_part.columns = [[c]*len(MeanStd_df_part.columns), MeanStd_df_part.columns]
        MeanStd_df = pd.concat([MeanStd_df, MeanStd_df_part], axis=1)
    Result_df = pd.concat([base, MeanStd_df], axis=1)
    return Result_df


# DataFrame.groupby.Mean/Std + Cpk_Range
def fun_Concat_Group_Cpk_Range(base, on, criteria, n_min=5, point='auto'):
    if point == 'auto':
        on_mean = base[on]['mean'].mean()
        if np.isnan(on_mean) == False:
            point_c = fun_Decimalpoint(on_mean)   # 자릿수 구하는 함수
        else:
            point_c = 0
    else:
        point_c = point

    if type(criteria) == list:
        c_list = criteria
    else:
        c_list = [criteria]
    base_copy = base.copy()
    Cpk_Range_df = pd.DataFrame()
    print(point_c)
    for c in c_list:
        Cpk_Range_df = base_copy[on].apply(lambda x: 
            str(int(round(x['mean'] - 3*c*x['std'], point_c))) + '~' + str(int(round(x['mean'] + 3*c*x['std'], point_c)))
            if point_c <= 0
            else str(round(x['mean'] - 3*c*x['std'], point_c)) + '~' + str(round(x['mean'] + 3*c*x['std'], point_c))
            if x['count']>=n_min else '', axis=1).to_frame(name=(on, 'CpkRange ' + str(c)))        
        base_copy = pd.concat([base_copy, Cpk_Range_df], axis=1)
    Result_df = base_copy.copy()
    return Result_df


# DataFrame.groupby.Mean/Std + Cpk
def fun_Concat_Group_Cpk(base, on, criteria, n_min=5):
    if type(criteria[0]) == list:
        c_list = criteria
    else:
        c_list = [criteria]
    Cpk_df = pd.DataFrame()
    for c in c_list:
        CpkValue = base.apply(lambda x : round(min( x[on]['mean'] - c[0], c[1] - x[on]['mean']) / (3 * x[on]['std']), 2)
            if x[on]['count']>=n_min and math.isnan(x[on]['std'])==False and x[on]['std']!=0 else np.nan, axis=1)

        CpkUL = base.apply(lambda x : ('U' if x[on]['mean'] - c[0] > c[1] - x[on]['mean'] else 'L')
            if x[on]['count']>=n_min and math.isnan(x[on]['std'])==False and x[on]['std']!=0 else np.nan, axis=1)
        Cpk_df_part = pd.concat([CpkValue, CpkUL], axis=1)
        Cpk_df_part.columns = ['Cpk '+'~'.join(map(lambda x: str(x), c)),
                        'CpkUL '+'~'.join(map(lambda x: str(x), c))]
        Cpk_df = pd.concat([Cpk_df, Cpk_df_part], axis=1)
    Cpk_df.columns = [[on]*len(Cpk_df.columns), Cpk_df.columns]
    Result_df = pd.concat([base, Cpk_df], axis=1)
    return Result_df


# DataFrame.groupby.count + RejectN /RejectRatio
def fun_Concat_Group_Reject(base, groupby, on, criteria):
    if type(criteria[0]) == list:
        c_list = criteria
    else:
        c_list = [criteria]
    Reject_df = pd.DataFrame()
    for c in c_list:
        filter_len_list =[]
        for i, v in groupby:        # print(i)        # print(v)
            filter_len_list.append(len(v[(v[on] < c[0] ) | (v[on] >= c[1])]))
        filter_len_list
        Reject_df_part = base.copy()[on]
        Reject_df_part['RejectN '+'~'.join(map(lambda x: str(x), c))] = filter_len_list
        Reject_df_part['RejectRatio '+'~'.join(map(lambda x: str(x), c))] = round(Reject_df_part['RejectN '+'~'.join(map(lambda x: str(x), c))] / Reject_df_part['count']*100,2)
        Reject_df_part = Reject_df_part[['RejectN '+'~'.join(map(lambda x: str(x), c)), 'RejectRatio '+'~'.join(map(lambda x: str(x), c))]] 
        Reject_df = pd.concat([Reject_df, Reject_df_part], axis=1)
    Reject_df.columns = [[on]*len(Reject_df.columns), Reject_df.columns]

    Result_df = pd.concat([base, Reject_df], axis=1)
    return Result_df


def fun_Drop_Var_Count(base, on):
    Drop_df = base.copy()

    if type(on) == list:
        on_list = on
    else:
        on_list = [on]
    for c in on_list:
        result_df_part = pd.DataFrame()
        if Drop_df[c]['count'].equals(Drop_df['Total']['count']) or Drop_df[c]['count'].values==0:
            Drop_df_part = Drop_df[c].drop('count', axis=1)
            Drop_df_part.columns = [[c]*len(Drop_df_part.columns),Drop_df_part.columns]
            Drop_df.drop(c, inplace=True, axis=1)
            Drop_df = pd.concat([Drop_df, Drop_df_part], axis=1)
        else :
            Drop_df = Drop_df
    return Drop_df




# -------------------------------------------------------------------------------------------------------------------------------------------------
# df에 대해 group에 따른 on항목 기준별 Cpk 및 불량률 산출 (Column_Wise)
def fun_Performance_Capability_Col (df, group, on, 
        group_tensile_dir=False, group_tensile_cd=False,
        first_test=True, cpk=True, rejectRatio=True):
    '''
    # df에 대해 group에 따른 on항목 기준별 Cpk 및 불량률 산출 (Column_Wise)

    < input >
    df : Base DataFrame
    group : Index_Group Variables
    on : Performance Variable  ex) YP, TS, EL...
    cpk : Cpk Result
    rejectRatio : Reject Ratio Result

    < output >
    DataFrame : df.groupby(group).agg({'count', 'mean', 'std', 'Cpk', 'RejectN', 'RejectRatio'}).unstack(on기준)
    '''

    perform_target = '초_' + on if first_test else on
    perfom_column_max = np.power(10, np.ceil(np.log10(df[perform_target].max()))) 

    if on + '_하한' in list(df.columns) == False:    df[on + '_하한'] = 0
    if on + '_상한' in list(df.columns) == False:    df[on + '_상한'] = perfom_column_max
    df[on + '_하한'] = df[on + '_하한'].fillna(0)
    df[on + '_상한'] = df[on + '_상한'].fillna(perfom_column_max)

    tensile_group = []
    if group_tensile_dir:
        tensile_group.append('인장_방향')

    if group_tensile_cd:
        if '인장_호수그룹_길이' not in df.columns:
            df['인장_호수그룹_길이'] = df['인장_호수'].apply(lambda x: '05 (80)' if x==5 else '04, 06 (50)' )
        tensile_group.append('인장_호수그룹_길이')

    perform_group1 = group + tensile_group
    perform_group2 = group + tensile_group + [on + '_하한', on + '_상한']      

        # (10) 제조기준 Group
    df_perform_table11 = df.groupby(perform_group1)[perform_target].agg(['count','mean','std'])

        # (20) 제조기준 + 보증기준 Group
    df_perform_group21 = df.groupby(perform_group2)
    df_perform_table21 = df_perform_group21[perform_target].count().to_frame()
    df_perform_table21.columns = [on + '_critiera_count']
    df_perform_table22 = df_perform_table21.reset_index([on + '_하한', on + '_상한'])

        # (30) 제조기준 + 보증기준 group별 : count, mean, std, cpk, reject
    df_perform_group31 = pd.DataFrame()
    n = 0
    for gi, gv  in df_perform_group21:
        n +=1;
        print(f"제조기준 + 보증기준 group별 '{perform_target}' 'count, mean, std, cpk, reject'  Calculate... ")
        print( f"Process: {round(n/df_perform_group21.ngroups*100,1)}%" )
        group = list(gi)
        group.pop()
        group.pop()

        g_obj = {}
        g_series = df_perform_table11.loc[tuple(group)] 
        g_obj['min'] = gv[on + '_하한'].max()
        g_obj['max'] = gv[on + '_상한'].min()
        g_obj['count'] = g_series['count']
        g_obj['mean'] = g_series['mean'] 
        g_obj['std'] = g_series['std'] 
        # print(n)
        if g_obj['count'] >=5 and np.isnan(g_obj['std'])==False and g_obj['std'] != 0:
            da_upper = g_obj['max'] - g_obj['mean']
            da_lower = g_obj['mean'] -  g_obj['min'] 
            if da_upper < da_lower:     # Upper
                g_obj['cpk'] = 0.01 if da_upper < 0 else round(da_upper / (3 * g_obj['std']), 2)
            else:                       # Lower
                g_obj['cpk'] = -0.01 if da_lower < 0 else -1* round(da_lower / (3 * g_obj['std']), 2)
        else:
            g_obj['cpk'] = np.nan
        g_obj['reject_count'] = len(gv)
        g_obj['rejectN'] = ((gv[perform_target] < g_obj['min']).sum() + (gv[perform_target] > g_obj['max']).sum())
        g_obj['rejectRatio'] = round( g_obj['rejectN'] / g_obj['reject_count'] * 100, 2)
        g_df = pd.Series(g_obj).to_frame().T
        df_perform_group31 = pd.concat([df_perform_group31, g_df], axis=0)
        clear_output(wait=True)

    df_perform_group31.index = df_perform_table22.index

    df_perform_group31['represent'] = df_perform_group31.groupby(df_perform_group31.index)['count'].rank(method='first').apply(lambda x: '*' if x==1 else np.nan).to_frame()
    df_perform_group30 = df_perform_group31.copy()
    df_perform_group30.to_clipboard()

        # (40) (row)제조기준 + (column)보증기준 unstack → Table화
    df_perform_group42 = df_perform_group30.copy()
    df_perform_group42['Item'] = on
    df42_index = df_perform_group42.index.names + ['min', 'max'] + ['Item']
    df_perform_group42.reset_index(inplace=True)
    df_perform_group42.set_index(df42_index, inplace=True)
    
    unstack_group = ['Item'] + tensile_group + ['min','max']
    # Cpk
    df_perform_group42_cpk = df_perform_group42[['cpk']].unstack(unstack_group).sort_index(level=0, axis=1)
    df_perform_group42_cpk = df_perform_group42_cpk.swaplevel(i=0,j=1, axis=1)
    # RejectRatio
    df_perform_group42_rejectRatio = df_perform_group42[['rejectRatio']].unstack(unstack_group).sort_index(level=0, axis=1)
    df_perform_group42_rejectRatio = df_perform_group42_rejectRatio.swaplevel(i=0,j=1, axis=1)

    result_df = pd.DataFrame()
    if cpk == True and rejectRatio == True:
        result_df = fun_Concat_MultiColumnDF(df_left=df_perform_group42_cpk, df_right=df_perform_group42_rejectRatio, fill='.', pre_fill=False)      # (function)
    elif cpk == True and rejectRatio == False:
        result_df = df_perform_group42_cpk
    elif cpk == False and rejectRatio == True:
        result_df = df_perform_group42_rejectRatio
    return result_df
# -------------------------------------------------------------------------------------------------------------------------------------------------


# df에 대해 group에 따른 on항목 기준별 Cpk 및 불량률 산출 (Row_Wise)
def fun_Performance_Capability_Row (df, on, 
        group=[], mtc_group=False,
        group_tensile_dir=False, group_tensile_cd=False,
        first_test=True, cpk=True, rejectRatio=True):
    '''
    # df에 대해 group에 따른 on항목 기준별 Cpk 및 불량률 산출 (Row_Wise)

    < input >
    df : Base DataFrame
    group : Index_Group Variables
    on : Performance Variable  ex) YP, TS, EL...
    cpk : Cpk Result
    rejectRatio : Reject Ratio Result

    < output >
    DataFrame : df.groupby(group).agg({'count', 'mean', 'std', 'Cpk', 'RejectN', 'RejectRatio'}).unstack(on기준)
    '''
  
    on_list = on if type(on) == list else [on]

    df_perform = df.copy()
    if first_test:
        on_list = ['초_'+oc for oc in on_list]

    tensile_group = []
    if group_tensile_dir:
        tensile_group.append('인장_방향')

    if group_tensile_cd:
        if '인장_호수그룹_길이' not in df_perform.columns:
            df_perform['인장_호수그룹_길이'] = df_perform['인장_호수'].apply(lambda x: '05 (80)' if x==5 else '04, 06 (50)' )
        tensile_group.append('인장_호수그룹_길이')
    
    mechanical_criteria = []
    if mtc_group:
        criteria_min = ['_하한', '_min']
        criteria_max = ['_상한', '_max']
        for oc in on_list:
            o = oc.replace('초_','') if '초_' in oc else oc
            for c_min in criteria_min:
                m_min = 0
                if len(fun_Search(on=o+c_min, data=df_perform, method='equal', case=False)) > 0:
                    mechanical_criteria.append(o+c_min)
                    df_perform[o+c_min] = df_perform[o+c_min].fillna(m_min)

            for c_max in criteria_max:
                m_max = np.power(10, np.ceil(np.log10(df_perform[o].mean()))) 
                if len(fun_Search(on=o+c_max, data=df_perform, method='equal', case=False)) > 0:
                    mechanical_criteria.append(o+c_max)
                    df_perform[o+c_max] = df_perform[o+c_max].fillna(m_max)

    group_index = group + tensile_group + mechanical_criteria
    if len(group_index) == 0:
        df_perform['total'] = 'total'
        group_index=['total']
    df_perform_group21 = df_perform.groupby(group_index)

    n = 0
    result = pd.DataFrame()
    for gi, gv in df_perform_group21:
        n +=1
        print( f"Process: {round(n/df_perform_group21.ngroups*100,1)}%")
        
        result_part = pd.DataFrame([len(gv)], columns=['count'])
        for oc in on_list:
            g_obj = {}
            o = oc.replace('초_','') if '초_' in oc else oc

            # min
            if len(fun_Search(on=o+'_하한', data=gv, method='equal', case=False))>0:
                g_obj['min'] = gv[o + '_하한'].max()
                if o+'_하한' not in group_index:
                    g_obj['하한'] = gv[o + '_하한'].max()
            elif len(fun_Search(on=o+'_min', data=gv, method='equal', case=False))>0:
                g_obj['min'] = gv[o + '_min'].max()
                if o+'_min' not in group_index:
                    g_obj['하한'] = gv[o + '_min'].max()
            else:
                g_obj['min'] = 0
            # max
            if len(fun_Search(on=o+'_상한', data=gv, method='equal', case=False))>0:
                g_obj['max'] = gv[o + '_상한'].min()
                if o+'_상한' not in group_index:
                    g_obj['상한'] = gv[o + '_상한'].min()
            elif len(fun_Search(on=o+'_max', data=gv, method='equal', case=False))>0:
                g_obj['max'] = gv[o + '_max'].min()
                if o+'_max' not in group_index:
                    g_obj['상한'] = gv[o + '_max'].min()
            else:
                g_obj['max'] = np.power(10, np.ceil(np.log10(gv[o].mean())))

            g_obj['count'] = len(gv)
            g_obj['mean'] = gv[oc].mean()
            g_obj['std'] = gv[oc].std()
            if g_obj['count'] >=5 and np.isnan(g_obj['std'])==False and g_obj['std'] != 0:
                da_upper = g_obj['max'] - g_obj['mean']
                da_lower = g_obj['mean'] -  g_obj['min'] 
                if da_upper < da_lower:     # Upper
                    g_obj['cpk'] = 0.01 if da_upper < 0 else round(da_upper / (3 * g_obj['std']), 2)
                else:                       # Lower
                    g_obj['cpk'] = -0.01 if da_lower < 0 else -1* round(da_lower / (3 * g_obj['std']), 2)
            else:
                g_obj['cpk'] = np.nan
            dec_point = fun_Decimalpoint(g_obj['mean'])
            g_obj['mean'] = round(g_obj['mean'], dec_point-1)
            g_obj['std'] = round(g_obj['std'], dec_point)
            g_obj['reject_count'] = len(gv)
            g_obj['rejectN'] = ((gv[oc] < g_obj['min']).sum() + (gv[oc] > g_obj['max']).sum())
            g_obj['rejectRatio'] = round( g_obj['rejectN'] / g_obj['reject_count'] * 100, 2)
            g_df = pd.Series(g_obj).to_frame().T
            g_df.drop(['count', 'min', 'max', 'reject_count', 'rejectN'], axis=1,inplace=True)

            if oc in group_index:
                 g_df.drop(['min', 'max'], axis=1,inplace=True)
            if not cpk:
                g_df.drop(['cpk'], axis=1,inplace=True)
            if not rejectRatio:
                g_df.drop(['rejectRatio'], axis=1,inplace=True)
            g_df.columns = [[oc]*len(g_df.columns), list(g_df.columns)]

            result_part = fun_Concat_MultiColumnDF(df_left=result_part, df_right=g_df, fill='Total', pre_fill=True)
        result = pd.concat([result, result_part], axis=0)    
        clear_output(wait=True)
    result.index = df_perform_group21.count().index

    return result
# -------------------------------------------------------------------------------------------------------------------------------------------------


