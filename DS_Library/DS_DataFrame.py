import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns

import scipy as sp
import statsmodels.api as sm

from IPython.core.display import display, HTML

import tqdm

import inspect
import re
import copy
from copy import deepcopy
import pyperclip

import collections
from collections import namedtuple
from functools import reduce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


from DS_Plot import distbox

# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
# df = pd.read_csv(absolute_path + 'dataset_city.csv')
# df = pd.read_csv(absolute_path + 'Bank.csv')

# DF_Summary(df)



# Return Instance Name
# import traceback
# class SomeObject():
#     def __init__(self, def_name=None):
#         if def_name == None:
#             (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
#             def_name = text[:text.find('=')].strip()
#         self.instance_name = def_name
#
# abc = SomeObject() 
# abc.instance_name         # abc


# 【 Data function 】  ################################################################################

# Dictionary를 보기좋게 Printing 해주는 함수
def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + '【'+ str(key) + '】')
            print_dict(value, indent+1)
        else:
            print('\t' * indent + '【'+ str(key) + '】', end=' : ')
            print(str(value))


# 정의된 변수명을 return하는 함수
def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


# DecimalPoint : 어떤 값에 대하여 자동으로 소수점 자리수를 부여
def fun_Decimalpoint(value):
    if value == 0:
        return 3
    try:
        point_log10 = np.floor(np.log10(abs(value)))
        point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
    except:
        point = 0
    return point

# 'num ~ num' format data → [num, num]
def criteria_split(criteria, error_handle=np.nan):
    try:
        if criteria == '' or pd.isna(criteria):
            return [np.nan, np.nan]
        else:
            criteria_list = list(map(lambda x: x.strip(), criteria.split('~')))
            criteria_list[0] = -np.inf if criteria_list[0] == '' else float(criteria_list[0])
            criteria_list[1] = np.inf if criteria_list[1] == '' else float(criteria_list[1])
            return criteria_list
    except:
        if error_handle == 'error':
            raise ValueError("An unacceptable value has been entered.\n .Allow format : num ~ num ")
        else:
            return [error_handle] * 2
    
# 'num ~ num' format series → min: [num...], max: [num...] seriess
def lsl_usl_split(criteria_data):
    if criteria_data.ndim == 1:
        splited_series = criteria_data.apply(lambda x: pd.Series(criteria_split(x))).apply(lambda x: x.drop_duplicates().apply(lambda x: x if -np.inf < x < np.inf else np.nan ).sort_values().dropna().to_list(), axis=0)
        splited_series.index = ['lsl','usl']
        splited_series.name = criteria_data.name
        return splited_series.to_dict()
    elif criteria_data.ndim == 2:
        result_dict = {}
        for c in criteria_data:
            splited_series = criteria_data[c].apply(lambda x: pd.Series(criteria_split(x))).apply(lambda x: x.drop_duplicates().apply(lambda x: x if -np.inf < x < np.inf else np.nan ).sort_values().dropna().to_list(), axis=0)
            splited_series.index = ['lsl','usl']
            splited_series.name = c
            result_dict[c] =  splited_series.to_dict()
        return result_dict




# 특정 값이나 vector에 자동으로 소수점 부여
# function auto_formating
def auto_formating(x, criteria='max', return_type=None, decimal=None, decimal_revision=0, thousand_format=True):
    special_case=False

    if type(x) == str:
        return x
    
    x_type_str = str(type(x))
    if 'int' in x_type_str or 'float' in x_type_str:
        if np.isnan(x) or x is np.nan:
            return np.nan
        x_array = np.array([x])
    else:
        x_array = np.array(x)
    x_Series_dropna = pd.Series(x_array[np.isnan(x_array) == False])

    # 소수점 자릿수 Auto Setting
    if decimal is None:
        if criteria == 'median':
            quantile = Quantile(q=0.5)
            criteria_num = quantile(x_Series_dropna)
        elif 'q' in criteria:
            quantile = Quantile(q=float(criteria.replace('q',''))/100)
            criteria_num = quantile(x_Series_dropna)
        elif criteria == 'mode':
            criteria_num = x_Series_dropna.mode()[0]
        else:
            criteria_num = eval('x_Series_dropna.' + criteria + '()')

        decimal = fun_Decimalpoint(criteria_num) + decimal_revision
    
    # 소수점 자릿수에 따른 dtype 변환
    if decimal < 1:
        if x_Series_dropna.min() == -np.inf or x_Series_dropna.max() == np.inf:
            result_Series = pd.Series(x_array).round().astype('float')
            special_case = 'inf'
        else:
            result_Series = pd.Series(x_array).round().astype('Int64')
    else:
        result_Series = pd.Series(x_array).apply(lambda x: round(x, decimal))
    
    # Output dtype 변환
    if return_type == 'str':
        result_Series = result_Series.apply(lambda x: '' if (type(x) == pd._libs.missing.NAType or np.isnan(x) or x is np.nan) else str(x) )
        if thousand_format:
            if decimal < 1:
                result_Series = result_Series.apply(lambda x: '' if x == '' else (str(x) if abs(float(x)) == np.inf else format(int(x), ',')) )
                # result_Series = result_Series.apply(lambda x: '' if x == '' else format(int(x), ','))
            else:    
                result_Series = result_Series.apply(lambda x: '' if x == '' else format(float(x), ','))
    elif return_type == 'float':
        if decimal < 1:
            result_Series = result_Series.round().astype('Int64')
        else:
            result_Series = result_Series.astype(float)
    elif return_type == 'int':
        result_Series = result_Series.round().astype('Int64')
       
    # Output Type에 따른 Return
    if 'str' in x_type_str or 'float' in x_type_str or 'int' in x_type_str:
        if return_type is None:
            if 'str' in x_type_str:
                return str(result_Series[0])
            elif 'float' in x_type_str:
                if decimal < 1:
                    if special_case == 'inf':
                        return float(result_Series[0])
                    else:
                        return int(result_Series[0])
                else:
                    return float(result_Series[0])
            elif 'int' in x_type_str:
                return int(result_Series[0])
        else:
            return eval(return_type + '(result_Series[0])')
    elif 'array' in x_type_str:
        return np.array(result_Series)
    elif 'Series' in x_type_str:
        return result_Series


# Series Paste from Clipboard
# import pandas as pd
def read_clipboard(copy=False, drop_duplicates=True, cut_string=None, return_type='tuple', sep=', '):
    pasted_frame = pyperclip.paste()
    if '\t' in pasted_frame:
        pasted_list = pasted_frame.replace('\r\n','').split('\t')
    else:
        pasted_list = pasted_frame.split('\r\n')[:-1]
    
    if cut_string:
        pasted_list = [e[:cut_string] for e in pasted_list]

    if drop_duplicates:
        pasted_list = list({e:e for e in pasted_list}.keys())

    if return_type == 'tuple':
        pasted_list = tuple(pasted_list)
    elif return_type == 'str':
        pasted_list = sep.join(pasted_list)
    elif 'series'  in return_type.lower():
        pasted_list = pd.Series(pasted_list)

    if copy:
        print(' * Copy pasted_list to the clipboard.')
        if return_type == 'series':
            pasted_list.to_clipboard(index=False, header=False)
        else:
            return pyperclip.copy(str(pasted_list))
    else:
        return pasted_list


# add new row
def add_row(data, content='index'):
    if content == 'index':
        addrow_list = list(data.columns)
        
    else:
        addrow_list = [content] * len(data.columns)
    addrow_DF = pd.Series(addrow_list).to_frame().T
    addrow_DF.columns = data.columns
    addrow_DF.index = [np.nan]
    return addrow_DF.append(data)


# print DataFrame to HTML
def print_DataFrame(data):
    display(HTML(data._repr_html_()))



# 【 String Appliable function 】 ################################################################################
# DataFrame의 Columns을 Search 해주는 함수
def search(data, query=None):
    # dc = np.array(list(map(lambda x: x.lower(), data.columns)))
    dc = data.columns
    if query is None:
        return np.array(dc)
    elif sum([c in query for c in ['>', '<', '==', '.isin(', '.str.contains(']]):
        return pd.DataFrame(dc, columns=['columns']).query('columns' + query).values.ravel()
    else:
        return pd.DataFrame(dc, columns=['columns']).query(f"columns.str.contains('{query}')").values.ravel()

# Series의 Column을 list로 바꿔주는 함수 (nan처리, autoformating 기능 포함)
def list_transform(series, columns, fillna=None, auto_format=False, return_type=None, decimal_revision=0, thousand_format=True):
    """
    < input >
        . series: pandas.Series
        . columns: listed columns
        . fillna (default None): fillna
        . auto_format (default False): if True, auto_formating is apply at each element
            .return_type (default None)
            .decimal_revision (default 0)
            .thousand_format (default None)
        
    < output >
        . element list
    """
    result_list = []
    for c in columns:
        if type(c) != str and np.isnan(series[c]):
            if fillna is not None:
                result_list.append(fillna)
            else:
                if auto_format:
                    result_list.append(auto_formating(series[c], return_type=return_type, decimal_revision=decimal_revision, thousand_format=thousand_format))
                else:
                    result_list.append(series[c])
        else:
            if auto_format:
                result_list.append(auto_formating(series[c], return_type=return_type, decimal_revision=decimal_revision, thousand_format=thousand_format))
            else:
                result_list.append(series[c])
    return result_list


# 선택된 Column들을 하나의 값으로 묶어주는 함수
def join_series(data, columns='all', join='~', fillna=None, decimal_revision=0):
    if columns is 'all':
        target_columns = list(data.columns)
    else:
        target_columns = columns
    data_auto_format = data[target_columns].apply(lambda x: auto_formating(x, return_type='str', decimal_revision=decimal_revision),axis=0)
    data_list = data_auto_format.apply(lambda x: list_transform(series=x, columns=target_columns), axis=1)
    data_join = data_list.apply(lambda x: '~'.join(np.array(x).astype(str)) )
    return data_join


# DataFrame split to object, numeric variables
def dtypes_split(data, return_type='columns'):
    """
    < input >
     . data : DataFrame
     . return_type : 'columns', 'columns_list', 'columns_all', 'data', 'tuple_data'
        └ columns : {'col1': 'dtype1', 'col2': 'dtype2' ...}
        └ columns_list : {'numeric': 'col1', 'col2'..., 'object': ..., 'time'}
        └ columns_all : {'col1': dtype1, 'col2': dtype2 ...}
        └ data : {'numeric': DataFrame, 'object': DataFrame, 'time': DataFrame}
        └ tuple_data : ((numeric)DataFrame, (object)DataFrame, (time)DataFrame}

    """
    result = {'numeric':[], 'object':[], 'time':[]}
    result_record = {}
    result_all = {}

    for c, d in data.dtypes.items():
        if sum([n in str(d).lower() for n in ['int', 'float']]) > 0:
            result['numeric'].append(c)
            result_record[c] = 'numeric'
            result_all[c] = {'dtype':d, 'dtype_group':'numeric'}
        elif sum([n in str(d).lower() for n in ['object', 'category']]) > 0:
            result['object'].append(c)
            result_record[c] = 'object'
            result_all[c] = {'dtype':d, 'dtype_group':'object'}
        elif 'time' in str(d).lower():
            result['time'].append(c)
            result_record[c] = 'time'
            result_all[c] = {'dtype':d, 'dtype_group':'time'}
    if return_type == 'columns':
        return result_record
    elif return_type == 'columns_list':
        return result
    elif return_type == 'columns_all':
        return result_all
    elif return_type == 'data':
        return {'numeric': data[result['numeric']], 'object': data[result['object']], 'time':data[result['time']]}
    elif return_type == 'tuple_data':
        return (data[result['numeric']], data[result['object']], data[result['time']])


# 이전데이터와 새로나온 데이터들을 key를 기준으로 위아래로 붙여주는 함수
def update_data(old_data, new_data, key, sort_values=None, ascending=False, verbose=1):
    key_list = [key] if type(key) != list else key
    
    new_key = new_data[key_list]
    new_key['new_old_'] = 'new'
    
    old_key = old_data[key_list]

    old_vlookup = pd.merge(left=old_key, right=new_key, on=key_list, how='left')
    filtered_old_data = old_data[old_vlookup['new_old_'].isna()]
    
    old_new = pd.concat([filtered_old_data, new_data], axis=0)

    if sort_values is not None:
        old_new_data = old_new.sort_values(by=sort_values, ascending=ascending, axis=0).reset_index(drop=True)
    else:
        old_new_data = old_new.reset_index(drop=True)
    
    if verbose:
        print(f'{old_data.shape} + {new_data.shape} → {old_new_data.shape}')
        
    return old_new_data




# 【 Operation function 】 ################################################################################
# def ttest_each
# 여러개의 Group별로 평균, 편차, ttest 결과를 Return 하는 함수
# import scipy as sp
# from collections import namedtuple
from itertools import combinations
def ttest_each(data, x, group, equal_var=False, decimal_point=4, return_result='all', return_type='vector'):
    """
    < input >
     . data (DataFrame): DataFrame
     . x (str): column name
     . group (str, list): grouping columns
     . equal_var (bool): whether variance is equal between group when processing ttest
     . decimal_point (int, None): pvalue decimal 
     . return_result (str): 'all', 'count', 'mean', 'std', 'ttest', 'plot'
     . return_type (str): 'matrix', 'vector'

    < output >
     . table by group (table)
    """
    result = namedtuple('ttest_each', ['count', 'mean', 'std', 'ttest'])

    if type(group) == list and len(group) > 1:
        group_unique = data[group].sort_values(by=group).drop_duplicates()
        # group_index_names = group_unique.apply(lambda x: ', '.join([f"{idx}: {v}" for idx, v in zip(x.index, x)]),axis=1).tolist()
        group_index = pd.MultiIndex.from_frame(group_unique)
        groups = group.copy()
    else:
        if type(group) == list:
            group_unique = data[group[0]].drop_duplicates()
            groups = group.copy()
        elif type(group) == str:
            group_unique = data[group].drop_duplicates()
            groups = [group].copy()
        # group_index_names = group_unique.copy()
        group_index = group_unique.to_list().copy()
    # print(group_index)
    # print(groups)

    group_table = pd.DataFrame(np.zeros(shape=(len(group_index), len(group_index))), index=group_index, columns=group_index)
    group_table[group_table== 0] = np.nan
    
    table_count = group_table.copy()
    table_mean = group_table.copy()
    table_std = group_table.copy()
    table_ttest = group_table.copy()
    table_plot = group_table.copy()

    groups_dict = {}
    for gi, gv in data.groupby(groups):
        groups_dict[gi] = np.array(gv[x])

    vector_table_list = []
    for g in combinations(group_index, 2):
        data_group = [groups_dict[g[1]], groups_dict[g[0]]]
        data_group_count = [int(len(x)) for x in data_group]
        data_group_mean = [auto_formating(np.mean(x)) for x in data_group]
        data_group_std = [auto_formating(np.std(x)) for x in data_group]

        group_count = f" {data_group_count[1]} - {data_group_count[0]}"
        group_mean = f" {data_group_mean[1]} - {data_group_mean[0]}"
        group_std = f" {data_group_std[1]} - {data_group_std[0]}"
        group_ttest = sp.stats.ttest_ind(data_group[1], data_group[0], equal_var=equal_var).pvalue
        if decimal_point is not None:
            group_ttest = round(group_ttest, decimal_point)
            
        table_count.loc[g[0], g[1]] = group_count
        table_mean.loc[g[0], g[1]] = group_mean
        table_std.loc[g[0], g[1]] = group_std
        table_ttest.loc[g[0], g[1]] = group_ttest
        
        # if return_result == 'plot':
        data_group1 = pd.Series(data_group[1], name=x).to_frame()
        data_group1[group] = g[1]
        data_group2 = pd.Series(data_group[0], name=x).to_frame()
        data_group2[group] = g[0]
        data_concat = pd.concat([data_group1, data_group2], axis=0)
        group_plot = distbox(data=data_concat, on=x, group=group)
        table_plot.loc[g[0], g[1]] = group_plot
        
        vector_table_list.append([g[0], g[1], group_count, group_mean, group_std, group_ttest, group_plot])
    vector_table = pd.DataFrame(vector_table_list, columns=['group1', 'group2', 'count','mean', 'std', 'ttest', 'plot']).set_index(['group1', 'group2'])
    
    if return_result == 'all':
        return result(table_count, table_mean, table_std, table_ttest) if return_type == 'matrix' else vector_table
    elif return_result == 'count':
        return table_count if return_type == 'matrix' else vector_table['count']
    elif return_result == 'mean':
        return table_mean if return_type == 'matrix' else vector_table['mean']
    elif return_result == 'std':
        return table_std if return_type == 'matrix' else vector_table['std']
    elif return_result == 'ttest':  
        return table_ttest if return_type == 'matrix' else vector_table['ttest']
    elif return_result == 'plot':  
        return table_plot if return_type == 'matrix' else vector_table['plot']





# class Quantile
class Quantile():
    def __init__(self, q=0.25):
        self.q_ = q
        self.set_name()
   
    def q(self, q=0.25):
        self.q_ = q
        self.set_name()
   
    def __call__(self, x, q=False):
        if type(q) == bool and q == False:
            q = self.q_
            self.set_name()
           
        self.quantile = x.quantile(q=q)
       
        return self.quantile
   
    def set_name(self):
        name = 'q' + format(str(int(self.q_*100)), '0>2s')
        self.__class__.__name__ = name
        return name

    def __repr__(self):
        try:
            return self.quantile
        except:
            return str(f'Quantile.object.{self.set_name()}')
   
    def __str__(self):
        self.__repr__()

# Describe Numeric Series Data
# class Describe()
class Describe():
    def __init__(self, x=False, mode='dict'):
        self.x = False if type(x) == bool and x == False else x
        self.mode = mode
   
    def set_x(self, x):
        self.x = x

    def calc_sigma(self, x=False, sigma=3):
        if type(x)==bool and x==False:
            x = self.x
        else:
            self.x = x

        df_describe = x.agg(['count','mean', 'std', 'min', 'max'])

        uf_sigma = df_describe['mean'] + sigma * df_describe['std']
        lf_sigma = df_describe['mean'] - sigma * df_describe['std']

        self.sigmaOutlier = pd.Series({'lf_sigma': lf_sigma, 'uf_sigma': uf_sigma})
        self.sigmaDescribe = pd.concat([df_describe, self.sigmaOutlier])

        self.df_describe = df_describe
        self.uf_sigma = uf_sigma
        self.lf_sigma = lf_sigma
        return self.sigmaDescribe

    def calc_quantile(self, x=False, q=[0, 0.01, 0.02, 0.03, 0.05, 0.1 , 0.25, 0.5 , 0.75, 0.9 , 0.95, 0.97, 0.98, 0.99, 1]):
        if type(x)==bool and x==False:
            x = self.x
        else:
            self.x = x
        # if x

        q_list = []
        if 0.25 not in q_list:
            q_list.append(0.25)
        if 0.75 not in q_list:
            q_list.append(0.75)

        for q_i in np.array(q):
            q_list.append(Quantile(q_i))
        quantiles = x.agg(q_list)
   
        iqr = quantiles['q75'] - quantiles['q25']

        self.uof_box = quantiles['q75'] + 3 * iqr    # upper inner fence
        self.uif_box = quantiles['q75'] + 1.5 * iqr    # upper inner fence
        self.lif_box = quantiles['q25'] - 1.5 * iqr    # lower outer fence
        self.lof_box = quantiles['q25'] - 3 * iqr    # lower outer fence

        self.boxOutlier = pd.Series({'lof_box': self.lof_box, 'lif_box': self.lif_box,
                                    'uif_box': self.uif_box, 'uof_box': self.uof_box})
        self.boxDescribe = pd.concat([quantiles, self.boxOutlier])

        self.quantiles = quantiles
        self.iqr = iqr
        return self.boxDescribe

    def describe(self, x=False, sigma=3, q=[0, 0.01, 0.02, 0.03, 0.05, 0.1 , 0.25, 0.5 , 0.75, 0.9 , 0.95, 0.97, 0.98, 0.99, 1]):
        if type(x)==bool and x==False:
            x = self.x
        else:
            self.x = x

        self.calc_quantile(x, q)
        self.calc_sigma(x, sigma)

        self.summary = pd.concat([self.sigmaDescribe, self.boxDescribe])
        self.outlier = pd.concat([self.sigmaOutlier, self.boxOutlier])
        self.all = pd.concat([self.sigmaDescribe, self.boxDescribe])
        if self.mode.lower() == 'series':
            return self.all
        if self.mode.lower() == 'dict':
            return self.all.to_dict()
   
    def __call__(self, x):
        if type(x)==bool and x==False:
            x = self.x
        else:
            self.x = x
        return self.describe(x=x)



# Series에 특정 format형태의 연산결과를 agg function을 활용해 사용자 정의 형태로 리턴
# class AggFormating
class AggFormating:
    """
    < input >
        ※ Appliable functions: 'min', 'max', 'std', 'mode', 'q??'
    < output >
    """
    def __init__(self, return_format=None, decimal=None, decimal_revision=0, thousand_format=True):
        self.return_format = return_format
        self.decimal = decimal
        self.decimal_revision = decimal_revision
        self.thousand_format = thousand_format
            
    def format(self, x, return_format=None, decimal=None, decimal_revision=None, thousand_format=None):
        if return_format is None:
            return_format = self.return_format
        if decimal is None:
            decimal = self.decimal
        if decimal_revision is None:
            decimal_revision = self.decimal_revision
        if thousand_format is None:
            thousand_format = self.thousand_format

        positions = [(o.start(),c.end()) for o,c in zip(re.finditer('{', return_format), re.finditer('}', return_format))]
        q_start = [qo.start() for qo in re.finditer('{q', return_format)]

        string_list = []
        start_index = 0
        q_positions = []

        calc_result = {}

        for i, (o, c) in enumerate(positions):
            string_list.append(return_format[start_index:o])    

            start_index = c
            formula_str = return_format[o+1:c-1]
            # operation
            while True:
                q_search = re.search('q[0-9][0-9]', formula_str)
                if bool(q_search):
                    q_start = q_search.start()
                    q_instance = Quantile(q = float(formula_str[q_start+1:q_start+3])/100)
                    formula_str = formula_str[:q_start] + str(x.agg(q_instance)) + formula_str[q_start+3:]
                else:
                    break
            formula_str = re.sub('mean', str(x.mean()), formula_str)
            formula_str = re.sub('std', str(x.std()), formula_str)
            try:
                formula_str = re.sub('mode', str(x.mode()[0]), formula_str)
            except:
                pass
            formula_str = re.sub('min', str(x.min()), formula_str)
            formula_str = re.sub('max', str(x.max()), formula_str)
            
            
            # auto_formating
            if 'nan' in formula_str:
                formula_result = ''
            elif decimal == 'auto':
                formula_result = auto_formating(eval(formula_str), decimal_revision=decimal_revision, return_type='str', thousand_format=thousand_format)
            elif decimal < 1:
                formula_result = str(int(round(eval(formula_str), decimal)))
            elif decimal is not None:
                formula_result = str(round(eval(formula_str), decimal))
            else:
                formula_result = str(eval(formula_str))

            string_list.append(formula_result)
        string_list.append(return_format[start_index:])

        self.agg_format = ''.join(string_list)
        return self.agg_format

    def __call__(self, x, return_format=None, decimal=None, decimal_revision=None, thousand_format=None):
        if return_format is None:
            return_format = self.return_format
        if decimal is None:
            decimal = self.decimal
        if decimal_revision is None:
            decimal_revision = self.decimal_revision
        if thousand_format is None:
            thousand_format = self.thousand_format

        self.format(x=x, return_format=return_format, decimal=decimal, decimal_revision=decimal_revision, thousand_format=thousand_format)
        return self.agg_format
        # return x.mean()








# outlier calculate
# class OutlierColumns  (Apply Just One Column)
class OutlierColumns():
    def __init__(self, x=False):
        self.x = False if type(x) == bool and x == False else x

    def set_x(self, x):
        self.x = x

    def make_outlier_table(self, x=False, sigma=3):
        if type(x) == bool and x == False:
            x = self.x
        else:
            x = x
            self.x = x

        x_describe = Describe(x)
        x_describe.describe(sigma=sigma)
        self.describ_instance = x_describe

        ol_dict = {}
        for of, ov in x_describe.outlier.items():
            if 'l' in of:
                ol_dict[of] = list(x[x < ov].index)
            elif 'u' in of:
                ol_dict[of] = list(x[x > ov].index)
        ol_vector = pd.Series(ol_dict)

        def vector_to_groupframe(vector):
            gf = vector.to_frame().reset_index()
            gf['ul'] = gf['index'].apply(lambda x: x[0])
            gf['index'] = gf['index'].apply(lambda x: x[1:])
            gf = gf.set_index(['index','ul']).unstack('ul')
            gf.columns = ['l', 'u']
            return gf

        self.outlier_index_tb = vector_to_groupframe(ol_vector)
        self.outlier_tb = vector_to_groupframe(x_describe.outlier)
        self.outlier_len = self.outlier_index_tb.applymap(len)

    def filter(self, x=False, sigma=3, method=['sigma', 'of_box']):

        if type(x) == bool and x == False:
            x = self.x
        else:
            self.x = x

        self.make_outlier_table(x=x, sigma=sigma)
        filter_criteria = list(map(lambda x: True if sum(list(map(lambda y: y in x, method))) else False, self.outlier_tb.index))
        outlier_criteria = self.outlier_tb[filter_criteria]
        outlier_lf = outlier_criteria['l'].min()
        outlier_uf = outlier_criteria['u'].max()

        outlier = x[(x < outlier_lf) | (outlier_uf < x)]
        normal = x[ (outlier_lf < x) & (x < outlier_uf)]

        self.outlier_criteria = outlier_criteria
        self.outlier_lf = outlier_lf
        self.outlier_uf = outlier_uf
        self.outlier = outlier
        self.normal = normal

    def outlier_plot(self, title=False, norm=True):
        try:
            self.outlier_criteria
            self.outlier_lf
            self.outlier_uf
        except:
            self.filter(self, x=self.x, sigma=3, method=['sigma', 'of_box'])

        outlier_color = {'f_sigma':(0,0,0), 'if_box':(0.8,0.6,0), 'of_box':(0.7,0.4,0)}

        N, bins, patches = plt.hist(self.x, bins=30, color='skyblue', edgecolor='grey')
        for i in np.argwhere((bins < self.outlier_lf) | (self.outlier_uf < bins)).ravel():
            if i >= len(patches):
                patches[-1].set_facecolor([1,0,0.5,0.7])
            else:
                patches[i].set_facecolor([1,0,0.5,0.7])
        bin_max, bin_min = bins.max(), bins.min()
        bin_padding = (bin_max-bin_min)*0.05
        plt.xlim(bin_min-bin_padding, bin_max+bin_padding)
        # lower_fence
        for li, lf in self.outlier_criteria['l'].items():
            plt.axvline(lf, ls='--', alpha=0.3, color=outlier_color[li], label='l'+li)
        # upper_fence
        for ui, uf in self.outlier_criteria['u'].items():
            plt.axvline(uf, ls='--', alpha=0.3, color=outlier_color[ui], label='u'+ui)
        plt.axvline(self.outlier_lf, ls='--', color=[1,0,0.5], alpha=0.7)
        plt.axvline(self.outlier_uf, ls='--', color=[1,0,0.5], alpha=0.7)
        plt.legend()
        plt.xlabel(self.x.name)

        if title:
            plt.title(title)
        elif self.title:
            plt.title(self.title)
        if norm:
            plt.plot(self.x.name, 'cpk', data=cpk_line(self.x), color='red', alpha=0.5)

    def fit(self, x=False, sigma=3, method=['sigma', 'of_box'], plot=False):
        if type(x) == bool and x == False:
            x = self.x
        else:
            self.x = x

        self.filter(x=x, sigma=sigma, method=method)

        if type(plot)==bool and plot==False:
            self.title = False
            self.norm = True
        else:
            if bool(plot) and plot==True:
                plot_options = {}
            else:
                plot_options = plot

            if 'title' not in plot_options.keys():
                plot_options['title'] = False
                self.title=False
            if 'norm' not in plot_options.keys():
                plot_options['norm'] = True
                self.norm=True
            self.outlier_plot(**plot_options)
        return (self.normal, self.outlier)



# outlier Remove
# class Outlier  (Apply Overall DataFrame)
class Outlier():
    def __init__(self, data=False, sigma=3, method=['sigma', 'of_box']):
        self.data = False if type(data) == bool and data == False else data
        self.sigma = sigma
        self.method = method
    
    def fit(self, data=False, on=False, plot=False, **kwargs):
        if type(data) == bool and data == False:
            data = self.data
        else:
            self.data = data
        if type(on) == bool and on==False:
            on = data.columns

        if 'sigma' in kwargs.keys():
            self.sigma = kwargs['sigma']
        if 'method' in kwargs.keys():
            self.method = kwargs['method']

        ncols = len(on)
        data_num, data_obj, data_time = dtypes_split(data, return_type='tuple_data')

        if plot:            
            fig_ncols = 4 if ncols > 4 else ncols
            fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.8
            fig = plt.figure(figsize=(fig_ncols * 5, fig_nrows * 4))
            fig.subplots_adjust(hspace=0.3)   # 위아래, 상하좌우 간격
        
        outlier_criteria_dict={}
        outlier_dict = {}
        target_cols = list(set(on) & set(list(data_num.columns)))
        for idx, col in enumerate(target_cols, 1):
            if plot:
                plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
            olc = OutlierColumns()
            normal, outlier = olc.fit(data_num[col], sigma=self.sigma,
                                    method=self.method, plot=plot)
            outlier_criteria_dict[col] = olc.outlier_criteria
            outlier_dict[col] = list(outlier.index)

        if plot:
            plt.show()
            self.plot = fig

        self.columns = on
        self.data_obj = data_obj
        self.data_num = data_num
        self.outlier_tb = pd.Series(outlier_dict)
        self.outlier_len = self.outlier_tb.apply(len)
        self.outlier_criteria = outlier_criteria_dict
        self.OutlierCoulumns_Obj = olc
    
    def outlier_plot(self, data=False, on=False, return_plot=False, **kwargs):
        if type(data) == bool and data == False:
            data = self.data
        else:
            self.data = data
        if type(on) == bool and on==False:
            on = data.columns
        
        try:
            self.plot
            plt.show()

            if return_plot:
                return self.plot
        except:
            if 'sigma' in kwargs.keys():
                self.sigma = kwargs['sigma']
            if 'method' in kwargs.keys():
                self.method = kwargs['method']

            ncols = len(on)
            data_num, data_obj, data_time = dtypes_split(data, return_type='tuple_data')
      
            fig_ncols = 4 if ncols > 4 else ncols
            fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.8
            fig = plt.figure(figsize=(fig_ncols * 5, fig_nrows * 4))
            fig.subplots_adjust(hspace=0.3)   # 위아래, 상하좌우 간격

            target_cols = list(set(on) & set(list(data_num.columns)))
            for idx, col in enumerate(target_cols, 1):
                plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
                olc = OutlierColumns()
                normal, outlier = olc.fit(data_num[col], sigma=self.sigma,
                                        method=self.method, plot=True)
            plt.show()
            
            if return_plot:
                return fig

    def transform(self, on=False, plot=False):
        if type(on) == bool and on==False:
            on = list(self.data_num.columns)
        target_cols = list(set(on) & set(list(self.outlier_tb.index)))
        
        outlier_idx = np.array(list(set(self.outlier_tb[target_cols].values.sum())))
        self.result = self.data_num.drop(outlier_idx, axis=0)

        if plot:
            self.outlier_plot(data=self.data_num, on=target_cols)

            ncols = len(target_cols)
            fig_ncols = 4 if ncols > 4 else ncols
            fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.8
            fig = plt.figure(figsize=(fig_ncols * 5, fig_nrows * 4))
            fig.subplots_adjust(hspace=0.3)   # 위아래, 상하좌우 간격
            
            for idx, col in enumerate(target_cols, 1):
                plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
                plt.hist(self.result[col], bins=30, color='skyblue', edgecolor='grey')
                plt.plot(col, 'cpk', data=cpk_line(self.result[col]), color='red', alpha=0.5)
            plt.show()
        
        print(f'(Result) Data_Length: {len(self.data):,} → {len(self.result):,}  ({len(self.data) - len(self.result):,} rows removed)')
        return self.result

    def fit_transform(self, data=False, on=False, plot=False, **kwargs):
        if type(data) == bool and data == False:
            data = self.data
        else:
            self.data = data
        if type(on) == bool and on==False:
            on = data.columns
        
        self.fit(data=data, on=on, plot=False, **kwargs)
        return self.transform(on=on, plot=plot)
    







#======== [ CPK, Capability ] ===================================================================================================================
# function calc cpk
def cpk(mean, std, lsl=None, usl=None, lean=False):
    if np.isnan(std) or std == 0:
        return np.nan
    
    if (lsl is None or np.isnan(lsl))and (usl is None or np.isnan(usl)):
        return np.nan
    lsl = -np.inf if (lsl is None or np.isnan(lsl)) else lsl
    usl = np.inf if (usl is None or np.isnan(usl)) else usl

    cpk = min(usl-mean, mean-lsl) / (3 * std)
    if lean:
       sign = 1 if usl-mean < mean-lsl else -1
       cpk = 0.01 if cpk < 0 else cpk
       cpk *= sign
    return cpk


# cpk_line in histogram
def cpk_line(x, bins=50, density=False):
    x_describe = x.describe()
    x_lim = x_describe[['min', 'max']]
    x_min = min(x_describe['min'], x_describe['mean'] - 3 * x_describe['std'])
    x_max = max(x_describe['max'], x_describe['mean'] + 3 * x_describe['std'])
    x_100Divide = np.linspace(x_min, x_max, 101)   # x 정의
    y_100Norm = (1 / (np.sqrt(2 * np.pi)*x_describe['std'])) * np.exp(-1* (x_100Divide - x_describe['mean'])** 2 / (2* (x_describe['std']**2)) )
    if not density:
        y_rev = len(x)/(bins) * (x_describe['max'] -x_describe['min'])
        y_100Norm *= y_rev
    return pd.DataFrame([x_100Divide,y_100Norm], index=[x.name, 'cpk']).T





# 【 Series, Vector function 】 ################################################################################
# 일정 범위구간에 따라 Level을 나눠주는 함수
# function cut_range
def cut_range(x, categories, right=False, labels=None, include_lowest=True, remove_unused_categories=True):
    if labels is not None and len(categories)+1 != len(labels):
        raise("labels must be the same length as categories")
    if (right==False and include_lowest==False) or (right==True and include_lowest==True):
        raise("Only the following combinations are possible.\ninclude_lowest=True, right=False or include_lowest=False, right=True")

    range_min = np.array([-np.inf] + list(categories))
    range_max = np.roll(range_min,-1)
    range_max[-1] = np.inf

    categories = [str(r_min) + ' ~ ' + str(r_max) for r_min, r_max in zip(range_min, range_max)]

    vector = x.copy()
    vector_temp = vector.copy()
    vector = vector.astype(str)

    categories = []
    for r_idx, (r_min, r_max) in enumerate(zip(range_min, range_max)):
        if labels is None:
            label = ('[' if include_lowest else '(') + str(r_min) + ' ~ ' + str(r_max) + (']'if right else ')') 
        else:
            label = labels[r_idx]

        if r_max == np.inf:
            if include_lowest:
                vector[vector_temp >= r_min] = label
            else:
                vector[vector_temp > r_min] = label
        else:
            if right:
                vector[(vector_temp <= r_max) & (vector_temp > r_min)] = label
            else:
                vector[(vector_temp < r_max) & (vector_temp >= r_min)] = label
        categories.append(label)
    
    if labels is not None:
        categories = labels

    if "<class 'pandas.core.series.Series'>" in str(type(vector)):
        vector = pd.Series(pd.Categorical(vector, categories=categories, ordered=True))
        vector.index = x.index
        if remove_unused_categories:
            vector = vector.cat.remove_unused_categories()
    elif "<class 'numpy.ndarray'>" in str(type(vector)):
        vector

    return vector




# cpk calculate
# class Capability
class Capability():
    """
    < Input >
    mode: 'cpk' Calculate Cpk
          'cpk (0.00..)' Calculate Cpk (0.00..) Range (ex. cpk 1.0 : cpk 1.0 range '0 ~ 0')
    """
    def __init__(self, lsl=-np.inf, usl=np.inf, mode='cpk', decimal=3, lean=False, name=None, sample_min=5):
        self.lsl = lsl
        self.usl = usl
        self.decimal = decimal
        self.lean = lean
        self.name = name
        self.sample_min = sample_min
        self.mode = mode
        self.mode_list = ['cpk', 'observe_lsl_reject_n', 'observe_usl_reject_n', 'observe_reject_n',
                        'observe_lsl_reject_prob', 'observe_usl_reject_prob', 'observe_reject_prob',
                        'gaussian_lsl_reject_prob', 'gaussian_usl_reject_prob', 'gaussian_reject_prob',
                        'cpk0.00_range']
        self.set_name()

    def set_lsl(self, lsl=-np.inf):
        self.lsl = lsl
        self.set_name()

    def set_usl(self, usl=-np.inf):
        self.usl = usl
        self.set_name()
   
    def set_decimal(self, decimal=3):
        self.decimal = decimal

    def set_lean(self, lean=False):
        self.lean = lean

    def performance_analysis(self, x, display=False, mode=None):
        if mode is None:
            mode = self.mode
        else:
            self.mode = mode

        x_agg = x.agg(['mean','std'])
        decimal = fun_Decimalpoint(x_agg['mean'])
        x_len = len(x)
        
        if x_len < self.sample_min:
            self.cpk = np.nan
        else:
            self.cpk = cpk(mean=x_agg['mean'], std=x_agg['std'], lsl=self.lsl, usl=self.usl, lean=self.lean)
        self.calc_performance(x, lsl=self.lsl, usl=self.usl)

        self.x = x.copy()
        self.agg = x_agg.copy()
        self.set_name()

        self.performance_summary_dict = {}
        self.performance_summary_dict['Spec_Limit'] = str('' if self.lsl == -np.inf else self.lsl) + ' ~ ' + str('' if self.usl == np.inf else self.usl)
        self.performance_summary_dict['Cpk'] = self.cpk
        self.performance_summary_dict['N'] = len(x)
        self.performance_summary_dict['Mean'] = round(x_agg['mean'], decimal)
        self.performance_summary_dict['Std'] = round(x_agg['std'], decimal)
        self.performance_summary_dict['observe_reject_n'] = self.observe_reject_n
        self.performance_summary_dict['observe_reject_prob'] = self.observe_reject_prob
        self.performance_summary_dict['gaussian_reject_prob'] = self.gaussian_reject_prob
        self.performance_summary = pd.Series(self.performance_summary_dict).to_frame().T

        if display:
            print(f"[{x.name}]  Cpk: {self.cpk:.3},  N: {len(x):,},  Mean: {round(x_agg['mean'],decimal)},  Std: {round(x_agg['std'],decimal)}")
            print(f"(Observe Performance) reject_count: {self.observe_reject_n:,},   reject_prob: {self.observe_reject_prob:.4f}")
            print(f"(Gaussian Performance) reject_prob: {self.gaussian_reject_prob:.4f}")
        
        return eval('self.' + mode)

    def calc_performance(self, x, lsl, usl):
        x_len = len(x)

        # observe performance
        observe_performance = {}
        self.observe_lsl_reject_n = len(x[x < self.lsl])
        self.observe_usl_reject_n = len(x[x > self.usl])
        self.observe_reject_n = self.observe_lsl_reject_n + self.observe_usl_reject_n
        self.observe_lsl_reject_prob = self.observe_lsl_reject_n/x_len
        self.observe_usl_reject_prob = self.observe_usl_reject_n/x_len
        self.observe_reject_prob = self.observe_lsl_reject_prob + self.observe_usl_reject_prob
        observe_performance['observe_lsl_reject_n'] = self.observe_lsl_reject_n
        observe_performance['observe_usl_reject_n'] = self.observe_usl_reject_n
        observe_performance['observe_reject_n'] = self.observe_reject_n
        observe_performance['observe_lsl_reject_prob'] = self.observe_lsl_reject_prob
        observe_performance['observe_usl_reject_prob'] = self.observe_usl_reject_prob
        observe_performance['observe_reject_prob'] = self.observe_reject_prob
        self.observe_performance = observe_performance

        # gaussian performance
        gaussian_performance = {}
        mean, std = x.agg(['mean','std'])
        decimal = fun_Decimalpoint(mean)
        if std == 0:
            self.gaussian_lsl_reject_prob = np.nan
            self.gaussian_usl_reject_prob = np.nan
            self.gaussian_reject_prob = np.nan
        else:
            lsl_sigma = (self.lsl - mean) / std
            usl_sigma = (self.usl - mean) / std
            self.gaussian_lsl_reject_prob = sp.stats.norm.cdf(lsl_sigma)
            self.gaussian_usl_reject_prob = 1 - sp.stats.norm.cdf(usl_sigma)
            self.gaussian_reject_prob = self.gaussian_lsl_reject_prob + self.gaussian_usl_reject_prob
        gaussian_performance['gaussian_lsl_reject_prob'] = self.gaussian_lsl_reject_prob
        gaussian_performance['gaussian_usl_reject_prob'] = self.gaussian_usl_reject_prob
        gaussian_performance['gaussian_reject_prob'] = self.gaussian_reject_prob
        self.gaussian_performance = gaussian_performance

    def plot(self, x=False, xlim=False, return_plot=True, title=None, line_kwargs={'color':'red', 'alpha':0.5}, **hist_kwargs):
        # try:
        if type(x)==bool and x==False:
            x = self.x
        else:
            self.x = x
        try:
            cpk = self.cpk
            if ~self.agg:
                self.agg = x.agg(['mean', 'std'])
        except:
            cpk = self.performance_analysis(x)

        mean, std = self.agg
        decimal = fun_Decimalpoint(mean)

        if return_plot:
            fig = plt.figure()
        if title is None:
            plt.title(f"< Capability Analysis for '{x.name}' >", fontsize=13)
        else:
            plt.title(title)
            
        if 'color' not in hist_kwargs.keys():
            hist_kwargs['color'] = 'skyblue'
        if 'edgecolor' not in hist_kwargs.keys():
            hist_kwargs['edgecolor'] = 'grey'
        N, plot_bins, patch = plt.hist(self.x, **hist_kwargs)
        plt.plot(self.x.name , 'cpk' ,data=cpk_line(self.x), **line_kwargs)
        plt.axvline(mean, color='blue', alpha=0.3)
        plt.axvline(self.lsl, color='red', ls='--', alpha=0.5)
        plt.axvline(self.usl, color='red', ls='--', alpha=0.5)
        plt.xlabel(f"{x.name}\nCpk: {self.cpk:.3},  N: {len(x):,},  Mean: {round(mean,decimal)},  Std: {round(std,decimal)}\n\
                    (Observe Performance) reject_count: {self.observe_reject_n:,},   reject_prob: {self.observe_reject_prob:.4f}\n\
                    (Gaussian Performance) reject_prob: {self.gaussian_reject_prob:.4f}", fontsize=12)
        if type(xlim)==bool and xlim == False:
            min_bin, max_bin = min(min(plot_bins), mean - 3*std), max(max(plot_bins), mean + 3*std)
            min_plot = min(min_bin, np.inf if self.lsl == -np.inf else self.lsl )
            max_plot = max(max_bin, -np.inf if self.usl == np.inf else self.usl)
            plt.xlim(min_plot - (max_plot-min_plot)*0.05, max_plot + (max_plot-min_plot)*0.05)
        else:
            plt.xlim(xlim[0], xlim[1])

        if return_plot:
            plt.close()
            self.cpk_plot = fig
            return self.cpk_plot

    def calc_capabilty(self, x, cpk):
        x_len = len(x)

        mean, std = x.agg(['mean','std'])
        decimal = fun_Decimalpoint(mean)
        lsl_cpk = round(mean - cpk * 3 * std, decimal)
        usl_cpk = round(mean + cpk * 3 * std, decimal)
        
        # observe performance
        observe_lsl_reject_n = len(x[x < lsl_cpk])
        observe_usl_reject_n = len(x[x > usl_cpk])
        observe_reject_n = observe_lsl_reject_n + observe_usl_reject_n
        observe_lsl_reject_proba = round(observe_lsl_reject_n/x_len, 4)
        observe_usl_reject_proba = round(observe_usl_reject_n/x_len, 4)
        observe_reject_proba = observe_lsl_reject_proba + observe_usl_reject_proba

        # gaussian performance
        gaussian_reject_prob = round((1 - sp.stats.norm.cdf(3 * cpk)) * 2, 4)

        capa = {}
        capa['cpk_range'] = str(lsl_cpk) + ' ~ ' + str(usl_cpk)
        capa['lsl_cpk'] = lsl_cpk
        capa['usl_cpk'] = usl_cpk
        capa['observe_lsl_reject_n'] = observe_lsl_reject_n
        capa['observe_usl_reject_n'] = observe_usl_reject_n
        capa['observe_reject_n'] = observe_reject_n
        capa['observe_lsl_reject_proba'] = observe_lsl_reject_proba
        capa['observe_usl_reject_proba'] = observe_usl_reject_proba
        capa['observe_reject_proba'] = observe_reject_proba
        capa['gaussian_reject_prob'] = gaussian_reject_prob

        return capa

    def capability_analysis(self, x=None, cpk=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3], include=None):
        if x is None:
            x = self.x
        else:
            self.x = x
        
        self.capa = {}
        for c in cpk:
            self.capa[c] = {}
            self.capa[c] = self.calc_capabilty(x=self.x, cpk=c)

        self.capability_summary = pd.DataFrame(self.capa)
        if include is None:
            self.capability_summary = self.capability_summary.drop(['lsl_cpk', 'usl_cpk', 'observe_lsl_reject_n', 'observe_usl_reject_n',
                                        'observe_lsl_reject_proba', 'observe_usl_reject_proba'], axis=0)
        else:
            self.capability_summary = self.capability_summary.loc[include,:]
        
        return self.capability_summary

    def reset(self):
        self.__init__()
        self.cpk = None
        self.x = False
        self.agg = None
        self.observe_performance = {}
        self.gaussian_performance = {}

        self.observe_lsl_reject_n = None
        self.observe_usl_reject_n = None
        self.observe_reject_n = None
        self.observe_lsl_reject_prob = None
        self.observe_usl_reject_prob = None
        self.observe_reject_prob = None

        self.gaussian_lsl_reject_prob = None
        self.gaussian_usl_reject_prob = None
        self.gaussian_reject_prob = None

        self.capa = None
        self.capability_summary = None
    
    def __call__(self, x, mode=None, display=False, **kwargs):
        if mode is None:
            mode = self.mode
        else:
            self.mode = mode
        if kwargs:
            for k, v in kwargs.items():
                arg = k.lower()
                if arg in ['lsl', 'usl', 'decimal', 'lean']:
                    exec('self.' + arg + ' = ' + 'v')
        
        if 'cpk' in mode[:3].lower() and len(mode) > 3:
            cpk_mode = mode[3:]
            if len(cpk_mode.split('_')) == 1:
                cpk = float(cpk_mode)
                self.capability_analysis(x=x, cpk=[cpk])
                result = self.capa[cpk]['cpk_range']
                self.set_name()
                return result
            elif len(cpk_mode.split('_')) > 1:
                cpk_idx = cpk_mode.index('_')
                cpk = float(cpk_mode[:cpk_idx])
                cpk_mode = cpk_mode[cpk_idx+1:]
                self.capability_analysis(x=x, cpk=[cpk])
                result = self.capa[cpk][cpk_mode]
                self.set_name()
                return result
        else:
            result = self.performance_analysis(x, display=display)
            self.set_name()
            return result

    def set_name(self):
        name = ''  if self.name is None else str(self.name)
        lsl = '' if self.lsl == -np.inf else str(self.lsl)
        usl = '' if self.usl == np.inf else str(self.usl)

        if 'cpk' in self.mode[:3].lower() and len(self.mode)>3:
            cpk_mode = self.mode[3:]
            if len(cpk_mode.split('_')) == 1:
                cpk = float(cpk_mode)
                name = f"{name} cpk {cpk}\nrange"
            elif len(cpk_mode.split('_')) > 1:
                cpk_idx = cpk_mode.index('_')
                cpk = float(cpk_mode[:cpk_idx])
                cpk_mode = cpk_mode[cpk_idx+1:]
                name = f"{name} cpk {cpk}\n{cpk_mode}"
        else:
            name = f"{name} {self.mode} \n({lsl} ~ {usl})"
        self.__class__.__name__ = name
        # return name

    def __repr__(self):
        self.info = {}
        for arg in ['lsl', 'usl', 'decimal', 'lean']:
            try:
                self.info[arg] = eval('self.'+arg)
            except:
                pass
        return str(self.info)

    def __str__(self):
        self.__repr__()
        return str(self.info)





# class CapabilityGroup
class CapabilityGroup():
    def __init__(self, capability=['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
                , statistics=['count','mean','std']):
        self.capability = capability
        self.statistics = statistics

    def analysis(self, data, criteria, target=None, value_vars=None, capability=None, statistics=None, lean=False, hist_kwargs={}, line_kwargs={}):
        if capability is None:
            capability = self.capability
        if statistics is None:
            statistics = self.statistics

        data_anlysis = data.copy()
        criteria_table = criteria.copy()
        index = list(criteria.index.names)

        if len(index) == 1 and index[0] is None:
            data_anlysis[''] = 0
            index = ['']

        capability_table = pd.DataFrame()
        capability_dict = {}
        capability_data = {}
        for sc in statistics + capability:
            capability_dict[sc] = criteria_table.copy()

        capability_list = capability.copy()
        plot=False

        if 'cpk_plot' in capability:
            plot=True
            capability_list.remove('cpk_plot')

        if value_vars is None:
            if criteria.columns.name is not None:
                value_vars = criteria.columns.name
        
        # criteria 각 Row별 iteration
        # ck: group_key, cv: values
        for cri, (ck, cv) in enumerate(criteria_table.T.items()):
            try:
                data_target = data_anlysis.loc[data_anlysis.groupby(index).groups[ck], :]
            except:
                data_target = data_anlysis[(data_anlysis.iloc[:,0] == False) & (data_anlysis.iloc[:,0] == True)]

            # criteria values 각 value별 iteration
            # cvk: values_key, cvv: value
            for cci, (cvk, cvv) in enumerate(cv.items()):
                group_key = tuple(list(ck)+[cvv])
                capability_dict_unit = {}
                capability_data[group_key] = {}

                cpk_range = cvv.split('~')
                cpk_lsl = -np.inf if cpk_range[0] == '' else float(cpk_range[0])
                cpk_usl = np.inf if cpk_range[1] == '' else float(cpk_range[1])

                #################################################
                if value_vars:  # Group별 특정 Target값을 분석할때
                    if target is None:
                        raise('Value Error: Analysis Target is not selected.')
                    data_target_detail = data_target[(data_target[value_vars]  == cvk) & (~data_target[target].isna())]
                    capability_dict_unit = data_target_detail[target].agg(statistics).apply(lambda x: round(x, fun_Decimalpoint(x)-1)).to_dict()
                else:   # Column값별로 분석할때
                    target = cvk
                    data_target_detail = data_target[~data_target[target].isna()]
                    capability_dict_unit = data_target_detail[target].agg(statistics).apply(lambda x: round(x, fun_Decimalpoint(x)-1)).to_dict()
                #################################################
                capability_data[group_key][cvk] = data_target_detail

                if len(data_target_detail) > 0:
                    # 각 Case별 Capability Calculation
                    for cp in capability_list:
                        capa = Capability(mode=cp)
                        if lean:
                            capa.lean = lean
                        capa.lsl = cpk_lsl
                        capa.usl = cpk_usl
                        capa_score = data_target_detail[target].agg(capa)
                        capability_dict_unit[cp] = round(capa_score, fun_Decimalpoint(capa_score)-1)

                        if plot:
                            fig = capa.plot(**hist_kwargs, **line_kwargs)
                            plt.close()
                            capability_dict_unit['cpk_plot'] = fig
                else:
                    for sc in statistics + capability:
                        if sc == 'count':
                            capability_dict_unit[sc] = 0
                        else:    
                            capability_dict_unit[sc] = np.nan
                
                for sc in statistics + capability:
                    capability_dict[sc].iloc[cri, cci] = capability_dict_unit[sc]

                # capatable 정리
                capa_table_unit = pd.Series(capability_dict_unit, name=target).to_frame().T
                capa_table_unit.reset_index(inplace=True)
                
                if value_vars is not None:      # Group별
                    capa_table_unit[value_vars] = cvk
                else:      # Column별
                    capa_table_unit['target'] = cvk
                capa_table_unit['criteria'] = cvv


                if index[0] == '':      # index 미지정
                    capa_table_unit[''] = ck
                    if value_vars is not None:      # Group별
                        capability_columns =  [value_vars] + ['criteria'] + statistics + capability
                    else:
                        capability_columns = ['target'] + ['criteria'] + statistics + capability
                    capa_table_unit = capa_table_unit[capability_columns]
                else:      # index 지정
                    for gkk, ckk in zip(index, ck):
                        capa_table_unit[gkk] = ckk
                    if value_vars is not None:      # Group별
                        capability_columns = ['index'] + index + [value_vars] + ['criteria'] + statistics + capability
                    else:      # Column별
                        capability_columns = index + ['target'] + ['criteria'] + statistics + capability
                    capa_table_unit = capa_table_unit[capability_columns]
                    capa_table_unit.columns = capability_columns


                # result Table에 정리
                capability_table = pd.concat([capability_table, capa_table_unit], axis=0)
                capability_table.reset_index(inplace=True)
                capability_table.drop('index', axis=1, inplace=True)
            
        self.capability_table = capability_table
        self.capability_dict = capability_dict
        self.capability_data = capability_data
        self.reset_dictgen()

    def reset_dictgen(self, key='all'):
        if key == 'all':
            self.iter_keys = iter(self.capability_dict.keys())
            self.len_iter_keys = len(self.capability_dict.keys())
        else:
            self.iter_keys = iter(key)
            self.len_iter_keys = len(key)
        self.count_iter = 0

    def bring_dictgen(self):
        self.count_iter += 1
        key = next(self.iter_keys)
        print(f'(dict_iteration) key : {key}   {self.count_iter}/{self.len_iter_keys}')
        return self.capability_dict[key]



# Capability Dataset from such groups
class CapabilityData():
    """

    criteria : {'YP': ['600~750'], 'TS':['980~'], 'EL':['12~']}
    criteria_column : {'YP':'YP_보증범위', 'TS':'TS_보증범위', 'EL':'EL_보증범위'}
    """
    def __init__(self, capability=['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
            , statistics=['count','mean','std']):
        self.capability = capability
        self.statistics = statistics
    
    def analysis(self, data, group=None, criteria=None, criteria_column=None):
        """
        criteria : {'YP': ['600~750'], 'TS':['980~'], 'EL':['12~']}
        criteria_column : {'YP':'YP_보증범위', 'TS':'TS_보증범위', 'EL':'EL_보증범위'}
        """
        data_copy = data.copy()
        data_copy['dummy'] = 'dummy'

        if group is None:
            group = 'dummy'
        data_group = data_copy.groupby(group, dropna=False)

        cg = CapabilityGroup(capability=self.capability, statistics=self.statistics)

        def criteria_dict(data, criteria_column):
            result_dict = {}
            for ck, cv in criteria_column.items():
                c_list = data_copy[cv].drop_duplicates().sort_values().values
                result_dict[ck] = c_list
            return result_dict

        group_result = []
        for gn, (gi, gv) in enumerate(data_group):
            print(f'{gn+1} / {data_group.ngroups} 번째 Group : ', end='')
            if criteria is not None:
                criteria_Y = criteria.copy()
            elif criteria_column is not None:
                criteria_Y = criteria_dict(gv, criteria_column)

            for cy, cc in criteria_Y.items():
                print(f'{cy}', end=' ')
                cy_frame = pd.DataFrame(cc)
                cy_frame.columns = [cy]
                cy_frame['dummy'] = 'dummy'
                cy_frame = cy_frame.set_index('dummy')
                cy_frame = cy_frame.applymap(lambda x: x.replace(' ~ ','~'))
                
                cg.analysis(gv, criteria=cy_frame)
                capa_result = cg.capability_table.drop('dummy', axis=1)
                if group != 'dummy':
                    if type(group) == str:
                        capa_result.insert(0, group, gi)
                    elif type(group)==list and len(group)==1:
                        capa_result.insert(0, group[0], gi)
                    else:
                        for gii, giv in enumerate(group[::-1]):
                            capa_result.insert(0, giv, gi[len(group)-gii-1])
                group_result.append(capa_result)
            print(' (done)')
        
        self.criteria_frame = cy_frame.copy()
        capa_result = pd.concat(group_result, axis=0)
        capa_result = capa_result.reset_index(drop=True)
        self.result = capa_result.copy()
        print(' . result: CapabilityData.result')
        return self.result



# Capability Dataset from such criteria_frame
class CapabilityFrame():
    """
     【required class】Capability
     【required function】lsl_usl_split

    < input >
     . data : History DataFrame
     . criteria_data : 
        
     . criteria_dict : 
        {'YP': {'lsl': [550.0, 580.0, 590.0, 600.0, 700.0],  'usl': [740.0, 750.0, 780.0, 850.0]},
         'TS': {'lsl': [980.0], 'usl': [1100.0, 1130.0, 1150.0]},
         'EL': {'lsl': [9.0, 10.0, 11.0, 12.0], 'usl': [21.0, 22.0]}}
     . mode: 'cpk', 'result_n', 'result_prob', 'plot', 'all'
    """
    def __init__(self, data=None, criteria_data=None, criteria_dict=None, criteria_group=None, mode=None):
        self.data = data
        self.criteria_data = criteria_data
        self.criteria_dict = criteria_dict
        self.criteria_group = criteria_group
        self.mode = mode

    def criteria_data_to_dict(self, criteria_data):
        return lsl_usl_split(criteria_data)

    def analysis(self, data=None, criteria_data=None, criteria_dict=None, criteria_group=None, mode=None, result_shape='vertical', display=True, return_result=False,
                bins=None, **capability_kwargs):

        if data is None:
            data = self.data
        if criteria_data is None:
            criteria_data = self.criteria_data
        if criteria_dict is None:
            criteria_dict = self.criteria_dict
        if criteria_group is None:
            criteria_group = self.criteria_group
        if mode is None:
            mode = self.mode
        
        if capability_kwargs:
            capa = Capability(**capability_kwargs)
        else:
            capa = Capability()

        if criteria_dict is None:
            if criteria_data is not None:
                criteria_dict = lsl_usl_split(criteria_data)

        cpk_dict = {}
        reject_n_dict = {}
        reject_prob_dict = {}
        plot_dict = {}
        for comp in criteria_dict:
            lsl_usl_comp = criteria_dict[comp]
            len_data = int(len(data[comp].dropna()))
            mean_data = auto_formating(data[comp].dropna().mean())
            std_data = auto_formating(data[comp].dropna().std())

            cpk_dict[(comp,'count','.')] = len_data
            reject_n_dict[(comp,'count','.')] = len_data
            reject_prob_dict[(comp,'count','.')] = len_data
            plot_dict[(comp,'count','.')] = len_data

            cpk_dict[(comp,'mean','.')] = mean_data
            reject_n_dict[(comp,'mean','.')] = mean_data
            reject_prob_dict[(comp,'mean','.')] = mean_data
            plot_dict[(comp,'mean','.')] = mean_data

            cpk_dict[(comp,'std','.')] = std_data
            reject_n_dict[(comp,'std','.')] = std_data
            reject_prob_dict[(comp,'std','.')] = std_data
            plot_dict[(comp,'std','.')] = std_data
            

            # lsl
            for c_lsl in lsl_usl_comp['lsl']:
                capa.lsl = c_lsl
                cpk_dict[(comp, 'lsl', c_lsl)] = round(capa(data[comp]), 2) if len_data > 0 else np.nan
                reject_n_dict[(comp, 'lsl', c_lsl)] = capa.observe_performance['observe_reject_n'] if len_data > 0 else np.nan
                reject_prob_dict[(comp, 'lsl', c_lsl)] = round(capa.observe_performance['observe_reject_prob'], 4) if len_data > 0 else np.nan
                plot_dict[(comp, 'lsl', c_lsl)] = capa.plot(bins=bins) if len_data > 0 else None
            capa.reset()

            # usl
            for c_usl in lsl_usl_comp['usl']:
                capa.usl = c_usl
                cpk_dict[(comp, 'usl', c_usl)] = round(capa(data[comp]), 2) if len_data > 0 else np.nan
                reject_n_dict[(comp, 'usl', c_usl)] = capa.observe_performance['observe_reject_n'] if len_data > 0 else np.nan
                reject_prob_dict[(comp, 'usl', c_usl)] = round(capa.observe_performance['observe_reject_prob'], 4) if len_data > 0 else np.nan
                plot_dict[(comp, 'usl', c_usl)] = capa.plot(bins=bins) if len_data > 0 else None
            capa.reset()
            
        # self.summary_data = data[criteria_dict].agg(['mean','std']).applymap(lambda x: auto_formating(x, return_type='str'))
        self.cpk = pd.Series(cpk_dict)
        self.reject_n = pd.Series(reject_n_dict)
        self.reject_prob = pd.Series(reject_prob_dict)
        self.plot = pd.Series(plot_dict)

        summary_frame = pd.concat([self.cpk, self.reject_n, self.reject_prob, self.plot],axis=1)
        summary_frame.columns = ['cpk', 'reject_n', 'reject_prob', 'plot']
        summary_frame.columns.name = 'result'
        summary_frame.index.names = ['contents', 'labels', 'values']
        self.summary_v = summary_frame
        self.summary_h = summary_frame.T

        if mode is None:
            mode = list(self.summary_v.columns)
        elif type(mode) == str:
            if mode == 'all':
                mode = list(self.summary_v.columns)
            else:
                mode = [mode]
        
        if 'v' in result_shape.lower():
            self.result = self.summary_v[mode]
        elif 'h' in  result_shape.lower():
            self.result = self.summary_h.loc[mode,:]

        if display:
            if 'v' in result_shape.lower():
                print(' . self.result / self.summary_v')
            elif 'h' in  result_shape.lower():
                print(' . self.result / self.summary_h')
            print_DataFrame(self.result)
        
        if return_result:
            return self.result

    def group_analysis(self, data=None, group=None, criteria_data=None, criteria_dict=None, criteria_group=None, bins=None, **capability_kwargs):
        
        result_cpk = pd.DataFrame()
        result_reject_n = pd.DataFrame()
        result_reject_prob = pd.DataFrame()
        result_plot = pd.DataFrame()

        
        for ei, (gi, gv) in enumerate(data.groupby(group)):
            print(f'({ei+1}/{data.groupby(group).ngroups})', end=' ')
            group_index = pd.MultiIndex.from_tuples([gi], names=group)    
            print({k: v for k, v in zip(group_index.names, group_index.values[0])}, end=': ')

            result_temp = self.analysis(data=gv, criteria_data=criteria_data, criteria_dict=criteria_dict, criteria_group=criteria_group,
                            mode='all', result_shape='vertical', display=False, return_result=True,
                            bins=bins, **capability_kwargs)
            
            result_cpk_temp = result_temp[['cpk']].T
            result_reject_n_temp = result_temp[['reject_n']].T
            result_reject_prob_temp = result_temp[['reject_prob']].T
            result_plot_temp = result_temp[['plot']].T

            result_cpk_temp.index = group_index
            result_reject_n_temp.index = group_index
            result_reject_prob_temp.index = group_index
            result_plot_temp.index = group_index
           
            result_cpk = pd.concat([result_cpk, result_cpk_temp], axis=0)
            result_reject_n = pd.concat([result_reject_n, result_reject_n_temp], axis=0)
            result_reject_prob = pd.concat([result_reject_prob, result_reject_prob_temp], axis=0)
            result_plot = pd.concat([result_plot, result_plot_temp], axis=0)

            print('done')
        
        self.group_cpk = result_cpk.copy()
        self.group_reject_n = result_reject_n.copy()
        self.group_reject_prob = result_reject_prob.copy()
        self.group_plot = result_plot.copy()

        print()
        print(f' . cpk: self.group_cpk {self.group_cpk.shape}')
        print(f' . reject_n: self.group_reject_n {self.group_reject_n.shape}')
        print(f' . reject_prob: self.group_reject_prob {self.group_reject_prob.shape}')
        print(f' . plot: self.group_plot {self.group_plot.shape}')


# 【 DataFrame function 】 ################################################################################

# [ EDA : Exploratory Data Analysis ] ==========================================================

# 빈도값을 Return해주는 함수
# function mode
def mode(x, seq=1, ascending=False):
    """
    < input > 
    x: Series
    seq (default 1): return counts (must be under 1 float or over 1 int)
    ascending (default False): order criteria

    < output >
    seq == 1: str
    seq >1 or float: list
    """
    value_counts = x.value_counts()
    value_counts = value_counts.sort_index(ascending=ascending)

    if seq == 'all':
        result = list(value_counts.index)
    elif seq == 1:
        result = x.mode()[0]
    elif seq < 1:
        value_counts_prob = value_counts.cumsum() / value_counts.sum()
        result = list(value_counts_prob[value_counts_prob < seq].index)
    elif type(seq) == int and seq >= 1:
        result = list(value_counts.index)[0:seq]
    else:
        raise('seq must be under 1 float or over 1 int.')
    return result


# 각 항목값을 오름차순 순서대로 column명과 항목값을 return 해주는 함수
def display_elements(x, seq='all', asceending=False, decimal_point=4, return_type='dict'):
    sort_x = np.round(x,decimal_point).sort_values(ascending=asceending)
    if seq != 'all' and seq != np.inf:
        if seq < 1 and len(sort_x) > 1:
                cumsum_x = (sort_x / sort_x.sum()).cumsum()
                filtered_x = sort_x[cumsum_x < seq]
                if len(filtered_x) == 0:
                    sort_x = sort_x[:1]
                else:
                    sort_x = sort_x[cumsum_x < seq]
        else:
            sort_x = sort_x[:seq]
    if return_type == 'str':
        return str(sort_x.to_dict()).replace('{','').replace('}','')
    elif return_type == 'dict':
        return sort_x.to_dict()


# 빈도값을 Return해주는 클래스
# class Mode
class Mode():
    """
    < input > 
    x: Series
    seq (default 1): return counts (must be under 1 float or over 1 int)
    ascending (default False): order criteria
    return_type(default 'auto'): list or join_seperator (return_type.join(...))
    return_format(default {i}):  {i} index, {c} counts, {cs} count_cumsum, 
                                 {p} proportional, {ps} proportional_cumsum

    < output >
    seq == 1: str
    seq >1 or float: list
    """
    def __init__(self, seq=1, ascending=False, return_type='auto', return_format='{i}'):
        self.seq=seq
        self.ascending=ascending
        self.return_type = return_type
        self.return_format = return_format
    
    def calculate(self, x, seq=None, ascending=None, return_format=None, return_type=None):
        if seq is None:
            seq = self.seq
        if ascending is None:
            ascending = self.ascending
        if return_format is None:
            return_format = self.return_format
        if return_type is None:
            return_type = self.return_type


        value_count = x.value_counts()
        value_count = value_count.sort_values(ascending=ascending)
        value_count_cumsum = value_count.cumsum()
        value_prob = value_count / value_count.sum()
        value_prob_cumsum = value_count.cumsum() / value_count.sum()
        value_count_frame = pd.concat([value_count,value_count_cumsum, value_prob, value_prob_cumsum ], axis=1)
        value_count_frame.columns = ['count', 'count_cumsum', 'prob', 'prob_cumsum']

        # printing filter
        if seq == 'all':
            mode_frame = value_count_frame
        elif seq < 1:
            mode_frame = value_count_frame[value_count_frame['prob_cumsum'] < seq]
        elif type(seq) == int and seq >= 1:
            mode_frame = value_count_frame.iloc[0:seq,:]
        else:
            raise('seq must be under 1 float or over 1 int.')
        self.value_count_frame = value_count_frame
        mode_frame['nvalues'] = len(mode_frame)
        self.mode_frame = mode_frame

        # printing format
        mode_format = [eval(f"f'{return_format}'") for i, c, cs, p, ps, n in zip(mode_frame.index, mode_frame['count'], mode_frame['count_cumsum'], mode_frame['prob'], mode_frame['prob_cumsum'], mode_frame['nvalues'])]
        self.mode = pd.Series(mode_format).drop_duplicates().values

        if return_type == 'auto':
            if len(mode_frame) == 1:
                return self.mode
            else:
                return list(self.mode)
        elif return_type == 'list':
            return list(self.mode)
        else:
            return return_type.join(list(np.array(self.mode).astype(str)))
        return mode
    
    def __call__(self, x, seq=None, ascending=None, return_format=None, return_type=None):
        if seq is None:
            seq = self.seq
        if ascending is None:
            ascending = self.ascending
        if return_format is None:
            return_format = self.return_format
        if return_type is None:
            return_type = self.return_type
        
        return self.calculate(x, seq=seq, ascending=ascending, return_format=return_format, return_type=return_type)





# (class) Data-Frame Information Plot
class SummaryPlot():
    def __init__(self, data, save_data=True):
        self.numeric_cols, self.obejct_cols, self.time_cols = map(lambda x: x.columns, dtypes_split(data, return_type='tuple_data'))
        self.missing_cols = len(data) == data.isna().sum(0)
        self.nunique_cols = data.nunique()

        self.save_data = save_data
        if save_data:
            self.data = data

    def attribute_plot(self, data, x, obejct_cols, numeric_cols, time_cols, nunique_cols, missing_cols, max_object):
        plt.title(x)
        if missing_cols[x]:
            plt.plot()
        else:
            if x in obejct_cols:
                if nunique_cols[x] == len(data):
                    plt.plot()
                    plt.text(-0.02, 0, 'Unique Value', fontsize=13)
                elif nunique_cols[x] > max_object:
                    data_value_counts = data[x].value_counts()
                    maxbar_value_counts = data_value_counts[:max_object]
                    maxbar_value_counts['other'] = data_value_counts[max_object:].sum()
                    maxbar_value_counts.plot.bar(color='mediumseagreen', edgecolor='grey')
                else:
                    data[x].value_counts().plot.bar(color='mediumseagreen', edgecolor='grey')
            elif x in numeric_cols:
                if nunique_cols[x] == 1:
                    fitted_line = None
                else:
                    fitted_line = sp.stats.norm
                sns.distplot(data[x], fit=fitted_line, kde=None, hist_kws={'edgecolor':'grey'}, fit_kws={'color':(1,0.5,0.5)})
            elif x in x in time_cols:
                plt.hist(data[x], edgecolor='grey', color='palegoldenrod')

    def summary_plot(self, on, dtypes='all', data=False, max_object=10, return_plot=False):
        if type(data)==bool and data == False:
            if self.save_data:
                data = self.data
            else:
                raise('data must be need.')
       
        columns = []
        if dtypes == 'all':
            columns = on.copy()
        else:
            if dtypes == 'numeric':
                for on_cols in on:
                    if on_cols in self.numeric_cols:
                        columns.append(on_cols)
            elif dtypes == 'object':
                for on_cols in on:
                    if on_cols in self.obejct_cols:
                        columns.append(on_cols)
            elif dtypes == 'time':
                for on_cols in on:
                    if on_cols in self.time_cols:
                        columns.append(on_cols)
       
        ncols = len(columns)

        fig_ncols = 6 if ncols > 6 else ncols
        fig_nrows = ((ncols // 6)+1) if ncols > 6 else 0.4

        if len(columns) >= 100:
            continue_YN = input(f'{len(on)} features are selected. Continue? (Y/N)')
       
        if len(columns) < 100 or continue_YN == 'Y':
            fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 5))
            fig.subplots_adjust(hspace=1)   # 위아래, 상하좌우 간격
            for idx, col in tqdm.tqdm_notebook( enumerate(columns, 1) ):
                # print(x, end=' ')         # Debugging Check
                plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
                self.attribute_plot(data=data, x=col,
                    obejct_cols=self.obejct_cols, numeric_cols=self.numeric_cols, time_cols=self.time_cols,
                    nunique_cols=self.nunique_cols, missing_cols=self.missing_cols,
                    max_object=max_object)
                # if col in self.numeric_cols and norm:
                #     plt.plot(data[col].name, 'cpk', data=cpk_line(data[col]), color='red', alpha=0.5)
            plt.show()
            if return_plot:
                return fig





# (class) Data-Frame Information
class DF_Summary:
    size_units = {'0':'bytes' ,'1':'KB', '2':'MB', '3':'GB', '4':'TB', '5':'PB'}

    def __init__(self, data, options=['dtype', 'missing', 'info', 'sample'], n_samples=20, display_counts=10):
        # data_save
        self.data = data

        # <head> ***
            # data_memory
        self.memory_usage = data.memory_usage().sum()
        data_unit = int(np.log2(self.memory_usage)/10)
        self.size_unit = DF_Summary.size_units[str(data_unit)]
        self.data_size = str(format(self.memory_usage / 2**(data_unit*10), '.1f')) + '+ ' + self.size_unit

            # data_shape
        self.shape = data.shape
       
        # <body> ***
        self.summary = pd.DataFrame()

        # dtype
        self.dtype = data.dtypes
        if 'dtype' in options:
            self.dtype.name = 'dtype'
            self.summary = pd.concat([self.summary, self.dtype], axis=1)  # **

        # missing_value
        if 'missing' in options:
            col_miss = data.isna().sum(axis=0)
            self.all_missing = len(data) == col_miss

            col_non_miss = (data.isna()==False).sum(axis=0)
            col_miss_ratio = (col_miss / (col_miss + col_non_miss)*100).apply(lambda x: round(x,1))
            self.missing = col_miss.to_frame().apply(lambda x: '-' if x[0] == 0 else str(format(x[0], ',')) + ' (' + format(str(col_miss_ratio[x.name]), '4s') +'%)' , axis=1)
            self.missing = self.missing
            self.missing.name = 'missing'
            self.summary = pd.concat([self.summary, self.missing], axis=1)   # **
       
        # column infomation
        if 'info' in options:
            data_unique = data.T.apply(lambda x: list(x.value_counts().index.astype(self.dtype[x.name])),axis=1)
            data_unique_length = data_unique.apply(lambda x: len(x))

            num_data, obj_data, time_data = dtypes_split(data, return_type='tuple_data')

            # number **
            if num_data.shape[1] > 0:
                self.describe = num_data.describe()
                num_describe = self.describe.T
                self.decimal = num_describe['50%'].apply(lambda x: np.nan if np.isnan(x) else self.fun_Decimalpoint(x))

                self.num_info = num_describe.apply(lambda x: str(data_unique_length[x.name]) + ' counts' if data_unique_length[x.name] <= display_counts else self.num_describe(x, self.decimal, self.dtype), axis=1)
            else:
                self.num_info = pd.Series()

            # object **
            if obj_data.shape[1] > 0:
                # obj_info = obj_data.apply(lambda x : self.obj_describe(x, self.shape[0]) ,axis=0).to_frame()
                self.obj_info = data_unique_length[obj_data.columns].apply(lambda x:
                                            'unique' if x == self.shape[0] else str(x) + ' levels')
            else:
                self.obj_info = pd.Series()

            # time **
            if time_data.shape[1] > 0:
                self.time_info = time_data.agg(['min','max']).T.apply(lambda x: x['min'].strftime('%Y-%m-%d') + ' ~ ' + x['max'].strftime('%Y-%m-%d') ,axis=1)
            else:
                self.time_info = pd.Series()
            
            print(self.obj_info.shape, self.num_info.shape, self.time_info.shape)
            self.info = pd.concat([self.obj_info, self.num_info, self.time_info], axis=0)[data.columns]
            self.info.name = 'info'
            self.summary = pd.concat([self.summary, self.info], axis=1)   # **

                # sample_data
        if 'sample' in options:
            if 'info' in options:
                self.sample = data_unique
                self.sample_print = self.sample.apply(lambda x: str(x)[:n_samples])
            else:
                self.sample = data.sample(3).T.apply(lambda x: list(x) if str(self.dtype[x.name]) == 'object' else list(map(eval('np.'+str(self.dtype[x.name])), x)),axis=1) #.apply(lambda x: list(x), axis=1)  
                self.sample_print = self.sample.to_frame().apply(lambda x: str(x[0]).replace(']', ''), axis=1).apply(lambda x: x[:20]+' ...')  #.to_frame()
            self.sample.name = 'sample'
            self.sample_print.name = 'sample'
            self.summary = pd.concat([self.summary, self.sample_print], axis=1)   # **

            # print-out
        print(self)
        # pd.set_option('display.expand_frame_repr',False)        # print-out option : not new-line
        # print(f'{self.data_size}, {str(self.shape[1])} columns, {str(self.shape[0])} obs')
        # print(self.summary)
        # pd.reset_option('display.expand_frame_repr')            # print-out option : reset
       
    def __str__(self):
        print(f'{self.data_size}  |  {str(format(self.shape[1], ","))} columns  {str(format(self.shape[0], ","))} obs')
        display(HTML(self.summary._repr_html_()))
        return ''
   
    def fun_Decimalpoint(self, value):
        if value == 0:
            return 3
        point_log10 = np.floor(np.log10(abs(value)))
        point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
        return point

    # object variables describe function
    def obj_describe(self, x, len_data):
        x_levles = list(set(x))
        len_levels = len(x_levles)

        if len_levels == len_data:
            return 'unique_value'
        else:
            return str(len_levels) + ' levels, ' + str(x_levles)[:20]

    # numeric variables describe function
    def num_describe(self, x, decimal_list, dtype_list):
        x_name = x.name

        if np.isnan(float(decimal_list[x_name])):
            return np.nan
        else:
            x_dtype = dtype_list[x_name]
            x_decimal = int(decimal_list[x_name])
            if 'int' in str(x_dtype):
                x_min = int(x['min'])
                x_max = int(x['max'])
                x_50 = int(x['50%'])
                if x_max - x_min == 1:
                    return '2 numbers, ' + str([x_min, x_max])
                else:
                    mean_str = format(format(int(x['mean']), ','), '6s')
                    std_str = format(format(round(x['std'], 1), ','), '6s')
                    min_str = format(format(int(x_min), ','), '6s')
                    max_str = format(format(int(x_max), ','), '6s')
            else:
                mean_str = format(format(round(x['mean'], x_decimal-1), ','), '6s')
                std_str = format(format(round(x['std'], x_decimal), ','), '6s')
                min_str = format(format(round(x['min'], x_decimal-1), ','), '6s')
                max_str = format(format(round(x['max'], x_decimal-1), ','), '6s')
            return mean_str + ' (' + std_str + ') ' + min_str + ' ~ ' + max_str

    # draw_summay_plot
    def summary_plot(self, on=False, dtypes='all', max_object=10, return_plot=False):
        plot_instance = SummaryPlot(self.data, save_data=False)

        if type(on) == bool and on == False:
            columns = self.data.columns
        else:
            columns = on

        fig = plot_instance.summary_plot(data=self.data, on=columns,
                dtypes=dtypes, max_object=max_object, return_plot=return_plot)
        if return_plot:
            return fig










# from collections import namedtuple
# import namedtuple
# from functools import reduce
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
################################################################################################
################################################################################################
################################################################################################


# Class 또는 Instance 객체를 String 형태나 Object 형태로 입력받았을때 즉시 실행해주는 함수
# import copy 
def class_object_execution(o=None, **kwargs):
    """
    【required (Library)】copy.deepcopy
    """
    params_str = ', '.join([k.replace("'","") + '=' + (f"'{v}'" if type(v)==str else str(v)) for k, v in kwargs.items()])

    if sum([t in str(type(o)) for t in ['int', 'float', 'list', 'tuple', 'dict']]) > 0:
        raise ValueError("Type Error: only require 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'.")
    
    elif 'str' in str(type(o)):
        if '(' in o and ')' in o:   # str instance
            try:
                try:      # Class Type
                    instance = eval(o)
                    isinstance(instance, eval(o[:o.find('(')]) )
                    return instance
                except:
                    return copy.deepcopy( eval(o[:o.find('(')]) )
            except:
                raise ValueError('incorrect string type object: confirm class/instance name or parameters')
        else:           # str class
            try:      # Class Type
                instance = eval(f"{o}({params_str})")
                isinstance(instance, eval(o) )
                return instance
            except:
                return copy.deepcopy(eval(o))

    elif '(' in str(o)  and ')' in str(o):
        return o

    else:
        try:      # Class Type
            if kwargs is None:
                instance = o()
            else:
                instance = o(**kwargs)
            isinstance(instance, o)
            return o()
        except:
            return copy.deepcopy(o)

        




# Data 형태 및 정보를 바꿔주는 Class
class DataHandler:
    """
    【required (Library)】 numpy, pandas, collections.namedtuple
    【required (Function)】vector_info

    < input >
    . data : Scalar, Vector, (1-D, 2-D) list, matrix, DataFrame
    . columns : user defined name(columns)  
                    * {i} auto name sequence
    . dtypes : setting dtypes manually
    . ndim : setting ndim manually 
    . reset_dtype : whether reset_dtype (if True, dtypes are automatically reassigned)
    . object_threshold : numeric column automatically transform to object (only operate when 'reset_dtype' is True)

    < output >
    . vector : ('vector_info' : ('data' : {'data': , 'index': , 'name': ]), kind, ndim, dtypes, nuniques)
    . matrix : ('matrix_info' : ('data' : {'data': , 'index': , 'columns': ]), kind', ndim, dtypes, nunique)
    """
    def __init__(self):
        pass
    
    # input data 의 정보(kind, ndim, shape, possible_vector)를 알려주는 함수
    def data_info(self, data, dict_type=None, save_data=True):
        data_info_object = namedtuple('data_info', ['frame', 'kind', 'ndim', 'shape', 'possible_vector'])

        if 'list' in str(type(data)):
            if type(data[0]) == dict:
                kind = 'dict_records'
                data_result = pd.DataFrame(data)
                input_ndim = 2
            else:
                kind = 'list'
                if type(data[0]) == list:
                    input_ndim = 2
                    data_result = pd.DataFrame(data)
                else:
                    input_ndim = 1
                    data_result = pd.Series(data)
        elif 'dict' in str(type(data)):
            dict_first_value = list(data.values())[0]
            if type(dict_first_value) == list:
                kind = 'dict_split' if list(data.keys()) == ['index', 'columns', 'data'] else 'dict_list'
                input_ndim = 2
                data_result = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns']) if kind == 'dict_split' else pd.DataFrame(data)
            elif 'series' in str(type(dict_first_value)):
                kind = 'dict_series'
                input_ndim = 2
                data_result = pd.DataFrame(data)
            elif type(dict_first_value) == dict:
                if dict_type is None:
                    kind = 'dict_index' if type(list(data.keys())[0]) == int else 'dict'
                    data = pd.DataFrame(data).T.infer_objects() if kind == 'dict_index' else pd.DataFrame(data)
                elif dict_type is not None:
                    kind = 'dict_' + dict_type
                input_ndim = 2
                data_result = pd.DataFrame(data)
            elif sum([d in str(type(dict_first_value)) for d in ['int','float','str','bool']]) > 0:
                kind = 'dict'
                input_ndim = 1
                data_result = pd.Series(data)

        elif 'numpy' in str(type(data)):
            kind = 'numpy'
            input_ndim = data.ndim
            data_result = pd.Series(data) if input_ndim == 1 else pd.DataFrame(data)
        elif 'pandas' in str(type(data)):
            kind = 'pandas'
            input_ndim = data.ndim
            data_result = pd.Series(data) if input_ndim == 1 else pd.DataFrame(data)

        if input_ndim == 1:
            possible_vector = True
        elif data_result.shape[1] == 1:
            possible_vector = True
        else:
            possible_vector = False

        if save_data:
            frame_data = data_result.to_frame() if input_ndim == 1 else data_result
        else:
            frame_data = data_result.shape
            
        return data_info_object(frame_data, kind, input_ndim, data_result.shape, possible_vector)

    def possible_vector_verify(self, x):
        data_info_object = self.data_info(x)
        if data_info_object.possible_vector == False:
            raise ValueError("x required only '(n,) Scalar' or '(n,1) Matrix'. ")

    # '(n,) Scalar' or '(n,1) Matrix' 의 name을 추출 및 자동 부여해주는 함수
    def vector_info_split(self, x, index=None, name=None, dtype=None, reset_dtype=False, object_threshold=3, save_data=True):
        """
        【required (Library)】 numpy
        """
        data_info_object = self.data_info(x)
        if data_info_object.possible_vector == False:
            raise ValueError("x required only '(n,) Scalar' or '(n,1) Matrix'. ")

        # try:
        vector_instance = namedtuple('vector_info', ['data', 'kind', 'ndim', 'dtypes', 'nuniques'])

        # kind ***
        kind = data_info_object.kind

        # data ***
        
        data_series = data_info_object.frame.iloc[:,0]
            
        # nunique ***
        unique_vector = data_series.drop_duplicates()
        data = np.array(data_series).ravel()
        
        # ndim ***
        ndim = 1


        # index ***
        if index is not None:
            if data_info_object.shape[0] != len(index):
                raise ValueError("The length of index is different.")
            else:
                index_result = index
        else:
            if type(x) == list:
                index_result = np.arange(len(x))
            else:
                try:
                    index_result = np.array(x.index)
                except:
                    index_result = np.arange(data_info_object.shape[0])
        
        # dtype ***
        if dtype is None:       # auto dtype
            try:
                dtype_result = x.dtype
            except:
                dtype_result = np.array(x).dtype
            if reset_dtype and sum([t in str(dtype_result) for t in ['int', 'float']]) > 0:
                if len(unique_vector) <= object_threshold:
                    dtype_result = np.array(list(map(str, np.unique(np.array(x))))).dtype
        elif type(dtype) == dict:
            dtype_result = list(dtype.values())[0]
            dtype_result = dtype_result if 'dtype(' in str(type(dtype_result)).lower() else pd.Series(False, dtype=dtype_result).dtype
        elif type(dtype) == list:
            dtype_result = list(dtype)[0]
            dtype_result = dtype_result if 'dtype(' in str(type(dtype_result)).lower() else pd.Series(False, dtype=dtype_result).dtype

        else:
            dtype_result = dtype if 'dtype' in str(type(dtype)) else pd.Series(True, dtype=dtype).dtype

        # name ***
        if name is not None:
            name_result = name[0] if type(name) == list else name
        else:
            if data_info_object.kind == 'pandas' or ('dict' in data_info_object.kind and data_info_object.ndim == 2):
                name_result = list(data_info_object.frame.columns)[0]
            else:
              name_result = 'x'
        
        # save_data
        if not save_data:
            data = data.shape
        else:
            data = np.array(data)

        # result ***
        vector_dict = {}
        vector_dict['data'] = data
        vector_dict['index'] = index_result
        vector_dict['name'] = name_result
        

        return vector_instance(vector_dict, kind, ndim, dtype_result, len(unique_vector))

    # Data Split : Vector, Matrix Data 를 data, index, name(columns), dtype(s) 로 나눠주는 함수
    def data_info_split(self, data, index=None, columns=None, dtypes=None, reset_dtypes=False, object_threshold=3, ndim=None, dict_type=None, save_data=True):

        data_info_object = self.data_info(data)

        #####
        if ndim == 1 or (ndim is None and data_info_object.ndim == 1):  # Scalar or Series
            return self.vector_info_split(x=data, index=index, name=columns, dtype=dtypes, 
                                reset_dtype=reset_dtypes, object_threshold=object_threshold)

        elif ndim == 2 or (ndim is None and data_info_object.ndim == 2):  # 2-dim DataFrame

            matrix_instance = namedtuple('matrix_info', ['data', 'kind', 'ndim', 'dtypes', 'nuniques'])
            # frame_instance = namedtuple('value', ['data', 'index', 'columns'])
            
            # kind ***
            kind = data_info_object.kind
            
            # data ***
            matrix_X = data_info_object.frame

            # ndim ***
            result_ndim = 2

            # index ***
            if index is not None:
                if data_info_object.shape[0] != len(index):
                    raise ValueError("The length of index is different.")
                else:
                    index_result = index
            else:
                try:
                    index_result = np.array(matrix_X.index)
                except:
                    index_result = np.arange(matrix_X.shape[0])
            
            # columns ***
            if columns is not None:
                if '{i}' in columns:
                    columns_result = [eval(f"f'{columns}'") for i in range(1, matrix_X.shape[1]+1)]
                else:
                    if 'str' in str(type(columns)) and data_info_object.possible_vector == True:
                        columns_result = [columns]
                    else:
                        if matrix_X.shape[1] != len(columns) or type(columns) != list:
                            raise ValueError("'columns' error : the number of input_data's columns is equal to length of 'columns' list.")
                        else:
                            columns_result = columns
            else:
                if data_info_object.kind == 'pandas' or ('dict' in data_info_object.kind and data_info_object.ndim == 2):
                    columns_result = list(data_info_object.frame.columns)
                elif data_info_object.possible_vector == True:
                    columns_result = ['x']
                else:
                    columns_result = ('x'+pd.Series(np.arange(1,matrix_X.shape[1]+1)).astype(str)).tolist()
                    # [f'x{c}' for c in range(1, np.array(matrix_X).shape[1]+1)]

            # nuniques ***
            nuniques = pd.DataFrame(matrix_X).apply(lambda x:len(x.value_counts().index) ,axis=0)
            nuniques.index = columns_result

            # dtypes ***
            if dtypes is not None:
                if type(dtypes) == dict:
                    dtypes_result = dtypes.copy()
                    try:
                        dtypes_origin = matrix_X.dtypes
                    except:
                        dtypes_origin = pd.DataFrame(np.array(matrix_X)).infer_objects().dtypes
                    dtypes_origin.index = columns_result
                    dtypes_origin_dict = dtypes_origin.to_dict()

                    dtypes_dict = {c: d if 'dtype' in str(type(d)) else pd.Series(True, dtype=d).dtype for c, d in dtypes_result.items()}
                    dtypes_origin_dict.update(dtypes_dict)
                    dtypes_result = dtypes_origin_dict.copy()
                elif type(dtypes) == list:
                    dtypes_result = dtypes.copy()
                    dtypes_list = [d if 'dtype' in str(type(d)) else pd.Series(True, dtype=d).dtype for d in dtypes_result]
                    dtypes_result = dict(zip(columns_result, dtypes_list))
                else:
                    dtypes_result = dtypes
                    dtypes_result = {c: pd.Series(True, dtype=dtypes_result).dtype for c in columns_result}
            else:
                try:
                    dtype_series = pd.DataFrame(data).dtypes
                    dtype_series.index = columns_result
                    dtypes_result = dtype_series.to_dict()
                except:
                    dtypes_origin = pd.DataFrame(np.array(matrix_X)).infer_objects().dtypes
                    dtypes_origin.index = columns_result
                    dtypes_result = dtypes_origin.to_dict().copy()
                
                if reset_dtypes:
                    numeric_columns_dict = dict(filter(lambda x: sum([t in str(x) for t in ['int', 'float']]) > 0, dtypes_result.items()))
                    dtypes_result.update({c: np.array(list(map(str, np.array(matrix_X)[:,list(dtypes_result.keys()).index(c)]))).dtype for c in numeric_columns_dict.keys() if c in nuniques[nuniques<=object_threshold].index})
            
            # save_data ***
            if not save_data:
                matrix_X = matrix_X.shape
            else:
                matrix_X = np.array(matrix_X)

            # result ***
            matrix_dict = {}
            matrix_dict['data'] = matrix_X
            matrix_dict['index'] = index_result
            matrix_dict['columns'] = columns_result
        
            return matrix_instance(matrix_dict, kind, result_ndim, dtypes_result, nuniques.to_dict())

    # Split Object를 data로 바꿔주는 함수
    def info_to_data(self, instance):
        """
        【required (Library)】 numpy, pandas, collections.namedtuple, copy
        """
        copy_instance = copy.deepcopy(instance)
        kind = copy_instance.kind
        # # kind ***
        # if kind is None:
        #     kind = copy_instance.kind
        
        # # # name (columns)
        # # if copy_instance.ndim == 1:
        # #     columns = [copy_instance.data['name']]
        # # elif copy_instance.ndim == 2:
        # #     columns = copy_instance.data['columns']

        # # ndim ***
        # if ndim == 2:
        #     if copy_instance.ndim == 1:
        #         copy_instance = self.data_info_split(copy_instance.data['data'], columns=copy_instance.data['name'], 
        #                                 ndim=2, dtypes=copy_instance.dtypes, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        #         copy_instance = copy_instance._replace(kind=kind)

        # elif ndim == 1:
        #     if copy_instance.ndim == 2:
        #         if copy_instance.data['data'].shape[1] > 1:
        #             raise ValueError("'vector' or 'Series' only allows 1D-Array")
        #         else:
        #             copy_instance = self.data_info_split(copy_instance.data['data'], columns=copy_instance.data['columns'][0], 
        #                 ndim=1, dtypes=copy_instance.dtypes, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        #             copy_instance = copy_instance._replace(kind=kind)

        # transform ***
        if kind == 'numpy':
            if copy_instance.ndim == 1:
                return np.array(pd.Series(**copy_instance.data).astype(copy_instance.dtypes))
            if copy_instance.ndim == 2:
                return np.array(pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes))
        elif kind == 'list':
            if copy_instance.ndim == 1:
                    return np.array(pd.Series(**copy_instance.data).astype(copy_instance.dtypes)).tolist()
            if copy_instance.ndim == 2:
                return np.array(pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)).tolist()
        elif kind == 'pandas':
            if copy_instance.ndim == 1:
                return pd.Series(**copy_instance.data).astype(copy_instance.dtypes)
            elif copy_instance.ndim == 2:
                return pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)
        elif 'dict' in kind:
            if copy_instance.ndim == 1:
                return pd.Series(**copy_instance.data).astype(copy_instance.dtypes).to_dict()
            if copy_instance.ndim == 2:
                pd_frame = pd.DataFrame(**copy_instance.data).astype(copy_instance.dtypes)
                return pd_frame.to_dict() if kind == 'dict' else pd_frame.to_dict(kind.split('_')[1])

    # Data를 특정 Format에 맞게 바꿔주는 함수
    def transform(self, data, apply_data=None, apply_instance=None, return_type='data',
            apply_kind=True, apply_ndim=True, apply_index=False, apply_columns=False, apply_dtypes=True,
            reset_dtypes=False, object_threshold=3):
        """
        < input >
          . data
          . apply_data
          . apply_instance
          . apply_options : ['name', 'dtypes', 'shape', 'kind'] are allowed
        """
        # input_data_info = self.data_info(data)
        # matrix_data = input_data_info.frame

        if apply_instance is not None:
            apply_instance = copy.deepcopy(apply_instance)
        elif apply_data is not None:
            apply_instance = self.data_info_split(apply_data, save_data=False)
        else:
            apply_instance = self.data_info_split(data, save_data=False)
        to_instance_dict = {}

        if apply_ndim:
            if type(apply_ndim) != bool:
                to_instance_dict['ndim'] = apply_ndim
            else:
                to_instance_dict['ndim'] = 1 if apply_instance.ndim == 1 else 2
        if apply_index is not False:
            if type(apply_index) != bool:
                to_instance_dict['index'] = apply_index
            else:
                to_instance_dict['index'] = apply_instance.data['index']
        if apply_columns is not False:
            if type(apply_columns) != bool:
                to_instance_dict['columns'] = apply_columns
            else:
                to_instance_dict['columns'] = apply_instance.data['name'] if apply_instance.ndim == 1 else apply_instance.data['columns']
        if apply_dtypes is not False and reset_dtypes is False:
            if type(apply_dtypes) != bool:
                to_instance_dict['dtypes'] = apply_dtypes
            else:
                to_instance_dict['dtypes'] = apply_instance.dtypes if apply_instance.ndim == 1 else list(apply_instance.dtypes.values())
        # elif apply_dtypes is True and  reset_dtypes is False:

        result_instance = self.data_info_split(data, **to_instance_dict, reset_dtypes=reset_dtypes, object_threshold=object_threshold)
        
        if apply_kind:
            if type(apply_kind) != bool:
                result_instance = result_instance._replace(kind=apply_kind)
            else:
                result_instance = result_instance._replace(kind=apply_instance.kind)

        if return_type == 'data':
            return self.info_to_data(result_instance)
        elif return_type == 'instance':
            return result_instance
        elif return_type == 'all':
            return {'data': self.info_to_data(result_instance), 'instance': result_instance}




# ('matrix_info' : ('frame_info' : ['data', 'index', 'columns']), kind, dtypes, nuniques)
# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 숫자형 데이터에 Scaler를 적용하는 Class
class ScalerVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . scaler : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, scaler='StandardScaler', **kwargs):
        self.name = 'Undefined'
        self.scaler = class_object_execution(scaler, **kwargs)
        self.DataHandler = DataHandler()

    def fit(self, x):
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        self.scaler.fit(self.fitted_object.data['data'].reshape(-1,1))
        self.name = self.fitted_object.data['name']
        self.transformed_names = [self.name]

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_data = self.scaler.transform(transformed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = transformed_object.data['name'] if apply_name is False else apply_name
            return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='float')
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = transformed_info.ndim if ndim is None else ndim
            return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='float')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.vector_info_split(x)
        inversed_data = self.scaler.inverse_transform(inversed_object.data['data'].reshape(-1,1))

        if fitted_format:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            # apply_kind = self.fitted_object.kind if kind is None else kind
            apply_ndim = self.fitted_ndim
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=apply_ndim,
                apply_dtypes=apply_dtypes, apply_index=inversed_object.data['index'])
        else:
            apply_name = inversed_object.data['name'] if apply_name is False else apply_name
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else self.fitted_object.dtypes
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def __repr__(self):
        return f"(ScalerInstance) {self.name}: {self.scaler}"


# Vector 형태 (1차원 vector 또는 (-1,1) Shpaed Matrix)의 문자형 데이터에 Encoder를 적용하는 Class
class EncoderVector:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy
    【required (Function)】DataHandler, class_object_execution

    < Input >
     . encoder : Scaler Object or String
                * required: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . x : '1dim vector' or '(-1, 1)shaped matrix'

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
     . get_params
     . get_feature_names
    """
    def __init__(self, encoder='OneHotEncoder', **kwargs):
        self.name='undefined'

        if 'OneHotEncoder' in str(encoder):
            if 'drop' not in kwargs.keys():
                kwargs.update({'drop':'first'})
            if 'sparse' not in kwargs.keys():
                kwargs.update({'sparse':False})
        # self.kwargs = kwargs
        self.DataHandler = DataHandler()
        self.encoder = class_object_execution(encoder, **kwargs)

    def fit(self, x):
        encoder_str = str(self.encoder)
        fitted_info = self.DataHandler.data_info(x, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.vector_info_split(x)
        fitted_series = pd.Series(**self.fitted_object.data)
        # fitted_series = pd.Series(**self.fitted_object.data).apply(lambda x: str(x))
        
        if 'OneHotEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            self.transformed_names = list(map(lambda x: str(self.fitted_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
        elif 'LabelEncoder' in encoder_str:
            self.encoder.fit(fitted_series)
            self.transformed_names = [self.fitted_object.data['name']]
        elif 'OrdinalEncoder' in encoder_str:
            self.encoder.fit(fitted_series.to_frame())
            self.transformed_names = [self.fitted_object.data['name']]

        self.name = self.fitted_object.data['name']

    def transform(self, x, fitted_format=False, apply_name=False, ndim=None, kind=None):
        encoder_str = str(self.encoder)
        transformed_info = self.DataHandler.data_info(x, save_data=False)
        transformed_object = self.DataHandler.vector_info_split(x)
        transformed_series = pd.Series(**transformed_object.data)
        # transformed_series = pd.Series(**transformed_object.data).apply(lambda x: str(x))
        if 'OneHotEncoder' in encoder_str or 'OrdinalEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series.to_frame())
        elif 'LabelEncoder' in encoder_str:
            transformed_data = self.encoder.transform(transformed_series)
        
        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.transformed_names
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                apply_name = list(map(lambda x: str(transformed_object.data['name']) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = [transformed_object.data['name']]
        else:
            if 'OneHotEncoder' in encoder_str:
                apply_name = list(map(lambda x: str(apply_name) + str(x)[2:], self.encoder.get_feature_names()))
            else:
                apply_name = apply_name

        # transform
        if 'OneHotEncoder' not in encoder_str:
            apply_name = apply_name[0]
            if fitted_format:
                return self.DataHandler.transform(transformed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                            apply_ndim=self.fitted_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = True if kind is None else kind
                apply_ndim = transformed_info.ndim if ndim is None else ndim
                return self.DataHandler.transform(transformed_data, apply_instance=transformed_object, apply_columns=True,
                    apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes='int')
        else:       # OneHotEncoder
            parmas = self.encoder.get_params()
            if parmas['sparse']:
                transformed_data = transformed_data.toarray()
            
            if fitted_format:
                # apply_kind = self.fitted_object.kind if kind is None else kind
                apply_ndim = self.fitted_ndim if transformed_data.shape[1] == 1 else 2
                return self.DataHandler.transform(transformed_data, apply_kind=self.fitted_object.kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')
            else:
                apply_kind = transformed_object.kind if kind is None else kind
                apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_data.shape[1] == 1 else 2

                return self.DataHandler.transform(transformed_data, apply_kind=apply_kind, apply_columns=apply_name,
                    apply_ndim=apply_ndim, apply_index=transformed_object.data['index'], apply_dtypes='int')

    def fit_transform(self, x, ndim=None, kind=None):
        self.fit(x)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(x, fitted_format=fitted_format, apply_name=True, ndim=ndim, kind=kind)

    def inverse_transform(self, x, fitted_format=True, apply_name=False, ndim=None, kind=None, dtypes=None):
        encoder_str = str(self.encoder)
        inversed_info = self.DataHandler.data_info(x, save_data=False)
        inversed_object = self.DataHandler.data_info_split(x, ndim=2)
        inversed_data = self.encoder.inverse_transform(inversed_object.data['data'])

        # name
        if (fitted_format is True) or (apply_name is True):
            apply_name = self.fitted_object.data['name']
        elif apply_name is False:
            if 'OneHotEncoder' in encoder_str:
                apply_name = list(map(lambda x: x[:x.rfind('_')], inversed_object.data['columns']))[0]
            else:
                apply_name = inversed_object.data['columns']
        else:
            apply_name = apply_name if type(apply_name) == list else [apply_name]

        if fitted_format:
            apply_dtypes = dtypes if dtypes is not None else True
            return self.DataHandler.transform(inversed_data, apply_instance=self.fitted_object, apply_columns=apply_name,
                apply_ndim=self.fitted_ndim, apply_index=inversed_object.data['index'], apply_dtypes=apply_dtypes)
        else:
            apply_kind = True if kind is None else kind
            apply_ndim = inversed_info.ndim if ndim is None else ndim
            apply_dtypes = dtypes if dtypes is not None else False
            
            return self.DataHandler.transform(inversed_data, apply_instance=inversed_object, apply_columns=apply_name,
                apply_kind=apply_kind, apply_ndim=apply_ndim, 
                apply_dtypes=apply_dtypes, apply_index=True)

    def get_params(self):
        return self.encoder.get_params()

    def get_feature_names(self):
        return np.array(self.transformed_names)

    def __repr__(self):
        return f"(EncoderInstance) {self.name}: {self.encoder}"



### ★★★ ###
# Matrix/Frame/DataFrame 데이터에 Scaler 또는 Encoder를 적용하는 Class
class ScalerEncoder:
    """
    【required (Library)】 numpy, pandas, sklearn.preprocessing.*, copy.deepcopy, functools.reduce
    【required (Function)】DataHandler, class_object_execution, ScalerVector, EncoderVector, dtypes_split

    < Input >
     . encoder : dictionay type {'columns' : Scaler/Encoder Object or String, ...}
                      (default) {'#numeric' : 'StandardScaler', '#object':'OneHotEncoder', '#time', 'StandardScaler'}
                * required Scaler/Encoder: 'str type Name_of_Class', 'str type Name_of_Instance', 'Class object', 'instance object'
     . X : 1dim, 2dim vector or matrix

    < Method >
     . fit
     . transform
     . fit_transform
     . inverse_transform
    """
    def __init__(self, encoder=None, **kwargs):
        self.apply_encoder = {'#numeric':'StandardScaler', '#object':'OneHotEncoder', '#time': 'StandardScaler'}
        if encoder is not None:
            self.apply_encoder.update(encoder)

        self.DataHandler = DataHandler()
        self.kwargs = kwargs
        self.encoder = {}
        self.match_columns = {}

    def fit(self, X):
        self.encoder = {}

        fitted_info = self.DataHandler.data_info(X, save_data=False)
        self.fitted_ndim = fitted_info.ndim
        self.fitted_object = self.DataHandler.data_info_split(X, ndim=2)

        fitted_DataFrame = pd.DataFrame(**self.fitted_object.data).astype(self.fitted_object.dtypes)
        
        self.columns_dtypes = pd.DataFrame(dtypes_split(fitted_DataFrame, return_type='columns_all')).T

        for c in fitted_DataFrame:
            if c in self.apply_encoder.keys():
                if 'scaler' in str(self.apply_encoder[c]).lower():
                    se = ScalerVector(scaler=self.apply_encoder[c])
                elif 'encoder' in str(self.apply_encoder[c]).lower():
                    se = EncoderVector(encoder=self.apply_encoder[c])
            else:
                apply_se = self.apply_encoder['#' + self.columns_dtypes.loc[c, 'dtype_group']]
                if 'scaler' in str(apply_se).lower():
                    se = ScalerVector(scaler=apply_se)
                elif 'encoder' in str(apply_se).lower():
                    se = EncoderVector(encoder=apply_se)
            se.fit(fitted_DataFrame[c])
            self.encoder[c] = copy.deepcopy(se)
            self.match_columns[c] = se.transformed_names

    def transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None):
        transformed_info = self.DataHandler.data_info(X, save_data=False)
        transformed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**transformed_object.data).astype(transformed_object.dtypes)
        if transformed_object.kind != 'pandas':
            X_DataFrame.columns = self.encoder.keys() if columns is None else columns
        
        # transform ***
        transformed_DataFrame = pd.DataFrame()
        for c in X_DataFrame:
            transformed_columnvector = pd.DataFrame(self.encoder[c].transform(X_DataFrame[c], fitted_format=True))
            transformed_DataFrame = pd.concat([transformed_DataFrame, transformed_columnvector], axis=1)
        
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=self.fitted_object.kind, apply_ndim=apply_ndim)
        else:
            apply_kind = transformed_object.kind if kind is None else kind
            apply_ndim = (transformed_info.ndim if ndim is None else ndim) if transformed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(transformed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim)
        
    def fit_transform(self, X, ndim=None, kind=None):
        self.fit(X)
        fitted_format = True if ndim is None and kind is None else False
        return self.transform(X, fitted_format=fitted_format, ndim=ndim, kind=kind)

    def inverse_transform(self, X, fitted_format=False, columns=None, ndim=None, kind=None, dtypes=None):
        inversed_info = self.DataHandler.data_info(X, save_data=False)
        inversed_object = self.DataHandler.data_info_split(X, ndim=2)
        
        X_DataFrame = pd.DataFrame(**inversed_object.data).astype(inversed_object.dtypes)
        if inversed_object.kind != 'pandas':
            X_DataFrame.columns = reduce(lambda x,y : x + y, self.match_columns.values()) if columns is None else columns

        Xcolumns = copy.deepcopy(X_DataFrame.columns)
        match_columns = copy.deepcopy(self.match_columns)

        # inverse_transform ***
        inversed_DataFrame = pd.DataFrame()
        while bool(len(Xcolumns)):
            inversed_target = pd.DataFrame()

            c = Xcolumns[0]
            Xcolumns = Xcolumns.drop(c)
            # print(c, Xcolumns)
            if type(dtypes) == dict:
                if c in dtypes:
                    apply_dtypes = dtypes[c]
                else:
                    apply_dtypes = None
            else:       # bool, str, dtype
                apply_dtypes = dtypes

            for fc, tc in match_columns.items():
                if c in tc:
                    inversed_target = X_DataFrame[tc]

                    del match_columns[fc]
                    tc.remove(c)
                    Xcolumns = Xcolumns.drop(tc)
                    
                    inversed_data = self.encoder[fc].inverse_transform(inversed_target, fitted_format=fitted_format, ndim=2, dtypes=apply_dtypes)
                    inversed_data.columns = [self.encoder[fc].name]

                    inversed_DataFrame = pd.concat([inversed_DataFrame, inversed_data], axis=1)
                    break
            
            if c == X_DataFrame.columns[-1]:
                break
            
        # return ***
        if fitted_format:
            apply_ndim = self.fitted_ndim if inversed_DataFrame.shape[1] == 1 else 2
            # apply_dtypes = dict(filter(lambda x: x[0] in X_DataFrame.columns, self.fitted_object.dtypes.items()))
            apply_dtypes = dict(filter(lambda x: x[0] in inversed_DataFrame.columns, self.fitted_object.dtypes.items()))
            # print(inversed_DataFrame, self.fitted_object.kind, apply_ndim, apply_dtypes)
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=self.fitted_object.kind, 
                    apply_ndim=apply_ndim, apply_dtypes=apply_dtypes)
        else:
            apply_kind = inversed_object.kind if kind is None else kind
            apply_ndim = (inversed_info.ndim if ndim is None else ndim) if inversed_DataFrame.shape[1] == 1 else 2
            return self.DataHandler.transform(inversed_DataFrame, apply_kind=apply_kind, apply_ndim=apply_ndim, apply_dtypes=dtypes)

    def __repr__(self):
        return f"(ScalerEncoder) {self.encoder}"









################################################################################################
################################################################################################
################################################################################################


# Trend Analysis Class
class TrendAnalysis():
    """
    【 Required Library 】
    import statsmodels.api as sm  (sm.tsa.filters.hpfilter)
    
    """
    def __init__(self, x=None, filter='hp_filter', rolling=2, **kwargs):
        self.x = x
        self.rolling = rolling
        
        self.cycle = None
        self.trend = None
        self.trend_slope_ = None
        self.trend_info_ = None
        
        self.params = {}
        self.params.update(kwargs)
        self.filter_result = None
        
        self.filter = filter
        
        if x is not None:
            if filter == 'hp_filter':
                self.filter_result = self.analysis(x=self.x, filter=self.filter, **self.params)
    
    # 【 filters 】
    def analysis(self, x=None, filter=None, **kwargs):
        self.params.update(kwargs)

        x = self.x if x is None else x
        filter = self.filter if filter is None else filter
        
        if filter == 'hp_filter':
            lamb = self.params['lamb'] if 'lamb' in self.params.keys() else 1600
            self.cycle, self.trend = self.hp_filter(x, lamb)
            self.filter_result = (self.cycle, self.trend)
        return self.filter_result
    
    def hp_filter(self, x=None, lamb=None, save_params=False):
        x = self.x if x is None else x
        lamb = (self.params['lamb'] if 'lamb' in self.params.keys() else 1600) if lamb is None else lamb
        
        if save_params:
            self.x = x
            self.params['lamb'] = lamb
        
        if x is not None:
            return sm.tsa.filters.hpfilter(x, lamb)

    # 【 trend_slope 】
    def calc_trend_slope(self, trend):
        return trend.tail(1).mean() - trend.head(1).mean()   
                    
    def trend_slope(self, x=None, trend=None, filter=None, rolling=None, **kwargs):
        self.params.update(kwargs)
        input_x = self.x if x is None else x
        input_trend = self.trend if trend is None else trend
        input_filter = self.filter if filter is None else filter
        input_rolling = self.rolling if rolling is None else rolling
        
        if input_x is None:
            if input_trend is None:
                raise('x, or trend is nessasary for working.')
        else:
            if input_trend is None:
                self.analysis(x=input_x, filter=input_filter, **self.params)
                input_trend = self.trend

        self.trend_slope_ = input_trend.rolling(input_rolling).agg(self.calc_trend_slope)
        return self.trend_slope_

    # 【 trend_info 】
    def calc_trend_info(self, trend_slope):
        trend_slope_shift = trend_slope.iloc[1:]
        info_list = []
        now_sign = trend_slope_shift.iloc[0] > 0
        for e, i in enumerate(trend_slope_shift):
            if i != 0:
                i_sign = i > 0
                if now_sign != i_sign:
                    if now_sign:
                        info_list.append('max')
                    else:
                        info_list.append('min')
                    now_sign = i_sign
                else:
                    info_list.append('up' if i_sign else 'down')
            else:
                info_list.append('keep')
        info_list.append('')

        return pd.Series(info_list, index=trend_slope.index, name=f'{trend_slope.name}_trend_info')
    
    def trend_info(self, x=None, trend=None, trend_slope=None, filter=None, rolling=None, **kwargs):
        self.params.update(kwargs)
        input_trend_slope = self.trend_slope_ if trend_slope is None else trend_slope
        if input_trend_slope is None:
            input_trend_slope = self.trend_slope(x, trend, filter, rolling, **self.params)
            
        self.trend_info_ = self.calc_trend_info(input_trend_slope)
        return self.trend_info_ 

    # 【 Summary 】
    def fit(self, x=None, trend=None, trend_slope=None, filter=None, rolling=None, **kwargs):
        self.params.update(kwargs)
        self.x = self.x if x is None else x
        self.trend_info(self.x, trend, trend_slope, filter, rolling, **self.params)
        
        self.summary = pd.concat([self.x, self.cycle, self.trend, self.trend_slope_, self.trend_info_], axis=1)
        self.summary.columns = [self.x.name, 'cycle', 'trend', 'trens_slope', 'trend_info']
        return self.summary

