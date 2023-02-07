import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns

import scipy as sp

from IPython.core.display import display, HTML

import copy
import pyperclip

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc    # 한글폰트사용
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



################################################################################################################################################################################################
class DataColumns():
    def __init__(self):
        cols_dict = {}
        # 【 Format 】: 주로사용format --------------------------------------------------------------------------------------------------------------
        cols_dict['format0'] = ['강종_소구분', '고객사_국가', 'ORDER_NO', '품종명', '규격약호', '출강목표N', '냉연코일번호', '주문두께', '주문폭', '진행재료_중량',
                            '시험위치L', '인장_방향호수', 'YP', 'TS', 'EL', 'YR', 'TN', 'HER_평균', 'BMB', '소둔공장', '소둔작업완료일시', 'LS', 'HS_TB', 'SS_TB', 'SCS_TB', 'RCS_TB', 'RHS_TB', 'OAS_TB', 'SPM_RollForce', '현공장공정N']
        cols_dict['format1'] = ['강종_소구분', '고객사_국가', 'ORDER_NO', '품종명', '규격약호', '출강목표N', '냉연코일번호', '주문두께', '주문폭', '진행재료_중량',
                                '시험위치L', '인장_방향호수', 'YP', 'TS', 'EL', 'YR', 'TN', 'HER_평균', 'HER_1', 'HER_2', 'HER_3', 'BMB', 'BH2_지수', 'Ra조도', 'B1',
                                '현공장공정N', '소둔공장', '소둔작업완료일시','LS', 'HS_TB', 'SS_TB', 'SCS_TB', 'RCS_TB', 'RHS_TB', 'OAS_TB', 'SPM_RollForce', 'CT', 'CT후냉각',
                                '열연코일_두께', '냉간압하율', 
                                '소둔_SS목표온도', '소둔_RCS목표온도', '소둔_OAS목표온도',
                                'HS_Range', 'SS_Range', 'SCS_Range', 'RCS_Range', 'RHS_Range', 'OAS_Range',
                                'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적',
                                '소둔_전코일번호', '소둔_현코일번호', '소둔_후코일번호',
                                '전후코일_HS_목표온도', '전후코일_SS_목표온도', '전후코일_SCS_목표온도', '전후코일_RCS_목표온도', '전후코일_OAS_목표온도']

        cols_dict['format2'] = ['강종_소구분', '고객사_국가', 'ORDER_NO', '품종명', '규격약호', '출강목표N', '냉연코일번호', '주문두께', '주문폭', '진행재료_중량',
                                '시험위치L', '인장_방향호수', 'YP', '인장2_YP', 'TS', '인장2_TS', 'EL', '인장2_EL', 'YR', 'TN', 'HER_평균', 'HER_1', 'HER_2', 'HER_3', 'BMB', 'BH2_지수', 'Ra조도', 'B1',
                                '현공장공정N', '소둔공장', '소둔작업완료일시','LS', 'HS_TB', 'SS_TB', 'SCS_TB', 'RCS_TB', 'RHS_TB', 'OAS_TB', 'SPM_RollForce', 'CT', 'CT후냉각',
                                '열연코일_두께', '냉간압하율', 
                                '소둔_SS목표온도', '소둔_RCS목표온도', '소둔_OAS목표온도',
                                'HS_Range', 'SS_Range', 'SCS_Range', 'RCS_Range', 'RHS_Range', 'OAS_Range',
                                'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적',
                                '소둔_전코일번호', '소둔_현코일번호', '소둔_후코일번호',
                                '전후코일_HS_목표온도', '전후코일_SS_목표온도', '전후코일_SCS_목표온도', '전후코일_RCS_목표온도', '전후코일_OAS_목표온도']


        cols_dict['format1_detail'] = ['강종_소구분', '고객사_국가', 'ORDER_NO', '품종명', '규격약호', '출강목표N', '냉연코일번호', '주문두께', '주문폭', '진행재료_중량',
                                '시험위치L', '인장_방향호수', 'YP', 'TS', 'EL', 'YR', 'TN', 'HER_평균', 'HER_1', 'HER_2', 'HER_3', 'BMB', 'BH2_지수', 'Ra조도', 'B1',
                                'YP_보증범위',	'TS_보증범위', 'EL_보증범위',
                                'TN_측정구분', 'BMB_방향', 'BMB_시험종류구분', 'BMB_시험굴곡각도', 'BMB_시험굴곡간격구분', 
                                '현공장공정N', '소둔공장', '소둔작업완료일시','LS_POS', 'HS_POS', 'SS_POS', 'RCS_POS', 'RHS_POS', 'OAS_POS', 'SPM_EL', 'SPM_RollForce', 'CT', 'CT후냉각',
                                '열연코일_두께', '냉간압하율', 
                                'HS', 'SS', 'RCS', 'OAS', 'HS_TB', 'SS_TB', 'SCS_TB', 'RCS_TB', 'RHS_TB', 'OAS_TB',
                                '소둔_SS목표온도', '소둔_RCS목표온도', '소둔_OAS목표온도',
                                'HS_Range', 'SS_Range', 'SCS_Range', 'RCS_Range', 'RHS_Range', 'OAS_Range',
                                'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적',
                                '소둔_전코일번호', '소둔_현코일번호', '소둔_후코일번호',
                                '전후코일_HS_목표온도', '전후코일_SS_목표온도', '전후코일_SCS_목표온도', '전후코일_RCS_목표온도', '전후코일_OAS_목표온도']

        cols_dict['format2_detail'] = ['강종_소구분', '고객사_국가', 'ORDER_NO', '품종명', '규격약호', '출강목표N', '냉연코일번호', '주문두께', '주문폭', '진행재료_중량',
                                '시험위치L', '인장_방향호수', 'YP', '인장2_YP', 'TS', '인장2_TS', 'EL', 'YR', 'TN', 'HER_평균', 'HER_1', 'HER_2', 'HER_3', 'BMB', 'BH2_지수', 'Ra조도', 'B1',
                                'YP_보증범위',	'TS_보증범위', 'EL_보증범위',
                                'TN_측정구분', 'BMB_방향', 'BMB_시험종류구분', 'BMB_시험굴곡각도', 'BMB_시험굴곡간격구분', 
                                '현공장공정N', '소둔공장', '소둔작업완료일시','LS_POS', 'HS_POS', 'SS_POS', 'RCS_POS', 'RHS_POS', 'OAS_POS', 'SPM_EL', 'SPM_RollForce', 'CT', 'CT후냉각',
                                '열연코일_두께', '냉간압하율', 
                                'HS', 'SS', 'RCS', 'OAS', 'HS_TB', 'SS_TB', 'SCS_TB', 'RCS_TB', 'RHS_TB', 'OAS_TB',
                                '소둔_SS목표온도', '소둔_RCS목표온도', '소둔_OAS목표온도',
                                'HS_Range', 'SS_Range', 'SCS_Range', 'RCS_Range', 'RHS_Range', 'OAS_Range',
                                'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적',
                                '소둔_전코일번호', '소둔_현코일번호', '소둔_후코일번호',
                                '전후코일_HS_목표온도', '전후코일_SS_목표온도', '전후코일_SCS_목표온도', '전후코일_RCS_목표온도', '전후코일_OAS_목표온도']


        # 【 PGBISDS 】: 신경영관리해석계(진행) -------------------------------------------------------------------------------------------
        cols_dict['PGBISDS'] = {}
        cols_dict['PGBISDS']['slab_tr'] = ['OrderNO','규격약호', '출강목표N', '재료번호', '진행상태', '현공정', '고객사명']
        cols_dict['PGBISDS']['hr'] = ['OrderNO','규격약호', '출강목표N', '재료번호', '진행상태', '주문두께T', '주문폭W', '열연코일_목표두께', '열연코일_목표폭']
        cols_dict['PGBISDS']['pcm'] = ['진행상태','열연코일번호', '열연코일_목표두께', '주문두께T', '주문폭W', 'PCM_통과공정', 'CAL_통과공정', 
                                    'NULL', 'NULL', 'NULL', 'NULL',  '강종_소구분', '규격약호', '출강목표N', '사내보증번호N',
                                    'NULL', 'NULL', 'CT', 'CT냉각',  'NULL', 'NULL', 'NULL', 'NULL', '관리대상']
        cols_dict['PGBISDS']['cg'] = ['고객사명', 'OrderNO', '강종_소구분', '품종', '확정통과공정','규격약호', '출강목표N', '재료번호', '진행상태', '현공정', '주문두께T', '주문폭W', '현중량']
        cols_dict['PGBISDS']['sample'] = ['OrderNO', '품종', '강종_소구분', 'NULL', 'OrderType', '재료번호', '진행상태', '주문두께T', '주문폭W', '현중량', '포장단중하한', '포장단중상한']

        self.cols_dict = cols_dict



############################################################################################################################################################################
####### DS_DataFrame #######################################################################################################################################################
############################################################################################################################################################################


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
        result_Series.index = x.index
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


# print DataFrame to HTML
def print_DataFrame(data):
    display(HTML(data._repr_html_()))




# 【 Series, Vector function 】 ################################################################################
# 일정 범위구간에 따라 Level을 나눠주는 함수
# function cut_range
def cut_range(x, categories, right=False, labels=None, include_lowest=True, remove_unused_categories=True, ordered=True):
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
        if ordered is True:
            vector = pd.Series(pd.Categorical(vector, categories=categories, ordered=True))
        elif ordered == 'descending':
            vector = pd.Series(pd.Categorical(vector, categories=categories[::-1], ordered=True))
        else:
            vector = pd.Series(pd.Categorical(vector, categories=categories))
        vector.index = x.index
        if remove_unused_categories:
            vector = vector.cat.remove_unused_categories()
    elif "<class 'numpy.ndarray'>" in str(type(vector)):
        vector

    return vector




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
    """
    【 Required Class 】 Capability
     
     . capability : ['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
     . statistics : ['count','mean','std']
    """
    def __init__(self, capability=['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
                , statistics=['count','mean','std']):
        self.capability = capability
        self.statistics = statistics

    def analysis(self, data, criteria, target=None, value_vars=None, capability=None, statistics=None, lean=False, hist_kwargs={}, line_kwargs={}):
        """
          . data : DataFrame
          . criteria : Criteria table (index: group, value: range of capability)
          . capability : ['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
          . statistics : ['count','mean','std']
          . lean : if True, sign mean direction of Cpk
          . hist_kwargs(dict) : histogram arguments
          . line_kwargs(dict) : cpk_line arguments
        """
        
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
        self.result = capability_table
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
    【 Required Class 】 Capability, CapabilityGroup
     
     . capability : ['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot']
     . statistics : ['count','mean','std']
    """
    def __init__(self, capability=['cpk', 'observe_reject_prob', 'gaussian_reject_prob', 'cpk_plot'],
            statistics=['count','mean','std']):
        self.capability = capability
        self.statistics = statistics
    
    def analysis(self, data, group=None, criteria=None, criteria_column=None, **kwargs):
        """
          . group : 'group' or ['group1', 'group2']
          . criteria : {'YP': ['600~750'], 'TS':['980~'], 'EL':['12~']}
          . criteria_column : {'YP':'YP_보증범위', 'TS':'TS_보증범위', 'EL':'EL_보증범위'}
          . **kwargs : 'CapabilityGroup' class 'analysis' method initialize arguments
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
                
                cg.analysis(gv, criteria=cy_frame, **kwargs)
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



















############################################################################################################################################################################
####### DS_Image ###########################################################################################################################################################
############################################################################################################################################################################



import os
from PIL import Image   # PIL는 이미지를 load 할 때 이용

from io import StringIO,  BytesIO
import win32clipboard

from IPython.display import clear_output
from IPython.core.display import display, HTML
# display(HTML(df._repr_html_()))


# [ Functions ] ------------------------------------------------------------------------------
# from PIL import Image   # PIL는 이미지를 load 할 때 이용
# from PIL import Image, ImageFilter, ImageGrab  # imports the library

# from io import StringIO,  BytesIO
# import win32clipboard

def fun_Send_To_Clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def img_to_clipboard(fig, format='jpeg', dpi='figure'):
    '''
    fig: pyplot figure
    '''
    fig.savefig(f'pyplot_temper_img.{format}', bbox_inches='tight', dpi=dpi)    # png파일로저장
    PIL_img = Image.open(f'pyplot_temper_img.{format}').copy()   #png파일 PIL image형태로 불러오기
    os.remove(f'pyplot_temper_img.{format}')  # png파일 지우기
    output = BytesIO()
    PIL_img.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    fun_Send_To_Clipboard(win32clipboard.CF_DIB, data)




















############################################################################################################################################################################
####### DS_Plot ############################################################################################################################################################
############################################################################################################################################################################
# def ttest_each
# 여러개의 Group별로 평균, 편차, ttest 결과를 Return 하는 함수
# import scipy as sp
from collections import namedtuple
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


# -----------------------------------------------------------------------------------------------------
# function jitter (make jitter list)
def jitter(x, ratio=0.6, method='uniform', sigma=5, transform=True):
    type_string = str(type(x))
    if 'Series' in type_string:
        dtype = 'Series'
        series_x = x.copy()
    else:
        if 'list' in type_string:
            dtype = 'list'
        elif 'ndarray' in type_string:
            dtype = 'ndarray'
        series_x = pd.Series(x)
    x1 = series_x.drop_duplicates().sort_values()
    x2 = series_x.drop_duplicates().sort_values().shift()
    jitter_range = (x1-x2).min()*ratio

    # apply distribution
    if method == 'uniform':
        jitter = pd.Series(np.random.rand(len(x))*jitter_range - jitter_range/2)
    elif method == 'gaussian' or method == 'normal':
        jitter = pd.Series(np.random.randn(len(x))*(jitter_range/sigma))
    if dtype == 'Series':
        jitter.index = x.index
    
    if transform:
        jitter += series_x
    
    # transform return type
    if dtype == 'Series':
        result = jitter
    if dtype == 'list':
        result = jitter.tolist()
    elif dtype == 'ndarray':
        result = jitter.values
    
    return result


# -----------------------------------------------------------------------------------------------------
# Dist_Box Plot Graph Function
def distbox(data, on, group=None, figsize=[5,5], title='auto', bins=None,
            mean_line=None, axvline=None, lsl=None, usl=None, xscale='linear',
            xlim=None, ylim=None,
            equal_var=False, return_plot='close'):
    # group = change_target
    # on = 'YP'
    # title = 'abc'
    normal_data = data.copy()
    # box_colors = ['steelblue','orange']
    box_colors = sns.color_palette()

    figs, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
    

    # distplot
    if title is not None and title != 'auto':
        figs.suptitle(title, fontsize=13)
    elif title == 'auto':
        title_name = on + '_Plot'
        if group is not None:
            title_name += ' (group: ' + group + ')'
        figs.suptitle(title_name, fontsize=13)

    if group is not None:
        # group_mean
        group_mean = normal_data.groupby(group)[on].mean()
        len_group_mean = len(group_mean)
        group_mean.sort_index(ascending=True, inplace=True)

        # distplot
        data_group = []
        for i, (gi, gv) in enumerate(normal_data.groupby(group)):
            data_group.append(gv[on].dropna())
            try:
                sns.distplot(gv[on], label=gi, ax=axes[0], bins=bins)
                if mean_line is not None:
                    axes[0].axvline(x=group_mean[gi], c=box_colors[i], alpha=0.5)
            except:
                pass
        axes[0].legend()
        axes[0].set_xscale(xscale)
        
        # boxplot
        boxes = sns.boxplot(x=on, y=group, data=normal_data, 
                orient='h', color='white', linewidth=1, ax=axes[1],
                order=sorted(normal_data[group].unique()) )
        axes[1].set_xscale(xscale)

        # mean_point
        axes[1].scatter(x=group_mean, y=list(range(0,len_group_mean)), 
                        color=box_colors[:len_group_mean], edgecolors='white', s=70)
        
        if len(data_group) == 2:
            pavlues = sp.stats.ttest_ind(*data_group, equal_var=equal_var).pvalue
        else:
            pavlues = sp.stats.f_oneway(*data_group).pvalue
        label_name = 'Anova Pvalue: ' + format(pavlues, '.3f')

        summary_dict = normal_data.groupby(group)[on].agg(['count','mean','std']).applymap(lambda x: auto_formating(x)).to_dict('index')

        if lsl is not None or usl is not None:
            cpk_list = ['-' if v['count'] < 5 else str(round(cpk(v['mean'], v['std'], lsl, usl),2)) for k, v in summary_dict.items()]
            iter_object = zip(summary_dict.items(), cpk_list)  
            label_summary = '\n'.join(['* ' + str(k) + ': ' + str(v).replace('{','').replace('}','').replace("'",'').replace(':','') + ' (cpk: '+cpk_value+')' for (k,v), cpk_value in iter_object ])
        else:
            iter_object = summary_dict.items()
            label_summary = '\n'.join(['* ' + str(k) + ': ' + str(v).replace('{','').replace('}','').replace("'",'').replace(':','') for k,v in iter_object ])
        
        label_name = label_name + '\n' + label_summary
        plt.xlabel(label_name, fontsize=11)
    else:
        # group_mean
        group_mean, group_std = normal_data[on].agg(['mean','std'])

        # distplot
        sns.distplot(normal_data[on], ax=axes[0], bins=bins)
        if mean_line:
            axes[0].axvline(x=group_mean, c=box_colors[0], alpha=0.5)
        # boxplot
        axes[0].set_xscale(xscale)
        boxes = sns.boxplot(data=normal_data, x=[on], orient='h', color='white', linewidth=1, ax=axes[1])
        
        # mean_points
        plt.scatter(x=group_mean, y=[0], color=box_colors[0], edgecolors='white', s=70)
        axes[1].set_xscale(xscale)

        summary_dict = normal_data[on].agg(['count','mean', 'std']).apply(lambda x: auto_formating(x)).to_dict()
        label_summary = '* All: ' + ', '.join([ k + ' ' + str(v) for k,v in summary_dict.items() ])
        label_name = '\n' + label_summary

        if lsl is not None or usl is not None:
            if len(normal_data) < 5:
                cpk_value = '-'
            cpk_value = str(round(cpk(group_mean, group_std, lsl=lsl, usl=usl),2))
            label_name = label_name + ' (cpk: ' + cpk_value +')'
        plt.xlabel(label_name, fontsize=11)

    # Box-plot option
    for bi, box in enumerate(boxes.artists):
        box.set_edgecolor(box_colors[bi])
        for bj in range(6*bi,6*(bi+1)):    # iterate over whiskers and median lines
            boxes.lines[bj].set_color(box_colors[bi])
    plt.grid(alpha=0.1)
    figs.subplots_adjust(hspace=0.5)
    # figs.subplots_adjust(bottom=0.2)
    
    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
        
    if ylim is not None:
        axes[0].set_ylim(ylim)

    # axvline
    if axvline is not None and type(axvline) == list:
        for vl in axvline:
            axes[0].axvline(vl, color='orange', ls='--', alpha=0.3)
            axes[1].axvline(vl, color='orange', ls='--', alpha=0.3)

    if lsl is not None:
        axes[0].axvline(lsl, color='red', ls='--', alpha=0.3)
        axes[1].axvline(lsl, color='red', ls='--', alpha=0.3)
    if usl is not None:
        axes[0].axvline(usl, color='red', ls='--', alpha=0.3)
        axes[1].axvline(usl, color='red', ls='--', alpha=0.3)

    if return_plot == 'close':
        plt.close()
    elif return_plot == 'show':
        plt.show()
    elif return_plot is None or return_plot == False:
        pass
    return figs

    
# Histogram Compare Graph Function
def hist_compare(data1, data2, figsize=None, title=None, bins=30, label=None, hist_alpha=0.5, histtype='stepfilled',
    legend=True, color=['skyblue','orange'], cpk_color=None, lsl=None, usl=None,
    cpk_alpha=0.7, legend_loc='upper right', return_plot=True, axvline=None, axvline_color='red', **hist_kwargs):

    hist_data1 = pd.Series(data1).dropna().astype('float')
    hist_data2 = pd.Series(data2).dropna().astype('float')

    len_data1 = len(hist_data1)
    len_data2 = len(hist_data2)


    mean_data1 = auto_formating(hist_data1.mean()) if len_data1 > 1 else hist_data1.iloc[0]
    mean_data2 = auto_formating(hist_data2.mean()) if len_data2 > 1 else hist_data2.iloc[0]
    std_data1 = auto_formating(hist_data1.std()) if len_data1 > 1 else np.nan
    std_data2 = auto_formating(hist_data2.std()) if len_data2 > 1 else np.nan
    # pvalue = sp.stats.ttest_ind(hist_data1, hist_data2, equal_var=False)[1]
    if len_data1 == 1 and len_data2 == 1:
        pavlue = np.nan
    elif len_data1 == 1:
        pvalue = round(sp.stats.ttest_1samp(hist_data1.iloc[0], hist_data2).pvalue, 3)   # x1 Column의 평균이 4와 같은가?
    elif len_data2 == 1:
        pvalue = round(sp.stats.ttest_1samp(hist_data1, hist_data2.iloc[0]).pvalue, 3) 
    else:
        pvalue = round(sp.stats.ttest_ind_from_stats(*hist_data1.agg(['mean','std','count']),
                *hist_data2.agg(['mean','std','count']), equal_var=False).pvalue, 3)

    try:
        name_data1 = data1.name
    except:
        name_data1 = 'Group1'
    try:
        name_data2 = data2.name
    except:
        name_data2 = 'Group2'

    if label:
        name_data1 = label[0]
        name_data2 = label[1]

    cpk_color = color if cpk_color is None else cpk_color

    if return_plot:
        fig = plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    elif name_data1 == name_data2:
        plt.title(f'{name_data1} Histogram')
    else:
        plt.title(f'{name_data2} - {name_data2} Histogram')
    
    if label:
        label_content = label
    else:
        label_content = [name_data1, name_data2]
    plt.hist([hist_data1, hist_data2], histtype=histtype, bins=bins, edgecolor='darkgray', 
            color=color, alpha=hist_alpha, label=label_content, **hist_kwargs)
    if 'density' in hist_kwargs.keys() and hist_kwargs['density'] is True:
        pass
    else:
        plt.plot(*np.array(cpk_line(hist_data1)).T, color=cpk_color[0])
        plt.plot(*np.array(cpk_line(hist_data2)).T, color=cpk_color[1])
    plt.axvline(mean_data1, color=cpk_color[0], ls='dashed', alpha=cpk_alpha)
    plt.axvline(mean_data2, color=cpk_color[1], ls='dashed', alpha=cpk_alpha)
    
    if lsl is not None or usl is not None:
        for l in [lsl, usl]:
            if l is not None:
                plt.axvline(l, color='red', alpha=0.7)
        cpk_value1 = round(cpk(mean_data1, std_data1, lsl=lsl, usl=usl),3)
        cpk_value2 = round(cpk(mean_data2, std_data2, lsl=lsl, usl=usl),3)
        xlabel = f'{name_data1}: (n) {len(hist_data1)}   (mean) {mean_data1}   (std) {std_data1}   (cpk) {cpk_value1}\n{name_data2}: (n) {len(hist_data2)}   (mean) {mean_data2}   (std) {std_data2}   (cpk) {cpk_value2}\np-value: {round(pvalue,3)}'
    else:
        xlabel = f'{name_data1}: (n) {len(hist_data1)}   (mean) {mean_data1}   (std) {std_data1}\n{name_data2}: (n) {len(hist_data2)}   (mean) {mean_data2}   (std) {std_data2}\np-value: {round(pvalue,3)}'
    plt.xlabel(xlabel, fontsize=12)

    if legend is True:
        plt.legend(loc=legend_loc)
    
    
    if type(axvline) == list:
        if len(axvline) > 0 and axvline[0] is not None:
            for axvl in axvline:
                plt.axvline(axvl, color=axvline_color, alpha=0.5, ls='--')
    elif axvline is not None:
        plt.axvline(axvline, color=axvline_color, alpha=0.5, ls='--')

    if return_plot:
        plt.close()
        return fig
    elif return_plot == 'show':
        plt.show()
    else:
        pass


# violin_box_plot
def violin_box_plot(x=None, y=None, data=None, group=None, figsize=None,
    title=None, color=None, label=None, return_plot=True):
    if data is None:
        if len(x) != len(y):
            raise 'Different length error between x and y'
        else:
            violin_box_data = pd.concat([pd.Series(x).astype('str'), pd.Series(y)], axis=1)
            if x is None:
                try:
                    x = group.name
                except:
                    x = 'x'
            elif group is None:
                try:
                    x = x.name
                except:
                    x = 'x'
            try:
                y = y.name
            except:
                y = 'y'   
    else:
        if x is None:
            x = group
        elif group is None:
            x = x
        violin_box_data = data[[x,y]]
        
    violin_box_data.columns = [x, y]


    def decimal(x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))
    
    def auto_decimal(x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            return round(x, decimal(x, rev=rev))

    def describe_string(x):
        mean = auto_decimal(x.mean())
        std = auto_decimal(x.std())
        return f'mean {mean},  std {std}'

    box_data_dict = {gi: np.array(gv[y]) for gi, gv in violin_box_data.groupby(x)}
    box_describe_dict = {gi: describe_string(gv[y]) for gi, gv in violin_box_data.groupby(x)}

    if return_plot:
        fig = plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    else:
        plt.title(f'{y} by {x} Violin Box Plot', fontsize=14)

    top_violin = plt.violinplot(box_data_dict.values(), showextrema=True, widths=0.7)

    for tv in top_violin['bodies']:
        tv.set_alpha(0.13)
        if color is not None:
            tv.set_facecolor(color)
    top_violin['cbars'].set_color('none')
    top_violin['cmaxes'].set_color('none')
    top_violin['cmins'].set_color('none')
    
    box_props = {}
    box_props['meanprops'] = {'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'none'}
    if color is not None:
        box_props['boxprops'] = {'color': color}
        box_props['capprops'] = {'color': color}
        box_props['whiskerprops'] = {'color': color}
        box_props['meanprops']['markerfacecolor'] = color
        box_props['medianprops'] = {'color': color}
        
    top_box = plt.boxplot(box_data_dict.values(), labels=box_data_dict.keys(),
        showmeans=True, widths=0.2, **box_props)
    
    if violin_box_data[x].nunique() == 1:
        xlabel = f'{x}\n\n' + '\n'.join([f'({di})  {dv}' for di, dv in box_describe_dict.items()])
    elif violin_box_data[x].nunique() > 1:
        pvalues = sp.stats.f_oneway(*box_data_dict.values()).pvalue
        xlabel = f'{x}\n\n' + '\n'.join([f'({di})  {dv}' for di, dv in box_describe_dict.items()]) + f'\npvalues: {str(round(pvalues,3))}'
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(f'{y}')
    
    if return_plot == 'show':
        plt.show()
    elif return_plot is True:
        plt.close()
        return fig





############################################################################################################################################################################
####### DS_MachineLearning ############################################################################################################################################################
############################################################################################################################################################################

from scipy import stats

# ModelEvaluate
class ModelEvaluate():
    '''
    < input >
    X : DataFrame or 2-Dim matrix
    y : Series
    model : sklearnModel
    '''
    def __init__(self, X, y, model, model_type=None, verbose=1):
        model_name = str(type(model)).lower()
        X_frame = pd.DataFrame(X)
        y_true = np.array(y).ravel()
        y_pred = np.array(model.predict(X)).ravel()
        n_data, dof = X.shape

        if model_type is None:
            if ('regress' in model_name) or ('lasso' in model_name) or ('ridge' in model_name) or ('elasticnet' in model_name):
                model_type = 'regressor'
            if 'classi' in model_name:
                model_type = 'classifier'

        if model_type == 'regressor':
            #### sum_square ****
            sum_square_instance = namedtuple('sum_square', ['sst', 'ssr', 'sse'])
            self.sst = sum((y_true-y_true.mean())**2)
            self.ssr = sum((y_true.mean() - y_pred)**2)
            self.sse = sum((y_true - y_pred)**2)            

            sum_square_list = [self.sst, self.ssr, self.sse]
            self.sum_square = sum_square_instance(*[self.auto_decimal(m) for m in sum_square_list]) 

            #### metrics ****
            metrics_instance = namedtuple('metrics', ['r2_score', 'r2_adj', 'mse', 'rmse', 'mae', 'mape'])

            self.r2_score = 1 - self.sse/self.sst
            # self.r2_adj = 1 - ((n_data-1) * self.ssr/self.sst) / (n_data - dof - 1)
            self.r2_adj = 1 - ((n_data-1) * self.sse/self.sst) / (n_data - dof - 1)
            

            self.mse = self.sse/(n_data-2)
            self.rmse = np.sqrt(self.mse)
            self.mae = sum(np.abs(y_true - y_pred)) / n_data
            mape_series = pd.Series(1 - np.abs( (y_true - y_pred) / y_true ))
            mape_series = mape_series[~((mape_series == np.inf) | (mape_series == -np.inf))]
            self.mape = mape_series.mean()
            
            metrics_list = [self.r2_score, self.r2_adj, self.mse, self.rmse, self.mae, self.mape]
            self.metrics = metrics_instance(*[self.auto_decimal(m) for m in metrics_list]) 

            if verbose > 0:
                print(' .', self.sum_square) 
                print(' .', self.metrics) 

            #### hypothesis ****
            try:
                if sum([i.lower() in model_name for i in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']]):
                    hypothesis_instance = namedtuple('hypothesis', ['tvalues', 'pvalues'])

                    params = np.array([model.intercept_] + list(model.coef_))
                    newX = pd.DataFrame({"Constant": np.ones(n_data)}, index=X_frame.index).join(X_frame)
                    std_b = np.sqrt(self.mse*(np.linalg.inv(np.dot(newX.T,newX)).diagonal()))
                    t = params/ std_b               
                    p = 2 * (1 - stats.t.cdf(np.abs(t), n_data - dof))

                    try:
                        X_names = ['const'] + list(X.columns)
                    except:
                        X_names = ['const'] + ['x' + str(i+1) for i in np.arange(dof)]
                    try:
                        y_name = y.name
                    except:
                        y_name = 'Y'


                    self.tvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, t)}
                    self.pvalues = {k: self.auto_decimal(v) for k, v in zip(X_names, p)}

                    self.hypothesis = hypothesis_instance(self.tvalues, self.pvalues)

                    #### linear ****
                    linear_instance = namedtuple('linear', ['coef', 'formula'])
                    self.coef = {k: self.auto_decimal(v) for k, v in zip(X_names, params)}
                    self.formula = y_name + ' = ' + ''.join([ f'{str(v)}·{k}' if i == 0 else (' + ' if v > 0 else ' - ') + (str(abs(v)) if k=='const' else f'{abs(v)}·{str(k)}') for i, (k, v) in enumerate(self.coef.items())])

                    self.linear = linear_instance(self.coef, self.formula)
                    if verbose > 0:
                        print(' .', self.hypothesis) 
                        print(' .', self.linear)
            except:
                pass

        elif model_type == 'classifier':
            pass

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self, x, rev=0):
        if pd.isna(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))



################################################################################################################
# ['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']
class FeatureInfluence():
    """
    【required (Library)】 numpy, pandas
    【required (Class)】DataHandler, Mode
    【required (Function)】class_object_execution, auto_formating, dtypes_split

    < Input >

    < Output >
    
    """
    def __init__(self, train_X=None, estimator=None, n_points=5, encoder=None, encoderX=None, encoderY=None, conditions={}, y_name=None, confidential_interval=None):
        self.estimator=estimator
        self.train_X = train_X
        self.n_points = n_points
        self.conditions = conditions

        self.encoder = encoder
        self.encoderX = encoderX
        self.encoderY = encoderY
        self.y_name =y_name

        self.DataHandler = DataHandler()
        self.grid_X = None
        
        self.confidential_interval = confidential_interval
        
    # define train, grid data handler instance
    def define_train_grid_data(self, train_X=None, grid_X=None, conditions={}, n_points=None):
        # train_data
        if train_X is not False:
            if train_X is None:
                if self.train_X is None:
                    raise Exception('train_X must be required to predict')
                else:
                    train_X = self.train_X

            self.train_X_info = self.DataHandler.data_info(train_X, save_data=False)
            if self.train_X_info.kind == 'pandas':
                self.train_X_info_split = self.DataHandler.data_info_split(train_X)
            else:
                self.train_X_info_split = self.DataHandler.data_info_split(train_X, columns=list(range(self.train_X.shape[1])))
        
        # grid_data
        if grid_X is not False:
            if grid_X is None:
                if self.grid_X is None:
                    grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
                else:
                    grid_X = self.grid_X

            self.grid_X_info = self.DataHandler.data_info(grid_X, save_data=False)
            if self.grid_X_info.kind == 'pandas':
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X)
            else:
                self.grid_X_info_split = self.DataHandler.data_info_split(grid_X, columns=list(range(self.grid_X.shape[1])))

    # checking unpredictable
    def predictable_check(self, target_data=None, criteria_data=None):
        target_frame = self.DataHandler.transform(target_data, apply_kind='pandas')
        criteria_frame = self.DataHandler.transform(criteria_data, apply_kind='pandas')
        
        numeric_columns = dtypes_split(target_frame, return_type='columns_list')['numeric']
        target_numeric = target_frame[numeric_columns]
        criteria_numeric = criteria_frame[numeric_columns]

        criteria_min = criteria_numeric.min()
        criteria_max = criteria_numeric.max()

        if target_numeric.shape == (1,0):
            return pd.DataFrame(columns=['unpred', 'unpred_columns', 'lower_columns', 'upper_columns'])
        else:
            unpredictable = pd.Series((target_numeric < criteria_min).any(axis=1) | (target_numeric > criteria_max).any(axis=1))
            unpredictable_cols_lower = pd.DataFrame(target_numeric < criteria_min).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols_upper = pd.DataFrame(target_numeric > criteria_max).apply(lambda x: [i for i, xc in zip(x.index, x) if xc==True] ,axis=1)
            unpredictable_cols = unpredictable_cols_lower + unpredictable_cols_upper

            unpredictable_result = pd.concat([unpredictable, unpredictable_cols, unpredictable_cols_lower, unpredictable_cols_upper], axis=1)
            unpredictable_result.columns = ['unpred', 'unpred_columns', 'lower_columns', 'upper_columns']
            return unpredictable_result

    # generate grid_table *
    def make_grid(self, train_X=None, conditions={}, n_points=None, save_result=True, return_result=False):
        re_num = re.compile('\d')

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)
        
        if n_points is None:
            n_points = self.n_points

        X_analysis = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        X_mean = X_analysis.apply(lambda x: x.mean() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else x.value_counts().index[0] ,axis=0)
        X_std = X_analysis.apply(lambda x: x.std() if 'int' in str(x.dtype) or 'float' in str(x.dtype) else np.nan, axis=0)
        X_dict = X_mean.to_dict()
        dtypes = self.train_X_info_split.dtypes

        for xc in conditions:
            el = conditions[xc]
            if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]):
                if type(el) == str:
                    if 'monte' in el.lower():
                        if '(' in el:
                            monte_str, X_mean_el = el.split('(')
                            X_mean_el = int(X_mean_el.replace(')',''))
                        else:
                            monte_str = el
                            X_mean_el = X_mean[xc]
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        el_list = np.random.randn(monte_n_points) * X_std[xc] + X_mean_el
                    elif '~' in el:
                        x_min, x_max = map(str, X_analysis[xc].agg(['min','max']))
                        el_split = el.split('~')
                        if len(el_split) == 2:
                            split_list = list(map(lambda x: str(x).strip().replace('min', x_min).replace('max',x_max), el_split))
                            if split_list[0] == '':
                                split_list[0] = x_min
                            if split_list[1] == '':
                                split_list[1] = x_max
                            el_list = np.linspace(*map(float, split_list), n_points)
                        elif len(el_split) == 1:
                            el_list = float(el_split[0].strip().replace('min', x_min).replace('max',x_max))
                elif 'int' in str(type(el)) or 'float' in str(type(el)):
                    el_list = conditions[xc]
                elif 'numpy' in str(type(el)):
                    el_list = conditions[xc]
            else:       # object dtype
                el_unique_vc = X_analysis[xc].value_counts()/len(X_analysis[xc])
                
                if type(el) == str:
                    if el.strip() in ['all', '~', 'min~max', 'min ~ max']:
                        el_list = list(el_unique_vc.index)
                    elif 'monte' in el.lower(): 
                        monte_str = el
                        monte_n_list = re_num.findall(monte_str)
                        monte_n_points = int(''.join(monte_n_list)) if monte_n_list else n_points
                        if monte_n_points > len(el_unique_vc):
                            monte_n_points = len(el_unique_vc)
                        el_list = list(np.random.choice(el_unique_vc.index, size=monte_n_points, replace=False, p=el_unique_vc))
                    else:
                        el_list = el
                elif type(el) == list:
                    el_list = el.copy()
                else:
                    el_list = el
            X_dict[xc] = el_list
            # break
        # X_dict 
        
        X_dict_array =  {k: v for k, v in X_dict.items() if type(v) == np.ndarray or type(v) == list}
        if len(X_dict_array) > 0:
            grid_X_frame_temp = pd.DataFrame(np.array(np.meshgrid(*X_dict_array.values())).reshape(len(X_dict_array),-1).T, columns=X_dict_array.keys())
                       
            for k, v in X_dict.items():
                if k not in X_dict_array.keys():
                    grid_X_frame_temp[k] = v
            grid_X_frame = grid_X_frame_temp[list(X_dict.keys())]
        else:
            grid_X_frame = pd.Series(X_dict).to_frame().T
        grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.train_X_info_split.kind)
        
        for xc in grid_X:
            apply_dtype = 'float' if 'int' in str(dtypes[xc]) or 'float' in str(dtypes[xc]) else 'object'
            if apply_dtype == 'object':
                obj_type = type(X_analysis[xc].iloc[0])
                if 'str' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: str(x))
                if 'int' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: int(x))
                if 'float' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: float(x))
                if 'bool' in str(obj_type):
                    grid_X[xc] = grid_X[xc].apply(lambda x: bool(x))
                grid_X[xc] = grid_X[xc].astype(self.train_X_info_split.dtypes[xc])
            else:
                grid_X[xc] = grid_X[xc].astype(apply_dtype)

        if save_result:
            self.n_points = n_points
            self.conditions = conditions
            self.grid_X = grid_X

        if return_result:
            return grid_X
        else:
            return self

    # apply model to grid_table
    def grid_apply_model(self, grid_X=None, x=None, y=None, model='LinearRegression', train_X=None, conditions={}, n_points=None,
        inplace=True, return_result=False):
        
        # define train, grid instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        # grid, train frame
        grid_X_frame = pd.DataFrame(**self.grid_X_info_split.data).astype(self.grid_X_info_split.dtypes)
        train_X_frame = pd.DataFrame(**self.train_X_info_split.data).astype(self.train_X_info_split.dtypes)

        # x, y → list
        if x is not None:
            if type(x) == list:
                x_list = x.copy()
            else:
                x_list = [x]
        
        if y is not None:
            if type(y) == list:
                y_list = y.copy()
            else:
                y_list = [y]

        # grid_apply by model
        for yc in y_list:
            model_instance = class_object_execution(model)
            model_instance.fit(train_X_frame[x_list], train_X_frame[yc])
            grid_X_frame[yc] = model_instance.predict(grid_X_frame[x_list])

        result = grid_X_frame[y_list + x_list]

        # inplace
        if inplace is True:
            self.grid_X = self.DataHandler.transform(grid_X_frame, apply_kind=self.grid_X_info.kind)
        
        # return
        if return_result is True:
            return self.DataHandler.transform(result, apply_kind=self.grid_X_info.kind)
        else:
            return self

    # predict_from_grid(train_X)
    def predict_from_train_X(self, train_X=None, grid_X=None, conditions={}, n_points=None,
                    estimator=None, encoder=None, encoderX=None, encoderY=None):
        if grid_X is None:
            if self.grid_X is None:
                grid_X = self.make_grid(train_X=train_X, conditions=conditions, n_points=n_points, return_result=True, save_result=False)
            else:
                grid_X = self.grid_X
        
        # if estimator is None:
        #     estimator = self.estimator
        # if encoder is None:
        #     encoder = self.encoder
        # if encoderX is None:
        #     encoderX = self.encoderX
        # if encoderY is None:
        #     encoderY = self.encoderY
        grid_instance = self.DataHandler.data_info_split(grid_X)
        grid_frame = pd.DataFrame(**grid_instance.data).astype(grid_instance.dtypes)
        
        if encoder is not None:
            grid_frame_apply = encoder.transform(grid_frame)
        elif encoderX is not None:
            grid_frame_apply = encoderX.transform(grid_frame)
        else:
            grid_frame_apply = grid_frame.copy()

        pred_y_temp = estimator.predict(grid_frame_apply)
        pred_y_temp = self.DataHandler.transform(pred_y_temp, apply_kind='pandas', apply_index=grid_frame.index, apply_columns=['pred_y'])

        if encoderY is None and encoder is not None:
            encoderY = copy.deepcopy(encoder)
            for xc in self.train_X_info_split.data['columns']:
                del encoderY.encoder[xc]
            encoderY = list(encoderY.encoder.values())[0]

        if encoderY is not None:
            pred_y = self.DataHandler.transform(encoderY.inverse_transform(pred_y_temp), apply_kind=grid_instance.kind)
        else:
            pred_y = self.DataHandler.transform(pred_y_temp, apply_kind=grid_instance.kind)
        return {'grid_frame': grid_frame, 'pred_y':pred_y}

    # predict ***
    def predict(self, grid_X=None, train_X=None, conditions={}, n_points=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        unpredictable=True,
        return_all=False):

        if train_X is None:
            train_X = self.train_X
        
        # grid_X auto_filling        
        if grid_X.shape[1] < train_X.shape[1] :
            train_mean = self.make_grid(train_X, save_result=False, return_result=True)
            grid_X_temp = pd.DataFrame()
            for i in train_mean:
                if i in grid_X.columns:
                    grid_X_temp[i] = grid_X[i]
                else:
                    grid_X_temp[i] = [train_mean[i].values[0]] * grid_X.shape[0]
            grid_X = grid_X_temp.copy()
        
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points)

        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions=conditions, n_points=n_points,
            estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
        grid_frame = result['grid_frame']

        pred_y = pd.Series(result['pred_y'], name=y_name, index=result['pred_y'].index)
    
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)
        if len(predictable_frame) > 0:
            unpredict = pred_y[predictable_frame['unpred']]

        if unpredictable is True and len(unpredict) > 0:
            print(f"{list(unpredict.index)} index data is unpredictable date (from estimator).")
        if return_all is False:
            return self.DataHandler.transform(pred_y, apply_ndim=1, apply_kind=self.train_X_info_split.kind)
        elif return_all is True:
            if predictable_frame:
                return pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
            else:
                return pd.concat([grid_frame, pred_y], axis=1)

    # plot one element
    def plot_element(self, x=None, x2=None, train_X=None, grid_X=None, conditions={}, estimator=None, 
        encoder=None, encoderX=None, encoderY=None, y_name=None,
        n_points=None, xlim=None, ylim=None,
        figsize=None, title=None, contour=True, display_unpredictable=True, text_points = 7,
        decimals = None,
        return_plot=True):

        if train_X is None:
            train_X = self.train_X

        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)


        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
        else:
            self.estimator = estimator
        if encoder is not None:
            self.encoder = encoder
        if encoderX is not None:
            self.encoderX = encoderX
        if encoderY is not None:
            self.encoderY = encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name
        
        if len(conditions) == 0:
            conditions = self.conditions
        
        if x2 is None:
            apply_condition = {x:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
        else:
            apply_condition = {x:'min~max', x2:'min~max'}
            apply_condition.update(dict(filter(lambda el: el[0]==x, conditions.items())))
            apply_condition.update(dict(filter(lambda el: el[0]==x2, conditions.items())))

        # data_set
        pred_result = self.predict_from_train_X(train_X=train_X, conditions=apply_condition, n_points=n_points,
                    estimator=self.estimator, encoder=self.encoder, encoderX=self.encoderX, encoderY=self.encoderY)
        
        grid_frame = pred_result['grid_frame']
        pred_y = pd.Series(pred_result['pred_y'], name=y_name)
        predictable_frame = self.predictable_check(target_data=grid_frame, criteria_data=train_X)

        plot_table_1D = pd.concat([grid_frame, pred_y, predictable_frame], axis=1)
        plot_table_lower_1D = plot_table_1D[plot_table_1D['lower_columns'].apply(lambda k: False if len(k) == 0 else True)]
        plot_table_upper_1D = plot_table_1D[plot_table_1D['upper_columns'].apply(lambda k: False if len(k) == 0 else True)]

        outlier_colors = {True:'orange', False:'steelblue'}

        if return_plot:
            f = plt.figure()
        
        # plot
        if x2 is None:       # 1D
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x} (y_range: {auto_formating(pred_y.max() - pred_y.min())})")
            
            # plot_scatter
            if display_unpredictable:
                for gi, gv in plot_table_1D.groupby('unpred'):
                    plt.scatter(gv[x], gv[y_name], edgecolor='white', alpha=0.3, color=outlier_colors[gi])
            else:
                plt.scatter(plot_table_1D[x], plot_table_1D[y_name], edgecolor='white', alpha=0.3)
            
            # plot_line
            plt.plot(plot_table_1D.groupby(x)[y_name].mean(), alpha=0.7)
            
            text_points_list = [int(i) for i in np.linspace(0, len(plot_table_1D)-1, text_points)]
            for pei, (pti, ptd) in enumerate(plot_table_1D.iterrows()):
                if pei in text_points_list:
                    plt.text(ptd[x], ptd[y_name], auto_formating(ptd[y_name]))
            
            if self.confidential_interval is not None:
                plt.fill_between(plot_table_1D[x], plot_table_1D[y_name]+self.confidential_interval, plot_table_1D[y_name]-self.confidential_interval,
                                 alpha=0.1, facecolor='green')
            
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            if display_unpredictable:
                if len(plot_table_lower_1D) > 0:
                    plt.plot(plot_table_lower_1D.groupby(x)[y_name].mean(), color='orange')
                if len(plot_table_upper_1D) > 0:
                    plt.plot(plot_table_upper_1D.groupby(x)[y_name].mean(), color='orange')
            plt.xlabel(x)
            plt.ylabel(y_name)
            
        else:
            plot_table_2D = plot_table_1D.groupby([x2,x])[y_name].mean().unstack(x).sort_index(ascending=False)
            self.trainX = train_X
            self.apply_condition = apply_condition
            self.confirm = plot_table_2D
            
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{y_name} by {x}~{x2} (y_range: {auto_formating(plot_table_2D.max().max() - plot_table_2D.min().min())})")
            
            vmin = None if ylim is None else ylim[0]
            vmax = None if ylim is None else ylim[1]
            CTMap = plt.contourf(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            # plt.contour(plot_table_2D.columns, plot_table_2D.index, plot_table_2D, cmap='jet', vmin=vmin, vmax=vmax)
            
            if decimals is None:
                decimals = 0 if -np.log10(plot_table_2D.mean().mean())+1 < 0 else int(-np.log10(plot_table_2D.mean().mean())+1)
            plt.clabel(CTMap, inline=True, colors ='grey', fmt=f'%.{decimals}f', fontsize=15)
            CTbar = plt.colorbar(extend = 'both')
            CTbar.set_label(y_name)
            plt.xlabel(x)
            plt.ylabel(x2)
        
        if return_plot:
            plt.close()
            self.plot = f
            return self.plot

        
        # return pd.concat([plot_table_1D, plot_table_lower_1D, plot_table_upper_1D])
        
        # pass

    # influence_summary ***
    def influence_summary(self, train_X=None, conditions={}, n_points=None, grid_X=None,
        estimator=None, encoder=None, encoderX=None, encoderY=None, y_name=None, 
        feature_importances=True, summary_plot=True, summary_table=True, sort=False):
        '''
         . *train_X : training Dataset
         . *estimator : model
         . (if need) encoder : apply both X and y variables.
         . (if need) encoderX : apply only X variables.
         . (if need) encoderY : apply only y variable.
        '''
        if train_X is None:
            if self.train_X is not None:
                train_X = self.train_X
        # define train instance
        self.define_train_grid_data(train_X=train_X, grid_X=False)

        # define train, grid instance
        conditions_dict = {c:'min~max' for c in self.train_X_info_split.data['columns']}
        if conditions:
            conditions_dict.update(conditions)

        # estimator, encoderX, encoderY, y_name
        if estimator is None:
            if  self.estimator is None:
                raise Exception("estimator must be required.")
            else:
                estimator = self.estimator
        if encoder is None:
            encoder = self.encoder
        if encoderX is None:
            encoderX = self.encoderX
        if encoderY is None:
            encoderY = self.encoderY
        if y_name is None:
            if self.y_name is None:
                y_name = 'pred_y' 
            else:
                y_name = self.y_name

        if n_points is None:
            n_points = self.n_points

        # influence analysis
        self.influence_dict_all = {}
        self.influence_dict = {}
        feature_influence = {}

        dtypes = self.train_X_info_split.dtypes

        y_min_total = np.inf
        y_max_total = -np.inf
        for ic in conditions_dict:
            grid_X = self.make_grid(train_X=train_X, conditions={ic: conditions_dict[ic]}, n_points=n_points, return_result=True, save_result=False)
            pred_result = self.predict_from_train_X(train_X=train_X, grid_X=grid_X, conditions={ic: conditions_dict[ic]}, n_points=n_points,
                    estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY)
            
            grid_temp_frame = pred_result['grid_frame']
            grid_temp_frame[y_name] = pred_result['pred_y']
            self.influence_dict_all[ic] = self.DataHandler.transform(grid_temp_frame, apply_kind=self.train_X_info.kind)
            self.influence_dict[ic] = self.DataHandler.transform(grid_temp_frame[[ic]+[y_name]], apply_kind=self.train_X_info.kind)

            # summary_plot
            y_argmin_idx = grid_temp_frame[y_name].argmin()
            y_argmax_idx = grid_temp_frame[y_name].argmax()
            x_argmin = grid_temp_frame[ic].iloc[y_argmin_idx]
            x_argmax = grid_temp_frame[ic].iloc[y_argmax_idx]
            x_min = grid_temp_frame[ic].min()
            x_max = grid_temp_frame[ic].max()
            y_min = grid_temp_frame[y_name].min()
            y_max = grid_temp_frame[y_name].max()
            
            if 'int' in str(train_X[ic].dtype).lower() or 'float' in str(train_X[ic].dtype).lower():
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': x_argmax - x_argmin,
                        'delta_x': x_max - x_min, 'max_slope': (y_max - y_min) / (x_argmax - x_argmin)}
            else:
                feature_influence[ic] = {'delta_y': y_max - y_min, 'delta_x_ymax': f"{x_argmin} ~ {x_argmax}",
                        'delta_x': np.nan, 'max_slope': np.nan}
            feature_influence[ic] = {k: auto_formating(v, return_type='str') for k, v in feature_influence[ic].items()}
            feature_influence[ic]['y_range'] = auto_formating(y_min, return_type='str') + ' ~ ' + auto_formating(y_max, return_type='str')
            feature_influence[ic]['x_ymax_range'] = auto_formating(x_argmin, return_type='str') + ' ~ ' + auto_formating(x_argmax, return_type='str')
            feature_influence[ic]['x_range'] = auto_formating(x_min, return_type='str') + ' ~ ' + auto_formating(x_max, return_type='str')

            y_min_total = min(y_min_total, y_min)
            y_max_total = max(y_max_total, y_max)
             
        # feature_plot
        self.feature_plot = {}
        for ic in conditions_dict:
            f = plt.figure()
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
            plt.close()
            self.feature_plot[ic] = f
        
        # feature_influence (table)
        self.summary_table = pd.DataFrame(feature_influence).T
        self.summary_table['plot'] = pd.Series(self.feature_plot)
        
        if feature_importances:
            print(f'==== < Feature Importances Plot > ====')
            print(f' → self.feature_importances_plot')

        self.feature_importances_plot = plt.figure(figsize=(5, self.train_X.shape[1]*0.13+2) )
        plt.barh(self.summary_table.index[::-1], self.summary_table['delta_y'].apply(lambda x: x.replace(',','')).astype('float')[::-1])
        
        if feature_importances:
            plt.show()
        else:
            plt.close()
        

        # summary_plot
        if summary_plot:
            print(f'==== < Feature Influence Summary Plot > ====')
            print(f' → self.summary_plot')
        
        ncols = len(conditions_dict)
        fig_ncols = 4 if ncols > 4 else ncols
        fig_nrows = ((ncols // 4)+1) if ncols > 4 else 0.4

        fig = plt.figure(figsize=(fig_ncols * 4, fig_nrows * 4))
        fig.subplots_adjust(hspace=0.5)   # 위아래, 상하좌우 간격

        for idx, ic in enumerate(conditions_dict, 1):
            plt.subplot(int(fig_nrows)+1, fig_ncols, idx)
            self.plot_element(x=ic, train_X=train_X, grid_X=self.influence_dict_all[ic], conditions={ic:'min~max'},
                estimator=estimator, encoder=encoder, encoderX=encoderX, encoderY=encoderY, y_name=y_name, return_plot=False)
            plt.ylim(y_min_total*0.95, y_max_total*1.05)
        if summary_plot:
            plt.show()
        else:
            plt.close()
        self.summary_plot = fig

        if sort:
            self.summary_table = self.summary_table.sort_values('delta_y', ascending=False)

        print(f'==== < Feature Influence Summary Table > ====')
        print(f' → self.summary_table')
        if summary_table:
            print_DataFrame(self.summary_table)



import time
from IPython.display import clear_output

class EarlyStopping():
    """
    【 Required Library 】numpy, pandas, matplotlib.pyplot, time, from IPython.display import clear_output
     < Initialize(patience=4, optimize='minimize') >
      . patience: 1,2,3,4 ...
      . optimize: minimize / maximize 
     
     < early_stop(score, save=None, label=None, reference_score=None, reference_save=None, reference_label=None, verbose=0, sleep=0.05, save_all=False) >
      (input)
       . score: metrics_score
       . save: anything that would like to save at optimal point
       . label: plot label
       
       . reference_score: reference metrics score
       . reference_save: reference_save value
       . reference_label: plot reference_label
       
       . verbose: 0, 1, 'plot', 'all'
       . sleep: when plotting, sleeping time(seconds).
       . save_all:
     
    """
    def __init__(self, patience=4, optimize='miminize'):
        self.patience = np.inf if patience is None else patience
        self.optimize = optimize
        
        self.metrics = []       # (epoch, event, score, save, r_score, r_save)
        self.metrics_frame = pd.DataFrame()
        self.patience_scores = []
        self.optimum = (0, np.inf if 'min' in optimize else -np.inf, '', None, None)    # (epoch, score, save, r_score, r_save)
    
    def reset_patience_scores(self):
        self.patience_scores = []
    
    def early_stop(self, score, save=None, label=None,
                   reference_score=None, reference_save=None, reference_label=None,
                   verbose=0, sleep=0, save_all=False):
        
        result = 'none'
        epoch = len(self.metrics)+1
        label_score = 'score' if label is None else label
        label_r_score = 'r_score' if reference_label is None else reference_label
        
        if 'min' in self.optimize:
            if score < self.optimum[1]:     # optimum
                self.patience_scores = []
                result = 'optimum'
            else:
                self.patience_scores.append(score)
                if len(self.patience_scores) > self.patience:
                    result = 'break'
                else:
                    result = 'patience'
        elif 'max' in self.optimize:
            if score > self.optimum[1]:     # optimum
                self.patience_scores = []
                result = 'optimum'
            else:
                self.patience_scores.append(score)
                if len(self.patience_scores) > self.patience:
                    result = 'break'
                else:
                    result = 'patience'
        
        # state save
        state = (epoch, result, score, save, reference_score, reference_save) if (save_all is True or result == 'optimum') else (epoch, result, score, '', reference_score, '')
        self.metrics.append(state)

        # update state metrics
        if result == 'optimum':
            if  self.optimum[0] > 0:
                prev_optim_index = self.metrics.index( list(filter(lambda x: x[0]==self.optimum[0], self.metrics))[0] )
                if save_all is True:
                    self.metrics[prev_optim_index] = tuple( ('none' if ei==1 else element) for ei, element in enumerate(self.metrics[prev_optim_index]) )
                else:
                    self.metrics[prev_optim_index] = tuple( ('none' if ei==1 else ('' if ei in [3,5] else element) ) for ei, element in enumerate(self.metrics[prev_optim_index]) )
            self.optimum = (epoch, score, save, reference_score, reference_save)
        
        # metrics_frame = pd.concat([self.metrics_frame, pd.Series(state, index=['epoch', 'event', label_score, 'save', 'r_score', 'r_save'], name=len(self.metrics_frame)).to_frame().T], axis=0)
        metrics_frame = pd.DataFrame(self.metrics, columns=['epoch', 'event', label_score, 'save', label_r_score, 'r_save'])
        metrics_frame['event'] = pd.Categorical(metrics_frame['event'], categories=['none', 'patience', 'break', 'optimum'], ordered=True)
        metrics_frame[label_score] = metrics_frame[label_score].astype('float')
        metrics_frame[label_r_score] = metrics_frame[label_r_score].astype('float')
        
        # plot        
        if verbose == 'plot' or verbose=='all':
            clear_output(wait=True)
        self.plot = plt.figure()
        
        # reference_score
        if reference_score is not None:
            plt.plot(metrics_frame['epoch'], metrics_frame[label_r_score], 'o-', alpha=0.5, color='orange', label='reference' if reference_label is None else reference_label)
            
        plt.plot(metrics_frame['epoch'], metrics_frame[label_score], alpha=0.5, color='steelblue', label='estimate' if label is None else label)
        plt.legend(loc='upper right')
        
        metrics_colors = ['steelblue', 'gold', 'red', 'green']
        for me, (mgi, mgv) in enumerate(metrics_frame.groupby('event')):
            plt.scatter(mgv['epoch'], mgv[label_score], color=metrics_colors[me])            
        for mi, mg in metrics_frame[metrics_frame['event'] != ''].iterrows():
            event_name = 'p' if mg['event'] == 'patience' else ('★' if mg['event']=='optimum' else ('break' if mg['event'] == 'break' else ''))
            plt.text(mg['epoch'], mg[label_score], event_name)
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.yscale('symlog')
        if verbose == 'plot' or verbose=='all':
            plt.show()
            time.sleep(sleep)
            plt.close()
        
        # print state
        if (type(verbose)==int and verbose > 1) or verbose=='all':
            if (verbose in ['plot', 'all']) and result != 'optimum':
                print(f"(Optimum) epoch: {self.optimum[0]}, {label_score}: {str(self.optimum[1])[:6]}, {label_r_score}: {str(self.optimum[3])[:6]}")
            
            if reference_score is not None:
                print(f"epoch: {len(self.metrics)}, {label_score}: {str(score)[:6]}, {label_r_score}: {str(reference_score)[:6]} {f'**{result}' if result != 'none' else ''}")
            else:
                print(f"epoch: {len(self.metrics)}, {label_score}: {str(score)[:6]} {f'**{result}' if result != 'none' else ''}")
        elif verbose == 1:
            if result != 'break':
                print(epoch, end=' ')
            else:
                print(epoch, end=' *break\n')
                print(f"(Optimum) epoch: {self.optimum[0]}, {label_score}: {str(self.optimum[1])[:6]}, {label_r_score}: {str(self.optimum[3])[:6]}") 
        
        self.metrics_frame = metrics_frame.copy()
        return result



############################################################################################################################################################################
####### DS_Plot ############################################################################################################################################################
############################################################################################################################################################################


from sklearn.linear_model import LinearRegression

def model_plot(X, y, model, xcols=None, model_evaluate=None, fitted_data=50,
            title=None, figsize=None, x_name=None, y_name=None, c=None, vmin=None, vmax=None,
            return_plot=True):
    if return_plot:
        fig = plt.figure(figsize=figsize)
    
    if model_evaluate is not None:
        me = model_evaluate
    else:
        me = ModelEvaluate(X, y, model, verbose=0)
    
    if title is None:
        title = str(model)
    
    if sum([i.lower() in str(model).lower() for i in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']]):
        plt.title(f'{title}\n{me.linear.formula}\n(r2: {me.metrics.r2_score}, rmse: {me.metrics.rmse}, mape: {me.metrics.mape}) \n pvalues: {me.hypothesis.pvalues}')
    else:
        plt.title(f'{title}\n(r2: {me.metrics.r2_score}, rmse: {me.metrics.rmse}, mape: {me.metrics.mape})')

    if not xcols:
        plot_dim = X.shape[1]
    else:
        plot_dim = len(xcols)

    if plot_dim == 1:
        x_array = np.array(X).ravel()
        y_array = np.array(y).ravel()
        
        if type(fitted_data) == int:
            x_min = x_array.min()
            x_max = x_array.max()
            x_interval = np.linspace(x_min, x_max, fitted_data)
        elif type(fitted_data) == list or type(fitted_data) == np.ndarray:
            x_interval = np.array(fitted_data)
        elif type(fitted_data) == dict:
            x_interval = np.array(list(fitted_data.values())[0])
        y_interval = model.predict(x_interval.reshape(-1,1))

        if c is None:
            plt.scatter(x_array, y_array, edgecolor='white', alpha=0.7)
        else:
            if 'int' in str(c.dtype) or 'float' in str(c.dtype):
                plt.scatter(x_array, y_array, edgecolor='white', alpha=0.7, c=c, vmin=vmin, vmax=vmax)
                plt.colorbar()
            # else:
            #     for gi, gv in 
        plt.plot(x_interval, y_interval, color='red', alpha=0.5)

        if x_name is None:
            try:
                x_name = X.columns[0]
            except:
                pass
        if y_name is None:
            try:
                x_name = y.name
            except:
                pass
        if y_name:
            plt.ylabel(y_name)
    elif plot_dim > 1:
        pass

    if x_name:
        plt.xlabel(x_name)

    if return_plot:
        return fig
        
    


# ------------------------------------
class FittedModel():
    '''
    < Input > 
     . x : Series or 1D-array
     . y : Series or 1D-array
    \n
    < Output >
     . model : instance of linear model
     . summary : summary
     . coef : coefficient parameters
     . formula : formula of linear-regression 
     . pvales : pvalues of each coefficient 
     . fitted_data : list type data to draw a fitted-plot
     . predict : predict function
    '''
    def __init__(self, X, y, model='LinearRegression', model_type=None, fitted_point=50, print_summary=False):
        
        self.X_shape = X.shape
        self.X_dim = 1 if (np.array(X).ndim == 1 or (np.array(X).ndim == 2 and self.X_shape[1] == 1) ) else 2

        y_frame = self.series_to_frame(y)
        X_frame = self.series_to_frame(X)
        Xy_frame = pd.concat([y_frame, X_frame], axis=1).dropna()
        
        # DataSet
        self.X = Xy_frame[[Xy_frame.columns[1]]]
        self.y = Xy_frame[Xy_frame.columns[0]]
        
        # ColumnName
        try:
            self.y_name = self.y.name
        except:
            self.y_name = 'Y'
        if self.X_dim == 1:
            try:
                self.x_name = self.X.columns[0]
            except:
                self.x_name = 'X'
        elif self.X_dim == 2:
            self.x_name = list(self.X.columns)
        
        if type(model) == str:
            exec(f'self.model = {model}()')
        else:
            self.model = model
        
        # Model
        self.model_name = str(model)
        self.model.fit(self.X, self.y)
        self.model_evaluate = ModelEvaluate(self.X, self.y, self.model, model_type=model_type, verbose=0)
        self.sum_square = self.model_evaluate.sum_square
        self.metrics = self.model_evaluate.metrics
        try:
            self.hypothesis = self.model_evaluate.hypothesis
            self.linear = self.model_evaluate.linear
        except:
            pass
        
        # FittedData
        self.fitted_point = fitted_point

        if self.X_dim == 1:
            x_min = np.array(self.X).min()
            x_max = np.array(self.X).max()
            x_interval = np.linspace(x_min, x_max, fitted_point)

        y_interval = self.model.predict(x_interval.reshape(-1,1))
        
        if self.X_dim == 1:
            self.fitted_data = np.concatenate([x_interval.reshape(-1,1), y_interval.reshape(-1,1)], axis=1).T

        if print_summary:
            print(' .', self.sum_square) 
            print(' .', self.metrics) 
            try:
                print(' .', self.hypothesis) 
                print(' .', self.linear) 
            except:
                pass      

    def series_to_frame(self, X):
        X_shape = X.shape
        X_dim = 1 if (np.array(X).ndim == 1 or (np.array(X).ndim == 2 and X_shape[1] == 1) ) else 2

        # DataSet
        if np.array(X).ndim == 1:
            X = pd.Series(X).to_frame()
        else:
            X = pd.DataFrame(X)
            if type(X) == np.ndarray:
                X.columns = ['x'+str(i+1) for i in np.arange(X_shape)]
        return X

    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))

    def auto_decimal(self,x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            return round(x, self.decimal(x, rev=rev))

    def predict(self, X, decimal_revision=True):
        X = self.series_to_frame(X)

        if decimal_revision:
            return self.model.predict(X).apply(lambda x: self.auto_decimal(x))
        else:
            return self.model.predict(X)
    
    def plot(self, X=None, y=None, figsize=None, model_evaluate=None, fitted_data=None,
            title=None, c=None, vmin=None, vmax=None,
            return_plot=True):
        if X is None:
            X = self.X
        else:
            X = self.series_to_frame(X)
        if y is None:
            y = self.y
        if model_evaluate is None:
            model_evaluate = self.model_evaluate
        if fitted_data is None:
            fitted_data = self.fitted_point
        if title is None:
            title = self.model_name

        if return_plot:
            self.fig = plt.figure(figsize=figsize)

        model_plot(X=X, y=y, model=self.model, model_evaluate=model_evaluate, fitted_data=fitted_data,
                title=title, x_name=self.x_name, y_name=self.y_name, c=c, vmin=vmin, vmax=vmax, 
                return_plot=False)

        if return_plot:
            plt.close()
            return self.fig
        else:
            pass
            # plt.show()









from bs4 import BeautifulSoup
# Special Test ###############################################################################################################################

# from bs4 import BeautifulSoup
# 특별시험 HTML Script에서 Coil번호를 추출해주는 함수
def extract_coils(html=None, html_from_clipboard=False, first_filters=['H', 'C']):
    if html is None and html_from_clipboard is True:
        html = pyperclip.paste()
    soup = BeautifulSoup(html)
    soup_result1 = soup.find_all('div', {'class', 'grid-input-render__field'})      # coil_no
    soup_result2 = soup.find_all('div', {'class', 'ag-custom-select-list'})         # LOC: T/B
    soup_result3 = soup.select("div[col-id='matSpcTeTePicStkDt']")                  # Date
    # soup_result3 = soup.find_all('div', {'class', 'ag-cell ag-cell-not-inline-editing ag-cell-with-height ag-cell-value ag-cell-range-right'})    # Date
    
    series_coil_no = pd.Series(list(filter(lambda y: len(y) > 0 and y[0][0] in first_filters, map(lambda x: x.contents, soup_result1)) ), name='coil_no').apply(lambda x: x[0])
    n_list = []
    for e in soup_result1:
        try:
            n_list.append( int(e.getText()) )
        except:
            pass
    series_n = pd.Series(n_list, name='n')                                      # coil_no
    series_loc = pd.Series([e.getText() for e in soup_result2], name='loc')     # LOC: T/B
    series_date = pd.Series([e.getText() for e in soup_result3 if '-' in e.getText() ], name='date')   # Date
    # series_date = pd.Series([e.getText() for e in soup_result3], name='date')   # Date
    
    result = pd.concat([series_coil_no, series_n, series_loc, series_date], axis=1, ignore_index=True)
    result.columns = ['coil_no','n', 'loc', 'date']
    # coils.columns = ['CoilNo']
    # coils = coils.sort_index()
    result.to_clipboard()
    return result

# # Special Specimen ---------------------------------------------------------------------------------
# ecs = extract_coils(html_from_clipboard=True)
# # ecs = pd.read_clipboard()
# ecs['parent_coil_no'] = ecs.coil_no.apply(lambda x: x[:7])
# ecs_group = ecs.groupby(['parent_coil_no'])[['loc', 'date']].agg({'loc':'sum', 'date':'min'})
# ecs_group.to_clipboard()




# Special Test ###############################################################################################################################

# 특별시험 인장실적 정리
def special_test(data, mode='tensile', reverse=True, dir_cd='-', loc={'001':'WS','002':'1W','003':'2W','004':'3W','005':'DS'}):
    '''
    dir_cd = 'C04'
    loc = {'001':'WS','002':'1W','003':'2W' ...}
    '''
    special_name = {'재질인장시험실적YP': 'YP', '인장시험Upper_YP실적치': 'YP_U', '인장시험Low_YP실적치': 'YP_L',
                '인장시험YP02실적치': 'YP_02', '인장시험YP05실적치': 'YP_05', '인장시험YP1실적치': 'YP_1',
                '재질인장시험실적TS': 'TS', '재질인장시험실적EL': 'EL', '인장시험RA실적치': 'TS_RA',
                '인장시험YR실적치': 'YR', '인장시험YP_EL실적치': 'YP_EL', '인장시험영율실적치': 'YM',
                '인장시험N가공경화지수': 'TN', '인장시험U_EL실적치': 'U_EL',
                '재질구멍확장성시험실적평균구멍확장률': 'HER_평균', '재질구멍확장성시험실적구멍확장률1': 'HER_1', '재질구멍확장성시험실적구멍확장률2': 'HER_2',
                '재질구멍확장성시험실적구멍확장률3': 'HER_3', '재질구멍확장성시험실적구멍확장률4': 'HER_4', '재질구멍확장성시험실적구멍확장률5': 'HER_5'}

    # df_special = pd.read_clipboard()
    df_special = data.copy()
    df_special_1 = df_special.rename(columns=special_name)

    
    if mode.lower() == 'tensile':
        df_special_1.insert(loc=0, column='시편_SEQ', value=df_special_1['시편번호'].apply(lambda x: str(x[-3:])))
        if loc is not None:
            if type(loc) == list:
                df_special_1['위치'] = df_special_1['시편_SEQ'].apply(lambda x: loc[x-1])
                df_special_1['위치'] = pd.Categorical(df_special_1['위치'], categories=loc, ordered=True)
            if type(loc) == dict:
                df_special_1['위치'] = df_special_1['시편_SEQ'].apply(lambda x: loc[str(x)])
                df_special_1['위치'] = pd.Categorical(df_special_1['위치'], categories=list(set(loc.values())), ordered=True)
                           
            if  type(dir_cd) == str:
                df_special_1['인장_방향호수'] = dir_cd
            elif type(dir_cd) == dict:
                df_special_1['인장_방향호수'] = df_special_1['시편_SEQ'].apply(lambda x: dir_cd[str(x)])
                df_special_1['인장_방향호수'] = pd.Categorical(df_special_1['인장_방향호수'], categories=list(set(dir_cd.values())), ordered=True)
        
    df_special_1.insert(loc=0, column='시험위치L', value=df_special_1['시편번호'].apply(lambda x: x[-4:-3]))
    # df_special_1.insert(loc=0, column='시험위치L', value=df_special_1['시편번호'].apply(lambda x: 'T' if 'T' in x else 'B'))
    
    if reverse:
        df_special_1['시험위치L'] = df_special_1['시험위치L'].apply(lambda x: 'B' if x == 'T' else ('T' if x == 'B' else x))
    df_special_1.insert(loc=0, column='재료번호', value=df_special_1['시편번호'].apply(lambda x: x[:-3]))
    
    df_special_2 = df_special_1.drop(['시편번호','채취\n위치','시험\n항목'],axis=1)
    # df_special_2 = df_special_1.drop(['시편번호','부위','MODE'],axis=1)

    if mode.lower() == 'tensile':
        df_special_3 = df_special_2.set_index(['재료번호','시험위치L','인장_방향호수', '위치'])
    elif mode.lower() == 'her':
        df_special_3 = df_special_2.set_index(['재료번호','시험위치L'])
        return df_special_3
    
    df_special_4 = df_special_3.dropna(axis=1, how='all')
    df_special_4.sort_index
    
    # return df_special_4
    df_result = pd.DataFrame()
    for v in df_special_4.columns:
        df_unstack = df_special_4[v].unstack(['위치', '인장_방향호수'])
        df_unstack.columns  = pd.MultiIndex.from_tuples([tuple([v] + list(c)) for c in df_unstack.columns], names=['Content', 'DIR_CD', 'LOC'])
        df_result = pd.concat([df_result, df_unstack], axis=1)
        # df_result.columns.names = ['Content', 'DIR_CD', 'LOC']
        
    # print_DataFrame(df_result)
    # df_result.to_clipboard()
    df_result = df_result.sort_values(['재료번호','시험위치L'], axis=0, ascending=[True, False])

    return df_result






# Micro Data #########################################################################################
# Micro Plot
def micro_plot(df, line=None, ylim=None, label=None, fill=False, return_plot=True):
    zones = df['ZONE'].drop_duplicates()
    if return_plot:
        fig = plt.figure(figsize=(13,len(zones)*2.5))
    for i, c in enumerate(zones):
        df_t = df[df['ZONE'] == c]
        plt.subplot(len(zones),1,i+1)
        plt.ylabel(c, fontsize=15)
        plt.plot(df_t['LEN_POS'], df_t['VALUE'], label=label)
        if type(line) == dict:
            if c in line:
                if type(line[c]) == list:
                    for cc in line[c]:
                        plt.axhline(cc, color='mediumseagreen', ls='--', alpha=0.5)
                        plt.text(df_t['LEN_POS'].max(), cc, cc, color='red', fontsize=13)
                    if len(line[c]) == 2:
                        if fill is not False:
                            plt.fill_between(df_t['LEN_POS']
                                             , np.linspace(line[c][0], line[c][1], len(df_t['LEN_POS'])) - fill
                                             , np.linspace(line[c][0], line[c][1], len(df_t['LEN_POS'])) + fill
                                             , color='mediumseagreen', alpha=0.15)        
                else:
                    plt.axhline(line[c], color='mediumseagreen', ls='--', alpha=0.5)
                    if fill is not False:
                        plt.fill_between(df_t['LEN_POS'], df_t['LEN_POS']*0+line[c]-fill, df_t['LEN_POS']*0+line[c]+fill, color='mediumseagreen', alpha=0.15)
                    plt.text(df_t['LEN_POS'].max(), line[c], line[c], color='red', fontsize=13)
        plt.text(df_t['LEN_POS'].min(), df_t['VALUE'].mean(), df_t['MTL_NO'].head(1).values[0], fontsize=13, color='blue')
        if ylim is not None:
            if type(ylim) == list:
                plt.ylim(ylim[0], ylim[1])
            if type(ylim) == dict:
                if c in ylim.keys():
                    plt.ylim(ylim[c][0], ylim[c][1])
    if return_plot:
        plt.close()
        return fig
    


# # Micro Plot ---------------------------------------------------------------------------------
# df_sql = pd.read_clipboard()
# # df_sql = DB.result.copy()
# df_sql['ZONE'] = pd.Categorical(df_sql['ZONE'], ['CT','LS','HS','SS','SCS','RCS','OAS','RHS','FCS','GA_Furnace', 'GA_IH', 'GI_ACE','SPM_EL','SPM_RF', 'SPM_RF_ST1','SPM_RF_ST2'], ordered=True)
# df_sql.sort_values(['MTL_NO','ZONE','LEN_POS'], axis=0, inplace=True)
# # df_sql.sort_values('LEN_POS', inplace=True)
# df_sql_index = ', '.join(list(df_sql['MTL_NO'].value_counts().index))
# print(df_sql_index); pyperclip.copy(str(df_sql_index))

# img_to_clipboard( micro_plot(df_sql.query("MTL_NO == 'CQM3899'")
#                              ,line=dict(HS=790, SS=790, SCS=650, RCS=490), fill=10) )













# INQ Review #########################################################################################
def cummax_summary(data, x, group, title=None, annotation=True, rotation=0, return_plot=True):
    result = {}
    data_agg = data.groupby(group)[x].agg('max')
    data_group = data_agg.agg('cummax').to_frame()
    data_group[x + '_Min'] = data_group[x].shift()
    data_group.reset_index(inplace=True)

    data_group_melt = pd.melt(data_group, id_vars=[group], value_vars=[x,x + '_Min'])
    data_group_melt.sort_values([group,'value'], ascending=[True,True], inplace=True)
    data_group_melt.dropna(inplace=True)

    if return_plot:
        fig = plt.figure()
    if title is not None:
        plt.title(title)
    plt.scatter(data[group], data[x], edgecolor='white', alpha=0.3)
    plt.plot(data_group_melt[group], data_group_melt['value'], color='navy')
    plt.xlabel(group)
    plt.ylabel(x)
    if annotation:
        annotation_data = data_group_melt.groupby('value')[group].min().reset_index()
        for r in annotation_data.iterrows():
            plt.text(r[1][group], r[1]['value'], f"{r[1][group]}t×{int(r[1]['value'])}w",rotation=rotation)
    if return_plot:
        plt.close()

    result['data_agg'] = data.groupby(group)[x].agg(['count','min','max'])
    result['data_group'] = data_group
    result['data_group_melt'] = data_group_melt
    if return_plot:
        result['plot'] = fig
        return result



# # 생산가능 주문폭 / 주문두께 #################################################################################
# cs_object = cummax_summary(data=df01,x='주문폭',group='주문두께')

# cs_object.keys()
# cs_object['data_agg'].to_
# cs_object['data_group']
# cs_object['data_group_melt']

# for gi, gv in df01.groupby("소둔공장"):
#     plt.scatter(gv['주문두께'], gv['주문폭'], label=gi, alpha=0.3)
# plt.legend(loc='upper right')
# plt.plot(cs_object['data_group_melt']['주문두께'], cs_object['data_group_melt']['value'])
# plt.xlabel('주문두께')
# plt.ylabel('주문폭')

# cs_object['plot']
# ########################################################################################################################




# -----------------------------------------------------------------------
# 열연목표두께 Plot
def hr_reduction_plot(data=None, figsize=(8,5), title=None, 
                    from_clipboard=False, return_plot=True):
    if from_clipboard:
        data = pd.read_clipboard()
    data_use = data[['두께이상', '두께미만', '폭이상', '폭미만', '열연목표두께']]
    data_use['두께이상'].drop_duplicates().to_clipboard()
    pd.Series(list(set(data_use['폭이상'].tolist()  + data_use['폭미만'].tolist()))).sort_values().to_clipboard()

    data_use['두께'] =  data_use[['두께이상', '두께미만']].mean(1)
    data_use['폭'] =  data_use[['폭이상', '폭미만']].mean(1)

    data_use[['두께','폭']]
    data_use[['두께','폭','열연목표두께']].set_index(['두께','폭']).unstack('두께')

    data_use['두께이상_압하율'] = 1- data_use['두께이상']/data_use['열연목표두께']
    data_use['두께미만_압하율'] = 1- data_use['두께미만']/data_use['열연목표두께']

    max_press_ratio = max(data_use['두께이상_압하율'].max(), data_use['두께미만_압하율'].max())
    min_press_ratio = min(data_use['두께이상_압하율'].min(), data_use['두께미만_압하율'].min())

    # 열연목표두께 Plot
    if return_plot is not False:
        f = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)

    plt.plot(np.nan,np.nan)
    for ri, rd in data_use.iloc[:-1,:].iterrows():
        rmax = 1-rd['두께이상']/rd['열연목표두께']
        rmin = 1-rd['두께미만']/rd['열연목표두께']
        rmean = (rmax+rmin)/2 
        color_grade = 1 - (rmean - min_press_ratio) / (max_press_ratio - min_press_ratio)

        thick = (rd['두께이상'] + rd['두께미만'])/2
        width = (rd['폭이상'] + rd['폭미만'])/2

        plt.fill_between( [rd['두께이상'], rd['두께미만']],  rd['폭미만'], rd['폭이상']
                ,facecolor=(color_grade, color_grade, color_grade)
                    ,edgecolor='gray', linewidth=1
                    ,alpha=0.5)

        # ---------------------------------------------------------------------------------------------
        # from matplotlib.patches import Rectangle
        # import matplotlib.patches as patches
        # plt.gca().add_patch(
        #     
        #         patches.Rectangle((rd['두께이상'], rd['폭이상']), rd['두께미만'] - rd['두께이상'], rd['폭미만'] - rd['폭이상']
        #         , fill=True
        #     #     color='black',
        #     #     ,facecolor='orange'
        #         ,facecolor=(color_grade, color_grade, color_grade)
        #         ,edgecolor='gray', linewidth=1
        #         ,alpha=0.5)
        #         )
        # plt.text(thick, width, round(rmean,2))
        # ---------------------------------------------------------------------------------------------
        plt.text(thick-0.05, width, f"{round(rmean,2)}\n({rd['열연목표두께']}t)")
        # plt.text(rd['두께이상'], width, f"{round(rmean,2)}\n{round(rmin,2)}~{round(rmax,2)}")

    plt.xlabel('주문두께')
    plt.ylabel('주문폭')
    
    if return_plot is not False:
        plt.show()
        if return_plot == 'clipboard':
            print('* plot is move to clipboard.')
            img_to_clipboard(f)














# Steel Compoent Formula ###############################################################################################################################

# Calculate Component Score
def calc_soluted_Nb_proba(C, SolAl, Nb, N, **kwargs):
    return Nb - 7.75*C - 6.65*(N/10000 - SolAl/1.93)

def calc_solute_Nb_temp(C, Nb, N, **kwargs):
    if 'Series' in str(type(C))  and 'Series' in str(type(Nb)) and 'Series' in str(type(N)):
        return -10400/((Nb * (N/10000)**0.65 * C ** 0.24).apply(lambda x: np.nan if np.isclose(x, 0) else np.log10(x)) -4.09) -273.5
    elif 'float' in str(type(C))  and 'float' in str(type(Nb)) and 'float' in str(type(N)) :
        if np.isclose(Nb * N * C, 0):
            return np.nan
        else:
            return -10400/(np.log10((Nb * (N/10000)**0.65 * C ** 0.24)) -4.09) -273.5
    else:
        return np.nan

def calc_theroical_rolling_TS(C, Si, Mn, SolAl, Cu, Nb, Cr, Mo, V, slab_thick=250, product_thick=0.6, **kwargs):
    return ( (24.5 if product_thick >= 8 else 25) + 70*C + 13*Si + 8.7*Mn + 5*Cr + 13*Mo + 7*Cu 
        + 0.086*(slab_thick/product_thick) - 0.066*product_thick 
        + 22*V + 165*Nb 
        + (SolAl.apply(lambda x: 1 if x >=0.015 else 0) if 'Series' in str(type(SolAl)) else 1 if SolAl >= 0.015 else 0)
        - 0.2 + 3*np.log10(product_thick/10)**2 ) * 9.80665


# 용접 Crack관련
# **Fp=2.5*(0.5-(C+0.04Mn-0.14Si+0.1Ni-0.04Cr-0.1Mo-0.24Ti+0.7N))
# **CP= C + 0.04Mn + 0.1Ni + 0.7N - 0.14Si-0.04Cr-0.1Mo-0.24Ti

def calc_RST(C, Si, SolAl, Nb, Ti, V, **kwargs):
    return 887 + 464*C + (6445*Nb - 644*np.sqrt(Nb)) + (732*V - 230*np.sqrt(V)) + 890*Ti + 363*SolAl - 357*Si

def calc_Ar3(C, Mn, Cu, Ni, Cr, Mo, product_thick=0.6, **kwargs):
    return 910 - 310*C - 80*Mn - 20*Cu - 15*Cr - 55*Ni - 80*Mo - 0.35*(product_thick-8)
    # (NEW) AR3 = 910-310XC-80XMn-20Cu-15Cr-55Ni-80Mo+0.35(t-8)

def calc_Ar1(Si, Mn, Ni, Cr, **kwargs):
    return 723 - 10.7*Mn - 16.9*Ni - 29.1*Si + 16.9*Cr

def calc_Ac3(C, Si, Ni, Mo, V, **kwargs):
    return 910 - 203*np.sqrt(C) - 15.2*Ni + 44.7*Si + 104*V + 31.5*Mo
    # (NEW) AC3 = 912-203X√C-30XMn-15.2XNi-11XCr+44.7Si+31.5XMo-20XCu+13.1W+104V+120As+400Ti+400Al+700P

def calc_Ac1(Si, Mn, Ni, Cr, **kwargs):
    return 723 - 10.7*Mn - 16.9*Ni - 29.1*Si + 16.9*Cr

def calc_Bs(C, Mn, Ni, Cr, Mo, **kwargs):
    return 830 - 270*C - 90*Mn -37*Ni - 70*Cr -83*Mo

def calc_Ms(C, Mn, Ni, Cr, Mo, **kwargs):
    return 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr - 7.5*Mo

def calc_CEQ(C=None, Si=None, Mn=None, Cu=None, Nb=None, B=None, Ni=None, Cr=None, Mo=None, Ti=None, V=None, N=None,
    code='G', **kwargs):
    if code == 'A': return C + Mn/6
    elif code == 'B': return C + Mn/10
    elif code == 'C': return C + Mn/6 + Si/24
    elif code == 'D': return C + Mn/6 + Si/24 + Cr/5 + V/14
    elif code == 'E': return C + Mn/6 + Si/24 + Cr/5 + V/14 + Ni/40
    elif code == 'F': return C + Mn/6 + Si/24 + Cr/5 + V/14 + Ni/40 + Mo/4
    elif code == 'G': return C + Mn/6 + (Ni + Cu)/15 + (Cr + Mo + V)/5
    elif code == 'H': return C + Mn/6 + Cr/10 + Mo/50 + V/10 + Ni/20 + Cu/40
    elif code == 'J': return C + Mn/6 + Si/24 + Ni/40 + Mo/4 + V/14
    elif code == 'K': return C + Mn/3
    elif code == 'L': return C + Mn/4
    elif code == 'M': return C + Mn/8
    elif code == 'N': return C + (Mn + Si)/4
    elif code == 'P': return C + Mn/6 + Si/24 + Ni/40 + Cr/5 + Mo/4 + V/14 + Cu/13
    elif code == 'Q': return C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cr)/15
    elif code == 'R': return C + (Mn + Si + Cr + Mo)/6 + (Ni + Cu)/16
    elif code == 'S': return C + Mn/5
    elif code == 'T': return C + Mn/6 + (Cu + Ni)/15 + (Cr + Mo + V)/5 + Si/24
    # elif code == 'U': return C + Mn/4 + Cr/10 ? V/10 + Ni/20 + Cu/20 ? Mo/50
    # elif code == 'V': return C + ※F * {Mn/6 + Si/24 + Cu/15 + Ni/20 + (Cr+Mo+V+Nb)/5 + 5B}
    elif code == 'W': return C + Mn/20 + Si/30 + Ni/60 + Cr/20 + Mo/15 + V/10 + Cu/20 + 5*B
    elif code == 'X': return C + Mn/6 +(Cr+Ti+Mo+Nb+V)/5 + (Ni+Cu)/15 + 15*B
    elif code == 'Y': return C + Si/6 + Mn/4.5 + Cu/15 + Ni/15 + Cr/4 + Mo/2.5 + 1.8*V
    # elif code == 'Z': return Al/N
