import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d1 = pd.read_clipboard(sep='\t')
d1['MTL_NO'].value_counts()

d2 = d1.query("MTL_NO == 'HRC160210'")




class CT_Handler:
    """
    【 Required Library 】: numpy, pandas, matplotlib.pyplot
    
    【 methods 】
      . __init__(data..):
      . remove_abnormal_CT : remove outlier
      . fit : split position (head, mid, tail)
      . plot : draw CT chart
        
    【 attribute 】
      . data : overall CT data
      . data_head : CT_head data
      . data_mid : CT_mid data
      . data_tail : CT_tail data
      . ct : CT (mid)
      . ct_std : CT std (mid)
    """
    def __init__(self, data):
        self.data = data
        self.length = None
        self.ct = None
        self.trend_info = False
        self.decide_from_filter = False
        
    def remove_abnormal_CT(self, data=None, normal_region=200, use_function=False, **kwargs):
        data = self.data if data is None else data
        data_25, data_50, data_75 = data['VALUE'].describe()[['25%','50%','75%']]
        data_mean, data_std = data[(data['VALUE'] >= data_25) & (data['VALUE'] <= data_75)]['VALUE'].agg(['mean','std'])
        data_low_3sigma = data_mean - 3* data_std
        data_high_3sigma = data_mean + 3* data_std
        
        data_len_min, data_len_max = data['LEN_POS'].agg(['min','max'])
        data_new = data[data.apply(lambda x: ((x['VALUE'] >= data_low_3sigma) ) 
                                if (x['LEN_POS'] < data_len_min + normal_region) or (x['LEN_POS'] > data_len_max - normal_region) 
                                else True, axis=1)]
        data_new['LEN_POS'] = data_new['LEN_POS'] - data_new['LEN_POS'].min()
        self.length = data_new['LEN_POS'].max()
        
        if use_function is False:
            self.data = data_new
            return self
        elif use_function is True:
            return data_new

    def fit(self, data=None, pos_criteria=200, pattern_factor='trend', use_as_function=False, 
                  filter=None, lamb=1000, decide_from_filter=True, **kwargs):
        '''
        【 required Class 】TrendAnalysis
          filter : None (optional)
           . 'hp_filter' : (lambda=1600)
        '''
        data = self.data if data is None else data
        if self.length is None:
            self.remove_abnormal_CT(data=data, **kwargs)
        
        if filter is not None:
            data_new = data.copy()
            TA_Object = TrendAnalysis(filter=filter, lamb=lamb, **kwargs)
            filtered_data = TA_Object.fit(data['VALUE'])
            filtered_data_dropna = filtered_data[~filtered_data['VALUE'].isna()].drop('VALUE',axis=1)
            data_new = pd.concat([data_new, filtered_data_dropna], axis=1)
            self.trend_info = True
        else:
            data_new = data.copy()
        
        pos_dict = {}
        if type(pos_criteria) == float or type(pos_criteria) == int:
            pos_dict = {k: pos_criteria for k in ['head','tail']}
        elif type(pos_criteria) == list:
            pos_dict = {k: pos_criteria[ei] for ei, k in enumerate(['head','tail'])}
        data_new['POS_GROUP'] = data_new.apply(lambda x: 'head' if x['LEN_POS'] <= pos_dict['head'] 
                else ('tail' if x['LEN_POS'] >= data_new['LEN_POS'].max() - pos_dict['tail'] else 'mid'),axis=1)
        
        if use_as_function is False:
            self.data = data_new
            self.data_head = data_new[data_new['POS_GROUP'] == 'head']
            self.data_mid = data_new[data_new['POS_GROUP'] == 'mid']
            self.data_tail = data_new[data_new['POS_GROUP'] == 'tail']
            
            if filter is not None and decide_from_filter is True:
                self.ct = self.data_mid['trend'].mean()
                self.ct_std = self.data_mid['trend'].std()
                
                self.head_max = self.data_head.iloc[self.data_head['trend'].argmax()]
                self.tail_max = self.data_tail.iloc[self.data_tail['trend'].argmax()]
                
                if 'float' in str(type(pattern_factor)) or 'int' in str(type(pattern_factor)):
                    self.head_pattern = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head['trend'] <= self.ct + pattern_factor*self.ct_std)].iloc[0]
                    self.tail_pattern = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail['trend'] >= self.ct + pattern_factor*self.ct_std)].iloc[0]
                elif pattern_factor == 'trend':
                    self.head_pattern = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head['trend'] <= self.ct + 3*self.ct_std) & (self.data_head['trend_info'] == 'min') ].iloc[0]
                    self.tail_pattern = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10)  & (self.data_tail['trend'] <= self.ct + 3*self.ct_std) & (self.data_tail['trend_info'] == 'min')].iloc[-1]
                
                self.decide_from_filter = decide_from_filter
            else:
                self.ct = self.data_mid['VALUE'].mean()
                self.ct_std = self.data_mid['VALUE'].std()
                
                self.head_max = self.data_head.iloc[self.data_head['VALUE'].argmax()]
                self.tail_max = self.data_tail.iloc[self.data_tail['VALUE'].argmax()]
                
                if filter is not None and ( 'float' in str(type(pattern_factor)) or 'int' in str(type(pattern_factor)) ):
                    self.head_pattern = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head['VALUE'] <= self.ct + pattern_factor*self.ct_std)].iloc[0]
                    self.tail_pattern = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail['VALUE'] >= self.ct + pattern_factor*self.ct_std)].iloc[0]
                elif pattern_factor == 'trend':
                    self.head_pattern = self.data_head[(self.data_head['LEN_POS'] > 10) & (self.data_head['VALUE'] <= self.ct + 3*self.ct_std) & (self.data_head['trend_info'] == 'min') ].iloc[0]
                    self.tail_pattern = self.data_tail[(self.data_tail['LEN_POS'] < self.length-10) & (self.data_tail['VALUE'] <= self.ct + 3*self.ct_std) & (self.data_tail['trend_info'] == 'min')].iloc[-1]
            
            return self
        elif use_as_function is True:
            return data_new

    def plot(self, data=None, figsize=(10,3), ylim=None, color=None, return_plot=True):
        data = self.data if data is None else data
        
        if return_plot is True:
            f= plt.figure(figsize=figsize)
        plt.plot(data['LEN_POS'], data['VALUE'], color=color)   
        
        if ylim is not None:
            plt.ylim(ylim)

        if return_plot is True:
            plt.close()
            return f
    
    def info_plot(self, figsize=(10,3), ylim=None, color=None, 
                  trend_plot=True, trend_color=None, return_plot=True, **kwargs):
        if self.ct is None:
            if self.length is None:
                self.remove_abnormal_CT(**kwargs)
            self.fit(**kwargs)
            
        if return_plot is True:
            f = plt.figure(figsize=(10,3))
        
        plt.plot(self.data['LEN_POS'], self.data['VALUE'], color=color)

        display_point = 'VALUE'
        if self.trend_info is True:
            if trend_plot is True:
                plt.plot(self.data['LEN_POS'], self.data['trend'], color=trend_color, alpha=0.7)
            if self.decide_from_filter is True:
                display_point = 'trend'
        
        # head_max
        plt.scatter(self.head_max['LEN_POS'], self.head_max[display_point], color='purple')
        plt.text(self.head_max['LEN_POS'], self.head_max[display_point], 
                f"head_max: {self.head_max['LEN_POS']:0.0f}m / {self.head_max[display_point]:0.0f}℃")

        # tail_max
        plt.scatter(self.tail_max['LEN_POS'], self.tail_max[display_point], color='purple')
        plt.text(self.tail_max['LEN_POS'], self.tail_max[display_point], 
                f"tail_max: {self.tail_max['LEN_POS']:0.0f}m / {self.tail_max[display_point]:0.0f}℃")

        # head_pattern
        plt.scatter(self.head_pattern['LEN_POS'], self.head_pattern[display_point], color='red')
        plt.text(self.head_pattern['LEN_POS'], self.head_pattern[display_point], 
                f"head_pattern: {self.head_pattern['LEN_POS']:0.0f}m / {self.head_pattern[display_point]:0.0f}℃")

        # tail_pattern
        plt.scatter(self.tail_pattern['LEN_POS'], self.tail_pattern[display_point], color='red')
        plt.text(self.tail_pattern['LEN_POS'], self.tail_pattern[display_point], 
                f"tail_pattern: {self.tail_pattern['LEN_POS']:0.0f}m / {self.tail_pattern[display_point]:0.0f}℃")

        # CT mean/std
        plt.text(self.length/2, self.ct, f"CT: {self.ct:0.0f}℃ (std: {self.ct_std:0.1f})")
        plt.axhline(self.ct, color='mediumseagreen', alpha=0.5, ls='--')
        plt.axhline(self.ct + 3*self.ct_std, color='mediumseagreen', alpha=0.3)
        plt.axhline(self.ct - 3*self.ct_std, color='mediumseagreen', alpha=0.3)

        # CT head/mid/tail
        plt.axvline(self.data_mid.head(1)['LEN_POS'].iloc[0], color='orange',alpha=0.5)
        plt.axvline(self.data_mid.tail(1)['LEN_POS'].iloc[0], color='orange',alpha=0.5)

        if ylim is not None:
            plt.ylim(ylim)

        if return_plot is True:
            plt.close()
            return f

