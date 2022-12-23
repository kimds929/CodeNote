import numpy as np
import pandas as pd
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc    # 한글폰트사용
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import seaborn as sns

from DS_OLS import *
from DS_DataFrame import *




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
# Data Transform : plot → group multi plot
def transform_group_array(x, group, data=None):
    data_type_str = str(type(data))
    dtype_error_msg = 'Only below arguments are available \n . data: DataFrame, x: colname in data, group: colname in data \n  or\n . x: list/ndarray/Series, group: list/ndarray/Series'
    list_types = ['str', 'list', 'ndarray', 'Series', 'DataFrame']
    x_type_list = [lt in str(type(x))  for lt in list_types]
    group_type_list = [lt in str(type(group))  for lt in list_types]
    x_type_str = list_types[np.argwhere(x_type_list)[0][0]]
    group_type_str = list_types[np.argwhere(group_type_list)[0][0]]
   
    # dtype error check
    if data is None:
        if not (sum(x_type_list) * sum(group_type_list)):
            raise(dtype_error_msg)
        elif len(x) != len(group):
            raise('Unequal array lenght between x and group.')
    elif 'DataFrame' in data_type_str:
        if not (sum(x_type_list) * sum(group_type_list)):
            raise(dtype_error_msg)
        elif 'int' not in str(data[x].dtype) and data[x].dtype != float:
            raise('x column type must be numeric.')
    else:
        raise(dtype_error_msg)

    if 'DataFrame' not in data_type_str:
        if x_type_str == 'Series':
            if group_type_str != 'DataFrame':
                group = pd.Series(group)
            group.index = x.index
        elif group_type_str == 'Series':
            x = pd.Series(x)
            x.index = group.index
        else:
            x = pd.Series(x)
            group = pd.Series(group)

        data = pd.concat([group, x], axis=1)
        if group_type_str == 'DataFrame':
            data.columns = list(group.columns) + ['x']
            group = list(group.columns)
        else:
            data.columns = ['group', 'x']
            group = 'group'
        x = 'x'
   
    if type(group) == list: # group to list
        group_list = group.copy()
    else:
        group_list = [group].copy()
    group_obj = {}
    group_obj['index'] = []
    group_obj['value'] = []
    group_obj['count'] = []
    group_obj['mean'] = []
    group_obj['std'] = []
    for i, g in data[group_list + [x]].groupby(group_list):
        # group_array = np.array(g[x])
        group_obj['index'].append(i)

        if x_type_str == 'list':
            group_obj['value'].append(g[x].tolist())
        elif x_type_str == 'ndarray':
            group_obj['value'].append(g[x].values)
        else:
            group_obj['value'].append(g[x])

        group_obj['count'].append(len(g[x].dropna()))
        group_obj['mean'].append(g[x].mean())
        group_obj['std'].append(g[x].std())
    group_obj['pvalue'] = anova(*group_obj['value'], equal_var=False).pvalue

    return group_obj


# -----------------------------------------------------------------------------------------------------
# Dist_Box Plot Graph Function
def distbox(data, on, group=None, figsize=[5,5], title='auto',
            mean_line=None, xscale='linear'):
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
        data_gruop = []
        for i, (gi, gv) in enumerate(normal_data.groupby(group)):
            data_gruop.append(gv[on])
            try:
                sns.distplot(gv[on], label=gi, ax=axes[0])
                if mean_line is not None:
                    axes[0].axvline(x=group_mean[gi], c=box_colors[i], alpha=1, linestyle='--')
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
       
        pavlues = sp.stats.f_oneway(*data_gruop).pvalue
        label_name = 'Anova Pvalue: ' + format(pavlues, '.3f')

        summary_dict = normal_data.groupby(group)[on].agg(['count','mean','std']).applymap(lambda x: auto_formating(x)).to_dict('index')
        label_summary = '\n'.join(['* ' + str(k) + ': ' + str(v).replace('{','').replace('}','').replace("'",'').replace(':','') for k,v in summary_dict.items()])
        label_name = label_name + '\n' + label_summary
        plt.xlabel(label_name, fontsize=11)
    else:
        # group_mean
        group_mean = normal_data[on].mean()

        # distplot
        sns.distplot(normal_data[on], ax=axes[0])
        if mean_line:
            axes[0].axvline(x=group_mean, c=box_colors[0], alpha=1, linestyle='--')
        # boxplot
        axes[0].set_xscale(xscale)
        boxes = sns.boxplot(data=normal_data, x=[on], orient='h', color='white', linewidth=1, ax=axes[1])
       
        # mean_points
        plt.scatter(x=group_mean, y=[0], color=box_colors[0], edgecolors='white', s=70)
        axes[1].set_xscale(xscale)

        summary_dict = normal_data[on].agg(['count','mean', 'std']).apply(lambda x: auto_formating(x)).to_dict()
        label_summary = '* All: ' + ', '.join([ k + ' ' + str(v) for k,v in summary_dict.items() ])
        label_name = '\n' + label_summary
        plt.xlabel(label_name, fontsize=11)

    # Box-plot option
    for bi, box in enumerate(boxes.artists):
        box.set_edgecolor(box_colors[bi])
        for bj in range(6*bi,6*(bi+1)):    # iterate over whiskers and median lines
            boxes.lines[bj].set_color(box_colors[bi])
    plt.grid(alpha=0.1)
    figs.subplots_adjust(hspace=0.5)
    # figs.subplots_adjust(bottom=0.2)
    plt.close()
    return figs



# Multi histogram
# x_multi = [np.random.randn(n) for n in [10, 50, 20]]
# plt.hist(x_multi, stacked=True, edgecolor='grey', linewidth=1.2, label=['a','b','c'])
# plt.legend()
def fun_Hist(data, x, figsize=[6,4], bins=10, density=False,
            color=None, xtick=0, alpha=1,
            norm=False,
            group=False, group_type='dodge',
            spec=False,
            spec_display='all',
            line_x=False,
            line_x_display='all',
            title=False,
            legend=False,
            xlim=False,
            ylim=False,
            grid=True):
    '''
    data(DataFrame): Histogram Root Data
    x(str, list): Target Variable
    bins(int): The Number of bar in histogram
    hist_alpha(float 0~1): Histogram bar alpha

    '''
    result_obj = {}
    if type(x) == list: # x to list
        x_list = x.copy()
    else:
        x_list = [x].copy()

    # spec
    for cx in x_list:
        normal_data = data[data[cx].isna()==False].copy()

        # plot figure
        hist_fig = plt.figure(figsize=figsize)

        if not group:
            plt.hist(x=normal_data[cx], bins=bins, density=density, color=color, edgecolor='grey', alpha=alpha, label=cx)
            if legend:
                plt.legend()
        else:
            group_obj = fun_Group_Array(data=normal_data, x=cx, group=group)

            if group_type == 'identity':
                for j, v in enumerate(group_obj['value']):
                    plt.hist(x=v, bins=bins, density=density, edgecolor='grey', label=group_obj['index'][j], color=color, alpha=alpha)
            elif group_type == 'stack':
                plt.hist(x=group_obj['value'], bins=bins, density=density, edgecolor='grey', stacked=True, label=group_obj['index'], color=color, alpha=alpha)
            else:
                plt.hist(x=group_obj['value'], bins=bins, density=density, edgecolor='grey', stacked=False, label=group_obj['index'], color=color, alpha=alpha)

            # if type(group) == list: # group to list
            #     group_list = group.copy()
            # else:
            #     group_list = [group].copy()
            # group_value = []
            # group_idx = []
            # for i, g in normal_data[group_list + [cx]].groupby(group_list):
            #     group_array = np.array(g[cx])
            #     group_idx.append(i)
            #     group_value.append(group_array)

            # if group_type == 'identity':
            #     for j, v in enumerate(group_value):
            #         plt.hist(v, edgecolor='grey', label=group_idx[j], color=color, alpha=alpha)
            # elif group_type == 'stack':
            #     plt.hist(group_value, edgecolor='grey', stacked=True, label=group_idx, color=color, alpha=alpha)
            # else:
            #     plt.hist(group_value, edgecolor='grey', stacked=False, label=group_idx, color=color, alpha=alpha)
            plt.legend()
       
        # norm
        if norm:
            x_Summary = normal_data[cx].describe().T
            norm_left = np.min([x_Summary['min'], x_Summary['mean'] - 3*x_Summary['std']])
            norm_right = np.max([x_Summary['max'], x_Summary['mean'] + 3*x_Summary['std']])
            x_100Divide = np.linspace(norm_left, norm_right, 101)   # x 정의
            y_100Norm = (1 / (np.sqrt(2 * np.pi)*x_Summary['std'])) * np.exp(-1* (x_100Divide - x_Summary['mean'])** 2 / (2* (x_Summary['std']**2)) )
            # y = (1 / np.sqrt(2 * np.pi)) * np.exp(- x ** 2 / 2 )
            # y = stats.norm(0, 1).pdf(x)
            if not density:
                y_rev = len(normal_data[cx])/(bins*1.2) * (x_Summary['max'] -x_Summary['min'])
                y_100Norm *= y_rev
            plt.plot(x_100Divide, y_100Norm, c='tomato', linewidth=1 )

        # Spec-line
        spec_n = 0
        while(spec_n == 0):
            if spec:
                if type(spec) == dict:
                    try:
                        if type(spec[cx]) == list:
                            spec_list = spec[cx].copy()
                        else:
                            spec_list = [spec[cx]].copy()
                    except:
                        break       # while Loop escape
                else:
                    if type(spec) == list:
                        spec_list = spec.copy()
                    else:
                        spec_list = [spec].copy()

                for sl in spec_list:
                    if type(sl) == int or type(sl) == float:
                        s = sl
                        plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                    elif sl in normal_data.columns and normal_data[sl].dtype!=str :
                        if spec_display=='auto':
                            if '하한' in sl or 'min' in sl.lower():
                                s = normal_data[sl].max()
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                            elif '상한' in sl or 'max' in sl.lower():
                                s = normal_data[sl].min()
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                            else:
                                for s in list(normal_data[sl].dropna().drop_duplicates()):
                                    plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                        elif spec_display=='all':
                            for s in list(normal_data[sl].dropna().drop_duplicates()):
                                plt.axvline(x=s, c='r', alpha=0.7, linestyle='--')
                    else:
                        print('spec must be numeric or numeric_columns.')
                        pass
            spec_n = 1


        # Sub-line x-Axis
        line_x_n = 0
        while(line_x_n == 0):
            if line_x:
                if type(line_x) == dict:
                        try:
                            if type(line_x[cx]) == list:
                                line_x_list = line_x[cx].copy()
                            else:
                                line_x_list = [line_x[cx]].copy()
                        except:
                            break       # while Loop escape
                else:
                    if type(line_x) == list:
                        line_x_list = line_x.copy()
                    else:
                        line_x_list = [line_x].copy()

                for slx in line_x_list:
                    if type(slx) == int or type(slx) == float:
                        lx = slx
                        plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                    elif slx in normal_data.columns and normal_data[slx].dtype!=str :
                            if line_x_display=='auto':
                                if '하한' in slx or 'min' in slx.lower():
                                    lx = normal_data[slx].max()
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                                elif '상한' in slx or 'max' in slx.lower():
                                    lx = normal_data[slx].min()
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                                else:
                                    for lx in list(normal_data[slx].dropna().drop_duplicates()):
                                        plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                            elif line_x_display=='all':
                                for lx in list(normal_data[slx].dropna().drop_duplicates()):
                                    plt.axvline(x=lx, c='tomato', alpha=0.5, linestyle='--')
                    else:
                        print('spec must be numeric or numeric_columns.')
                        pass
            line_x_n = 1

                

        #title
        if title:
            plt.title(title)
        # Label
        plt.xlabel(cx)

        # Axis
        plt.xticks(rotation=xtick)

        # x_Limit
        if xlim and type(xlim) == list:
            if type(xlim[0]) == float or type(xlim[0]) == int:
                if type(xlim[1]) == float or type(xlim[1]) == int:
                    plt.xlim(left=xlim[0], right=xlim[1])
                else:
                    plt.xlim(left=xlim[0])
            else:
                if type(xlim[1]) == float or type(xlim[1]) == int:
                    plt.xlim(right=xlim[1])
                else:
                    pass
       
        # y_Limit
        if ylim and type(ylim) == list:
            if type(ylim[0]) == float or type(ylim[0]) == int:
                if type(ylim[1]) == float or type(ylim[1]) == int:
                    plt.ylim(bottom=ylim[0], top=ylim[1])
                else:
                    plt.ylim(bottom=ylim[0])
            else:
                if type(ylim[1]) == float or type(ylim[1]) == int:
                    plt.ylim(top=ylim[1])
                else:
                    pass

        # Grid
        if grid:
            plt.grid(alpha=0.2)
        # plt.show()

        result_obj[cx] = hist_fig   # Result save to object

    return result_obj




# 데이터의 X, Y값에 따른 Scatter Plot, Histogram, Regression Line을 동시에 Display해주는 함수 ------------------------------------------------------
def fun_Group_OLS_Plot(df, y, x, group=[],
        figsize = [5,3],
        PointPlot=True, fitLine=True, histY=True, histX=True,
        specY=[], specX=[], spec_display='auto',
        lineX=[], lineY=[],
        xlim=False, ylim=False,
        alpha=0.7
        ):
    '''
    # 데이터의 X, Y값에 따른 Scatter Plot, Histogram, Regression Line을 동시에 Display해주는 함수

    < Input >
    df (DataFrame) : Input Raw Data
    y (Str) : Y variable
    x (Str) : X variable
    group (list) : Grouping variable List

    PointPlot, fitLine, histY, histX (Boolean) : Plot Display
    SpecY, SpecX (list) : Spec Line Display (it can be variable name)
    Spec_display ('auto', 'all') : if Spec Line has variable name, it display all? or auto?
    lineX, lineY (list) : subline Display
    xlim, ylim (list) : x, y axis Display Limit
    alpha(float number) : graph alpha

    < Output >
    Object['OLS'] (DataFrame) : OLS Regression Result Summary
    Object['plot'] (DataFrame) :  Plot, X_Histogram, Y_Histogram
        Object['plot']['scatter'] : scatter plot
        Object['plot']['histY'] : Y variable Histogram
        Object['plot']['histX'] : X variable Histogram
    '''
    df_plot = df.copy()
    if not group:
        df_plot['total'] = 'total'
        group = ['total']
   
    group = group if type(group) == list else [group]
    df_group = df_plot.groupby(group)    # Group

    if fitLine:
        groupOLS_Base = df_group.count().iloc[:,0].to_frame()   # Group Count
        groupOLS_Base.columns = [['Total'],['count']]
        groupOLS_df = fun_Concat_Group_OLS(base=groupOLS_Base, groupby=df_group, y=y, x=x, const=True)  # OLS Function
   
    if xlim:
        xlim_revision = (df_plot[x].max().item() - df_plot[x].min().item())*0.05
        xlim = [df_plot[x].min().item() - xlim_revision, df_plot[x].max().item() + xlim_revision]

    if ylim:
        ylim_revision = (df_plot[y].max().item() - df_plot[y].min().item())*0.05
        ylim = [df_plot[y].min().item() - ylim_revision, df_plot[y].max().item() + ylim_revision]


    result_Obj={}
    df_part_Object={}
    result_Plot = pd.DataFrame()

    for gi, gv in df_group:
        result_Plot_part = {}
        print(gi)
        print(len(gv))
        df_part_Object[gi] = gv

        spec_listX =[]
        spec_listY =[]
        line_x_list = []
        line_y_list = []
        if specY:
            if type(specY)==dict:
                try:
                    spec_listY = specY[gi]
                except:
                    spec_listY = []
            else:
                spec_listY = specY

        if specX:
            if type(specX)==dict:
                try:
                    spec_listX = specX[gi]
                except:
                    spec_listX = []
        else:
            spec_listX = specX

        if PointPlot:     # 1D OLS
            groupPlot = plt.figure(figsize=figsize)
            plt.scatter(x=x, y=y, data=gv, alpha=alpha, edgecolors='black', linewidth=0.5)

            if fitLine:
                if groupOLS_df.loc[gi].swaplevel(i=0,j=1)['nTrain'].values != '':
                    try:
                        df_part = groupOLS_df.loc[gi]
                        df_part = df_part.reset_index().drop('level_0', axis=1)
                        df_part = df_part.set_index('level_1').T
                        pred_y = df_part_Object[gi][x]*df_part['coef_' + x].values + df_part['coef_const'].values
                        plt.plot(gv[x], pred_y, 'r')
                        result_Obj['OLS'] = groupOLS_df
                        # pass
                    except:
                        pass

            if specY:
                for s in spec_listY:
                    plt.axhline(y=s, c='r', alpha =0.7, linestyle='--')
            if specX:
                for s in spec_listX:
                    plt.axvline(x=s, c='r', alpha =0.7, linestyle='--')
            if ylim:
                plt.ylim(bottom=ylim[0], top=ylim[1])
            if xlim:
                plt.xlim(left=xlim[0], right=xlim[1])

            plt.title(gi)
            plt.ylabel(y)
            plt.xlabel(x)
            plt.grid(alpha=0.2)
            plt.show()
            result_Plot_part['scatter'] = groupPlot

        # line Y
        if type(lineY) == dict:
            try:
                line_y_list = lineY[gi]
            except:
                line_y_list = []
        else:
            line_y_list = lineY

        # line X
        if type(lineX) == dict:
            try:
                line_x_list = lineX[gi]
            except:
                line_x_list = []
        else:
            line_x_list = lineX

        if histY:
            result_Plot_part['histY'] = fun_Hist(data=gv, x=y, figsize=figsize, title=gi, xtick=45, norm=True,
                                    color='mediumseagreen', alpha=0.7,
                                    spec=spec_listY, spec_display=spec_display,
                                    line_x=line_y_list,
                                    xlim=ylim)[y]

        if histX:
            result_Plot_part['histX']  = fun_Hist(data=gv, x=x, figsize=figsize, title=gi, xtick=45, norm=True,
                                    color='skyblue', alpha=0.7,
                                    spec=spec_listX, spec_display=spec_display,
                                    line_x=line_x_list,
                                    xlim=xlim)[x]
        result_Plot = pd.concat([result_Plot, pd.DataFrame([result_Plot_part], index=[gi]).T], axis=1)

    result_Obj['plot'] = result_Plot
    return result_Obj