import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
import statsmodels.api as sm


# cpk calculate
# class Cpk
class Cpk():
    def __init__(self, lsl=-np.inf, usl=np.inf, decimal=3, lean=False):
        self.lsl = lsl
        self.usl = usl
        self.decimal = decimal
        self.lean = lean

    def lsl(self, lsl=-np.inf):
        self.lsl = lsl
    
    def usl(self, usl=-np.inf):
        self.usl = usl
    
    def decimal(self, decimal=3):
        self.decimal = decimal

    def lean(self, lean=False):
        self.lean = lean

    def calculation(self, x):
        x_agg = x.agg(['mean','std'])
        self.cpk = min(x_agg['mean']-self.lsl, self.usl-x_agg['mean']) / (3*x_agg['std'])
        if self.decimal:
            self.cpk = round(self.cpk, self.decimal)
        
        if self.lean:
            sign = -1 if x_agg['mean'] - self.lsl < self.usl - x_agg['mean'] else 1
            self.cpk = sign * 0.01 if self.cpk < 0 else sign * abs(self.cpk)
        return self.cpk

    def reset(self):
        self.__init__()

    def __call__(self, x, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                arg = k.lower()
                if arg in ['lsl', 'usl', 'decimal', 'lean']:
                    exec('self.' + arg + ' = ' + 'v')

        self.calculation(x)
        return self.cpk

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


# cpk_line in histogram
def cpk_line(X, bins=10, density=False):
    X_describe = X.describe()
    X_lim = X_describe[['min', 'max']]
    X_min = min(X_describe['min'], X_describe['mean'] - 3 * X_describe['std'])
    X_max = max(X_describe['max'], X_describe['mean'] + 3 * X_describe['std'])
    x_100Divide = np.linspace(X_min, X_max, 101)   # x 정의
    y_100Norm = (1 / (np.sqrt(2 * np.pi)*X_describe['std'])) * np.exp(-1* (x_100Divide - X_describe['mean'])** 2 / (2* (X_describe['std']**2)) )
    if not density:
        y_rev = len(X)/(bins) * (X_describe['max'] -X_describe['min'])
        y_100Norm *= y_rev
    return pd.DataFrame([x_100Divide,y_100Norm], index=[X.name, 'cpk']).T


# DecimalPoint
def fun_Decimalpoint(value):
    if value == 0:
        return 3
    point_log10 = np.floor(np.log10(abs(value)))
    point = int((point_log10 - 3)* -1) if point_log10 >= 0 else int((point_log10 - 2)* -1)
    return point



# Dist_Box Plot Graph Function
def distboxplot(data, on, group=False, figsize=[5,5], title=False, mean_line=False, xscale='linear', 
    hist=True, kde=True, fit=None, hist_kws=None, kde_kws=None, fit_kws=None):
    # group = change_target
    # on = 'YP'
    # title = 'abc'
    normal_data = data.copy()
    # box_colors = ['steelblue','orange']
    box_colors = sns.color_palette()

    figs, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
    # distplot
    if title:
        figs.suptitle(title)
    if group:
        # group_mean
        group_mean = normal_data.groupby(group)[on].mean()
        len_group_mean = len(group_mean)
        group_mean.sort_index(ascending=True, inplace=True)

        # distplot
        for i, (gi, gv) in enumerate(normal_data.groupby(group)):
            try:
                sns.distplot(gv[on], label=gi, ax=axes[0],
                    hist=hist, kde=kde, fit=fit, hist_kws=hist_kws, kde_kws=kde_kws, fit_kws=fit_kws)
                if mean_line:
                    axes[0].axvline(x=group_mean[gi], c=box_colors[i], alpha=1, linestyle='--')
            except:
                pass
        axes[0].legend()
        axes[0].set_xscale(xscale)
        
        # boxplot
        boxes = sns.boxplot(x=on, y=group, data=normal_data, orient='h', color='white', linewidth=1, ax=axes[1])
        axes[1].set_xscale(xscale)

        # mean_point
        axes[1].scatter(x=group_mean, y=list(range(0,len_group_mean)), 
                        color=box_colors[:len_group_mean], edgecolors='white', s=70)
        
    else:
        # group_mean
        group_mean = normal_data[on].mean()

        # distplot
        sns.distplot(normal_data[on], ax=axes[0],
            hist=hist, kde=kde, fit=fit, hist_kws=hist_kws, kde_kws=kde_kws, fit_kws=fit_kws)
        if mean_line:
            axes[0].axvline(x=group_mean, c=box_colors[0], alpha=1, linestyle='--')
        # boxplot
        axes[0].set_xscale(xscale)
        boxes = sns.boxplot(data=normal_data[on], orient='h', color='white', linewidth=1, ax=axes[1])
        
        # mean_points
        plt.scatter(x=group_mean, y=[0], color=box_colors[0], edgecolors='white', s=70)
        axes[1].set_xscale(xscale)

    # Box-plot option
    for bi, box in enumerate(boxes.artists):
        box.set_edgecolor(box_colors[bi])
        for bj in range(6*bi,6*(bi+1)):    # iterate over whiskers and median lines
            boxes.lines[bj].set_color(box_colors[bi])
    plt.grid(alpha=0.1)
    
    return figs



# OLS Class
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
class sm_OLS:
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    def __init__(self, intercept=True):
        self.intercept = intercept
        
    def fit(self, X, y):
        LR_X = X.copy()
        if type(LR_X) == pd.Series:
            LR_X = LR_X.to_frame()
        if self.intercept:
            LR_X['const'] = 1
        self.model = sm.OLS(y, LR_X).fit()
        return self

    def predict(self, X):
        LR_X = X.copy()
        if type(LR_X) == pd.Series:
            LR_X = LR_X.to_frame()
        if self.intercept:
            LR_X['const'] = 1
        return self.model.predict(LR_X)
    
    def summary(self):
        return self.model.summary()

    def predict_features(self, X, y, n_points=30):
        LR_X = X.copy()
        if type(LR_X) == pd.Series:
            LR_X = LR_X.to_frame()
        self.predict_features_df = {}
        X_agg = LR_X.agg(['mean','std','min','max'])
        influence_init = pd.concat([X_agg.loc[['mean'],:]]*n_points, ignore_index=True)
        influence_linspace = pd.DataFrame(np.linspace(X_agg.loc['min',:], X_agg.loc['max',:], n_points), columns=LR_X.columns)
        
        for Xc in LR_X:
            temp_df = influence_init.copy()
            temp_df[Xc] = influence_linspace[Xc]
            temp_df['predict'] = self.predict(temp_df)
            self.predict_features_df[Xc] = temp_df
        return self.predict_features_df

    def features_plot(self, X, y, n_points=30, figsize='auto', alpha=1, line_alpha=0.7):
        LR_X = X.copy()

        if type(LR_X) == pd.Series:
            LR_X = LR_X.to_frame()
        if type(y) == pd.Series:
            y_name = y.name
        elif type(y) == pd.DataFrame:
            y_name = y.columns[0]
        else:
            y_name = 'y'

        # Model Plot
        model_r2 = format(self.model.rsquared, '.3f')
        model_r2_adj = format(self.model.rsquared_adj, '.3f')
        model_rmse = round(np.sqrt(self.model.mse_resid), fun_Decimalpoint(np.sqrt(self.model.mse_resid)))
        formula_list = []
        for i, (mi, mv) in enumerate(zip(self.model.params.index, self.model.params.values)):
            coef_name = '' if mi == 'const' else '*' + mi
            if mv > 0:
                if i > 0:
                    formula_list.append(' + ')
            else:
                formula_list.append(' - ')
            formula_list.append(str(round(abs(mv),fun_Decimalpoint(mv))) + coef_name)
        

        self.plot = {}
        self.plot['model'] = plt.figure(figsize=(6,4))
        plt.title('Model Perfomance\n'+ y_name + ' = ' + ''.join(formula_list) + '\n R2 ' + str(model_r2) + ' / R2_ajd '+ str(model_r2_adj) + ' / RMSE ' + str(model_rmse)) 
        plt.scatter(self.predict(LR_X).sort_values(), y, edgecolors='white', alpha=alpha)
        plt.plot(self.predict(LR_X).sort_values(), self.predict(LR_X).sort_values(), 'r--', alpha=line_alpha)
        plt.ylabel('true ' + y_name)
        plt.xlabel('predict ' + y_name)
        plt.show()

        # Features Plot
        feature_pvalues = round(self.model.pvalues,3)
        self.predict_features_df = self.predict_features(X=LR_X, y=y, n_points=n_points)
        len_predict_features_df = len(self.predict_features_df)
        if figsize=='auto':
            if len_predict_features_df == 1:
                figsize = (4.5, 3)
            elif len_predict_features_df == 2:
                figsize = (9, 3)
            else:
                figsize = (13.5, 3.3 * ((len_predict_features_df-1)//3+1))

        self.plot['features'] = plt.figure(figsize=figsize)
        self.plot['features'].subplots_adjust(hspace=0.5, wspace=0.3)
        for Xi, Xc in enumerate(self.predict_features_df):
            if len_predict_features_df == 2:
                plt.subplot((len_predict_features_df-1)//2+1, 2, Xi+1)
            elif len_predict_features_df > 2:
                plt.subplot((len_predict_features_df-1)//3+1, 3, Xi+1)
            plt.title(y_name + ' - ' + Xc + ' Plot \n pvalue ' + str(feature_pvalues[Xc]))
            plt.scatter(LR_X[Xc], y, edgecolors='white', alpha=alpha)
            plt.plot(self.predict_features_df[Xc][Xc], self.predict_features_df[Xc]['predict'], 'r--', alpha=line_alpha)
            plt.ylabel(y_name)
            plt.xlabel(Xc)
        plt.show()
        return self.plot
