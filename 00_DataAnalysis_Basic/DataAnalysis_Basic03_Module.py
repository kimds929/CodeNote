import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns

# from DataAnalysis_Module import Cpk, cpk, cpk_line, fun_Decimalpoint, DF_Summary, SummaryPlot distboxplot, sm_LinearRegression
# from DataAnalysis_Module import *

# Capability / Performance Analysis ===========================================================
from DataAnalysis_Module import cpk, Capability

# cpk Function (Cumstomizing Function)
cpk(mean=df['x3'].mean(), std=df['x3'].std(), lsl=5)
cpk(mean=df['x3'].mean(), std=df['x3'].std(), lsl=5, usl=10)


# Cpk Class (Cumstomizing Class)
# 사용법 1
cpk_class = Capability()
cpk_class(df['x3'], lsl=5, usl=10)

# 사용법 2
cpk_class = Capability()
cpk_class.lsl = 5
cpk_class.usl = 10
cpk_class(df['x3'])
cpk_class(df['x3'], display=True)    # Detail Printing

# visualization
cpk_class.plot()


# 추가기능
cpk_class.cpk
cpk_class.lsl_reject_prob
cpk_class.usl_reject_prob
cpk_class.reset()


# 여러 값에 동시에 적용
df['x3'].agg(cpk_class)
df[['x1','x3']].agg(cpk_class)


# 능력 검토
cpk_class.capability_analysis(df['x3'])
cpk_class.capability_analysis(df['x3'], cpk=[0.3, 0.5, 0.7, 1.0])


# grouping Analysis
from DataAnalysis_Module import CapabilityGroup

# test_criteria = pd.read_clipboard()
test_criteria = pd.read_csv('test_criteria.csv')
ct = test_criteria.set_index('x2')
ct

cg = CapabilityGroup()
cg.analysis(data=df, criteria=ct)
cg.capability_table['cpk_plot'][0]


# titanic_criteria = pd.read_clipboard()
titanic_criteria = pd.read_csv('titanic_criteria.csv')
ct = titanic_criteria.set_index(['Sex','Embarked'])
ct

cg_titanic = CapabilityGroup()
cg_titanic.analysis(data=df_titanic, criteria=ct)
cg_titanic.capability_table

cg_titanic.capability_table['cpk_plot'][1]

cg_titanic.capability_dict['count']
cg_titanic.capability_dict['mean']
cg_titanic.capability_dict['cpk']
cg_titanic.capability_dict['observe_reject_prob']
cg_titanic.capability_dict['gaussian_reject_prob']



# Groupby Apply ------------------------------------------------
from DataAnalysis_Module import Cpk
# Cpk (Cumstomizing Function)

cpk_calc = Cpk()
cpk_calc.lsl = 10
cpk_calc.usl = 20
cpk_calc.decimal = 4
cpk_calc.lean = True
cpk_calc

df_group['y'].agg(cpk_calc)
df_group['y'].agg(['mean', 'std', cpk_calc])
df_group['y'].agg(lambda x: cpk_calc(x, lsl=12, usl=18))
df_group['y'].describe()







# Dataset Summary Library ------------------------------------------------------------
# 【 Cumstomizing Module 】
from DataAnalysis_Module import SummaryPlot, DF_Summary

# SummaryPlot
sm_plt = SummaryPlot(df)

sm_plt.summary_plot(on=['x1'])
sm_plt.summary_plot(on=['x1', 'x2'])
sm_plt.summary_plot(on=df.columns)
sm_plt.summary_plot(on=df.columns, dtypes='numeric')
sm_plt.summary_plot(on=df.columns, dtypes='object')


# DF_Summary
sm_df = DF_Summary(df)
sm_df.summary
sm_df.summary.to_clipboard()

sm_df.summary_plot()        # Summary Plot in DF_Summary
sm_df.summary_plot(on=['x3', 'x4'])
sm_df.summary_plot(dtypes='numeric')
sm_df.summary_plot(dtypes='object')






# Graph histogram kde ----------------------------------------------------------
from DataAnalysis_Module import cpk_line

# df['x1'].plot.hist()

plt.hist(df['x1'])
plt.show()

plt.hist(df_wine['Aroma'], edgecolor='white')
plt.hist(df_wine['Aroma'], bins=20, edgecolor='white')
plt.hist(df_wine['Aroma'], bins=20, edgecolor='white', color='orange')


# Gaussian Graph ---------
# cpk_line(df['x1'])
# plt.hist('x1', data=df, edgecolor='grey')
plt.plot('x1', 'cpk', data=cpk_line(df['x1']), color='red')
plt.grid(alpha=0.1)
plt.show()



# Graph distbox kde ----------------------------------------------------------
from DataAnalysis_Module import distboxplot

distboxplot(data=df, on='x1')
distboxplot(data=df, on='x1', group='x2')
distboxplot(data=df, on='x1', group='x2', mean_line=True)




# 【 Linear_Regression  】==========================================================
import statsmodels.api as sm


# Learning
# X_add = sm.add_constant(X)
# LR = sm.OLS(y, X_add).fit()


# LR.summary()
# LR.model.params

# # predict
# LR_pred = LR.predict(X_add)

# LR_pred_tb = LR_pred.to_frame(name='pred')
# LR_pred_tb['true_y'] = y


# # evaluate
# LR.rsquared
# LR.rsquared_adj
# np.sqrt(LR.mse_resid)





# 【 Cumstomizing Module: Regression 】
from DataAnalysis_Module import sm_LinearRegression

# Learning
# LR = sm_LinearRegression()
# LR.fit(X, y)

LR = sm_LinearRegression()
LR.fit(X,y)

LR.summary()
LR.model
LR.model.params



# predict
LR_OLS_pred = LR.predict(X)

OLS_pred_tb = LR_OLS_pred.to_frame(name='pred')
OLS_pred_tb['true_y'] = y


# evaluate
LR.model.rsquared               # R2
LR.model.rsquared_adj           # R2_adj
LR.model.mse_resid              # MSE (statics)
np.sqrt(LR.model.mse_resid)     # RMSE
# np.sqrt(( (y['y'] - LR_OLS_pred)**2).sum() / 2 )  # RMSE
LR.model.ssr / len(y)           # MSE (ML)


# OLS_wine = sm_LinearRegression().fit(df.iloc[:,:-1], df.iloc[:,-1].to_frame())
# OLS_wine.features_plot(df.iloc[:,:-1], df.iloc[:,-1].to_frame())


# visualization
LR.features_plot(X, y)



