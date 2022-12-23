# import pyperclip

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# from plotnine import *      # R: ggplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
import statsmodels.api as sm

df = pd.read_clipboard()
df.shape

df.to_clipboard(index=False)


# Regression ------------------------
    # DataSet준비
reg_df_origin = df
# reg_df_origin = df_final

regColumn_df = list(pd.read_clipboard().columns)        # X값 붙여넣기
# regColumn_df
# reg_df_origin[['냉연코일번호'] + regColumn_df].to_clipboard(index=False)
# reg_df_origin[['냉연코일번호','주문두께'] + regColumn_df].to_clipboard(index=False)

    # Data 전처리
reg_df = reg_df_origin[regColumn_df]
reg_df.info()

reg_df = reg_df[reg_df['초_YP'].isna()==False] 
reg_df['냉연정정_SPM횟수'] = reg_df['냉연정정_SPM횟수'].fillna(0)
reg_df['냉연정정_SPM_EL'] = reg_df['냉연정정_SPM_EL'].fillna(0)
reg_df.info()
len(reg_df[reg_df['냉연정정_SPM_EL'].isna()])
len(reg_df[reg_df['냉연정정_SPM_EL'].isna()])

    # DataSet준비
y = '초_YP'
df_y = reg_df[y]

df_x = reg_df[regColumn_df]
df_x = df_x.drop([y], axis=1)
df_x = df_x.drop([ 'SS'], axis=1)
df_x.columns

df_y.head()
df_y.shape

df_x.head()
df_x.shape


pyperclip.copy(str(df_x.columns.to_list()))
# pyperclip.paste()


# 상관관계
df_cor = df_x.corr() # (default) method='pearson'

fig, ax = plt.subplots( figsize=(7,7) ) # 그림 사이즈 지정
mask = np.zeros_like(df_cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True     # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
sns.heatmap(df_cor, 
            cmap = 'RdYlBu_r', 
            fmt = '.2f',
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           ) 

# 학습, 테스트 데이터
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3)

train_coef_x = sm.add_constant(train_x)
# train_coef_x = train_x

reg_formula =  df_y.name + '~' + ' + '.join(train_coef_x.columns) + '+0'
df_train = pd.concat([train_y.to_frame(name=df_y.name), train_coef_x], axis=1)
df_train.head().T

reg_model = sm.OLS.from_formula(reg_formula, data = df_train).fit()    # regression model 생성

reg_model.summary()     # regression 결과 보기
# reg_model.summary().tables[0]
# reg_model.summary().tables[1]
# reg_model.summary().tables[2]

reg_model.params        # coefficient parameter
reg_model.bse           # standard Error

reg_model.tvalues       # t-values   (each parameter)
reg_model.pvalues       # p-values   (each parameter)
model.fvalue            # f-value    (model)

reg_model.resid         # residuals     (each data)

reg_model.rsquared      # r-square
reg_model.rsquared_adj  # r-square adj
reg_model.df_model      # Degree Of Freedom 
reg_model.aic           # AIC
reg_model.bic           # BIC
reg_model.resid         # 각데이터별 Residual

reg_model.ssr           # SSE
reg_model.ess           # SSR
reg_model.centered_tss  # SST

reg_model.mse_resid     # MSE
reg_model.mse_model     # MSS

model.df_model          # dof SSR
model.df_resid          # dof SSE


    # 예측구간 (신뢰구간)****
# ?reg_model.get_prediction
prediction = reg_model.get_prediction(df_x_add_const)
# dir(prediction)
# prediction.summary_frame()
# prediction.summary_frame(alpha=0.05)
prediction.summary_frame(alpha=0.01)

prediction = reg_model.get_prediction(np.array([1, 2]))     # 예측할 Data
prediction.summary_frame()


pd.DataFrame(reg_model.summary().tables[0]).to_clipboard(index=False)
pd.DataFrame(reg_model.summary().tables[1]).to_clipboard(index=False)
pd.DataFrame(reg_model.summary().tables[2]).to_clipboard(index=False)


    # Regression Validation Test(Test Data)
test_coef_x = sm.add_constant(test_x)   # 상수항 결합
# test_coef_x = test_x
reg_predict_test = reg_model.predict(test_coef_x)  # Train Data를 이용해 만든 Model을 활용해 Test Data Y값 예측
reg_predict_test

    # 모델 평가
TSS = reg_model.uncentered_tss        # TSS
ESS = reg_model.mse_model             # ESS
RSS = reg_model.ssr                   # RSS
reg_validation = pd.DataFrame([{'TSS': TSS, 'ESS': ESS, 'RSS': RSS}])

r2_score(test_y, reg_predict_test)

mae = sum( abs(reg_predict_test - test_y) / len(reg_predict_test))      # 평균절대오차
rmse = math.sqrt( sum((reg_predict_test - test_y)**2) / len(reg_predict_test))  # 평균제곱근오차
mape = ((abs(reg_predict_test - test_y) / test_y)*100).mean()   # 평균절대퍼센티지오차
reg_validation = pd.DataFrame([{'mae' : mae, 'rmse' : rmse, 'mape': mape}])
reg_validation
reg_validation.to_clipboard(index=False)
df_x.describe().T[['mean','std']].to_clipboard()





    # VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# df_x = sm.add_constant( df[[x1,x2,x3]] )
# df_x =  df[[x1,x2,x3]] 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
vif["features"] = df_x.columns
vif
# df[[x1,x2,x3]].values     # DataFrame → [ [...], [...], [...] ]
# df[[x1,x2,x3]].shape      # (Length of Rows, Length Of Columns)





