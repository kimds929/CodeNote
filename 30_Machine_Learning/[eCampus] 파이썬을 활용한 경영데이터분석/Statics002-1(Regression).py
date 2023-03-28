import pandas as pd
import numpy as np
import scipy as sp
import math

from matplotlib import pyplot as plt
import matplotlib
from plotnine import *
import seaborn as sns
from sklearn import datasets

# https://datascienceschool.net/notebook/ETC/    # 참고사이트 : 데이터 사이언스 스쿨

# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/supermarket_sales.csv')

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

pd.set_option('display.float_format', '{:.3f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None

df.head()

# 【 Data Sampling : Train / Test 】 --------------------------------------------------------------------------------------------------------

# Data Sampling (random suffle) : 학습데이터(Train)와 테스트데이터(Test)
# df.sample(n=None, frac=None, replace=False, weight=None, random_state=None, axis=None)
    # n : 추출할 샘플갯수
    # frac : 전체 개수의 비율 지정하여 추출(n과 중복사용 불가)
    # replace : 샘플링시 중복되게 추출할것인지?
    # weights : 샘플추출시 샘플마다 뽑힐 확률을 조정
    # random_state : 랜덤샘플 추출시 시드를 입력받아 같은시드는 항상 같은 결과를 도출
    # axis: 샘플추출할방향 (0:행, 1:열)
df_train = df.sample(frac=0.7)       # df_train Data
df_test = df.loc[~df.index.isin(df_train.index)]      # Test Data


# from sklearn.model_selection import train_test_split
# train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
    # arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
    # test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)
    # train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
    # random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
    # shuffle : 셔플여부설정 (default = True)
    # stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.

# 『 Example 』
# X = [[0,1],[2,3],[4,5],[6,7],[8,9]]
# Y = [0,1,2,3,4]
# # 데이터(X)만 넣었을 경우
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)
#     # X_train : [[0,1],[6,7],[8,9],[2,3]]
#     # X_test : [[4,5]]
# # 데이터(X)와 레이블(Y)을 넣었을 경우
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=321)
#     # X_train : [[4,5],[0,1],[6,7]]
#     # Y_train : [2,0,3]
#     # X_test : [[2,3],[8,9]]
#     # Y_test : [1,4]

# -------------------------------------------------------------------------------------------------------------------


# 【 회귀분석을 위한 전처리 기능 】 -------------------------------------------------------------------------------------
# https://datascienceschool.net/view-notebook/c642f2d81d134c9f9b6e3b88135a3393/
# -------------------------------------------------------------------------------------------------------------------


# 【 Regression 】 --------------------------------------------------------------------------------------------------------
    # 변수선택
y = 'petal length (cm)'
x1 = 'sepal length (cm)'

    # 그래프 그리기
plt.scatter(df_train[x1], df_train[y])
plt.xlabel(x1)
plt.ylabel(y)


    # 변수간의 상관관계확인
from scipy.stats.stats import pearsonr
import statsmodels.api as sm

pearsonr(df_train[x1], df_train[y])     # scipy 활용
df_train[[y, x1]].corr()        # pandas 활용

    # Regression Model(Train Data)
train_y = df_train[y]
train_x1 = df_train[x1]
# from sklearn.model_selection import train_test_split
# train_x1, test_x1, train_y, test_y = train_test_split(df[x1], df[y], test_size=0.3)

train_coef_x1 = sm.add_constant(train_x1)   # 상수항 결합

reg_model = sm.OLS(train_y, train_coef_x1).fit()    # regression model 생성
# reg_model = sm.OLS.from_formula('petal length (cm)~sepal length (cm)', data = df).fit()    # regression model 생성
# reg_model = sm.OLS(formula = 'petal length (cm)~sepal length (cm)', data = df).fit()    # regression model 생성

# dir(reg_model)          # 사용할 수 있는 command & method List
reg_model.summary()     # regression 결과 보기
reg_model.params        # coefficient parameter

reg_model.bse           # standard Error
reg_model.tvalues       # t-values      (each parameter)
reg_model.pvalues       # p-values      (each parameter)

reg_model.rsquared      # r-square
reg_model.rsquared_adj  # r-square adj

reg_model.df_model      # Degree Of Freedom 
reg_model.aic           # AIC
reg_model.bic           # BIC
reg_model.resid         # 각데이터별 Residual

reg_model.resid         # residuals

reg_model.mse_model     # MSS
reg_model.mse_resid     # MSE

reg_model.ssr           # SSE
reg_model.ess           # SSR
reg_model.centered_tss  # SST


model.df_model          # dof SSR
model.df_resid          # dof SSE


    # 예측구간 (신뢰구간)****
# ?model.get_prediction
prediction = model.get_prediction(df_x_add_const)
# dir(prediction)
# prediction.summary_frame()
# prediction.summary_frame(alpha=0.05)
prediction.summary_frame(alpha=0.01)

prediction = model.get_prediction(np.array([1, 2]))     # 예측할 Data
prediction.summary_frame()



reg_predict = model.predict(train_coef_x1) # regression model을 통한 y값예측

    # Graph
plt.scatter(df_train[x1], df_train[y])
plt.plot(df_train[x1], reg_predict, color='r')
plt.xlabel(x1)
plt.ylabel(y)


    # Regression Validation Test(Test Data)
test_y = df_test[y]
test_x1 = df_test[x1]
test_coef_x1 = sm.add_constant(test_x1)   # 상수항 결합

reg_predict_test = reg_model.predict(test_coef_x1)  # Train Data를 이용해 만든 Model을 활용해 Test Data Y값 예측

    # Graph
plt.scatter(df_test[x1], df_test[y])
plt.plot(df_test[x1], reg_predict_test, color='r')
plt.xlabel(x1)
plt.ylabel(y)


    # 모델 평가
mae = sum( abs(reg_predict_test - test_y) / len(reg_predict_test))      # 평균절대오차
rmse = math.sqrt( sum((reg_predict_test - test_y)**2) / len(reg_predict_test))  # 평균제곱근오차
mape = ((abs(reg_predict_test - test_y) / test_y)*100).mean()   # 평균절대퍼센티지오차

print('mse: ' + str(mae) + ' / rmse: ' + str(rmse) + ' / mape: ' + str(mape))
# -------------------------------------------------------------------------------------------------------------------




# 【 다중회귀분석(Multi-Regression) 】 -------------------------------------------------------------------------------
y = 'petal length (cm)'
x1 = 'sepal length (cm)'
x2 = 'sepal width (cm)'
x3 = 'petal width (cm)'
reg_formula = y + ' ~ ' + x1 + ' + ' + x2 + ' + ' + x3 + ' + 0'

train_y = df_train[y]
train_x = df_train[[x1,x2,x3]]
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(df[x1,x2,x3], df[y], test_size=0.3)
train_coef_x = sm.add_constant(train_x)
train_coef_x = train_x

    # Regression Model(Train Data)
# reg_model = sm.OLS(train_y, train_coef_x).fit()    # regression model 생성
reg_model = sm.OLS.from_formula(reg_formula, data = df_train).fit()    # (회귀식을 이용한 모델 생성) regression model 생성
# dir(reg_model)          # 사용할 수 있는 command & method List
reg_model.summary()     # regression 결과 보기
reg_model.params        # coefficient parameter
reg_model.bse           # standard Error
reg_model.tvalues       # t-values
reg_model.pvalues       # p-values
reg_model.rsquared      # r-square
reg_model.rsquared_adj  # r-square adj
reg_model.df_model      # Degree Of Freedom 
reg_model.aic           # AIC
reg_model.bic           # BIC
reg_model.resid         # 각데이터별 Residual


    # Regression Validation Test(Test Data)
test_y = df_test[y]
test_x = df_test[[x1,x2,x3]]
test_coef_x = sm.add_constant(test_x)   # 상수항 결합

reg_predict_test = reg_model.predict(test_coef_x)  # Train Data를 이용해 만든 Model을 활용해 Test Data Y값 예측

    # 모델 평가
mae = sum( abs(reg_predict_test - test_y) / len(reg_predict_test))      # 평균절대오차
rmse = math.sqrt( sum((reg_predict_test - test_y)**2) / len(reg_predict_test))  # 평균제곱근오차
mape = ((abs(reg_predict_test - test_y) / test_y)*100).mean()   # 평균절대퍼센티지오차
print(f"mse: {mae} / rmse: {rmse} / mape: {mape}")

# -------------------------------------------------------------------------------------------------------------------



# 【 기저함수 모형과 과최적화 】
# https://datascienceschool.net/view-notebook/ff9458c7156c4961b012c60e5fc97301/

# 【 분산분석과 모형성능 】
# https://datascienceschool.net/view-notebook/a60e97ad90164e07ad236095ca74e657/
# TSS=ESS+RSS
# 설명된 제곱합 (ESS, explained sum of squares ) : 모형에서 나온 예측값의 움직임의 범위
# 잔차제곱합(RSS, Residula Sum of Square) : 잔차의 움직임의 범위, 즉 오차의 크기
# 전체제곱합 TSS(total sum of square)
dir(reg_model)
TSS = reg_model.uncentered_tss        # TSS
ESS = reg_model.mse_model             # ESS
RSS = reg_model.ssr                   # RSS
print(f"TSS: {TSS} / ESS: {ESS} / RSS: {RSS}")


# 【 다중공선성 제거 : VIF (Variance Inflation Factor) 】 -------------------------------------------------------------
# https://datascienceschool.net/view-notebook/36176e580d124612a376cf29872cd2f0/
# 각계수중에 p-value가 0.5이상인경우 다중공선성 의심


# -------------------------------------------------------------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

# df_x = sm.add_constant( df[[x1,x2,x3]] )
df_x =  df[[x1,x2,x3]] 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
vif["features"] = df_x.columns
vif
# df[[x1,x2,x3]].values     # DataFrame → [ [...], [...], [...] ]
# df[[x1,x2,x3]].shape      # (Length of Rows, Length Of Columns)


# 【 독립변수 선정 】 ------------------------------------------------------------------------------------------------
# ○ 후진제거법



# ○ 전진선택법



# ○ 단계적 방법



# -------------------------------------------------------------------------------------------------------------------

