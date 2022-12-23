import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# pandas Display 옵션
pd.set_option('display.float_format', None)  # 초기화
pd.reset_option('display.float_format')  # 초기화
pd.set_option('display.float_format', '{:.2g}'.format)  # 적당히 알아서




os.getcwd()

boston = pd.read_csv('./Dataset/Boston_house.csv')
boston.shape
boston.info()
boston.head()

boston_data = boston.drop('Target', axis=1)
boston_data

boston_data.describe()

#【 단순선형 회귀분석 】 -------------------------------------------------------------------------------------
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]


# target ~ crim 선형회귀분석
    # 상수항추가
crim1 = sm.add_constant(crim, has_constant = 'add')

    # sm.OLS에 적합시키기
model1 = sm.OLS(target, crim1)
fit_model1 = model1.fit()

fit_model1.summary()
fit_model1.params

    # 예측값 구하기
np.dot(crim1, fit_model1.params)
pred1 = fit_model1.predict(crim1)

    # 예측구간 (신뢰구간)****
# ?fit_model1.get_prediction
prediction = fit_model1.get_prediction(df_x_add_const)
# dir(prediction)
# prediction.summary_frame()
# prediction.summary_frame(alpha=0.05)
prediction.summary_frame(alpha=0.01)

prediction = fit_model1.get_prediction(np.array([1, 2]))
prediction.summary_frame()


# 회귀모형 Graph
import matplotlib.pyplot as plt
plt.yticks(fontname='Arial')
plt.scatter(x=crim, y=target, label='Data')
plt.plot(crim, pred1, color='r', label='result')
plt.legend()
plt.show()

# 실제 Y와 예측Y 비교
plt.scatter(x=target, y=pred1)
plt.xlabel('real_y')
plt.ylabel('pred_y')
plt.show()

# 잔차 residual 시각화
fit_model1.resid.plot(alpha=0.5)
plt.xlabel('residual')
plt.show()


# 잔차합 계산
fit_model1.resid.sum()

fit_model1.summary()
fit_model1.f_pvalue



#【 다중선형 회귀분석 】 -------------------------------------------------------------------------------------
x_data = boston[['CRIM','RM','LSTAT']]
x_data1 = sm.add_constant(x_data, has_constant='add')

multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()

fitted_multi_model.summary()



# 행렬 연산을 활용하여 회귀계수 구하기      (XX')-1X'Y
from numpy import linalg

XXT = np.dot(x_data1.T, x_data1)      # XX'
XXT_1 = linalg.inv(XXT)
XXT_1XT = np.dot(XXT_1,x_data1.T)
coef_b = np.dot(XXT_1XT, target)
coef_b


# 예측 y구하기
pred_multi = fitted_multi_model.predict(x_data1)


# Residual Plot
fitted_multi_model.resid.plot(alpha=0.5)
plt.xlabel('Residual')
plt.show



#【 다중공선성 】 ----------------------------------------------------------------------------------------
x_data2 = boston_data.drop(['CHAS', 'DIS','PTRATIO', 'RAD'],axis=1)
x_data2.shape

x_data2_c = sm.add_constant(x_data2, has_constant='add')
x_data2_c.head()

multi_model2 = sm.OLS(target, x_data2_c)
fitted_multi_model2 = multi_model2.fit()

fitted_multi_model2.summary()


# 세변수만 추가된 모델의 회귀계수
fitted_multi_model.params.to_frame().T

# 9개 변수가 사용된 모델의 회귀계수
fitted_multi_model2.params.to_frame().T
fitted_multi_model2.pvalues.to_frame().T

# 잔차 Plot 확인
import matplotlib.pyplot as plt
fitted_multi_model.resid.plot(alpha=0.5)
fitted_multi_model2.resid.plot(alpha=0.5)
plt.legend()
plt.show()


# 다중공선성
# 상관행렬
x_data2_corr = x_data2.corr()
x_data2_corr

    # 상관행렬 시각화
import seaborn as sns

    # 흑백 기본
cmap = sns.light_palette('darkgray', as_cmap=True)
sns.heatmap(x_data2_corr, annot=True, cmap=cmap)
plt.show()

    # Blue 계열
sns.heatmap(data = x_data2_corr, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

    # 아래쪽만 그리기
mask = np.zeros_like(x_data2_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True     # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
sns.heatmap(x_data2_corr, 
            cmap = 'RdYlBu_r', 
            fmt = '.2f',
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .8},# 컬러바 크기 0.8배 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           ) 
plt.show()

# Pair Plot     # 시간이 오래걸림
sns.pairplot(x_data2)
plt.show()




# VIF를 통한 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
vif = pd.DataFrame()
vif['features'] = x_data2.columns
vif['VIF_Factor'] = [VIF(x_data2.values, i) for i in range(x_data2.shape[1])]
vif


# 다중공선성이 높은 NOX변수를 제거한 후 VIF확인
x_data3 = x_data2.drop('NOX', axis=1)
vif = pd.DataFrame()
vif['features'] = x_data3.columns
vif['VIF_Factor'] = [VIF(x_data3.values, i) for i in range(x_data3.shape[1])]
vif


x_data3_c = sm.add_constant(x_data3, has_constant='add')
multi_model3 = sm.OLS(target, x_data3_c)
fitted_multi_model3 = multi_model3.fit()
fitted_multi_model3.summary()


# 다중공선성이 높은 RM변수를 추가로 제거한 후 VIF확인
x_data4 = x_data3.drop('RM', axis=1)
vif = pd.DataFrame()
vif['features'] = x_data4.columns
vif['VIF_Factor'] = [VIF(x_data4.values, i) for i in range(x_data4.shape[1])]
vif


x_data4_c = sm.add_constant(x_data4, has_constant='add')
multi_model4 = sm.OLS(target, x_data4_c)
fitted_multi_model4 = multi_model4.fit()
fitted_multi_model4.summary()




#【 학습데이터 테스트데이터 분할 】 ----------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X = x_data2
y = target
train_y, test_y, train_x, test_x = train_test_split(y, X, test_size=0.3, random_state=1)
print(f"y : {train_y.shape} / {test_y.shape}  ||  x : {train_x.shape} / {test_x.shape}")

# train데이터 회귀모델 적합
train_x.head()
traintest_model = sm.OLS(train_y, train_x)
fit_traintest_model = traintest_model.fit()

# 검증데이터에 대한 예측값과 true값 비교
plt.plot(np.array(fit_traintest_model.predict(test_x)), label='pred')
plt.plot(np.array(test_y), label='true_y')
plt.legend()
plt.show()

plt.scatter(test_y, fit_traintest_model.predict(test_x))
plt.plot(fit_traintest_model.predict(test_x), fit_traintest_model.predict(test_x), c='r')
plt.show()




#【 MSE를 통한 검증데이터 성능비교 】 ----------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true=test_y['Target'], y_pred=fit_traintest_model.predict(test_x))







#【 변수선택법 실습 】 -------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt



corolla = pd.read_csv('./Dataset/ToyotaCorolla.csv')
corolla.shape
corolla.head()

# 데이터의 수와 변수의 수 확인하기
ncar = corolla.shape

# 명목형(문자형)변수 → Dummy변수 생성
corolla['Fuel_Type'].value_counts()

# help(pd.get_dummies)
corolla_ = corolla.drop(['Id', 'Model'], axis=1, inplace=False)
mir_data = pd.get_dummies(data=corolla_, columns=['Fuel_Type'], drop_first=False)
mir_data.info()

dummy_p = np.repeat(0, ncar[0])
dummy_d = np.repeat(0, ncar[0])
dummy_c = np.repeat(0, ncar[0])

p_idx = np.array(corolla['Fuel_Type']=='Petrol')
d_idx = np.array(corolla['Fuel_Type']=='Diesel')
c_idx = np.array(corolla['Fuel_Type']=='CNG')

dummy_p[p_idx] = 1
dummy_d[d_idx] = 1
dummy_c[c_idx] = 1
help(pd.DataFrame)

Fuel = pd.DataFrame({'Fuel_Type_Petrol':dummy_p, 'Fuel_Type_Diesel':dummy_d, 'Fuel_Type_CNG':dummy_c})
# Fuel = pd.DataFrame([dummy_p, dummy_d, dummy_c], index=['Petrol', 'Diesel', 'CNG']).T

corolla_ = corolla.drop(['Id', 'Model', 'Fuel_Type'], axis=1, inplace=False)
mir_data = pd.concat((corolla_, Fuel), axis=1)
mir_data.info()


# bias추가(상수항 추가)
mir_data2 = sm.add_constant(data=mir_data, has_constant='add')
mir_data2.head().T




# 설명변수(x), 타겟변수(y)로 분리 및 학습데이터와 평가데이터 분할

feature_columns = list(mir_data2.columns.difference(['Price']))
feature_columns
# feature_columns = list(mir_data2.columns.drop(['Price']).sort_values())
# feature_columns

X = mir_data2[feature_columns]
y = mir_data2['Price']

# help(train_test_split)
train_y, test_y, train_x, test_x = train_test_split(y, X, test_size=0.3, random_state=0)

# help(sm.OLS)
full_model = sm.OLS(train_y, train_x)
fitted_full_model = full_model.fit()
fitted_full_model.summary()

fitted_full_model.pvalues.to_frame().reset_index().sort_values(0, ascending=False)


# 다중공선성 확인(VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
vif = pd.DataFrame()
vif['features'] = feature_columns
vif['VIF_Factor'] = [VIF(train_x.values, i) for i in range(train_x.shape[1])]
vif.sort_values(['VIF_Factor'], ascending=False)

# 학습데이터의 잔차 확인
res = fitted_full_model.resid
fig = sm.qqplot(res, fit=True, line='45')

# residual pattern 확인
pred_y = fitted_full_model.predict(train_x)
fig = plt.scatter(pred_y, res, s=4)
plt.xlabel('Fitted values')
plt.ylabel('Residual')
plt.show()


# 검증데이터 예측
pred_testy = fitted_full_model.predict(test_x)
plt.plot(np.array(test_y - pred_testy), label='pred_test')
plt.legend()
plt.show()


# MSE성능
mean_squared_error(y_true=test_y, y_pred=pred_testy)


# 변수 선택법 ****************
    # y, X에 따른 OLS.fit()과 AIC, BIC를 호출하는 함수
def processSubset(X, y, feature_set):
    model = sm.OLS(y, X[list(feature_set)])     # Modeling
    fit_model = model.fit()     # model fit
    AIC = fit_model.aic     # AIC of Model
    BIC = fit_model.bic     # AIC of Model
    return {'model': fit_model, 'AIC':AIC, 'BIC':BIC}

processSubset(X=train_x, y=train_y, feature_set=feature_columns[0:5])


import time
import itertools


# 모든Case에 대한 model 성능 검증
for combo in itertools.combinations(['a','b','c','d'], 2):      # 해당 List내에 2개씩 모든 조합의 수를 Iterator 형태로 Return
    print(combo)

    # 예측 y에 대해 x의 k개 조합에 따른 모든 경우의 수를 OLS모델링하고, AIC, BIC의 Best 모델을 도출하는 함수
def getAllCaseBest(X, y, k):
    tic = time.time()   # 시작시간
    results = []     # 결과저장공간
    for combo in itertools.combinations(X.columns.drop(['const']), k):
        combo= (list(combo) + ['const'])

        results.append(processSubset(X, y, feature_set=combo))  #모델링 결과를 저장
    models = pd.DataFrame(results)  # 결과를 DataFrame형태로 변환
    best_aic = models.loc[models['AIC'].argmin()]     # index
    best_bic = models.loc[models['BIC'].argmin()]     # index
    toc = time.time()
    print(f"process: {models.shape[0]}  / model on: {k}  / predictors in : {toc - tic} seconds")
    return {'models': models, 'best_AIC':best_aic, 'best_BIC':best_bic}


allCase = getAllCaseBest(X=train_x, y=train_y, k=2)

allCase['models']
allCase['best_AIC']['model'].summary()
allCase['best_BIC']['model'].summary()

a = allCase['best_AIC'].to_frame().T


    # 예측 y에 대해 x의 변수의 갯수를 1개씩 늘려가며 각 변수 갯수별로 AIC를 기준으로 최적의 모델결과를 보여주는 함수
AICModel = pd.DataFrame(columns=['AIC', 'BIC', 'features', 'model'])
tic = time.time()
for i in range(1,4):
    best_aic = getAllCaseBest(X=train_x, y=train_y, k=i)['best_AIC'].to_frame().T
    best_aic['features'] = str(list(best_aic['model'].iloc[0].params.index))
    best_aic.index = [i]
    AICModel = pd.concat( [AICModel, best_aic], axis=0) 

toc = time.time()
print(f"'Total elapsed time: {toc-tic} seconds")
AICModel

AICModel.loc[3, 'model'].summary()


AICModel.apply(lambda row: print(row[3].mse_total - fitted_full_model.mse_total), axis=1)



# Plot the result (Plot 평가지표)
plt.figure(figsize=(12,8))
plt.rcParams.update({'font.size': 13, 'lines.markersize': 8})

## Mallow Cp
# help(plt.subplot)
plt.subplot(2, 2, 1)
Cp= AICModel.apply(lambda row: (row[3].params.shape[0]+(row[3].mse_total-
                               fitted_full_model.mse_total)*(train_x.shape[0]-
                                row[3].params.shape[0])/fitted_full_model.mse_total
                               ), axis=1)

plt.plot(Cp)
plt.plot(Cp.index[Cp.argmin()], Cp.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('Cp')

# adj-rsquared plot
# adj-rsquared = Explained variation / Total variation
adj_rsquared = AICModel.apply(lambda row: row[3].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(adj_rsquared)
plt.plot(adj_rsquared.index[adj_rsquared.argmax()], adj_rsquared.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# aic
aic = AICModel.apply(lambda row: row[3].aic, axis=1)
plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.index[aic.argmin()], aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

# bic
bic = AICModel.apply(lambda row: row[3].bic, axis=1)
plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.index[bic.argmin()], bic.min(), "or")
plt.xlabel(' # Predictors')
plt.ylabel('BIC')









########전진선택법(step=1)
def forward(X, y, predictors):  
    # 기존에 선택된 변수 + 새로운 가장 영향력있는 변수를 찾기위해  기존변수+모든변수의 조합을 실행, 가장 AIC가 낮은 모델을 선택
    remaining_predictors = [p for p in X.columns.difference(['const']) if p not in predictors]  # 기존변수를 제외한 나머지 변수들 List
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X=X, y= y, feature_set=predictors+[p]+['const'] ))   # 기존선택된 Column + 각Column(1개씩 iter) + constant
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)

    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()] # index
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model['AIC'] )
    return best_model


#### 전진선택법 모델
def forward_model(X,y):
    Fmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X,y=y,predictors=predictors)     # 변수조합별로 OLS를 실행하여 AIC가 가장 낮은 모델만 선정
        if i > 1:
            if Forward_result['AIC'] > Fmodel_before:       # 기존모델(변수추가 전)의 AIC보다 신규모델(변수추가 후)의 AIC가 높을경우 중지
                break
        Fmodels.loc[i] = Forward_result
        predictors = Fmodels.loc[i]["model"].model.exog_names
        Fmodel_before = Fmodels.loc[i]["AIC"]
        predictors = [ k for k in predictors if k != 'const']
        print('-------------------------------------')
    toc = time.time()
    print('--- [finish] --------------------------')
    print("Total elapsed time:", (toc - tic), "seconds.")

    return(Fmodels['model'][len(Fmodels['model'])])

Forward_best_model = forward_model(X=train_x, y=train_y)
Forward_best_model.summary()
Forward_best_model.model.exog_names




######## 후진선택법(step=1)
def backward(X,y,predictors):
    tic = time.time()
    results = []
    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(X=X, y= y, feature_set=list(combo)+['const']))
    models = pd.DataFrame(results)
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in",
          (toc - tic))
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model['AIC'] )
    print(f"minus predictor: {[minusColumn for minusColumn in predictors if minusColumn not in best_model['model'].model.exog_names ]}")
    return best_model


# 후진 소거법 모델
def backward_model(X, y):
    Bmodels = pd.DataFrame(columns=["AIC", "model"], index = range(1,len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X,y,predictors)['AIC']        # 초기 모든 변수를 대상으로 모델링하여 AIC를 도출
    while (len(predictors) > 1):
        Backward_result = backward(X=train_x, y= train_y, predictors = predictors)
        if Backward_result['AIC'] > Bmodel_before:
            break
        Bmodels.loc[len(predictors) - 1] = Backward_result
        predictors = Bmodels.loc[len(predictors) - 1]["model"].model.exog_names
        Bmodel_before = Backward_result['AIC']
        predictors = [ k for k in predictors if k != 'const']
        print('-------------------------------------')

    toc = time.time()
    print('--- [finish] --------------------------')
    print("Total elapsed time:", (toc - tic), "seconds.")
    print(Bmodels)
    return (Bmodels['model'].dropna().iloc[0])

Backward_best_model = backward_model(X=train_x,y=train_y)


# 단계적 방법
def Stepwise_model(X,y):
    Stepmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []
    Smodel_before = processSubset(X,y,predictors+['const'])['AIC']  # 초기 상수항만으로 AIC를 계산
    # 변수 1~10개 : 0~9 -> 1~10
    for i in range(1, len(X.columns.difference(['const'])) + 1):
        Forward_result = forward(X=X, y=y, predictors=predictors) # Forward Algorism
        print('forward')
        Stepmodels.loc[i] = Forward_result
        predictors = Stepmodels.loc[i]["model"].model.exog_names    # Forward를 통해 1개의 변수가 추가된 상태
        predictors = [ k for k in predictors if k != 'const']
        Backward_result = backward(X=X, y=y, predictors=predictors) # Backward Algorism

        if Backward_result['AIC'] < Forward_result['AIC']:      # Backward의 AIC값이 낮으면 Backward결과로 대체
            Stepmodels.loc[i] = Backward_result
            predictors = Stepmodels.loc[i]["model"].model.exog_names
            Smodel_before = Stepmodels.loc[i]["AIC"]
            predictors = [ k for k in predictors if k != 'const']
            print('backward')

        if Stepmodels.loc[i]['AIC'] > Smodel_before:     # Forward, Backward를 통해 결정된 모델의 AIC값이 이전 모델의 AIC값보다 크면 종료
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
        print('-------------------------------------')
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")
    return (Stepmodels['model'][len(Stepmodels['model'])])


Stepwise_best_model = Stepwise_model(X=train_x,y=train_y)

# the number of params
print(Forward_best_model.params.shape, Backward_best_model.params.shape, Stepwise_best_model.params.shape)

# 모델에 의해 예측된/추정된 값 <->  test_y
pred_y_full = fitted_full_model.predict(test_x)
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names])
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])

perf_mat = pd.DataFrame(columns=["ALL", "FORWARD", "BACKWARD", "STEPWISE"],
                        index =['MSE', 'RMSE','MAE', 'MAPE'])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn import metrics

perf_mat.loc['MSE']['ALL'] = metrics.mean_squared_error(test_y,pred_y_full)
perf_mat.loc['MSE']['FORWARD'] = metrics.mean_squared_error(test_y,pred_y_forward)
perf_mat.loc['MSE']['BACKWARD'] = metrics.mean_squared_error(test_y,pred_y_backward)
perf_mat.loc['MSE']['STEPWISE'] = metrics.mean_squared_error(test_y,pred_y_stepwise)

perf_mat.loc['RMSE']['ALL'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_full))
perf_mat.loc['RMSE']['FORWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_forward))
perf_mat.loc['RMSE']['BACKWARD'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_backward))
perf_mat.loc['RMSE']['STEPWISE'] = np.sqrt(metrics.mean_squared_error(test_y, pred_y_stepwise))

perf_mat.loc['MAE']['ALL'] = metrics.mean_absolute_error(test_y, pred_y_full)
perf_mat.loc['MAE']['FORWARD'] = metrics.mean_absolute_error(test_y, pred_y_forward)
perf_mat.loc['MAE']['BACKWARD'] = metrics.mean_absolute_error(test_y, pred_y_backward)
perf_mat.loc['MAE']['STEPWISE'] = metrics.mean_absolute_error(test_y, pred_y_stepwise)

perf_mat.loc['MAPE']['ALL'] = mean_absolute_percentage_error(test_y, pred_y_full)
perf_mat.loc['MAPE']['FORWARD'] = mean_absolute_percentage_error(test_y, pred_y_forward)
perf_mat.loc['MAPE']['BACKWARD'] = mean_absolute_percentage_error(test_y, pred_y_backward)
perf_mat.loc['MAPE']['STEPWISE'] = mean_absolute_percentage_error(test_y, pred_y_stepwise)

print(perf_mat)
