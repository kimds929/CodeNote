import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# 데이터 Load : 집값을 예측하는 문제
os.getcwd()     # 현재경로 확인
data_load = pd.read_csv("./Dataset/kc_house_data.csv")   # 데이터 불러오기
data_load.head().T
'''
id: 집 고유아이디
date: 집이 팔린 날짜 
price: 집 가격 (타겟변수)
bedrooms: 주택 당 침실 개수
bathrooms: 주택 당 화장실 개수
floors: 전체 층 개수
waterfront: 해변이 보이는지 (0, 1)
condition: 집 청소상태 (1~5)
grade: King County grading system 으로 인한 평점 (1~13)
yr_built: 집이 지어진 년도
yr_renovated: 집이 리모델링 된 년도
zipcode: 우편번호
lat: 위도
long: 경도
'''

nCar = data_load.shape[0] # 데이터 개수
nVar = data_load.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )


# 데이터 전처리
    # 필요없는 변수 제거
data = data_load.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1) # id, date, zipcode, lat, long  제거
data.describe().T

# 범주형 변수를 이진형 변수로 변환 : 범주형 변수는 waterfront 컬럼 뿐이며, 이진 분류이기 때문에 0, 1로 표현한다. 
#                                 데이터에서 0, 1로 표현되어 있으므로 과정 생략


# 설명변수와 타겟변수를 분리, 학습데이터와 평가데이터 분리
feature_columns = list(data.columns.difference(['price'])) # Price를 제외한 모든 행
X = data[feature_columns]
y = data['price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) # 학습데이터와 평가데이터의 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인


# < 일반모델과 Bagging을 비교 > -----------------------------------------------------------------------------------------------------------------------------------------------
# 학습 데이터를 선형 회귀 모형에 적합 후 평가 데이터로 검증 (Stats_Models) ---------------------------------------------
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

sm_train_x = sm.add_constant(train_x, has_constant = 'add') # Bias 추가
sm_model = sm.OLS(train_y, sm_train_x) # 모델 구축
fitted_sm_model = sm_model.fit() # 학습 진행
fitted_sm_model.summary() # 학습 모델 구조 확인


# 결과 확인
sm_test_x = sm.add_constant(test_x, has_constant = 'add') # 테스트 데이터에 Bias 추가
sm_model_predict = fitted_sm_model.predict(sm_test_x) # 테스트 데이터 예측

sm_rmse = sqrt(mean_squared_error(sm_model_predict, test_y))
print(f'RMSE: {sm_rmse}') # RMSE
sm_r2_score = r2_score(y_pred=sm_model_predict, y_true=test_y)
print(f'r2_score: {sm_r2_score}')  # r2_score

print(fitted_sm_model.params) # 회귀계수




# Bagging 한 결과가 일반적인 결과보다 좋은지 확인  ------------------------------------------------------------------------
import random
bagging_model_fit = {}
bagging_predict_result = [] # 빈 리스트 생성
bagging_rmse_result = []
bagging_r2_score_result = []
for _ in range(10):
    np.random.seed(_)
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0], replace=True)    # 복원 랜덤 샘플링 (약 63%)
        # np.random.choice(target_list, amount, replace=Boolean)    # target_list: 추출할 List, amount: 추출샘플갯수, replace: 복원추출여부
    random_data_idx_set = list(set(random_data_index))
    print(len(random_data_idx_set))

    bagging_train_x = train_x.iloc[random_data_idx_set, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 설명 변수
    bagging_train_y = train_y.iloc[random_data_idx_set, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 종속 변수
    bagging_train_x_add_const = sm.add_constant(bagging_train_x, has_constant = 'add') # Bias 추가
    bagging_sm_model = sm.OLS(bagging_train_y, bagging_train_x_add_const) # 모델 구축
    bagging_fitted_sm_model = bagging_sm_model.fit() # 학습 진행
    
    bagging_test_x = sm.add_constant(test_x, has_constant = 'add') # 테스트 데이터에 Bias 추가
    bagging_model_predict = bagging_fitted_sm_model.predict(bagging_test_x) # 테스트 데이터 예측
    bagging_predict_result.append(bagging_model_predict) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

    bagging_rmse = sqrt(mean_squared_error(bagging_model_predict, test_y))
    bagging_rmse_result.append(bagging_rmse)
    print(f'({_} iter) bagging rmse: {bagging_rmse}')

    bagging_r2_score = r2_score(y_pred=bagging_model_predict, y_true=test_y)
    bagging_r2_score_result.append(bagging_r2_score)
    print(f'({_} iter) bagging r2_score: {bagging_r2_score}')

    bagging_model_fit[_] = bagging_fitted_sm_model
    
bagging_rmse_result
np.mean(bagging_rmse_result)

bagging_r2_score_result
np.mean(bagging_r2_score_result)
len(bagging_predict_result)



# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_predict_result[lst_index].values[lst2_index])   # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가

bagging_predict[0:10]
    # bagging_predict_df = pd.DataFrame(np.array(bagging_predict_result).T)
    # bagging_predict_total = bagging_predict_df.apply(lambda x: np.mean(x), axis=1)
    # bagging_predict_total.iloc[0:10,]

sqrt(mean_squared_error(y_pred=bagging_predict, y_true=test_y))
    # sqrt(mean_squared_error(y_pred=bagging_predict_total, y_true=test_y))

r2_score(y_pred=bagging_predict, y_true=test_y)
    # r2_score(y_pred=bagging_predict_total, y_true=test_y)


# 평균 Coefficient를 활용한 모델 예측결과
bagging_coef = pd.DataFrame([bagging_model_fit[i].params.values for i in bagging_model_fit]).T
bagging_coef.index = list(bagging_model_fit[0].params.index)
bagging_coef_total = bagging_coef.apply(lambda x: np.mean(x), axis=1).to_frame().T
bagging_coef_total      # 모델별 coefficient 평균값

bagging_pred = bagging_test_x.apply(lambda x: x * bagging_coef_total[x.name].values)
bagging_pred_y = bagging_pred.apply(lambda x: np.sum(x), axis=1)         # 모델 coefficient 평균값을 적용한 결과 (predicted y)

sqrt(mean_squared_error(y_pred=bagging_pred_y, y_true=test_y))
r2_score(y_pred=bagging_pred_y, y_true=test_y)




# Scikit-Learn을 활용한 LinearRegression ------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression() # 선형 회귀 모형
linear_model1 = regression_model.fit(train_x, train_y) # 학습 데이터를 선형 회귀 모형에 적합
predict1 = linear_model1.predict(test_x) # 학습된 선형 회귀 모형으로 평가 데이터 예측

sklearn_rmse1 = sqrt(mean_squared_error(predict1, test_y))
print(f'RMSE OLS1: {sklearn_rmse1}') # RMSE 결과

sklearn_r2_score = r2_score(y_pred=predict1, y_true=test_y)
print(f'r2_score OLS1: {sklearn_r2_score}') # RMSE 결과


# Scikit-Learn을 활용한 Bagging - LinearRegression ------------------------------------------------------------------------
from sklearn.ensemble import BaggingRegressor
# ?BaggingRegressor
# BaggingRegressor(
#     base_estimator=None,
#     n_estimators=10,
#     *,
#     max_samples=1.0,
#     max_features=1.0,
#     bootstrap=True,
#     bootstrap_features=False,
#     oob_score=False,
#     warm_start=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
# )
bagging_model = BaggingRegressor(base_estimator = regression_model, # 선형회귀모형
                                 n_estimators = 5, # 5번 샘플링
                                 verbose = 1) # 학습 과정 표시
linear_model2 = bagging_model.fit(train_x, train_y) # 학습 진행

predict2 = linear_model2.predict(test_x) # 학습된 Bagging 선형 회귀 모형으로 평가 데이터 예측

bagging_rmse2 = sqrt(mean_squared_error(predict2, test_y))
print(f'RMSE Bagging2: {bagging_rmse2}') # RMSE 결과

bagging_r2_score2 = r2_score(y_pred=predict2, y_true=test_y)
print(f'r2_score Bagging2: {bagging_r2_score2}') # RMSE 결과


    # Sampling을 많이 해보자 
bagging_model2 = BaggingRegressor(base_estimator = regression_model, # 선형 회귀모형
                                  n_estimators = 30, # 30번 샘플링
                                  verbose = 1) # 학습 과정 표시
linear_model3 = bagging_model2.fit(train_x, train_y) # 학습 진행
predict3 = linear_model3.predict(test_x) # 학습된 Bagging 선형 회귀 모형으로 평가 데이터 예측

bagging_rmse3 = sqrt(mean_squared_error(predict3, test_y))
print(f'RMSE Bagging3: {bagging_rmse3}') # RMSE 결과

bagging_r2_score3 = r2_score(y_pred=predict3, y_true=test_y)
print(f'r2_score Bagging3: {bagging_r2_score3}') # RMSE 결과



# Tree Model 기반의 Regression ------------------------------------------------------------------------
# 학습 데이터를 의사결정나무모형에 적합 후 평가 데이터로 검증
from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor() # 의사결정나무 모형
tree_model1 = decision_tree_model.fit(train_x, train_y) # 학습 데이터를 의사결정나무 모형에 적합
    # dir(tree_model1)
    # tree_model1.feature_importances_

predict4 = tree_model1.predict(test_x) # 학습된 의사결정나무 모형으로 평가 데이터 예측

bagging_rmse4 = sqrt(mean_squared_error(predict4, test_y))
print(f'RMSE Tree4: {bagging_rmse4}') # RMSE 결과

bagging_r2_score4 = r2_score(y_pred=predict4, y_true=test_y)
print(f'r2_score Tree4: {bagging_r2_score4}') # RMSE 결과


# tree model bagging 적용
import random
tree_bagging_model_fit = {}
tree_bagging_predict_result = [] # 빈 리스트 생성
for _ in range(30):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    tree_train_x = train_x.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 설명 변수
    tree_train_y = train_y.iloc[random_data_index, ] # 랜덤 인덱스에 해당되는 학습 데이터 중 종속 변수
    decision_tree_model = DecisionTreeRegressor() # 의사결정나무 모형
    tree_bagging_model = decision_tree_model.fit(tree_train_x, tree_train_y) # 학습 데이터를 의사결정나무 모형에 적합
 
    tree_bagging_predict = tree_bagging_model.predict(test_x) # 테스트 데이터 예측
    tree_bagging_predict_result.append(tree_bagging_predict) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

    tree_bagging_model_fit[_] = tree_bagging_model

tree_bagging_model_fit
tree_bagging_predict_result


# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
tree_bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(tree_bagging_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(tree_bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    tree_bagging_predict.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가

tree_bagging_predict

tree_bagging_rmse = sqrt(mean_squared_error(tree_bagging_predict, test_y))
print(f'RMSE Tree4: {tree_bagging_rmse}') # RMSE 결과

tree_bagging_r2_score = r2_score(y_pred=tree_bagging_predict, y_true=test_y)
print(f'r2_score Tree4: {tree_bagging_r2_score}') # RMSE 결과


# Scikit-Learn을 활용한 Bagging - tree기반 모델 ------------------------------------------------------------------------
bagging_decision_tree_model1 = BaggingRegressor(base_estimator = decision_tree_model, # 의사결정나무 모형
                                                n_estimators = 5, # 5번 샘플링
                                                verbose = 1) # 학습 과정 표시
tree_model2 = bagging_decision_tree_model1.fit(train_x, train_y) # 학습 진행
predict5 = tree_model2.predict(test_x) # 학습된 Bagging 의사결정나무 모형으로 평가 데이터 예측

bagging_rmse5 = sqrt(mean_squared_error(predict5, test_y))
print(f'RMSE Tree_Bagging5: {bagging_rmse5}') # RMSE 결과

bagging_r2_score5 = r2_score(y_pred=predict5, y_true=test_y)
print(f'r2_score Tree_Bagging5: {bagging_r2_score5}') # RMSE 결과



    # Sampling을 많이 해보자 
bagging_decision_tree_model2 = BaggingRegressor(base_estimator = decision_tree_model, # 의사결정나무 모형
                                                n_estimators = 30, # 30번 샘플링
                                                verbose = 1) # 학습 과정 표시
tree_model3 = bagging_decision_tree_model2.fit(train_x, train_y) # 학습 진행
predict6 = tree_model3.predict(test_x) # 학습된 Bagging 의사결정나무 모형으로 평가 데이터 예측

bagging_rmse6 = sqrt(mean_squared_error(predict6, test_y))
print(f'RMSE Tree_Bagging6: {bagging_rmse6}') # RMSE 결과

bagging_r2_score6 = r2_score(y_pred=predict6, y_true=test_y)
print(f'r2_score Tree_Bagging6: {bagging_r2_score6}') # RMSE 결과
