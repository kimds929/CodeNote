import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

from math import sqrt

# 데이터 로드
os.getcwd()     # 현재경로 확인
data_load = pd.read_csv("./Dataset/otto_train.csv")   # 데이터 불러오기
data_load.head().T
'''
id: 고유 아이디
feat_1 ~ feat_93: 설명변수
target: 타겟변수 (1~9)
'''

nCar = data_load.shape[0] # 데이터 개수
nVar = data_load.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )


# 의미가 없다고 판단되는 변수 제거
data = data_load.drop(['id'], axis = 1) # id 제거


# 타겟 변수의 문자열을 숫자로 변환
mapping_dict = {"Class_1": 1,
                "Class_2": 2,
                "Class_3": 3,
                "Class_4": 4,
                "Class_5": 5,
                "Class_6": 6,
                "Class_7": 7,
                "Class_8": 8,
                "Class_9": 9}
after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])
after_mapping_target.value_counts().sort_index()



# 설명변수와 타겟변수를 분리, 학습데이터와 평가데이터 분리
feature_columns = list(data.columns.difference(['target']))     # target을 제외한 모든 행
X = data[feature_columns] # 설명변수
y = after_mapping_target # 타겟변수
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) # 학습데이터와 평가데이터의 비율을 8:2 로 분할| 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인


# 1. XG Boost --------------------------------------------------------------------------------
import xgboost as xgb
# https://3months.tistory.com/368
start = time.time() # 시작 시간 지정
xgb_dtrain = xgb.DMatrix(data = train_x, label = train_y) # 학습 데이터를 XGBoost 모델에 맞게 변환
xgb_dtest = xgb.DMatrix(data = test_x) # 평가 데이터를 XGBoost 모델에 맞게 변환
xgb_param = {'max_depth': 10, # 트리 깊이
         'learning_rate': 0.01, # Step Size
         'n_estimators': 100, # Number of trees, 트리 생성 개수
         'objective': 'multi:softmax', # 목적 함수 (y class가 다수이므로 multi:softmax사용)
        #  'eval_metric':       # 평가척도
        'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
xgb_model = xgb.train(params = xgb_param, dtrain = xgb_dtrain) # 학습 진행
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

xgb_model_predict = xgb_model.predict(xgb_dtest) # 평가 데이터 예측
accuracy1 = format(accuracy_score(test_y, xgb_model_predict) * 100, '.2f')
print(f'Accuracy 1: {accuracy1} %') # 정확도 % 계산

xgb_model_predict


# 2. LightGBM --------------------------------------------------------------------------------
import lightgbm as lgb

start = time.time() # 시작 시간 지정
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(train_y)) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

lgb_model_predict = lgb_model.predict(test_x)
lgb_model_predict_max = np.argmax(lgb_model_predict, axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측

accuracy2 = format(accuracy_score(test_y, lgb_model_predict_max) * 100, '.2f')
print(f'Accuracy 2: {accuracy2} %') # 정확도 % 계산


# 3. Catboost --------------------------------------------------------------------------------
import catboost as cb

start = time.time() # 시작 시간 지정
cb_dtrain = cb.Pool(data = train_x, label = train_y) # 학습 데이터를 Catboost 모델에 맞게 변환
cb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 100, # Number of trees, 트리 생성 개수
            'eval_metric': 'Accuracy', # 평가척도
            'loss_function': 'MultiClass'} # 손실 함수, 목적 함수
cb_model = cb.train(pool = cb_dtrain, params = cb_param) # 학습 진행
print("Time: %.2f" % (time.time() - start), "seconds") # 코드 실행 시간 계산

cb_model_predict = cb_model.predict(test_x)
cb_model_predict_max = np.argmax(cb_model.predict(test_x), axis = 1) + 1 # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측, 인덱스의 순서를 맞추기 위해 +1

accuracy3 = format(accuracy_score(test_y, cb_model_predict_max) * 100, '.2f')



print(f'Accuracy 1 (XG Boost): {accuracy1} %') # 정확도 % 계산
print(f'Accuracy 2 (Light Gradient Boost): {accuracy2} %') # 정확도 % 계산
print(f'Accuracy 3 (Cat Boost): {accuracy3} %') # 정확도 % 계산





# House Data 예측문제 ---------------------------------------------------------------------------------------
# 데이터 Load : 집값을 예측하는 문제
os.getcwd()     # 현재경로 확인
house_data_load = pd.read_csv("./Dataset/kc_house_data.csv")   # 데이터 불러오기
house_data_load.head().T
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

nCar = house_data_load.shape[0] # 데이터 개수
nVar = house_data_load.shape[1] # 변수 개수
print('nCar: %d' % nCar, 'nVar: %d' % nVar )


# 데이터 전처리
    # 필요없는 변수 제거
house_data = house_data_load.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis = 1) # id, date, zipcode, lat, long  제거
house_data.describe().T

# 범주형 변수를 이진형 변수로 변환 : 범주형 변수는 waterfront 컬럼 뿐이며, 이진 분류이기 때문에 0, 1로 표현한다. 
#                                 데이터에서 0, 1로 표현되어 있으므로 과정 생략

feature_columns = list(house_data.columns.difference(['price'])) # Price를 제외한 모든 행
X = house_data[feature_columns]
y = house_data['price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) # 학습데이터와 평가데이터의 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인


# light gradient boost 적용 ----------------------------------------------------------------
import lightgbm as lgb

start = time.time() # 시작 시간 지정
house_lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
house_lgb_param = {'max_depth': 10, # 트리 깊이
                    'learning_rate': 0.01, # Step Size
                    'n_estimators': 500, # Number of trees, 트리 생성 개수
                    'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
house_lgb_model = lgb.train(params = house_lgb_param, train_set = house_lgb_dtrain) # 학습 진행
house_lgb_predict = house_lgb_model.predict(test_x)

house_lgb_rmse = sqrt(mean_squared_error(house_lgb_predict, test_y))
print(f'house_lgb RMSE: {house_lgb_rmse}') # RMSE
house_lgb_r2_score = r2_score(y_pred=house_lgb_predict, y_true=test_y)
print(f'house_lgb r2_score: {house_lgb_r2_score}')  # r2_score



# bagging + lgb 적용 ----------------------------------------------------------------
import random
bagging_lgb_predict_result = [] # 빈 리스트 생성
for _ in range(10):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    bagging_lgb_dtrain = lgb.Dataset(data = train_x.iloc[random_data_index,], label = train_y.iloc[random_data_index,]) # 학습 데이터를 LightGBM 모델에 맞게 변환
    bagging_lgb_param = {'max_depth': 14, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 500, # Number of trees, 트리 생성 개수
            'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
    bagging_lgb_model = lgb.train(params = bagging_lgb_param, train_set = bagging_lgb_dtrain) # 학습 진행
    bagging_predict = bagging_lgb_model.predict(test_x) # 테스트 데이터 예측
    bagging_lgb_predict_result.append(bagging_predict) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장


# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
bagging_predict_list = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(bagging_lgb_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(bagging_lgb_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    bagging_predict_list.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가


# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가
house_bagging_lgb_rmse = sqrt(mean_squared_error(bagging_predict_list, test_y))
print(f'house_bagging_lgb RMSE: {house_bagging_lgb_rmse}') # RMSE
house_bagging_lgb_r2_score = r2_score(y_pred=bagging_predict_list, y_true=test_y)
print(f'house_bagging_lgb r2_score: {house_bagging_lgb_r2_score}')  # r2_score

bagging_predict

# lbg ↔ bagging + lgb 평가지표 비교
print(f'house_lgb RMSE: {house_lgb_rmse}') # RMSE
print(f'house_lgb r2_score: {house_lgb_r2_score}')  # r2_score
print(f'house_bagging_lgb RMSE: {house_bagging_lgb_rmse}') # RMSE
print(f'house_bagging_lgb r2_score: {house_bagging_lgb_r2_score}')  # r2_score