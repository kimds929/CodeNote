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


# Ensemble의 Ensemble (bagging + light gradient boosting) --------------------------------------------------------
import random
import lightgbm as lgb      # light gradient boost 적용 

ensemble_predict_result = []        # y_predict 빈 리스트 생성
ensemble_predict_rmse = []          # rmse 빈 리스트 생성
ensemble_predict_r2_score = []      # r2_score 빈 리스트 생성

for _ in range(30):
    data_index = [data_index for data_index in range(train_x.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, train_x.shape[0]) # 데이터의 1/10 크기만큼 랜덤 샘플링, // 는 소수점을 무시하기 위함
    print(len(set(random_data_index)))
    ensemble_bagging_train_x = train_x.iloc[random_data_index,]
    ensemble_bagging_train_y = train_y.iloc[random_data_index,]

    ensemble_dtrain = lgb.Dataset(data = ensemble_bagging_train_x, label = ensemble_bagging_train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
    ensemble_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 500, # Number of trees, 트리 생성 개수
            'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
    lgb_model = lgb.train(params = ensemble_param, train_set = ensemble_dtrain) # 학습 진행

    ensemble_predict = lgb_model.predict(test_x) # 테스트 데이터 예측
    ensemble_predict_result.append(ensemble_predict) # 반복문이 실행되기 전 빈 리스트에 결과 값 저장

    ensemble_rmse = sqrt(mean_squared_error(ensemble_predict, test_y))
    ensemble_predict_rmse.append(ensemble_rmse)
    print(f'rmse : {ensemble_rmse}')

    ensemble_r2_score = r2_score(y_pred=ensemble_predict, y_true=test_y)
    ensemble_predict_r2_score.append(ensemble_r2_score)
    print(f'r2_score : {ensemble_r2_score}')




# Bagging을 바탕으로 예측한 결과값에 대한 평균을 계산
ensemble_predict_list = [] # 빈 리스트 생성
for lst2_index in range(test_x.shape[0]): # 테스트 데이터 개수만큼의 반복
    temp_predict = [] # 임시 빈 리스트 생성 (반복문 내 결과값 저장)
    for lst_index in range(len(ensemble_predict_result)): # Bagging 결과 리스트 반복
        temp_predict.append(ensemble_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    ensemble_predict_list.append(np.mean(temp_predict)) # 해당 인덱스의 30개의 결과값에 대한 평균을 최종 리스트에 추가


# 예측한 결과값들의 평균을 계산하여 실제 테스트 데이트의 타겟변수와 비교하여 성능 평가
ensemble_lgb_rmse = sqrt(mean_squared_error(ensemble_predict_list, test_y))
print(f'ensemble_lgb RMSE: {ensemble_lgb_rmse}') # RMSE
ensemble_lgb_r2_score = r2_score(y_pred=ensemble_predict_list, y_true=test_y)
print(f'ensemble_lgb r2_score: {ensemble_lgb_r2_score}')  # r2_score


