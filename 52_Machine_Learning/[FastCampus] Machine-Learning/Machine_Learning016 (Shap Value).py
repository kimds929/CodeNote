import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

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

# Light Gradient Boosting --------
import lightgbm as lgb

start = time.time() # 시작 시간 지정
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
                    'learning_rate': 0.01, # Step Size
                    'n_estimators': 500, # Number of trees, 트리 생성 개수
                    'objective': 'regression'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_predict = lgb_model.predict(test_x)
end = time.time() # 시작 시간 지정
print(f'delay time: {end-start}')

lgb_rmse = sqrt(mean_squared_error(lgb_predict, test_y))
print(f'lgb RMSE: {lgb_rmse}') # RMSE
lgb_r2_score = r2_score(y_pred=lgb_predict, y_true=test_y)
print(f'lgb r2_score: {lgb_r2_score}')  # r2_score



# plt.bar(x=np.arange(len(test_x.columns)), height=lgb_model.feature_importance)
# xlabel = list(test_x.columns)


# Shap Value를 이용하여 변수 별 영향도 파악 -----------------------
    # conda install scikit-image
    # conda install -c conda-forge shap
import skimage
import shap
explainer = shap.TreeExplainer(lgb_model) # 트리 모델 Shap Value 계산 객체 지정
shap_values = explainer.shap_values(test_x) # Shap Values 계산

shap.initjs() # 자바스크립트 초기화 (그래프 초기화)
shap.force_plot(explainer.expected_value, shap_values[0,:], test_x.iloc[0,:]) # 첫 번째 검증 데이터 인스턴스에 대해 Shap Value를 적용하여 시각화
shap.force_plot(explainer.expected_value, shap_values[1,:], test_x.iloc[1,:]) # 두 번째 검증 데이터 인스턴스에 대해 Shap Value를 적용하여 시각화
shap.force_plot(explainer.expected_value, shap_values[0:2,:], test_x.iloc[0:2,:]) # 0~2 번째 검증 데이터 인스턴스에 대해 Shap Value를 적용하여 시각화
# 빨간색이 영향도가 높으며, 파란색이 영향도가 낮음

shap.force_plot(explainer.expected_value, shap_values, test_x) # 전체 검증 데이터 셋에 대해서 적용

shap.summary_plot(shap_values, test_x)
# grade : 변수의 값이 높을 수록, 예상 가격이 높은 경향성이 있다.
# yr_built : 변수의 값이 낮을 수록, 예상 가격이 높은 경향성이 있다.
# bathrooms : 변수의 값이 높을 수록, 예상 가격이 높은 경향성이 있다.
# bedrooms : 변수의 값이 높을 수록, 예상 가격이 높은 경향성이 있다.
# condition : 변수의 값이 높을 수록, 예상 가격이 높은 경향성이 있다
# waterfront : 변수의 값이 높을 수록, 예상 가격이 높은 경향성이 있다.
# floors : 해석 모호성 (Feature Value에 따른 Shap Values의 상관성 파악 모호)
# yr_renovated : 해석 모호성 (Feature Value에 따른 Shap Values의 상관성 파악 모호)

shap.summary_plot(shap_values, test_x, plot_type = "bar") # 각 변수에 대한 Shap Values의 절대값으로 중요도 파악

# 자동으로 가장 연관이 된 column과 묶어줌
shap.dependence_plot("yr_built", shap_values, test_x)
shap.dependence_plot("grade", shap_values, test_x)
shap.dependence_plot("condition", shap_values, test_x)

## https://github.com/slundberg/shap