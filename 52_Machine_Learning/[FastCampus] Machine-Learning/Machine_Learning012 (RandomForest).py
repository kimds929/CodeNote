import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


# Random Forest ---------------------------------------------------------------------------------------------------------------------------
# 학습 데이터를 랜덤포레스트 모형에 적합 후 평가 데이터로 검증 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ?RandomForestClassifier
# RandomForestClassifier(
#     n_estimators=100,
#     *,
#     criterion='gini',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features='auto',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
#     class_weight=None,
#     ccp_alpha=0.0,
#     max_samples=None,
# )

    # 반복 20 / 깊이 5
random_forest_model1 = RandomForestClassifier(n_estimators = 20, # 20번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 42) # 시드값 고정
model1 = random_forest_model1.fit(train_x, train_y) # 학습 진행
predict1 = model1.predict(test_x) # 평가 데이터 예측
accuracy1 = format(accuracy_score(test_y, predict1) * 100, '.2f')
print(f'Accuracy 1: {accuracy1} %') # 정확도 % 계산


    # 반복 300 / 깊이 5     (반복횟수 증대) 
random_forest_model2 = RandomForestClassifier(n_estimators = 300, # 300번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 42) # 시드값 고정
model2 = random_forest_model2.fit(train_x, train_y) # 학습 진행
predict2 = model2.predict(test_x) # 평가 데이터 예측
accuracy2 = format(accuracy_score(test_y, predict2) * 100, '.2f')
print(f'Accuracy 2: {accuracy2} %') # 정확도 % 계산


    # 반복 20 / 깊이 20     (Depth 증대) 
random_forest_model3 = RandomForestClassifier(n_estimators = 20, # 20번 추정
                                             max_depth = 20, # 트리 최대 깊이 20
                                             random_state = 42) # 시드값 고정
model3 = random_forest_model3.fit(train_x, train_y) # 학습 진행
predict3 = model3.predict(test_x) # 평가 데이터 예측
accuracy3 = format(accuracy_score(test_y, predict3) * 100, '.2f')
print(f'Accuracy 3: {accuracy3} %') # 정확도 % 계산


    # 반복 300 / 깊이 20     (반복횟수, Depth 증대) 
random_forest_model4 = RandomForestClassifier(n_estimators = 300, # 300번 추정
                                             max_depth = 20, # 트리 최대 깊이 20
                                             random_state = 42) # 시드값 고정
model4 = random_forest_model4.fit(train_x, train_y) # 학습 진행
predict4 = model4.predict(test_x) # 평가 데이터 예측
accuracy4 = format(accuracy_score(test_y, predict4) * 100, '.2f')
print(f'Accuracy 4: {accuracy4} %') # 정확도 % 계산


# 정확도 비교
print(f'Accuracy 1 (iter 20  / depth  5): {accuracy1} %') # 정확도 % 계산
print(f'Accuracy 2 (iter 300 / depth  5): {accuracy2} %') # 정확도 % 계산
print(f'Accuracy 3 (iter 20  / depth 20): {accuracy3} %') # 정확도 % 계산
print(f'Accuracy 4 (iter 300 / depth 20): {accuracy4} %') # 정확도 % 계산

# ** Result **
# Accuracy 1 (iter 20  / depth  5): 60.16 %
# Accuracy 2 (iter 300 / depth  5): 61.73 %
# Accuracy 3 (iter 20  / depth 20): 77.42 %
# Accuracy 4 (iter 300 / depth 20): 78.09 %


# 다른 하이퍼파라미터에 대한 정보를 얻고싶으면 링크를 참조
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html