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


# Adaboost -----------------------------------------------------------------------------------------
# 학습 데이터를 에이다부스트 모형에 적합 후 평가 데이터로 검증
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


    # 반복 20 / 깊이 5
tree_model = DecisionTreeClassifier(max_depth = 5) # 트리 최대 깊이 5
# ?AdaBoostClassifier
# AdaBoostClassifier(
#     base_estimator=None,
#     *,
#     n_estimators=50,
#     learning_rate=1.0,
#     algorithm='SAMME.R',
#     random_state=None,
# )
Adaboost_model1 = AdaBoostClassifier(base_estimator = tree_model, # 트리모델을 기본으로 추정
                                     n_estimators = 20, # 20회 추정
                                     random_state = 42) # 시드값 고정
model1 = Adaboost_model1.fit(train_x, train_y) # 학습 진행
predict1 = model1.predict(test_x) # 평가 데이터 예측
accuracy1 = format(accuracy_score(test_y, predict1) * 100, '.2f')
print(f'Accuracy 1: {accuracy1} %') # 정확도 % 계산




    # 반복 100 / 깊이 5     (반복횟수 증대) 
Adaboost_model2 = AdaBoostClassifier(base_estimator = tree_model, # 트리모델을 기본으로 추정
                                    n_estimators = 100, # 100회 추정
                                    random_state = 42) # 시드값 고정
model2 = Adaboost_model2.fit(train_x, train_y) # 학습 진행
predict2 = model2.predict(test_x) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, predict2) * 100), "%") # 정확도 % 계산
accuracy2 = format(accuracy_score(test_y, predict2) * 100, '.2f')
print(f'Accuracy 2: {accuracy2} %') # 정확도 % 계산



    # 반복 20 / 깊이 20     (Depth 증대) 
tree_model2 = DecisionTreeClassifier(max_depth = 20) # 트리 최대 깊이 20으로 새로 정의
Adaboost_model3 = AdaBoostClassifier(base_estimator = tree_model2, # 새 트리 모델을 기본으로 추정
                                     n_estimators = 20, # 20회 추정
                                     random_state = 42) # 시드값 고정
model3 = Adaboost_model3.fit(train_x, train_y) # 학습 진행
predict3 = model3.predict(test_x) # 평가 데이터 예측
accuracy3 = format(accuracy_score(test_y, predict3) * 100, '.2f')
print(f'Accuracy 3: {accuracy3} %') # 정확도 % 계산



    # 반복 100 / 깊이 20     (반복횟수, Depth 증대) 
tree_model2 = DecisionTreeClassifier(max_depth = 20) # 트리 최대 깊이 20으로 새로 정의
Adaboost_model4 = AdaBoostClassifier(base_estimator = tree_model2, # 새 트리 모델을 기본으로 추정
                                     n_estimators = 100, # 300회 추정
                                     random_state = 42) # 시드값 고정
model4 = Adaboost_model4.fit(train_x, train_y) # 학습 진행
predict4 = model4.predict(test_x) # 평가 데이터 예측
accuracy4 = format(accuracy_score(test_y, predict4) * 100, '.2f')
print(f'Accuracy 4: {accuracy4} %') # 정확도 % 계산




# 정확도 비교
print(f'Accuracy 1 (iter 20  / depth  5): {accuracy1} %') # 정확도 % 계산
print(f'Accuracy 2 (iter 100 / depth  5): {accuracy2} %') # 정확도 % 계산
print(f'Accuracy 3 (iter 20  / depth 20): {accuracy3} %') # 정확도 % 계산
print(f'Accuracy 4 (iter 100 / depth 20): {accuracy4} %') # 정확도 % 계산

# ** Result **
# Accuracy 1 (iter 20  / depth  5): 61.40 %
# Accuracy 2 (iter 100 / depth  5): 59.61 %
# Accuracy 3 (iter 20  / depth 20): 74.13 %
# Accuracy 4 (iter 100 / depth 20): 78.26 %


# 다른 하이퍼파라미터에 대한 정보를 얻고싶으면 링크를 참조
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html