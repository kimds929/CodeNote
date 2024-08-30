import numpy as np
import pandas as pd
import missingno as msno

import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 탐색 -----------------------------------------------------------------------------------------------------------------------
# 데이터셋 불러오기
creditcard_data = pd.read_csv("Dataset/creditcard.csv")
creditcard_data.head()

creditcard_df = creditcard_data.drop(['Time','Amount'],axis=1)
creditcard_df.shape
creditcard_df.head()
creditcard_df.describe() # 요약 통계량

# 데이터 내 NA값 여부 확인
msno.matrix(creditcard_df)
msno.bar(creditcard_df)
creditcard_df.isnull().any() # 만약 존재한다면 0으로 대체 혹은, 해당 열을 제외하고 진행


 # y값 데이터 분포 확인
from collections import Counter
Counter(creditcard_df['Class'])
# creditcard_df['Class'].value_counts()


# EDA Plot (변수별 Box-plot)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot') # Using ggplot2 style visuals 
f, ax = plt.subplots(figsize = (11, 15)) # 그래프 사이즈

ax.set_facecolor('#fafafa') # 그래프 색상값
ax.set(xlim = (-5, 5)) # X축 범위
plt.ylabel('Variables') # Y축 이름
plt.title("Overview creditcard_df Set") # 그래프 제목
ax = sns.boxplot(data=creditcard_df.drop(columns = ['Class']), # V1 ~ V28 확인
                 orient = 'h', 
                 palette = 'Set2')
plt.show()



# 각 데이터별 y값의 분포 확인(distplot)
var = creditcard_df.columns.values[:-1] # V1 ~ V28
i = 0
t0 = creditcard_df.loc[creditcard_df['Class'] == 0] # Class : 0 인 행만 추출
t1 = creditcard_df.loc[creditcard_df['Class'] == 1] # Class : 1 인 행만 추출

sns.set_style('whitegrid') # 그래프 스타일 지정
plt.figure()
fig, ax = plt.subplots(8, 4, figsize = (16, 28)) # 축 지정

for feature in var:
    i += 1
    plt.subplot(7, 4, i) # 28개의 그래프
    sns.kdeplot(t0[feature], bw = 0.5, label = "Class = 0")
    sns.kdeplot(t1[feature], bw = 0.5, label = "Class = 1")
    plt.xlabel(feature, fontsize = 12) # 라벨 속성값
    locs, labels = plt.xticks()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.show();


# #### 각 변수 별 그래프를 타겟변수에 대해서 그려보았을 때 차이가 있는 변수들은 다음과 같이 정의할 수 있다.
# - 1) 타겟 변수에 대해 분포 차이가 많이 나는 변수 : V4, V11
# - 2) 타겟 변수에 대해 분포 차이가 비교적 많이 존재하는 변수 : V12, V14, V18
# - 3) 타겟 변수에 대해 분포 차이가 비교적 적게 존재하는 변수 : V1, V2, V3, V10



# y값(target), X값(features) 분리 → array 형태로 변환
X = np.array(creditcard_df.iloc[:, creditcard_df.columns != 'Class'])
y = np.array(creditcard_df.iloc[:, creditcard_df.columns == 'Class'])
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))


# Test데이터, Training  Data분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# 모델 평가지표 함수 정의
def model_evaluation(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    Accuracy = (cf_matrix[0][0] + cf_matrix[1][1]) / sum(sum(cf_matrix))
    Precision = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1])
    Recall = cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[1][0])
    F1_Score = (2 * Recall * Precision) / (Recall + Precision)
    print("Model_Evaluation with Label:1")
    print("Accuracy: ", Accuracy)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1-Score: ", F1_Score)



# Light Gradient Boosting을 활용하여 모델 생성 ------------------------------------------------------------------------------------
    # [ multiclass lgb ]
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
lgb_dtrain = lgb.Dataset(data = pd.DataFrame(X_train), label = pd.DataFrame(y_train)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
                        # len(pd.DataFrame(y_train).drop_duplicates())
                # num_class multiclass인경우 class의 수도 정해주어야 함
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_model_predict = np.argmax(lgb_model.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
                    # predict결과가 y값의 Class마다의 확률로 나타나는데, 확률이 가장 높은 값의 column번호를 예측값으로 간주
# lgb_model_predict_df = pd.DataFrame(lgb_model.predict(X_test))
# lgb_model_predict_result = lgb_model_predict_df.idxmax(axis=1).to_frame()
# lgb_model_predict_result.iloc[1155]

model_evaluation(y_true=y_test, y_pred=lgb_model_predict) # 모델 분류 결과 평가 ***

    # [ binary lgb ]
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
lgb_dtrain2 = lgb.Dataset(data = pd.DataFrame(X_train), label = pd.DataFrame(y_train)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param2 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'binary'} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model2 = lgb.train(params = lgb_param2, train_set = lgb_dtrain2) # 학습 진행

pred = np.repeat(0, len(y_test))            # 길이가 y_test인 0 벡터
pred[lgb_model2.predict(X_test) >0.5] =1    # model 예측결과가 0.5초과인 데이터의 위치값을 1로 치환
model_evaluation(y_test, pred) # 모델 분류 결과 평가        * multiclass lgb와 결과가 같음



# [ SMOTE ]을 이용해서 Oversampling을 진행해보자! ---------------------------------------------------------------------------------
# 기존의 X_train, y_train, X_test, y_test의 형태 확인
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) # y_train 중 레이블 값이 1인 데이터의 개수
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) # y_train 중 레이블 값이 0 인 데이터의 개수

smote3 = SMOTE(random_state = 42, sampling_strategy=0.3) # SMOTE 알고리즘, 비율 증가 (1:0.3)
X_train_res3, y_train_res3 = smote3.fit_sample(X_train, y_train.ravel()) # Over Sampling 진행

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res3==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res3==0)))

print("Before OverSampling, the shape of X_train: {}".format(X_train.shape)) # SMOTE 적용 이전 데이터 형태
print("Before OverSampling, the shape of y_train: {}".format(y_train.shape)) # SMOTE 적용 이전 데이터 형태
print('After OverSampling, the shape of X_train: {}'.format(X_train_res3.shape)) # SMOTE 적용 결과 확인
print('After OverSampling, the shape of y_train: {}'.format(y_train_res3.shape)) # # SMOTE 적용 결과 확인


# SMOTE된 데이터를 바탕으로 lgb 재실시
lgb_dtrain3 = lgb.Dataset(data = pd.DataFrame(X_train_res3), label = pd.DataFrame(y_train_res3)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param3 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train_res3))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model3 = lgb.train(params = lgb_param3, train_set = lgb_dtrain3) # 학습 진행
lgb_model3_predict = np.argmax(lgb_model3.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
model_evaluation(y_test, lgb_model3_predict) # 모델 분류 평가 결과 ***



# 그렇다면, Oversampling을 더 많이 해보자. (1:0.6) ------------------------------------------------------------------------------
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) # y_train 중 레이블 값이 1인 데이터의 개수
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) # y_train 중 레이블 값이 0 인 데이터의 개수

smote4 = SMOTE(random_state = 42, sampling_strategy = 0.6) # SMOTE 알고리즘, 비율 60%
X_train_res4, y_train_res4 = smote4.fit_sample(X_train, y_train.ravel()) # Over Sampling 진행

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res4==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res4==0)))

lgb_dtrain4 = lgb.Dataset(data = pd.DataFrame(X_train_res4), label = pd.DataFrame(y_train_res4)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param4 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train_res4))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model4 = lgb.train(params = lgb_param4, train_set = lgb_dtrain4) # 학습 진행
lgb_model4_predict = np.argmax(lgb_model4.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
model_evaluation(y_test, lgb_model4_predict) # 모델 분류 평가 결과 ***


# 아예, 1:1 비율로 Oversampling을 해보자. (1:1) ------------------------------------------------------------------------------
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) # y_train 중 레이블 값이 1인 데이터의 개수
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) # y_train 중 레이블 값이 0 인 데이터의 개수

smote5 = SMOTE(random_state = 42) # SMOTE 알고리즘, Default: 동등
X_train_res5, y_train_res5 = smote5.fit_sample(X_train, y_train.ravel()) # Over Sampling 진행

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res5==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res5==0)))

lgb_dtrain5 = lgb.Dataset(data = pd.DataFrame(X_train_res5), label = pd.DataFrame(y_train_res5)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param5 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train_res5))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model5 = lgb.train(params = lgb_param5, train_set = lgb_dtrain5) # 학습 진행
lgb_model5_predict = np.argmax(lgb_model5.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
model_evaluation(y_test, lgb_model5_predict) # 모델 분류 평가 결과 ***



## 그럼 BLSM과 비교해보자! 
# BLSM (Borderline SMOTE) 비율 0.3 --------------------------------------------------------------------------
from imblearn.over_sampling import BorderlineSMOTE
blsm6 = BorderlineSMOTE(random_state = 42, sampling_strategy = 0.3) # BLSM 알고리즘 적용
X_train_res6, y_train_res6 = blsm6.fit_sample(X_train, y_train.ravel()) # Over Sampling 적용

lgb_dtrain6 = lgb.Dataset(data = pd.DataFrame(X_train_res6), label = pd.DataFrame(y_train_res6)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param6 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train_res6))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model6 = lgb.train(params = lgb_param6, train_set = lgb_dtrain6) # 학습 진행
lgb_model6_predict = np.argmax(lgb_model6.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
model_evaluation(y_test, lgb_model6_predict) # 모델 분류 평가 결과 ***


# BLSM (Borderline SMOTE) 비율 0.6은 어떨까?  --------------------------------------------------------------------------
blsm7 = BorderlineSMOTE(random_state = 42, sampling_strategy = 0.6) # BLSM 알고리즘 적용
X_train_res7, y_train_res7 = blsm7.fit_sample(X_train, y_train.ravel()) # Over Sampling 적용

lgb_dtrain7 = lgb.Dataset(data = pd.DataFrame(X_train_res7), label = pd.DataFrame(y_train_res7)) # 학습 데이터를 LightGBM 모델에 맞게 변환
lgb_param7 = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'objective': 'multiclass', # 목적 함수
            'num_class': len(set(pd.DataFrame(y_train_res6))) + 1} # 파라미터 추가, Label must be in [0, num_class) -> num_class보다 1 커야한다.
lgb_model7 = lgb.train(params = lgb_param7, train_set = lgb_dtrain7) # 학습 진행
lgb_model7_predict = np.argmax(lgb_model7.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측
model_evaluation(y_test, lgb_model7_predict) # 모델 분류 평가 결과 ***






# # BLSM보다 기본 SMOTE가 성능이 좋다. 이를 바탕으로 다양한 모델에 적용-----------------------------------------------------------


# - 선형회귀(로지스틱), Random Forest, CatBoost-----------------------------------------------------------
# BLSM을 이용해서 Oversampling 한 학습 데이터 셋 : X_train_res3, y_train_res3

from sklearn.linear_model import LogisticRegression

# 일반 로지스틱 회귀모형 학습   -----------------------------------------------------------
lr_model = LogisticRegression(C = 1e+10) 
# sklearn 의 Logistic Regression은 기본적으로 Ridge 정규화가 포함되어 있기 때문에, 
# 정규화 텀을 억제하는 C를 크게 적용한다 (C:Inverse of regularization strength)
lr_model.fit(X_train_res3, y_train_res3) # 로지스틱 회귀 모형 학습
lr_predict = lr_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
model_evaluation(y_test, lr_predict) # 모델 분류 평가 결과 ***

np.sum(lr_predict==1)

# Lasso(라쏘) 로지스틱 회귀모형 학습   -----------------------------------------------------------
lasso_model = LogisticRegression(penalty = 'l1', solver='liblinear') # Penalty = l1 Regularizer, C = 1.0 (Default))
lasso_model.fit(X_train_res3, y_train_res3) # 라쏘 로지스틱 회귀 모형 학습
lasso_predict = lasso_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
model_evaluation(y_test, lasso_predict) # 모델 분류 평가 결과 ***

np.sum(lasso_predict==1)
# solver='lbfgs'  (default option)
# The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation, or no regularization.
#  The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. 
# The Elastic-Net regularization is only supported by the ‘saga’ solver.

# Ridge(릿지) 로지스틱 회귀모형 학습   -----------------------------------------------------------
ridge_model = LogisticRegression(penalty = 'l2') # Default = LogisticRegression()
ridge_model.fit(X_train_res3, y_train_res3) # 릿지 로지스틱 회귀 모형 학습
ridge_predict = ridge_model.predict(X_test) # 학습 결과를 바탕으로 검증 데이터를 예측
model_evaluation(y_test, ridge_predict) # 모델 분류 평가 결과 ***

np.sum(ridge_predict==1)


# Random Forest -----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators = 50, # 50번 추정
                                             max_depth = 10, # 트리 최대 깊이 10
                                             random_state = 42) # 시드값 고정
rf_model = random_forest_model.fit(X_train_res3, y_train_res3) # 학습 진행
rf_predict = rf_model.predict(X_test) # 평가 데이터 예측
model_evaluation(y_test, rf_predict) # 모델 분류 평가 결과 ***


# catboost -----------------------------------------------------------
import catboost as cb
cb_dtrain = cb.Pool(data = X_train_res3, label = y_train_res3) # 학습 데이터를 Catboost 모델에 맞게 변환
cb_param = {'max_depth': 10, # 트리 깊이
            'learning_rate': 0.01, # Step Size
            'n_estimators': 50, # Number of trees, 트리 생성 개수
            'eval_metric': 'Accuracy', # 평가 척도
            'loss_function': 'MultiClass'} # 손실 함수, 목적 함수
cb_model = cb.train(pool = cb_dtrain, params = cb_param) # 학습 진행
cb_model_predict = np.argmax(cb_model.predict(X_test), axis = 1) # 평가 데이터 예측, Softmax의 결과값 중 가장 큰 값의 Label로 예측, 인덱스의 순서를 맞추기 위해 +1
model_evaluation(y_test, cb_model_predict) # 모델 분류 평가 결과 ***



# # Ensemble의 Ensemble -----------------------------------------------------------
# - 성능이 가장 좋은 Random Forest 모델을 바탕으로 진행

import random
bagging_predict_result = [] # 빈 리스트 생성
number_of_bagging = 5 # Bagging 횟수
for _ in range(number_of_bagging):
    data_index = [data_index for data_index in range(X_train_res3.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, X_train_res3.shape[0]) # 
    random_forest_model2 = RandomForestClassifier(n_estimators = 50, # 50번 추정
                                                 max_depth = 10, # 트리 최대 깊이 10
                                                 random_state = 42) # 시드값 고정
    rf_model2 = random_forest_model2.fit(X = pd.DataFrame(X_train_res3).iloc[random_data_index, ],
                                       y = pd.DataFrame(y_train_res3).iloc[random_data_index]) # 학습 진행
    rf_predict2 = rf_model2.predict(X_test) # 평가 데이터 예측
    bagging_predict_result.append(rf_predict2) # 예측 결과를 bagging_predict_result에 저장
    print(_+1, "Model Evaluation Result:", "\n") # 전체적인 성능 평가
    model_evaluation(y_test, rf_predict2) # 모델 분류 평가 결과 ***



# # Bagging을 바탕으로 예측한 결과값에 대해 다수결로 예측 -----------------------------------------------
bagging_predict = [] # 빈 리스트 생성
for lst2_index in range(X_test.shape[0]): # 테스트 데이터 개수만큼 반복
    temp_predict = [] # 반복문 내 임시 빈 리스트 생성
    for lst_index in range(len(bagging_predict_result)): # Bagging 결과 리스트 개수 만큼 반복
        temp_predict.append(bagging_predict_result[lst_index][lst2_index]) # 각 Bagging 결과 예측한 값 중 같은 인덱스를 리스트에 저장
    if np.mean(temp_predict) >= 0.5: # 0, 1 이진분류이므로, 예측값의 평균이 0.5보다 크면 1, 아니면 0으로 예측 다수결)
        bagging_predict.append(1)
    elif np.mean(temp_predict) < 0.5: # 예측값의 평균이 0.5보다 낮으면 0으로 결과 저장
        bagging_predict.append(0)
model_evaluation(y_test, bagging_predict) # 모델 분류 평가 결과 ***

rf_model2.predict(X_test) 



# 최적 cut-off값을 적용하여 모델 평가 -------------------------------------------------------------------------------------------------------
import random
bagging_predict_result = 0 # 
number_of_bagging = 5 # Bagging 횟수
for i in range(number_of_bagging):
    data_index = [data_index for data_index in range(X_train_res3.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    random_data_index = np.random.choice(data_index, X_train_res3.shape[0]) # 
    random_forest_model2 = RandomForestClassifier(n_estimators = 50, # 50번 추정
                                                 max_depth = 10, # 트리 최대 깊이 10
                                                 random_state = 42) # 시드값 고정
    rf_model2 = random_forest_model2.fit(X = pd.DataFrame(X_train_res3).iloc[random_data_index, ],
                                       y = pd.DataFrame(y_train_res3).iloc[random_data_index]) # 학습 진행
    rf_predict2 = rf_model2.predict_proba(X_test)[: , 1]
    bagging_predict_result=bagging_predict_result+(rf_predict2) # 예측 결과를 bagging_predict_result에 저장
    print(i)

pred= np.repeat(0,len(y_test))
pred[bagging_predict_result /2 > 0.5]=1
model_evaluation(y_test, pred) # 모델 분류 평가 결과 ***


def cut_off(y,threshold):
    Y =y.copy()
    Y[Y >threshold]=1
    Y[Y <=threshold]=0
    return(Y.astype(int))

threshold = np.arange(0,1,0.1)
threshold


for i in threshold :
    pred_y = cut_off(bagging_predict_result /2 ,i)
    print(i)
    model_evaluation(y_test, pred_y)





# 각각의 데이터에 대해 가중치를 다르게 부여 -------------------------------------------------------------------------------------------------------
    # 잘 맞추지 못하는 데이터에 대해 가중치를 높게 부여
p_list=np.repeat(1/X_train_res3.shape[0],X_train_res3.shape[0])
p_list[rf_model2.predict(X_train_res3) != y_train_res3] = p_list[rf_model2.predict(X_train_res3) != y_train_res3] * 2  # 맞추지 못한 데이터에 대한 가중치를 2배로 부여

p_list[rf_model2.predict(X_train_res3) != y_train_res3]
p_list = p_list/sum(p_list)     # 전체 확률값의 합을 1로 맞추기

sum(p_list)

import random
bagging_predict_result = 0 # 
number_of_bagging = 5 # Bagging 횟수
for i in range(number_of_bagging):
    data_index = [data_index for data_index in range(X_train_res3.shape[0])] # 학습 데이터의 인덱스를 리스트로 변환
    if i ==0 : 
        plist = np.repeat(1/X_train_res3.shape[0],X_train_res3.shape[0])
    else :
        p_list = np.repeat(1/X_train_res3.shape[0],X_train_res3.shape[0])
        p_list[rf_model2.predict(X_train_res3) != y_train_res3] = p_list[rf_model2.predict(X_train_res3) != y_train_res3]*2
    random_data_index = np.random.choice(data_index, X_train_res3.shape[0],p=plist) # 
    random_forest_model2 = RandomForestClassifier(n_estimators = 50, # 50번 추정
                                                 max_depth = 10, # 트리 최대 깊이 10
                                                 random_state = 42) # 시드값 고정
    rf_model2 = random_forest_model2.fit(X = pd.DataFrame(X_train_res3).iloc[random_data_index, ],
                                       y = pd.DataFrame(y_train_res3).iloc[random_data_index]) # 학습 진행
    rf_predict2 = rf_model2.predict_proba(X_test)[: , 1]
    bagging_predict_result=bagging_predict_result+(rf_predict2) # 예측 결과를 bagging_predict_result에 저장
    print(i)


for i in threshold :
    pred_y = cut_off(bagging_predict_result /2 ,i)
    print(i)
    model_evaluation(y_test, pred_y)