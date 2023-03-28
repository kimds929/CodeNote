import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import statsmodels.api as sm
import itertools
import time

os.getcwd()
ploan = pd.read_csv('./Dataset/Personal Loan.csv')
ploan.info()
ploan.shape
'''
Experience 경력
Income 수입
Famliy 가족단위
CCAvg 월 카드사용량 
Education 교육수준 (1: undergrad; 2, Graduate; 3; Advance )
Mortgage 가계대출
Securities account 유가증권계좌유무
CD account 양도예금증서 계좌 유무
Online 온라인계좌유무
CreidtCard 신용카드유무 
'''

len(ploan.dropna())
# 빈값이 포함된 row제거 + ID, ZIP Code Column제거
ploan_df = ploan.dropna().drop(['ID','ZIP Code'], axis=1, inplace=False)
ploan_df.shape

ploan_processed = sm.add_constant(ploan_df, has_constant='add')
ploan_processed.info()
ploan_processed.shape

# 학습데이터 / 테스트데이터 분리
y_column = 'Personal Loan'
feature_columns = ploan_processed.columns.difference([y_column])
# feature_columns = ploan_processed.columns.drop([y_column])
X = ploan_processed[feature_columns]
y = ploan_processed[y_column]
y.value_counts()

train_y, test_y, train_x, test_x = train_test_split(y, X, stratify=y, test_size=0.3, random_state=42)
# stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 
#         Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.



# Logistic 회귀모델 적합
logit_model = sm.Logit(train_y, train_x)
logit_results = logit_model.fit(method='newton')
logit_results.summary()

logit_results.params
np.exp(logit_results.params).apply(lambda x: round(x,2))

pred_y = logit_results.predict(test_x)
pred_y


# cut off
def cut_off(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return (Y.astype(int))


pred_Y = cut_off(pred_y, 0.5)


# confusion matrix
cfmat = confusion_matrix(test_y, pred_Y)
cfmat_ct = pd.crosstab(test_y, pred_Y)


accuracy = (cfmat[0][0] + cfmat[1][1])/len(pred_Y)
# accuracy = (cfmat_ct[0][0] + cfmat_ct[1][1])/len(pred_Y)

def acc(cfmat):
    if isinstance(cfmat, np.ndarray):
        acc = (cfmat[0][0] + cfmat[1][1])/cfmat.sum()
    if isinstance(cfmat, pd.DataFrame):
        acc = (cfmat[0][0] + cfmat[1][1])/cfmat.sum().sum()
    return acc

accuracy = acc(cfmat)
# accuracy = acc(cfmat_ct)


# 임계값(cut-off)에 따른 성능지표 비교
threshold = np.arange(0.1, 1, 0.1)
threshold
table = pd.DataFrame(columns=['ACC'])
for i in threshold:
    pred_Y = cut_off(pred_y, i)
    cfmat = confusion_matrix(test_y, pred_Y)
    table.loc[i] = acc(cfmat)
table.index.name = 'threshold'
table.columns.name = 'performance'
table

plt.figure()
plt.plot(table.index, table['ACC'])
plt.xlabel('threshold')
plt.ylabel('Accuracy')
plt.show()


# sklearn ROC 패키지 이용
fpr, tpr, thresholds = metrics.roc_curve(y_true=test_y, y_score=pred_y, pos_label=1)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

# auc값
auc = np.trapz(tpr, fpr)  # ROC Curve의 면적값
    # np.trapz : 합성 사다리꼴 규칙을 사용하여 주어진 축을 따라 적분합니다
auc



# 기존 모델에서 다중공선성이 높은 변수 제거 후 logistic 회귀분석 재실시
logit_results.summary()

ploan_processed
# Experience, Mortgage 제거
feature_columns2 = ploan_processed.columns.difference(['ID', 'ZIP Code','Personal Loan', 'Experience', 'Mortgage'])
feature_columns2

X2 = ploan_processed[feature_columns2]
X2
y


train_y2, test_y2, train_x2, test_x2 = train_test_split(y, X2, stratify=y, test_size=0.3, random_state=42)

logit_model2 = sm.Logit(train_y2, train_x2)
logit_results2 = logit_model2.fit(method='newton')
logit_results2.summary()

logit_results2.params
np.exp(logit_results2.params).apply(lambda x: round(x,2))

pred_y2 = logit_results2.predict(test_x2)
pred_y2

pred_Y2 = cut_off(pred_y2, 0.5)


cfmat2 = confusion_matrix(test_y2, pred_Y2)
acc(cfmat2)


# 임계값(cut-off)에 따른 성능지표 비교
threshold2 = np.arange(0.1, 1, 0.1)
table2 = pd.DataFrame(columns=['ACC'])
for i in threshold2:
    pred_Y2 = cut_off(pred_y2, i)
    cfmat2 = confusion_matrix(test_y2, pred_Y2)
    table2.loc[i] = acc(cfmat2)
table2.index.name = 'threshold'
table2.columns.name = 'performance'
table2

plt.figure()
plt.plot(table.index, table['ACC'], c='steelblue', alpha=0.5)
plt.plot(table2.index, table2['ACC'], c='coral', alpha=0.5)
plt.xlabel('threshold')
plt.ylabel('Accuracy')
plt.show()



# sklearn ROC 패키지 이용
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true=test_y2, y_score=pred_y2, pos_label=1)

plt.figure()
plt.plot(fpr, tpr, c='steelblue', alpha=0.5)
plt.plot(fpr2, tpr2, c='coral', alpha=0.5)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# auc값
auc2 = np.trapz(tpr2, fpr2)  # ROC Curve의 면적값
    # np.trapz : 합성 사다리꼴 규칙을 사용하여 주어진 축을 따라 적분합니다
auc2
auc





# 변수선택법 적용
import DS_OLS_Tools

Forward_best_model = forward_model(X=train_x, y= train_y)
Backward_best_model = backward_model(X=train_x,y=train_y)
Stepwise_best_model = Stepwise_model(X=train_x,y=train_y)

### 선택된 변수갯수
print(len(Forward_best_model.model.exog_names))
print(len(Backward_best_model.model.exog_names))
print(len(Stepwise_best_model.model.exog_names))

# Print ROC curve
fpr_full, tpr_full, thresholdsr_full = metrics.roc_curve(test_y, pred_y_full, pos_label=1)
fpr_forward, tpr_forward, thresholds_forward = metrics.roc_curve(test_y, pred_y_forward, pos_label=1)
fpr_backward, tpr_backward, thresholds_backward = metrics.roc_curve(test_y, pred_y_backward, pos_label=1)
fpr_stepwise, tpr_stepwise, thresholds_stepwise = metrics.roc_curve(test_y, pred_y_stepwise, pos_label=1)

plt.figure()
plt.plot(fpr_full, tpr_full, c='coral', alpha=0.5, label='full')
plt.plot(fpr_forward, tpr_forward, c='steelblue', alpha=0.5, label='forward')
plt.plot(fpr_forward, tpr_forward, c='mediumseagreen', alpha=0.5, label='backward')
plt.plot(fpr_forward, tpr_forward, c='purple', alpha=0.5, label='stepwise')
plt.legend()
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# AUC area
pred_y_full = logit_results.predict(test_x) # full model (전체변수)
pred_y_forward = Forward_best_model.predict(test_x[Forward_best_model.model.exog_names])
pred_y_backward = Backward_best_model.predict(test_x[Backward_best_model.model.exog_names])
pred_y_stepwise = Stepwise_best_model.predict(test_x[Stepwise_best_model.model.exog_names])

pred_Y_full= cut_off(pred_y_full,0.5)
pred_Y_forward = cut_off(pred_y_forward,0.5)
pred_Y_backward = cut_off(pred_y_backward,0.5)
pred_Y_stepwise = cut_off(pred_y_stepwise,0.5)

cfmat_full = confusion_matrix(test_y, pred_Y_full)
cfmat_forward = confusion_matrix(test_y, pred_Y_forward)
cfmat_backward = confusion_matrix(test_y, pred_Y_backward)
cfmat_stepwise = confusion_matrix(test_y, pred_Y_stepwise)

print(acc(cfmat_full))
print(acc(cfmat_forward))
print(acc(cfmat_backward))
print(acc(cfmat_stepwise))




