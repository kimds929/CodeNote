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

import DS_OLS_Tools



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


# Lasso & Ridge
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# 빈값이 포함된 row제거 + ID, ZIP Code Column제거
ploan_df = ploan.dropna().drop(['ID','ZIP Code'], axis=1, inplace=False)
ploan_df.shape

ploan_processed = ploan_df.copy()
# ploan_processed = sm.add_constant(ploan_df, has_constant='add')
# ploan_processed.shape

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


# Lasso -------------------------------------------------
lasso_model = Lasso(alpha=0.01)    # alpha(=lambda): hyper parameter
lasso_results = lasso_model.fit(train_x, train_y)

lasso_results
lasso_results.coef_

lasso_params = pd.Series(lasso_results.coef_, index=feature_columns, name='params')
lasso_params
np.exp(lasso_params).apply(lambda x: round(x,3))

lasso_pred_y = lasso_results.predict(test_x)
lasso_pred_y

# cut_off apply
lasso_pred_Y = cut_off(lasso_pred_y, 0.5)

# confusion matrix
lasso_cfmat = confusion_matrix(test_y, lasso_pred_Y)
lasso_cfmat

lasso_accuracy = acc(lasso_cfmat)
lasso_accuracy

# Cut-Off값에 따른 모형 정확도
fun_cutoff_plot(test_y, lasso_pred_y)



# sklearn ROC 패키지 이용
lasso_fpr, lasso_tpr, lasso_thresholds = metrics.roc_curve(y_true=test_y, y_score=lasso_pred_y, pos_label=1)

plt.figure()
plt.plot(lasso_fpr, lasso_tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

# auc값
auc = np.trapz(lasso_tpr, lasso_fpr)  # ROC Curve의 면적값
    # np.trapz : 합성 사다리꼴 규칙을 사용하여 주어진 축을 따라 적분합니다
auc


# Lasso -------------------------------------------------
lasso_model = Ridge(alpha=0.01)    # alpha(=lambda): hyper parameter
lasso_results = lasso_model.fit(train_x, train_y)

lasso_results
lasso_results.coef_

lasso_params = pd.Series(lasso_results.coef_, index=feature_columns, name='params')
lasso_params
np.exp(lasso_params).apply(lambda x: round(x,3))

lasso_pred_y = lasso_results.predict(test_x)
lasso_pred_y

# cut_off apply
lasso_pred_Y = cut_off(lasso_pred_y, 0.5)

# confusion matrix
lasso_cfmat = confusion_matrix(test_y, lasso_pred_Y)
lasso_cfmat

lasso_accuracy = acc(lasso_cfmat)
lasso_accuracy

# Cut-Off값에 따른 모형 정확도
fun_cutoff_plot(test_y, lasso_pred_y)



# sklearn ROC 패키지 이용
lasso_fpr, lasso_tpr, lasso_thresholds = metrics.roc_curve(y_true=test_y, y_score=lasso_pred_y, pos_label=1)

plt.figure()
plt.plot(lasso_fpr, lasso_tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

# auc값
lasso_auc = np.trapz(lasso_tpr, lasso_fpr)  # ROC Curve의 면적값
    # np.trapz : 합성 사다리꼴 규칙을 사용하여 주어진 축을 따라 적분합니다
lasso_auc



# Ridge -------------------------------------------------
ridge_model = Ridge(alpha=0.01)    # alpha(=lambda): hyper parameter
ridge_results = ridge_model.fit(train_x, train_y)

ridge_results
ridge_results.coef_

ridge_params = pd.Series(ridge_results.coef_, index=feature_columns, name='params')
ridge_params
np.exp(ridge_params).apply(lambda x: round(x,3))

ridge_pred_y = ridge_results.predict(test_x)
ridge_pred_y

# cut_off apply
ridge_pred_Y = cut_off(ridge_pred_y, 0.5)

# confusion matrix
ridge_cfmat = confusion_matrix(test_y, ridge_pred_Y)
ridge_cfmat

ridge_accuracy = acc(ridge_cfmat)
ridge_accuracy

# Cut-Off값에 따른 모형 정확도
fun_cutoff_plot(test_y, ridge_pred_y)



# sklearn ROC 패키지 이용
ridge_fpr, ridge_tpr, ridge_thresholds = metrics.roc_curve(y_true=test_y, y_score=ridge_pred_y, pos_label=1)

plt.figure()
plt.plot(ridge_fpr, ridge_tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

# auc값
ridge_auc = np.trapz(ridge_tpr, ridge_fpr)  # ROC Curve의 면적값
    # np.trapz : 합성 사다리꼴 규칙을 사용하여 주어진 축을 따라 적분합니다
ridge_auc


# Lasso와 Ridge비교 ------------------------------------------------------------
lasso_accuracy
ridge_accuracy

lasso_params
ridge_params

lasso_auc
ridge_auc


# lambda 값에 따른 회귀계수 / Accuracy 계산 ------------------------------------------------------------

# list(map(lambda x: round(x,3), np.logspace(-3,1,5)))
def fun_model_iterator(train_x, train_y, test_x, test_y, alpha=np.logspace(-3,1,5), method='lasso'):
    result_obj = {}
    data = []
    acc_table = []
    for i, a in enumerate(alpha):
        if method.lower() == 'lasso':
            model_fit = Lasso(alpha=a).fit(train_x, train_y)
        elif method.lower() == 'ridge':
            model_fit = Ridge(alpha=a).fit(train_x, train_y)
        data.append(pd.Series(np.hstack([model_fit.intercept_, model_fit.coef_])))
        pred_y = model_fit.predict(test_x) # full model
        pred_y= cut_off(pred_y,0.5)
        cfmat = confusion_matrix(test_y, pred_y)
        acc_table.append((acc(cfmat)))

    data_df = pd.DataFrame(data, index=alpha).T
    data_df.index = ['intercept'] + list(train_x.columns)
    result_obj['data'] = data_df
    acc_table_df = pd.DataFrame(acc_table, index=alpha).T
    acc_table_df.index = ['accuracy']
    result_obj['accuracy'] = acc_table_df
    return result_obj

alpha = np.logspace(-3,1,5)
lasso_result = fun_model_iterator(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, method='lasso')
ridge_result = fun_model_iterator(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, method='ridge')

lasso_result['data']
lasso_result['accuracy']

ridge_result['data']
ridge_result['accuracy']



plt.figure(figsize=(10,4))
ax1 = plt.subplot(121)
plt.semilogx(lasso_result['data'].T)
plt.title('Lasso')
plt.xticks(alpha)

ax2 = plt.subplot(122)
plt.semilogx(ridge_result['data'].T)
plt.title('Ridge')
plt.xticks(alpha)

plt.show()




