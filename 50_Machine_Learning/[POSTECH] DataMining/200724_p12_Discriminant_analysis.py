
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import *
from sklearn.preprocessing import *

from sklearn.naive_bayes import *
from sklearn.neighbors import *

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# ## Import dataset
# load dataset
df = pd.read_csv(absolute_path + 'breast_cancer.csv')
print(df.head())
df_info = DS_DF_Summary(df)

# Extract training set
X = df.loc[:,'radius_mean':].values
y_cat = df.loc[:,'result'].values
# 문자형 binery y값 → 0, 1
enc = OrdinalEncoder()
enc.fit(y_cat.reshape(-1,1))
y = enc.transform(y_cat.reshape(-1,1)).ravel()

X
y


# ## Model learning
# 모형 선언 후 학습
# ?LinearDiscriminantAnalysis
# LinearDiscriminantAnalysis(
#     *,
#     solver='svd',
#     shrinkage=None,
#     priors=None,
#     n_components=None,        # n_components*** 
#     store_covariance=False,
#     tol=0.0001,
# )
mdl_lda = LinearDiscriminantAnalysis().fit(X, y)
# ?QuadraticDiscriminantAnalysis
# QuadraticDiscriminantAnalysis(
#     *,
#     priors=None,
#     reg_param=0.0,
#     store_covariance=False,   # 공분산(covariance)를 나중에 확인하고 싶으면 True로 해야함
#     tol=0.0001,
# )
mdl_qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)

# 학습된 모형의 계수를 살펴보기
print('LDA weight vector:',mdl_lda.coef_)
print('LDA intercept:',mdl_lda.intercept_)
mdl_lda.predict_proba(X)        # 각 class별 확률값
mdl_lda.predict_log_proba(X)        # 각 class별 log 확률값

print('QDA class별 평균:',mdl_qda.means_)
# mdl_qda.covariance_ # 공분산 matrix


y_lda_prob = mdl_lda.predict_proba(X) # 각 class별 확률값
# np.argmax(y_lda_prob, axis=1)
y_lda_pred = mdl_lda.predict(X)

y_qda_prob = mdl_qda.predict_proba(X) # 각 class별 확률값
y_qda_pred = mdl_qda.predict(X)


# 각 클래스별 사후 확률 (각 class별 확률값)
print(y_lda_prob)
print(y_qda_prob)


# ## Evaluation
ctbl_lda = confusion_matrix(y.to_numpy(), y_lda_pred)
ctbl_qda = confusion_matrix(y, y_qda_pred)
print('LDA confusion matrix:')
print(ctbl_lda)
print('QDA confusion matrix:')
print(ctbl_qda)

acc_lda = accuracy_score(y,y_lda_pred)
print('LDA accuracy:', acc_lda)
acc_qda = accuracy_score(y,y_qda_pred)
print('QDA accuracy:', acc_qda)








#define function to compute various measures
def classification_metrics(label, prediction):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    
    (tn, fp, fn, tp) = confusion_matrix(label, prediction).reshape(-1)
    
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)    
    return sen, spe

sensitivity_lda, specificity_lda = classification_metrics(y, y_lda_pred)
auc_lda = roc_auc_score(y,y_lda_prob[:,1])

print('LDA Sensitivity:', sensitivity_lda)
print('LDA Specificity:', specificity_lda)
print('LDA AUC:', auc_lda)


sensitivity_qda, specificity_qda = classification_metrics(y, y_qda_pred)
auc_qda = roc_auc_score(y,y_qda_prob[:,1])

print('QDA Sensitivity:', sensitivity_qda)
print('QDA Specificity:', specificity_qda)
print('QDA AUC:', auc_qda)









# ------------------------------------------------------------------------

# ## 실습: Breast Cancer dataset
# 1) imbalnace 확인
# 2) KNN과 Naive Bayes 사용
# 3) Undrer vs oversampling 사용시 accuracy, AUC와 F1 score 비교
# 4) Train/test set split 10회 반복
# => 총 6가지 모형: KNN (기본, with US, with OS), Naive Bayes (기본, with US, with OS)


    # 평가모델
# recall : TP / (TP + FN)
# Precision : TP/(TP + FP)
# F1 = 2 * (precision * recall) / (precision + recall)

def accuracy_matrix(confusion_matrix):
    (tn, fp, fn, tp) = confusion_matrix.reshape(-1)
    return (tp + tn) / (tp + tn + fp + fn)

def recall_matrix(confusion_matrix):
    (tn, fp, fn, tp) = confusion_matrix.reshape(-1)
    return tp  / (tp + fn)

def precision_matrix(confusion_matrix):
    (tn, fp, fn, tp) = confusion_matrix.reshape(-1)
    return tp  / (tp + fp)

def f1_score_matrix(confusion_matrix):
    recall = recall_matrix(confusion_matrix)
    precision = precision_matrix(confusion_matrix)
    return 2 * (precision * recall) / (precision + recall)

# def evaluate_classfication_matrix(confusion_matrix):
#     result={}
#     result['confusion_matrix'] = confusion_matrix
#     # sensitivity = tp / (tp + fn)
#     # specificity = tn / (tn + fp)
#     # error_rate = 1 - accuracy
#     result['tn'], result['fp'], result['fn'], result['tp'] = confusion_matrix.reshape(-1)
#     result['accuracy'] = recall_matrix(confusion_matrix)
#     result['precision'] = precision_matrix(confusion_matrix)
#     result['f1_score'] = f1_score_matrix(confusion_matrix)
#     return result


bc = pd.read_csv(absolute_path + 'breast_cancer.csv')
bc_info = DS_DF_Summary(bc)

gnb_accuracy = []
knn_accuracy = []
lda_accuracy = []
qda_accuracy = []

result = {}
result['nb_f1'] = []
result['knn_f1'] = []
result['lda_f1'] = []
result['qda_f1'] = []
result['log_f1'] = []
result['logl1_f1'] = []
result['logl2_f1'] = []

for i in range(10):
    print(i, 'times')
    bc_train, bc_test = train_test_split(bc, test_size=0.3, random_state=i)

    bc_train_X = bc_train.loc[:,'radius_mean':]
    bc_train_y = bc_train.loc[:,'result']
    bc_test_X = bc_test.loc[:,'radius_mean':]
    bc_test_y = bc_test.loc[:,'result']

    # scale normalize
    bc_scaler_train = StandardScaler().fit(bc_train_X)
    bc_train_X_scale = bc_scaler_train.transform(bc_train_X)
    bc_test_X_scale = bc_scaler_train.transform(bc_test_X)

    # Naive_Base 학습
    gnb = GaussianNB()
    gnb.fit(bc_train_X, bc_train_y)

    y_pred_test_bc = gnb.predict(bc_test_X)

    conf_mat_nb = confusion_matrix(bc_test_y, y_pred_test_bc)
    # print(conf_mat_bc)
    
    result['nb_f1'].append(f1_score_matrix(conf_mat_nb))    # f1
    # gnb_accuracy.append(accuracy(conf_mat_nb))


    # KNN
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(bc_train_X_scale, bc_train_y)

    y_pred_test_knn = knn.predict(bc_test_X_scale)
    conf_mat_knn = confusion_matrix(bc_test_y, y_pred_test_knn)
    # print(conf_mat_knn)
    result['knn_f1'].append(f1_score_matrix(conf_mat_knn))    # f1
    # knn_accuracy.append(accuracy_matrix(conf_mat_knn))



    # LDA
    lda_model = LinearDiscriminantAnalysis().fit(bc_train_X, bc_train_y)
    y_pred_test_lda = lda_model.predict(bc_test_X)

    conf_mat_lda = confusion_matrix(bc_test_y, y_pred_test_lda)
    # print(conf_mat_lda)
    result['lda_f1'].append(f1_score_matrix(conf_mat_lda))    # f1
    # accuracy_matrix(conf_mat_lda)
    # lda_accuracy.append(accuracy_matrix(conf_mat_lda))



    # QDA
    qda_model = QuadraticDiscriminantAnalysis().fit(bc_train_X, bc_train_y)
    y_pred_test_qda = qda_model.predict(bc_test_X)

    conf_mat_qda = confusion_matrix(bc_test_y, y_pred_test_qda)
    # print(conf_mat_qda)
    result['qda_f1'].append(f1_score_matrix(conf_mat_qda))    # f1
    # accuracy_matrix(conf_mat_qda)
    # qda_accuracy.append(accuracy_matrix(conf_mat_qda))


    # Logistic Regression
        # Logistic
    log_model = LogisticRegression(max_iter=10000, penalty='none', random_state=1).fit(bc_train_X_scale, bc_train_y)
    y_pred_log = log_model.predict(bc_test_X_scale)

    conf_mat_log = confusion_matrix(bc_test_y, y_pred_log)
    # print(conf_mat_log)
    result['log_f1'].append(f1_score_matrix(conf_mat_log))    # f1


        # Logistic Lasso CV
    log_l1CV_model = LogisticRegression(max_iter=10000, penalty='l1', solver='liblinear', random_state=1).fit(bc_train_X_scale, bc_train_y)
    y_pred_logl1CV = log_l1CV_model.predict(bc_test_X_scale)

    conf_mat_logl1CV = confusion_matrix(bc_test_y, y_pred_logl1CV)
    # print(conf_mat_logl1CV)
    result['logl1_f1'].append(f1_score_matrix(conf_mat_logl1CV))    # f1


        # Logistic Ridge CV
    log_l2CV_model = LogisticRegression(max_iter=10000, penalty='l2', random_state=1).fit(bc_train_X_scale, bc_train_y)
    y_pred_logl2CV = log_l2CV_model.predict(bc_test_X_scale)

    conf_mat_logl2CV = confusion_matrix(bc_test_y, y_pred_logl2CV)
    # print(conf_mat_logl2CV)
    result['logl2_f1'].append(f1_score_matrix(conf_mat_logl2CV))    # f1



# result_df = pd.DataFrame([gnb_accuracy, knn_accuracy, lda_accuracy, qda_accuracy],
#                  index=['gnb','knn','lda','qda']).T
f1_result = pd.DataFrame(result)
f1_result
f1_result.mean(0)
plt.boxplot(f1_result.T, labels=f1_result.columns)

conf_mat_logl2CV