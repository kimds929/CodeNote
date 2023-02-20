import numpy as np
import pandas as pd

from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import *
from sklearn.preprocessing import *

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
X = df.loc[:,'radius_mean':]
X_np = X.to_numpy()

y = df.loc[:,'result'].to_frame()
y_np = y.to_numpy()

# 문자형 binery y값 → 0, 1
enc = OrdinalEncoder()
enc.fit(y_np)
y_enc = enc.transform(y_np)

X
y

# ## Model learning

# get_ipython().run_line_magic('pinfo', 'LogisticRegression')
# ?LogisticRegression
# LogisticRegression(
#     penalty='l2',     # (penalty-term) l1 : Lasso(|b|), l2: ridge(b**2), elasticnet : |b| + b**2
#     *,
#     dual=False,
#     tol=0.0001,
#     C=1.0,            # regulerization parameter : lambda inverse (1 / lambda)
#     fit_intercept=True,
#     intercept_scaling=1,
#     class_weight=None,
#     random_state=None,
#     solver='lbfgs',
#     max_iter=100,
#     multi_class='auto',
#     verbose=0,
#     warm_start=False,
#     n_jobs=None,
#     l1_ratio=None,
# )
# mdl = LogisticRegression(random_state=0).fit(X, y)
mdl = LogisticRegression(max_iter=10000).fit(X, y)
    # max_iter에서 warning message가 나오지 않는 숫자로 충분히 크게 해주어야 함
mdl.classes_

mdl.coef_   # 로지스틱 회귀의 계수 확인
mdl.intercept_  # 로지스틱 회귀의 절편
mdl.predict_proba(X)    # 각 변수별 확률
dir(mdl)

y_prob = mdl.predict_proba(X)   # 클래스에 대한 예측 확률 [0,1]
y_pred = mdl.predict(X)           # 예측된 label

y_prob
y_pred


# ## Evaluation
ctbl = confusion_matrix(y, y_pred)
print(ctbl)

acc = accuracy_score(y,y_pred)
print('accuracy:', acc)


#define function to compute various measures
def classification_metrics(label, prediction):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    
    (tn, fp, fn, tp) = confusion_matrix(label, prediction).reshape(-1)
    
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)    
    return sen, spe

y
y_pred



sensitivity, specificity = classification_metrics(y, y_pred)
precision = precision_score(y,y_pred)
recall = recall_score(y,y_pred) # sensitivity=recall)
auc = roc_auc_score(y,y_prob[:,1])
f1 = f1_score(y,y_pred)
print('Sensitivity:', sensitivity)
print('recall:', recall)
print('Specificity:', specificity)
print('Precision(=TP/(TP+FP)):', precision)
print('AUC:',auc)
print('f1:',f1)


sensitivity, specificity, auc = classification_metrics(y, y_pred)
print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('AUC:', auc)



# Logistic Regression Cross-validation -------------------------------------------------------------
# ?LogisticRegressionCV
# LogisticRegressionCV(
#     *,
#     Cs=10,                # ** regularization 후보군 (lambda inverse)
                            # 정수: 정수갯수(자동으로 부여)
                            # list: 후보군을 지정
#     fit_intercept=True,
#     cv=None,
#     dual=False,
#     penalty='l2',         # ** regularization-term
#     scoring=None,         # ** evaluation measure로 어떤걸 써줄껀지?
#     solver='lbfgs',
#     tol=0.0001,
#     max_iter=100,
#     class_weight=None,
#     n_jobs=None,
#     verbose=0,
#     refit=True,
#     intercept_scaling=1.0,
#     multi_class='auto',
#     random_state=None,
#     l1_ratios=None,
# )


from sklearn.model_selection import *
kf = KFold(n_splits=5)
mdl_cv = LogisticRegressionCV(max_iter=10000, cv=kf, penalty='l1', solver='liblinear').fit(X,y)
dir(mdl_cv)
mdl_cv.coef_
mdl_cv.coefs_paths_

mdl_cv.C_   # 최적의 C값
mdl_cv.Cs   # 고려한 C값의 갯수
mdl_cv.Cs_  # 고려한 C-inverse값들 (1/C)
# (목적) LogisticCV.C_


mdl_cv2 = LogisticRegressionCV(max_iter=10000, cv=None, random_state=1).fit(X,y)
mdl_cv2.C_
mdl_cv2.Cs
1/mdl_cv2.Cs_
mdl_cv2.coef_

mdl2 = LogisticRegression(max_iter=10000, C=mdl_cv2.C_[0], random_state=1).fit(X,y)
mdl2.coef_
mdl2.C


# ## Bank dataset

# load example dataset (bank)
df2 = pd.read_csv(absolute_path + 'Bank.csv')
print(df2.head())
df2_info = DS_DF_Summary(df2)

df2.dtypes

X_bk = df2.loc[:,'age':'poutcome']
y_bk = df2.loc[:,'y'].to_frame()


X_bk_np = X_bk.to_numpy()
y_bk_np = y_bk.to_numpy()
# y_bk_cat = df2.loc[:,'y'].values

enc_bk = OrdinalEncoder()
enc_bk.fit(y_bk_np)
y_bk_enc = enc_bk.transform(y_bk_np)

X_bk_obj, X_bk_num = fun_object_numeric_split(X_bk)             # DS_DataFrame
X_bk_obj_dummies = pd.get_dummies(X_bk_obj)

# idx_cat = X_bk.dtypes=='object'
# idx_contin = (X_bk.dtypes=='float64') | (X_bk.dtypes=='int64')
# X_bk_cat  = X_bk.loc[:,idx_cat]
# X_bk_cat_dummies = pd.get_dummies(X_bk_cat.loc[:,])
# X_bk_contin = X_bk.loc[:,idx_contin]
# X_bk_all = np.concatenate([X_bk_contin.values, X_bk_cat_dummies.values], axis=1)

X_bk_all = pd.concat([X_bk_obj_dummies, X_bk_num], axis=1)
X_bk.shape
X_bk_all.shape

mdl_bk = LogisticRegression(max_iter=500).fit(X_bk_all, y_bk)

y_bk_pred= mdl_bk.predict(X_bk_all)
y_bk_prob= mdl_bk.predict_proba(X_bk_all)

plot_confusion_matrix(mdl_bk, X_bk_all, y_bk)


# sensitivity, specificity = classification_metrics(y_bk, y_bk_pred)
# precision = precision_score(y_bk,y_bk_pred)
# recall = recall_score(y_bk,y_bk_pred) # sensitivity=recall)
# auc = roc_auc_score(y_bk,y_bk_prob[:,1])
# f1 = f1_score(y_bk,y_bk_pred)
# print('Accuracy:', accuracy_score(y_bk,y_bk_pred))
# print('Sensitivity:', sensitivity)
# print('recall:', recall)
# print('Specificity:', specificity)
# print('Precision(=TP/(TP+FP)):', precision)
# print('AUC:',auc)
# print('f1:',f1)










# ### Undersampling 
idx_0 = np.where(y_bk == 0)[0]
idx_1 = np.where(y_bk == 1)[0]
idx_0_under = np.random.choice(idx_0, size=len(idx_1), replace=True)
# concatenate indexes
idx_undersample = np.concatenate((idx_0_under, idx_1))


X_bk_us = X_bk_all[idx_undersample, :]
y_bk_us = y_bk[idx_undersample]



mdl_bk_us= LogisticRegression(max_iter=500).fit(X_bk_us, y_bk_us)

y_bk_us_pred = mdl_bk_us.predict(X_bk_all)
y_bk_us_prob = mdl_bk_us.predict_proba(X_bk_all)


ctbl_bk_us = confusion_matrix(y_bk, y_bk_us_pred)
print(ctbl_bk_us)


plot_confusion_matrix(mdl_bk_us, X_bk_all,y_bk)


sensitivity, specificity = classification_metrics(y_bk, y_bk_us_pred)
precision = precision_score(y_bk,y_bk_us_pred)
recall = recall_score(y_bk,y_bk_us_pred) # sensitivity=recall)
auc = roc_auc_score(y_bk,y_bk_us_prob[:,1])
f1 = f1_score(y_bk,y_bk_us_pred)
print('Accuracy:', accuracy_score(y_bk,y_bk_us_pred))
print('Sensitivity:', sensitivity)
print('recall:', recall)
print('Specificity:', specificity)
print('Precision(=TP/(TP+FP)):', precision)
print('AUC:',auc)
print('f1:',f1)

