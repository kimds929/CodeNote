import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import *

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


#define function to compute various measures
def classification_metrics(label, prediction):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    
    (tn, fp, fn, tp) = confusion_matrix(label, prediction).reshape(-1)
    
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)    
    return sen, spe


# load example dataset (bank)
df = pd.read_csv(absolute_path + 'Bank.csv')
X = df.loc[:,'age':'poutcome']
y_cat = df.loc[:,'y'].values
enc = OrdinalEncoder()
enc.fit(y_cat.reshape(-1,1))
y = enc.transform(y_cat.reshape(-1,1))

idx_cat = X.dtypes=='object'
idx_contin = (X.dtypes=='float64') | (X.dtypes=='int64')
X_cat  = X.loc[:,idx_cat]
X_cat_dummies = pd.get_dummies(X_cat.loc[:,])
X_contin = X.loc[:,idx_contin]
X_all = np.concatenate([X_contin.values, X_cat_dummies.values], axis=1)

X_all.shape


# ## Hold out split

from sklearn.model_selection import train_test_split
Num_obs = X.shape[0]
Num_test = 2000
Num_train = Num_obs - Num_test
idx = np.arange(Num_obs)

idx_train, idx_test = train_test_split(idx,
                                       train_size = Num_train,
                                       test_size=Num_test)

X_train= X_all[idx_train,:]
X_test= X_all[idx_test,:]
y_train = y[idx_train]
y_test = y[idx_test]

mdl_lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
mdl_lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
mdl_qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)


y_test_lr_pred = mdl_lr.predict(X_test)
y_test_lda_pred = mdl_lda.predict(X_test)
y_test_qda_pred = mdl_qda.predict(X_test)


acc_lr = accuracy_score(y_test,y_test_lr_pred)
sensitivity_lr, specificity_lr = classification_metrics(y_test, y_test_lr_pred)
f1_lr = f1_score(y_test,y_test_lr_pred)

print('Logistic Accuracy:', acc_lr)
print('Logistic Sensitivity:', sensitivity_lr)
print('Logistic Specificity:', specificity_lr)
print('Logistic f1', f1_lr)

acc_lda = accuracy_score(y_test,y_test_lda_pred)
sensitivity_lda, specificity_lda= classification_metrics(y_test, y_test_lda_pred)
f1_lda = f1_score(y_test,y_test_lda_pred)

print('LDA Accuracy:', acc_lda)
print('LDA Sensitivity:', sensitivity_lda)
print('LDA Specificity:', specificity_lda)
print('LDA f1', f1_lda)

acc_qda = accuracy_score(y_test,y_test_qda_pred)
sensitivity_qda, specificity_qda = classification_metrics(y_test, y_test_qda_pred)
f1_qda = f1_score(y_test,y_test_qda_pred)

print('QDA Accuracy:', acc_qda)
print('QDA Sensitivity:', sensitivity_qda)
print('QDA Specificity:', specificity_qda)
print('QDA f1', f1_qda)


# ## Cross Validation
from sklearn.model_selection import KFold
Num_fold=5
kf = KFold(n_splits=Num_fold, shuffle=True) 

cv_accuracy = np.zeros((Num_fold,3))
cv_f1= np.zeros((Num_fold,3))

for fold,(idx_train, idx_test) in enumerate(kf.split(X)):
    
    # train test split
    print("fold:",fold,"idx train:",idx_train, "idx test:",idx_test)
    X_train, X_valid = X_all[idx_train, :], X_all[idx_test, :]
    y_train, y_valid = y[idx_train], y[idx_test]
    
    # model learning
    mdl_lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
    mdl_lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    mdl_qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    
    # evaluation on test set
    y_test_lr_pred = mdl_lr.predict(X_test)
    y_test_lda_pred = mdl_lda.predict(X_test)
    y_test_qda_pred = mdl_qda.predict(X_test)

    acc_lr = accuracy_score(y_test,y_test_lr_pred)
    f1_lr = f1_score(y_test,y_test_lr_pred)
    cv_accuracy[fold,0] = acc_lr
    cv_f1[fold,0] = f1_lr

    acc_lda = accuracy_score(y_test,y_test_lda_pred)
    f1_lda = f1_score(y_test,y_test_lda_pred)
    cv_accuracy[fold,1] = acc_lda
    cv_f1[fold,1] = f1_lda
    
    acc_qda = accuracy_score(y_test,y_test_qda_pred)
    f1_qdq= f1_score(y_test,y_test_lr_pred)
    cv_accuracy[fold,2] = acc_qda
    cv_f1[fold,2] = f1_qda
    

print('cross validation accuracy:', cv_accuracy.mean(0))
print('cross validation f1:', cv_f1.mean(0))

