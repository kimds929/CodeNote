import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import *
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
import matplotlib.pyplot as plt

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *


# load example dataset (bank)
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
# df_city = pd.read_csv(absolute_path + 'dataset_city.csv')
# df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')

df = pd.read_csv(absolute_path + 'Bank.csv')
print(df.head())

bank_info = DS_DF_Summary(df)


# pre-processing
# transform continuous variables to categorical variables
numeric_columns = df.dtypes != 'object'

# ?KBinsDiscretizer
# KBinsDiscretizer(n_bins=5, *, encode='onehot', strategy='quantile')
    # n_bins : 구간갯수
    # strategy : 구간을 나누는 방법

# ?OrdinalEncoder
# OrdinalEncoder(*, categories='auto', dtype=<class 'numpy.float64'>)
discretizer = ('Discretizer', 
              KBinsDiscretizer(encode='ordinal', strategy='kmeans'), 
              numeric_columns)
labeling = ('Labeling', 
           OrdinalEncoder(), 
           ~numeric_columns)

ct = ColumnTransformer(transformers=[discretizer, labeling], 
                      remainder='passthrough')
data_tfm = ct.fit_transform(df)
print(data_tfm)

bank_tfm = DS_DF_Summary(pd.DataFrame(data_tfm, columns=df.columns))


# divide independent variables and label
X = data_tfm[:, :-1]
y = data_tfm[:, -1]



# declare NaiveBayes and train the model
cnb = CategoricalNB()
cnb.fit(X, y)

# predict label and compute accuracy
y_pred = cnb.predict(X)
# decision_function
y_proba = cnb.predict_proba(X)      # 각 class에 대한 확률값

acc = np.sum(y_pred == y) / len(y)
print(f'Accuracy: {np.around(acc, decimals=3)}')



# compute imbalance of label distribution
print('The number of instances:', len(y))
print('The number of positive instances:', np.sum(y == 1))
print('The number of negative instances:', np.sum(y == 0))

# trivial classifier
print('=' * 30)
y_pred_triv = np.zeros(len(y))

acc_triv = np.sum(y_pred_triv == y) / len(y)
print(f'Trivial accuracy: {np.around(acc_triv, decimals=3)}')

# compute label distribution of NB classifier
print('The number of instances:', len(y))
print('The number of positive predictions:', np.sum(y_pred == 1))
print('The number of negative predictions:', np.sum(y_pred == 0))


ctbl = confusion_matrix(y,y_pred)
print(ctbl)
(tn, fp, fn, tp) = confusion_matrix(y,y_pred).reshape(-1)
print('tp:',tp, 'fn:',fn)
print('fp:',fp, 'tn:',tn)

plot_confusion_matrix(cnb, X,y)
plt.title('Confusion_matrix')


plot_roc_curve(cnb, X,y)
plt.title('ROC curve)')


#define function to compute various measures
def classification_metrics(label, prediction):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    
    (tn, fp, fn, tp) = confusion_matrix(label, prediction).reshape(-1)
    
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)    
    return sen, spe


sensitivity, specificity = classification_metrics(y, y_pred)
precision = precision_score(y,y_pred)
recall = recall_score(y,y_pred) # sensitivity=recall)
# ?roc_auc_score
# roc_auc_score(
#     y_true,
#     y_score,
#     *,
#     average='macro',
#     sample_weight=None,
#     max_fpr=None,
#     multi_class='raise',
#     labels=None,
# )
auc = roc_auc_score(y,y_proba[:,1])
f1 = f1_score(y,y_pred)
print('NB with original (not sampling)')
print('Accuracy:', acc_triv)
print('Sensitivity:', sensitivity)
print('recall:', recall)
print('Specificity:', specificity)
print('Precision(=TP/(TP+FP)):', precision)
print('AUC:',auc)
print('f1(=2*precision*recall/(precision+recall)):',f1)







# # Sampling-based solution
# [ Libarary ] imbalanced-learn : https://imbalanced-learn.readthedocs.io/en/stable/install.html
# (install) conda install -c conda-forge imbalanced-learn

# ## Over-sampling
# over-sampling --------------------------------------------------------------
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]

idx_1_over = np.random.choice(idx_1, size=len(idx_0), replace=True)

print(idx_1_over.shape)
print(idx_1_over.tolist())


# concatenate indexes
idx_oversample = np.concatenate((idx_0, idx_1_over))
X_os = X[idx_oversample, :]
y_os = y[idx_oversample]

# train the model with over-sampling
cnb_os = CategoricalNB()
cnb_os.fit(X_os, y_os)


# make prediction and compute classification metrics
y_pred_os = cnb_os.predict(X)
y_proba_os= cnb_os.predict_proba(X)


print('NB with oversampling')
ctbl_os = confusion_matrix(y,y_pred_os)
print(ctbl_os)
(tn_os, fp_os, fn_os, tp_os) = confusion_matrix(y,y_pred_os).reshape(-1)
print('tp:',tp_os, 'fn:',fn_os)
print('fp:',fp_os, 'tn:',tn_os)


    # original
plot_confusion_matrix(cnb, X,y)
plt.title('Confusion_matrix')
    # oversampling
plot_confusion_matrix(cnb_os, X,y)
plt.title('Confusion_matrix (oversampling)')



acc_os = np.sum(y_pred_os == y) / len(y)
sensitivity_os, specificity_os = classification_metrics(y, y_pred_os)
precision_os = precision_score(y,y_pred_os)
recall_os = recall_score(y,y_pred_os) # sensitivity=recall)
auc_os = roc_auc_score(y,y_proba_os[:,1])
f1_os = f1_score(y,y_pred_os)

print('NB with oversampling')
print('Accuracy:', acc_os)
print('Sensitivity(=recall):', sensitivity_os)
print('Specificity:', specificity_os)
print('Precision:', precision_os)
print('AUC:',auc_os)
print('f1:',f1_os)






# ## Under-sampling --------------------------------------------------------------

# under-sampling
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]

idx_0_under = np.random.choice(idx_0, size=len(idx_1), replace=True)

# concatenate indexes
idx_undersample = np.concatenate((idx_0_under, idx_1))


# train the model with under-sampling
X_us = X[idx_undersample, :]
y_us = y[idx_undersample]

cnb_us = CategoricalNB()
cnb_us.fit(X_us, y_us)








# ## 실습: Breast Cancer dataset
# 1) imbalnace 확인
# 2) KNN과 Naive Bayes 사용
# 3) Undrer vs oversampling 사용시 accuracy, AUC와 F1 score 비교
# 4) Train/test set split 10회 반복
# => 총 6가지 모형: KNN (기본, with US, with OS), Naive Bayes (기본, with US, with OS)

def accuracy(confusion_matrix):
    (tn, fp, fn, tp) = confusion_matrix.reshape(-1)
    return (tp + tn) / (tp + tn + fp + fn)


bc = pd.read_csv(absolute_path + 'breast_cancer.csv')
bc_info = DS_DF_Summary(bc)


gnb_accuracy = []
knn_accuracy = []
for i in range(10):
    print(i, 'times')
    bc_train, bc_test = train_test_split(bc, test_size=0.3, random_state=i)

    bc_train_X = bc_train.loc[:,'radius_mean':]
    bc_train_y = bc_train.loc[:,'result']
    bc_test_X = bc_test.loc[:,'radius_mean':]
    bc_test_y = bc_test.loc[:,'result']


    # Naive_Base 학습
    gnb = GaussianNB()
    gnb.fit(bc_train_X, bc_train_y)

    y_pred_test_bc = gnb.predict(bc_test_X)

    conf_mat_bc = confusion_matrix(bc_test_y, y_pred_test_bc)
    # print(conf_mat_bc)
    gnb_accuracy.append(accuracy(conf_mat_bc))


    # KNN
    bc_scaler_train = StandardScaler().fit(bc_train_X)
    bc_train_X_scale = bc_scaler.transform(bc_train_X)
    bc_test_X_scale = bc_scaler.transform(bc_test_X)

    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(bc_train_X_scale, bc_train_y)

    y_pred_test_knn = knn.predict(bc_test_X_scale)
    conf_mat_knn = confusion_matrix(bc_test_y, y_pred_test_knn)
    # print(conf_mat_knn)
    accuracy(conf_mat_knn)
    knn_accuracy.append(accuracy(conf_mat_knn))

gnb_accuracy
knn_accuracy
result = pd.DataFrame([gnb_accuracy, knn_accuracy], index=['gnb','knn']).T
result.mean(0)
plt.boxplot(result.T, labels=result.columns)

