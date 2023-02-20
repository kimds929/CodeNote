import numpy as np
import pandas as pd

from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import KBinsDiscretizer

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
df_city = pd.read_csv(absolute_path + 'dataset_city.csv')
df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')


# load example dataset (Iris)
df1 = pd.read_csv(absolute_path + 'Iris.csv')
print(df1.head())

df_iris_info = DS_DF_Summary(df1)

# divide independent variables and label
X1 = df1.iloc[:, :-1].to_numpy()
y1 = df1.iloc[:, -1].to_numpy()


# [Naive_Bayse] : 분류문제에서만 사용*****
# Gaussian Naive Bayes with all continuous variables  -----------------------------------------------
# 독립변수들이 정규분포를 따른다고 가정
# 문자형변수가 섞여있을때에는 Dummy변수를 만들어서 사용
# get_ipython().run_line_magic('pinfo', 'GaussianNB')
# ?GaussianNB
    # GaussianNB(*, priors=None, var_smoothing=1e-09)
gnb = GaussianNB()
gnb.fit(X1, y1)

gnb.class_prior_    # f(y): probability of each class.
gnb.sigma_      # y값을 고정한 상태에서 x변수별 sigma
gnb.theta_      # y값을 고정한 상태에서 x변수별 theta

# make prediction and compute accuracy
y_pred = gnb.predict(X1)

acc = np.sum(y_pred == y1) / len(y1)
print(f'Accuracy: {np.around(acc, decimals=3)}')





# # CategoricalNB : Discretize all variables and apply multinomial NB   -----------------------------------------------
# 각 독립변수들이 multinomial Distribution을 따른다고 가정
# 각 숫자형 독립 변수들을 문자형 형태로 변환

# Discretize variables : 숫자값을 서열형 숫자 변수로 치환
    # ?KBinsDiscretizer
    # KBinsDiscretizer(n_bins=5, *, encode='onehot', strategy='quantile')
        # n_bins : 구간갯수
        # strategy : 구간을 나누는 방법
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
disc.fit(X1)
# dir(disc)
X_tfm = disc.transform(X1)
print(X_tfm)

# get_ipython().run_line_magic('pinfo', 'CategoricalNB')
# ?CategoricalNB
# CategoricalNB(*, alpha=1.0, fit_prior=True, class_prior=None)
cnb = CategoricalNB()
cnb.fit(X_tfm, y1)


# cnb.category_count_
# cnb.feature_log_prob_
y_pred2 = cnb.predict(X_tfm)

acc = np.sum(y_pred2 == y1) / len(y1)
print(f'Accuracy: {np.around(acc, decimals=3)}')

