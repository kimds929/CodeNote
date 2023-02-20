import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
df_X = pd.DataFrame(iris.data)
df_y = pd.DataFrame(iris.target)

df_X.head()
df_y.head()

# Gaussian NaiveBayes ------------------------------------------------------------------------------
gnb = GaussianNB()
fit_gnb = gnb.fit(X=df_X, y=df_y, sample_weight=None)
fit_gnb.predict_proba(df_X)[[1,48,51,100]]      # 각 Y값의 Level이 나올 확률에 대해 명시

y_pred = fit_gnb.predict(df_X)      # 예측값
confusion_matrix(y_pred, df_y)      # 예측결과

    
    # Prior(가중치) 지정
gnb2 = GaussianNB(priors=[1/100, 1/100, 98/100])    # 3번째 값에 대한 가중치를 증대 → 3번째 Level로 분류할 확률이 높아짐
fit_gnb2 = gnb2.fit(X=df_X, y=df_y, sample_weight=None)
fit_gnb2.predict_proba(df_X)[[1,48,51,100]]  # 각값이 나올 확률에 대해 명시

y_pred2 = fit_gnb2.predict(df_X)      # 예측값
confusion_matrix(y_pred2, df_y)       # 3번째 Level로 분류할 확률이 높아짐


# Multinomial naive bayes -------------------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB

X = np.random.randint(5, size=(6,100))      # x의 Column: 100개, row: 6개
y = np.array([1,2,3,4,5,6])

mnb = MultinomialNB()
fit_mnb = mnb.fit(X=X, y=y)

fit_mnb.predict(X)
fit_mnb.predict_proba(X)# 각 Y값의 Level이 나올 확률에 대해 명시

    
    # Prior(가중치) 지정
mnb2 = MultinomialNB(class_prior=[0.1, 0.1999, 0.0001, 0.1, 0.1, 0.1])
fit_mnb2 = mnb2.fit(X=X, y=y)

fit_mnb2.predict(X)
fit_mnb2.predict_proba(X)# 각 Y값의 Level이 나올 확률에 대해 명시
