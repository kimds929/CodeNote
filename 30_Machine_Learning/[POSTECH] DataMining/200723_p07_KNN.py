import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

# 독립변수의 갯수가 많아지면 KNN의 성능이 잘 안나옴: 차원이 증가하여 Distance가 너무 커짐

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
df_city = pd.read_csv(absolute_path + 'dataset_city.csv')
df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')

# load example dataset (Titanic)
df = pd.read_csv(absolute_path + 'Titanic.csv')
print(df.head())
titanic_info = DS_DF_Summary(df)
# Pclass: Ticket class
# SibSp: # of siblings and spouses aboard
# Parch: # of parents and children aboard
# Embarked: Port of Embarked (C=Cherbourg, Q=Queenstown, S=Southhampton)



# pre-processing before applying KNN

# 1. drop nominal variables
df = df.drop(columns='Embarked')

# 2. transform binary variables into numeric values
# declared LabelEncoder and transform bianry variable
le = LabelEncoder()
sex_tfm = le.fit_transform(df['Sex'])       # 'Sex' 변수에 대해서 문자 → 숫자로 변환
print(le.classes_)

# replace original column with transformed column
df['Sex'] = sex_tfm

# check the dataframe
print(df.head())

# divide independent variables and label
'int' in  df.dtypes.values
df.loc[:,('int' in  df.dtypes.values)]
titanic_info = DS_DF_Summary(df)
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

df.describe()

# declare classifier and train the model
# ?KNeighborsClassifier
# get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')
# KNeighborsClassifier(
#     n_neighbors=5,        # Nearest Neighbors K
#     *,
#     weights='uniform',    # uniform : Neighbor 동일한 가중치 / distance : 거리에 반비례하는 가중치 부여
#     algorithm='auto',
#       # - 'ball_tree' will use :class:`BallTree`
#       # - 'kd_tree' will use :class:`KDTree`
#       # - 'brute' will use a brute-force search.
#       # - 'auto' will attempt to decide the most appropriate algorithm
#       #      based on the values passed to :meth:`fit` method.
#     leaf_size=30,
#     p=2,
#     metric='minkowski',       # distance 방법
#     metric_params=None,
#     n_jobs=None,
#     **kwargs,
# )
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X, y)

# make prediction
y_pred = knn.predict(X)
y_pred


# compute accuracy
acc = np.sum(y == y_pred) / len(y)
print(f'Accuracy: {np.around(acc, decimals=3)}')

    # x값을 scaler를 통해 Normalize (StandardScaler)
X_original = np.copy(X)
scaler = StandardScaler()
scaler.fit(X_original)

X_scale = scaler.transform(X_original)

knn_std = KNeighborsClassifier(n_neighbors=100)
knn_std.fit(X_scale, y)

y_pred_std = knn_std.predict(X_scale)

acc_std = np.sum(y == y_pred_std) / len(y)
print(f'Accuracy_std: {np.around(acc_std, decimals=3)}')
