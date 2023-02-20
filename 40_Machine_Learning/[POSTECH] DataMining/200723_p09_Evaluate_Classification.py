import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
# df_city = pd.read_csv(absolute_path + 'dataset_city.csv')
# df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')


# load example dataset (Titanic)
df = pd.read_csv(absolute_path + 'Titanic.csv')
print(df.head())


# pre-processing (drop "embarked" and transform "sex")
df.drop(columns='Embarked', inplace=True)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df.head()

titanic_info = DS_DF_Summary(df)


# devide independent variables and label
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()


# declare classifier object and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# make predictions
y_pred = knn.predict(X)


# compute evaluation measures
tp = np.sum(np.logical_and(y == 1, y_pred == 1))
tn = np.sum(np.logical_and(y == 0, y_pred == 0))
fp = np.sum(np.logical_and(y == 0, y_pred == 1))
fn = np.sum(np.logical_and(y == 1, y_pred == 0))

accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f'Accuracy: {np.around(accuracy, decimals=3)}')
print(f'Error rate: {np.around(error_rate, decimals=3)}')
print(f'Sensitivity: {np.around(sensitivity, decimals=3)}')
print(f'Specificity: {np.around(specificity, decimals=3)}')

# ?confusion_matrix
# get_ipython().run_line_magic('pinfo', 'confusion_matrix')
    # confusion_matrix(
    #     y_true,
    #     y_pred,
    #     *,
    #     labels=None,
    #     sample_weight=None,
    #     normalize=None,
    # )


# using confusion matrix
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

(tn, fp, fn, tp) = conf_mat.reshape(-1)
print(tn, fp, fn ,fp)

# plot
plot_confusion_matrix(knn, X,y)
