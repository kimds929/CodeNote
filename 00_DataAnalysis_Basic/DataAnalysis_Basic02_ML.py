import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns


# Example Data
# n = 20
# test_df = {
# 'y1': np.random.randint(100, 200, n),
# 'y2': np.random.randint(0,2, n),
# 'x1' :(np.random.random(n)*100).astype(int),
# 'x2': np.array(list(map(chr, np.random.randint(97,99,n)))),
# 'x3': np.random.randint(1,50, n),
# 'x4': np.array(list(map(lambda x: 'g'+str(x), (np.random.randint(1, 5, n))))),
# 'x5': np.random.randint(50,150, n)
# }

test_df = {'y1': [140, 183, 150, 183, 154, 168, 163, 100, 123, 194, 166, 117, 129,
            112, 143, 197, 152, 199, 102, 132],
    'y2': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    'x1': [25, 95, 47, 36, 96, 21, 57, 50, 63, 60,  9, 13, 67, 33, 71,  1, 70,
            80, 23, 42],
    'x2': ['a', 'b', 'a', 'b', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'b',
            'a', 'a', 'b', 'a', 'b', 'a', 'b'],
    'x3': [ 6,  4,  8, 41, 39,  6, 48, 45, 23,  2, 16, 23, 14, 30, 21, 27,  6,
            17,  6,  8],
    'x4': ['g2', 'g2', 'g2', 'g3', 'g4', 'g2', 'g1', 'g2', 'g3', 'g3', 'g1',
            'g3', 'g1', 'g4', 'g4', 'g2', 'g4', 'g3', 'g3', 'g2'],
    'x5': [ 88,  55,  59, 128,  77,  87, 115, 108, 124, 110, 145, 130,  56,
        75, 108,  98,  56,  52, 123,  82]
    }


df = pd.DataFrame(test_df)
df.info()
df.shape

# Example Data
# path = 'D:\Python\Dataset'
path = r'C:\Users\USER\Desktop\Python\9) Dataset'
df = pd.read_csv(path + '\wine_aroma.csv')

df.info()
df.shape




# EDA
from DataAnalysis_Module import DF_Summary, SummaryPlot
df
df_summary = DF_Summary(df)
df_summary.summary_plot()




# 【 Machine_Learning  】==========================================================
# Test Data
y = df[['y1']]
X = df[['x1', 'x3', 'x5']]

# Example Data
y = df[['Aroma']]
X = df[['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']]





# Train_Test_Split ----------------
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, test_size=0.3, random_state=1)
# X_train, X_test = train_test_split(X_scale_df, test_size=0.3, shuffle=True, random_state=1)
X_train
X_test

y_train, y_test = train_test_split(y, test_size=0.3, random_state=1)
y_train
y_test


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Normalizing  ----------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_train_scale

X_train = pd.DataFrame(X_train_scale, columns=X.columns)

# scaler.inverse_transform(X_train)

X_test_scale = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scale, columns=X.columns)






# Modeling -----------------------
from DataAnalysis_Module import sm_LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 【 sm_LinearRegression 】
# modeling
LR_sm = sm_LinearRegression()
LR_sm.fit(X_train, y_train)
LR_sm.summary()

LR_sm.features_plot(X_train, y_train)

# predict
LR_sm_pred = LR_sm.predict(X_test)
LR_sm_pred

LR_sm_test = LR_sm_pred.to_frame(name ='pred')
LR_sm_test['true'] = y_test.values
LR_sm_test

# Test-set Evaluate
mean_squared_error(y_true=y_test, y_pred=LR_sm_pred)
r2_score(y_true=y_test, y_pred=LR_sm_pred)
# R ^ 2 = 1-RSS / TSS
#   . RSS = 실제 값 (yi)과 예측 값 (yi ^) 간의 차이 제곱의 합
#   . TSS = 실제 값 (yi)과 평균 값 (회귀 적용 전) 간의 차이 제곱합.


# Remove Variables ***
LR_sm.model.pvalues[LR_sm.model.pvalues < 0.6]
X_train2 = X_train[['Mo', 'Cr', 'Sr', 'Ca']]

LR_sm2 = sm_LinearRegression()
LR_sm2.fit(X_train2, y_train)
LR_sm2.summary()

LR_sm_pred2 = LR_sm2.predict(X_test[['Mo', 'Cr', 'Sr', 'Ca']])
mean_squared_error(y_true=y_test, y_pred=LR_sm_pred2)
r2_score(y_true=y_test, y_pred=LR_sm_pred2)



# Remove Variables2 ***
X_train3 = X_train[['Mo', 'Sr', 'Ca']]

LR_sm3 = sm_LinearRegression()
LR_sm3.fit(X_train3, y_train)
LR_sm3.summary()

LR_sm_pred3 = LR_sm3.predict(X_test[['Mo', 'Sr', 'Ca']])
mean_squared_error(y_true=y_test, y_pred=LR_sm_pred3)
r2_score(y_true=y_test, y_pred=LR_sm_pred3)




# 【 Decision Tree 】
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# modeling
DT_reg = DecisionTreeRegressor()
# DT_reg = DecisionTreeRegressor(max_depth=10)
DT_reg.fit(X_train, y_train)

tree.plot_tree(DT_reg)

# predict
DT_reg_pred = DT_reg.predict(X_test)
DT_reg_pred

DT_reg_test = pd.Series(DT_reg_pred).to_frame(name='pred')
DT_reg_test['true'] = y_test.values
DT_reg_test

# Test-set Evaluate
mean_squared_error(y_true=y_test, y_pred=DT_reg_pred)
r2_score(y_true=y_test, y_pred=DT_reg_pred)

# DT_reg_train_pred = DT_reg.predict(X_train)
# r2_score(y_true=y_train, y_pred=DT_reg_train_pred)




# 【 Random Forest 】
from sklearn.ensemble import RandomForestRegressor

# modeling
RF_reg = RandomForestRegressor()
RF_reg.fit(X_train, y_train)


# predict
RF_reg_pred = RF_reg.predict(X_test)
RF_reg_pred

RF_reg_test = pd.Series(RF_reg_pred).to_frame(name='pred')
RF_reg_test['true'] = y_test.values
RF_reg_test

# Test-set Evaluate
mean_squared_error(y_true=y_test, y_pred=RF_reg_pred)
r2_score(y_true=y_test, y_pred=RF_reg_pred)


# Feature Importance
RF_reg.feature_importances_
plt.barh(X.columns, RF_reg.feature_importances_)



















# --- Regressor --------------------------------------------------

y
X

# Dataset Split -------------------------------
# from sklearn.model_selection import train_test_split

# train_X, test_X = train_test_split(X, test_size=0.2)
# train_y, test_y = train_test_split(y, test_size=0.2)


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)



# ----- Modeling -------------
# Learn
OLS_wine = sm_LinearRegression()
OLS_wine.fit(X, y)

OLS_wine.summary()


# predict
LR_OLS2_pred = OLS_wine.predict(X)

OLS2_pred_tb = LR_OLS2_pred.to_frame(name='pred')
OLS2_pred_tb['true_y'] = y

OLS2_pred_tb


# Evaluate
OLS_wine.model.rsquared
OLS_wine.model.rsquared_adj
OLS_wine.model.mse_resid

OLS_wine.model.pvalues.plot.barh()
plt.axvline(0.3, color='red')
plt.show()


# Remove 'Ba', 'Pb', 'B', 'Mg', 'K'
X = df[['Mo', 'Cr', 'Sr', 'Ca']]
# Learn
OLS_wine2 = sm_LinearRegression()
OLS_wine2.fit(X, y)

OLS_wine2.summary()


# Remove 'Cr'
X = df[['Mo', 'Sr', 'Ca']]
# Learn
OLS_wine3 = sm_LinearRegression()
OLS_wine3.fit(X, y)

OLS_wine3.summary()









# --- Classifier --------------------------------------------------
path = r'C:\Users\USER\Desktop\Python\9) Dataset'
df = pd.read_csv(path + '\Titanic.csv')
df_summary = DF_Summary(df)
df_summary.summary









# ----- Modeling -------------











# 【 Decision Tree 】==========================================================



from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import r2_score, accuracy_score


# < Regressor >
y = df[['y']]
X = df[['x1', 'x3']]

# modeling
DT_reg = DecisionTreeRegressor().fit(X, y)

# predict & evaluate
DT_reg_pred = DT_reg.predict(X)
r2_score(y_true=y, y_pred=DT_reg_pred)

# visualization
plt.figure(figsize=(10,10))
tree.plot_tree(DT_reg)




# < Classifier >
y = df[['x2']]
X = df[['x1', 'x3']]

# modeling
DT_clf = DecisionTreeClassifier().fit(X, y)
DT_clf

# predict & evaluate
DT_clf_pred = DT_clf.predict(X)
accuracy_score(y_true=y, y_pred=DT_clf_pred)

# visualization
plt.figure(figsize=(10,10))
tree.plot_tree(DT_clf)





# ===== Real Data Practice ==============================================
from DataAnalysis_Module import DF_Summary, SummaryPlot

# Example Data
# path = 'D:\Python\Dataset'
path = r'C:\Users\USER\Desktop\Python\9) Dataset'
df = pd.read_csv(path + '\wine_aroma.csv')
df.info()

df_summary = DF_Summary(df)
df_summary.summary

df_summary.summary_plot()

y = df[['Aroma']]
X = df[['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']]

y
X


# ----- Modeling -------------
DT_wine = DecisionTreeRegressor().fit(X, y)

# predict & evaluate
DT_wine_pred = DT_wine.predict(X)
DT_pred_tb = pd.DataFrame(DT_wine_pred, columns=['pred'])
DT_pred_tb['true'] = y
DT_pred_tb

r2_score(y_true=y, y_pred=DT_wine_pred)


# visualization
plt.figure(figsize=(10,10))
tree.plot_tree(DT_wine)







# 【 Random Forest 】==========================================================
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

RF_reg = RandomForestRegressor().fit(X, y)

RF_reg.feature_importances_
plt.barh(X.columns, RF_reg.feature_importances_)



