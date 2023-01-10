import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/gothic.ttf").get_name()
rc('font', family=font_name)


# 【 Data Load 】 ###################################################
# Example Data
# n = 20
# sample_data_dict = {
# 'y1': np.random.randint(100, 200, n),
# 'y2': (np.random.rand(n) >0.5).astype(int),
# 'x1' :(np.random.random(n)*100).astype(int),
# 'x2': np.array(list(map(chr, np.random.randint(97,99,n)))),
# 'x3': np.random.randint(1,50, n),
# 'x4': np.array(list(map(lambda x: 'g'+str(x), (np.random.randint(1, 5, n))))),
# 'x5': np.random.randint(50,150, n)
# }
# 
# sample_data_dict = {'y1': [140, 183, 150, 183, 154, 168, 163, 100, 123, 194, 166, 117, 129,
#             112, 143, 197, 152, 199, 102, 132],
#     'x1': [25, 95, 47, 36, 205, 21, 57, np.nan, 63, 60,  9, 13, 67, 33, 71,  1, 70,
#             80, 23, 42],
#     'x2': ['a', 'b', 'a', 'b', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'b',
#             'a', 'a', 'b', 'a', 'b', 'a', 'b'],
#     'x3': [ 2, 12, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, np.nan],
#     'x4': [ 6,  4,  8, 41, 39,  6, 48, 45, 23,  2, 16, 23, np.nan, 155, 21, 27,  6,
#             17,  6,  8],
#     'x5': ['g2', 'g2', 'g2', 'g3', 'g4', 'g2', 'g1', 'g2', 'g3', 'g3', 'g1',
#             'g3', 'g1', 'g4', 'g4', 'g2', 'g4', 'g3', 'g3', 'g2'],
#     'x6': [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
#            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#     'x7': [ 88,  55,  59, 128,  77,  87, 115, 108, 124, 110, 145, 130,  56,
#         75, 108,  98,  56,  52, 123,  60]
#     }
# df = pd.DataFrame(sample_data_dict)
# df.to_csv('test_data5(ML).csv', index=False, encoding='utf-8-sig')


data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/00_DataAnalysis_Basic/'
df = pd.read_csv(data_url + "test_data5(ML).csv")
# df = pd.read_clipboard(sep='\t')      # clipboard에서 불러오기


# data 탐색
df

df.shape
df.head()

df.info()

df.dtypes
df.describe()           # 숫자데이터 요약


######################################################################################################
##### 【 Data 전처리 】 ###############################################################################
######################################################################################################

# 【 결측치 처리 】 ###################################################
# 결측치 확인
df.isna()
df.isna().sum(0)                         # 결측치 데이터의 갯수
len(df) - df.isna().sum(0)               # 결측치가 아닌데이터의 갯수

# conda install -c conda-forge missingno
import missingno as msno
msno.matrix(df)


# 결측치 처리 : 삭제
df.dropna()             # 전체 삭제 (row마다 하나라도 결측치가 있는경우 삭세)
df['x3'].dropna()

df['x1']
df['x1'].isna()         
df[~df['x1'].isna()]    # 특정 columns 결측지만 삭제

~df['x1'].isna()
df[~df['x1'].isna()]


df['x1'].fillna(30)     # 특정값으로 채우기
df['x1'].ffill()        # 앞의 값으로 채우기
df['x1'].bfill()        # 뒤의 값으로 채우기
df['x1'].fillna(df['x1'].mean())        # 평균값으로 채우기



## 결측치 처리
df.isna().sum(0) 
df.drop(['x3', 'x6'], axis=1)   # 결측값이 너무 많은 column 삭제

df.isna().sum(1) 
df.drop([12], axis=0)   # 결측값이 너무 많은 row 삭제

df01 = df.drop(['x3', 'x6'], axis=1).dropna()      # column중에 하나라도 결측치가 있는 raw 전체 제거

print(df.shape, df01.shape)




# 【 이상치 처리 】 ###################################################
import matplotlib.pyplot as plt
# EDA
df02 = df01.copy()

# Continuous Data Plot → Histogram
def hist_plot(x):
    f = plt.figure()
    plt.hist(x, edgecolor='gray')
    plt.close()
    return f


# Descrete Data Plot → Barplot
df02['x2'].value_counts()
bar_data = df02['x2'].value_counts().sort_index()
plt.bar(bar_data.index, bar_data.values)

def bar_plot(x):
    bar_data = x.value_counts().sort_index()
    
    f = plt.figure()
    plt.bar(bar_data.index, bar_data.values, edgecolor='gray')
    plt.close()
    return f

df02.dtypes
hist_plot(df02['x4'])
bar_plot(df02['x2'])

# dtype별로 plot을 dictionary에 저장
list(df02.dtypes.items())

hist_dict = {}
for c, cd in df02.dtypes.items():
    # c : y1
    # cd : dtype(int64)
    if str(cd) == 'object':
        hist_dict[c] = bar_plot(df02[c])
    else:
        hist_dict[c] = hist_plot(df02[c])
        
hist_dict['y1']

hist_dict['x1']
hist_dict['x2']
hist_dict['x4']
hist_dict['x5']
hist_dict['x7']

# import seaborn as sns
# sns.pairplot(df02)



# 이상치 처리
df02.dtypes

df02_des = df02.describe()
df02_des
df02_des.T

df02_des['y1']
df02_des['x1']

df02_des.T['25%']
df02_des.T['50%']
df02_des.T['mean']




# all (모두 True일경우에만 True) / any (하나라도 True가 있으면 True)
np.array([True, True, True]).all()      # True
np.array([True, False, True]).all()     # False
np.array([False, False, False]).all()   # False

np.array([True, True, True]).any()      # True
np.array([True, False, True]).any()     # True
np.array([False, False, False]).any()   # False


# 이상치처리 ---------------------------
df02 = df01.copy()      # original_data


def filter_outlier(data, lower=-np.inf, upper=np.inf):
    normal = (~((data < lower) | (upper < data))).all(1)
    return data[normal]

# sigma 기반 이상치 처리 --------------------
#   . Upper = Mean + 3.0 × Std
#   . Lower = Mean - 3.0 × Std
sigma_lower = df02_des.T['mean'] - 3.0 * df02_des.T['std']
sigma_upper = df02_des.T['mean'] + 3.0 * df02_des.T['std']

sigma_lower
sigma_upper


(df02 < sigma_lower)   # 하 한이상치
(sigma_upper < df02)   # 상한 이상치

(df02 < sigma_lower) | (sigma_upper < df02)  # 이상치 (상한or하한)
~((df02 < sigma_lower) | (sigma_upper < df02))    # 정상치 = ~이상치
(~((df02 < sigma_lower) | (sigma_upper < df02))).all(1)    # raw가 모두 정상치인경우
nomal_sigma = (~((df02 < sigma_lower) | (sigma_upper < df02))).all(1)


df03 = df02[nomal_sigma]
df03 = filter_outlier(df02, sigma_lower, sigma_upper)

print(df02.shape, df03.shape)



# IQR 기반 이상치 처리 --------------------
#   . IQR = Q(0.75) - Q(0.25)
#   . Upper = Q(0.75) + 1.5 × IQR
#   . Lower = Q(0.25) - 1.5 × IQR
IQR = df02_des.T['75%'] - df02_des.T['25%']
IQR_lower = df02_des.T['25%'] - 1.5 * IQR
IQR_upper = df02_des.T['75%'] + 1.5 * IQR

df03 = filter_outlier(df02, IQR_lower, IQR_upper)

print(df02.shape, df03.shape)



# MAD(Median Absolute Deviation) 기반 이상치 처리 --------------------
#   . MAD = Median( abs(x - Median(x)) )
median = df02_des.T['50%']

(df02 - median).quantile(0.5)
MAD = abs(df02[median.index] - median).quantile(0.5)

abs_median_lower = median - 1.486 * MAD
abs_median_upper = median + 1.486 * MAD

df03 = filter_outlier(df02, abs_median_lower, abs_median_upper)

print(df02.shape, df03.shape)





# 【 파생변수 설정 】 ###################################################
df03 = df02.copy()
# 기존 변수를 바탕으로 새로운 변수 만들기
#  . x8 = x1 + 2.5 * x4
df03['x1'] + 2.5 * df03['x4']

df03['x8'] = df03['x1'] + 2.5 * df03['x4']
df03





# log scale
hist_plot(df03['x4'])   # hist before

np.log(df03['x4'])
df03['x4_log'] = np.log(df03['x4'])

hist_plot(np.log(df03['x4']))   # hist after


# 연속형 변수를 범주형 변수로 만들기
def make_categories(x):
    if x >= 150:
        return 1
    else:
        return 0

df03['y2'] = df03['y1'].apply(lambda x: make_categories(x))
df03['y2'] = df03['y1'].apply(lambda x: 1 if x >= 150 else 0)


def x7_category(x):
    if x < 75 :
        return '050-074'
    elif x < 100:
        return '075-099'
    elif x < 125:
        return '100-125'
    else:
        return '125-150'
df03['x7'].apply(lambda x: x7_category(x))
df03['x7_cat'] = df03['x7'].apply(lambda x: x7_category(x))

df03
print(df03.shape)






######################################################################################################
##### 【 Modeling 】 ###############################################################################
######################################################################################################


# 【 sklearn 소개 】 ###################################################
# conda install -c anaconda scikit-learn
# ML000_sklearn.py

# '(model selection)';     from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
# '(전처리)';               from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# '(feature selection)';   from sklearn.feature_selection import f_regression, ...
# '(cross_decomposition)'; from sklearn.cross_decomposition import PLSRegression, ...

# '(선형모델)';             from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
# '(트리모델)';             from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# '(Ensemble)';            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

# '(모델 평가)';            from sklearn.metrics import mean_squared_error, r2_score, accuracy, f1_score


# [ 금번 교육시 사용될 Sklearn Module ]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score




# 【 DataSet준비 】 ###################################################
df04 = df03.copy()
df04


# 데이터셋 분리 (train_test_split)
from sklearn.model_selection import train_test_split
# train_test_split(
#     *arrays,
#     test_size=None,
#     train_size=None,
#     random_state=None,
#     shuffle=True,
#     stratify=None,
# )

train_test_split(df04)
train_test_split(df04, test_size=0.3)
train_test_split(df04, test_size=0.3, random_state=2)

train_set , test_set = train_test_split(df04, test_size=0.3, random_state=2)
train_set
test_set

print(train_set.shape, test_set.shape)



# 표준화 (dataset / normalizing) *
from sklearn.preprocessing import StandardScaler

y_col_reg = 'y1'    # regressor
y_col_clf = 'y2'    # regressor

train_y_reg = train_set[y_col_reg]
train_y_clf = train_set[y_col_clf]
test_y_reg = test_set[y_col_reg]
test_y_clf = test_set[y_col_clf]


X_cols = ['x1', 'x4_log', 'x7', 'x8']
train_X = train_set[X_cols]
test_X = test_set[X_cols]

scaler_X = StandardScaler()
scaler_X.fit(train_X)

scaler_X.transform(train_X)
train_X_norm = pd.DataFrame(scaler_X.transform(train_X), index=train_X.index, columns=train_X.columns)
test_X_norm = pd.DataFrame(scaler_X.transform(test_X), index=test_X.index, columns=test_X.columns)

print(train_y_reg.shape, train_y_clf.shape, train_X_norm.shape)
print(test_y_reg.shape, test_y_clf.shape, test_X_norm.shape)







# 【 Regressor 】 =================================================

# (LinearRegression) *** --------------------------------
from sklearn.linear_model import LinearRegression

# Learning *
LR_reg = LinearRegression()
LR_reg.fit(train_X_norm, train_y_reg)

# predict *
pred_train_LR_reg = LR_reg.predict(train_X_norm)
pred_test_LR_reg = LR_reg.predict(test_X_norm)

# metrics *
from sklearn.metrics import mean_squared_error
rmse_train_LR_reg = np.sqrt(mean_squared_error(train_y_reg, pred_train_LR_reg))
rmse_test_LR_reg = np.sqrt(mean_squared_error(test_y_reg, pred_test_LR_reg))
rmse_train_LR_reg
rmse_test_LR_reg

# feature_importances *
LR_reg.coef_

plt.barh(X_cols, LR_reg.coef_)
plt.show()



# (DecisionTree) *** --------------------------------
from sklearn.tree import DecisionTreeRegressor
DT_reg = DecisionTreeRegressor()
DT_reg.fit(train_X_norm, train_y_reg)

# predict *
pred_train_DT_reg = DT_reg.predict(train_X_norm)
pred_test_DT_reg = DT_reg.predict(test_X_norm)

# metrics *
rmse_train_DT_reg = np.sqrt(mean_squared_error(train_y_reg, pred_train_DT_reg))
rmse_test_DT_reg = np.sqrt(mean_squared_error(test_y_reg, pred_test_DT_reg))
rmse_train_DT_reg
rmse_test_DT_reg

# feature_importances *
DT_reg.feature_importances_

plt.barh(X_cols, DT_reg.feature_importances_)
plt.show()

# plot *
from sklearn.tree import plot_tree
# plt.rcParams['figure.dpi'] = 72   # default
plt.rcParams['figure.dpi'] = 300

plt.figure(figsize=(5,5))
plot_tree(DT_reg, feature_names=X_cols, filled=True)
plt.show()




# (RandomForest) *** --------------------------------
from sklearn.ensemble import RandomForestRegressor
RF_reg = RandomForestRegressor()
RF_reg.fit(train_X_norm, train_y_reg)

# predict *
pred_train_RF_reg = RF_reg.predict(train_X_norm)
pred_test_RF_reg = RF_reg.predict(test_X_norm)

# metrics *
rmse_train_RF_reg = np.sqrt(mean_squared_error(train_y_reg, pred_train_RF_reg))
rmse_test_RF_reg = np.sqrt(mean_squared_error(test_y_reg, pred_test_RF_reg))
rmse_train_RF_reg
rmse_test_RF_reg

# feature_importances *
RF_reg.feature_importances_

plt.barh(X_cols, RF_reg.feature_importances_)
plt.show()



# (Metrics Regressor) *** --------------------------------
metrics_reg = {
    'train' : [rmse_train_LR_reg, rmse_train_DT_reg, rmse_train_RF_reg],
    'test' : [rmse_test_LR_reg, rmse_test_DT_reg, rmse_test_RF_reg]
}

pd.DataFrame(metrics_reg, index=['LR_reg', 'DT_reg', 'RF_reg']).T


# graph *
plt.figure(figsize=(10,3))
plt.title('comparing of train_set model performance')
plt.plot(train_y_reg.values, label='real')
plt.plot(pred_train_LR_reg, alpha=0.3, label='LR_reg_pred')
plt.plot(pred_train_DT_reg, alpha=0.3, label='DT_reg_pred')
plt.plot(pred_train_RF_reg, alpha=0.3, label='RF_reg_pred')
plt.legend(loc='upper right')
plt.show()



plt.figure(figsize=(10,3))
plt.title('comparing of test_set model performance')
plt.plot(test_y_reg.values, label='real')
plt.plot(pred_test_LR_reg, alpha=0.3, label='LR_reg_pred')
plt.plot(pred_test_DT_reg, alpha=0.3, label='DT_reg_pred')
plt.plot(pred_test_RF_reg, alpha=0.3, label='RF_reg_pred')
plt.legend(loc='upper right')
plt.show()


# ----------------------------------------------------------










# 【 Classifier 】 =================================================

# (LogisticRegression) *** --------------------------------
from sklearn.linear_model import LogisticRegression

# Learning *
LR_clf = LogisticRegression()
LR_clf.fit(train_X_norm, train_y_clf)

# predict *
pred_train_LR_clf = LR_clf.predict(train_X_norm)
pred_test_LR_clf = LR_clf.predict(test_X_norm)

# metrics *
from sklearn.metrics import accuracy_score
acc_train_LR_clf = accuracy_score(train_y_clf, pred_train_LR_clf)
acc_test_LR_clf = accuracy_score(test_y_clf, pred_test_LR_clf)
acc_train_LR_clf
acc_test_LR_clf

# feature_importances *
LR_clf.coef_

plt.barh(X_cols, LR_clf.coef_[0])
plt.show()



# (DecisionTree) *** --------------------------------
from sklearn.tree import DecisionTreeClassifier
DT_clf = DecisionTreeClassifier()
DT_clf.fit(train_X_norm, train_y_clf)

# predict *
pred_train_DT_clf = DT_clf.predict(train_X_norm)
pred_test_DT_clf = DT_clf.predict(test_X_norm)

# metrics *
acc_train_DT_clf = accuracy_score(train_y_clf, pred_train_DT_clf)
acc_test_DT_clf = accuracy_score(test_y_clf, pred_test_DT_clf)
acc_train_DT_clf
acc_test_DT_clf

# feature_importances *
DT_clf.feature_importances_

plt.barh(X_cols, DT_clf.feature_importances_)
plt.show()

# plot *
from sklearn.tree import plot_tree
# plt.rcParams['figure.dpi'] = 72   # default
plt.rcParams['figure.dpi'] = 300

plt.figure(figsize=(5,5))
plot_tree(DT_clf, feature_names=X_cols, filled=True)
plt.show()




# (RandomForest) *** --------------------------------
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier()
RF_clf.fit(train_X_norm, train_y_clf)

# predict *
pred_train_RF_clf = RF_clf.predict(train_X_norm)
pred_test_RF_clf = RF_clf.predict(test_X_norm)

# metrics *
acc_train_RF_clf = accuracy_score(train_y_clf, pred_train_RF_clf)
acc_test_RF_clf = accuracy_score(test_y_clf, pred_test_RF_clf)
acc_train_RF_clf
acc_test_RF_clf

# feature_importances *
RF_clf.feature_importances_

plt.barh(X_cols, RF_clf.feature_importances_)
plt.show()



# (Metrics Regressor) *** --------------------------------
metrics_clf = {
    'train' : [acc_train_LR_clf, acc_train_DT_clf, acc_train_RF_clf],
    'test' : [acc_test_LR_clf, acc_test_DT_clf, acc_test_RF_clf]
}

pd.DataFrame(metrics_clf, index=['LR_clf', 'DT_clf', 'RF_clf']).T

# ----------------------------------------------------------















