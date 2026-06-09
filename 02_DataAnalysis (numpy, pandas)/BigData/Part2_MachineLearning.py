import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


train_set = pd.read_csv(f"{data_path}/part2/ch2/train.csv", encoding='utf-8-sig')
test_set = pd.read_csv(f"{data_path}/part2/ch2/test.csv", encoding='utf-8-sig')

# head
train_set.head(5)

# info
train_set.info()

# dtype 확인
train_set.dtypes
pd.concat([train_set.dtypes, train_set.iloc[0], train_set.describe(include='all').T], axis=1)

cols_numeric = []
cols_str = []
cols_datetime = []

for k, v in train_set.dtypes.items():
    v_str = str(v)
    if v_str in ['str', 'object']:
        cols_str.append(k)
    elif 'datetime' in str(v_str):
        cols_datetime.append(k)
    elif ('float' in v_str) | ('int' in v_str):
        cols_numeric.append(k)

qs = np.arange(1,10)/10
# decribe
train_set.describe(include='str')
train_set.describe(percentiles=qs, include='number')

train_set[cols_numeric].skew()
# train_set['race'].astype('category').cat.codes


# Na
train_set.isna().sum(axis=0)

#########################################################################################
from sklearn.preprocessing import OneHotEncoder
# oh = OneHotEncoder(sparse_output=False)
oh = OneHotEncoder(sparse_output=False, drop='first')


oh.fit_transform(train_set[['income']])

pd.DataFrame(oh.fit_transform(train_set[cols_str]), 
            index=train_set.index, 
            columns=oh.get_feature_names_out())


# -------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

transformed_list = []
le_list = []
for c_str in cols_str:
    le = LabelEncoder()    
    transformed = le.fit_transform(train_set[c_str])
    transformed_list.append(transformed)
    le_list.append(le)

train_set_str = pd.DataFrame(np.stack(transformed_list).T, 
            index=train_set.index,
            columns=cols_str)

# ------------------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
ms.fit_transform(train_set[cols_numeric])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(train_set[cols_numeric])


#########################################################################################
from sklearn.model_selection import train_test_split

train_train_set, train_valid_set = train_test_split(train_set, test_size=0.3)
label_transform = {'<=50K':0, '>50K':1}

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

mdl = LGBMClassifier()
mdl.fit(train_train_set[cols_numeric], train_train_set['income'].map(label_transform))

pred = mdl.predict(train_valid_set[cols_numeric])

# metrics
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)

accuracy_score(train_valid_set['income'].map(label_transform), pred)
recall_score(train_valid_set['income'].map(label_transform), pred)
precision_score(train_valid_set['income'].map(label_transform), pred)
f1_score(train_valid_set['income'].map(label_transform), pred)
roc_auc_score(train_valid_set['income'].map(label_transform), pred)
conf = confusion_matrix(train_valid_set['income'].map(label_transform), pred)
print(classification_report(train_valid_set['income'].map(label_transform), pred))

conf[1,1] / conf[1,:].sum()     # recall
conf[1,1] / conf[:,1].sum()     # precision
#########################################################################################

# 이진분류 데이터
y_true = pd.DataFrame([1, 1, 1, 0, 0, 1, 1, 1, 1, 0]) #실제값
y_pred = pd.DataFrame([1, 0, 1, 1, 0, 0, 0, 1, 1, 0]) #예측값

y_true_str = pd.DataFrame(['A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'B']) #실제값
y_pred_str = pd.DataFrame(['A', 'B', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'B']) #예측값

accuracy_score(y_true, y_pred)
recall_score(y_true, y_pred)
precision_score(y_true, y_pred)
f1_score(y_true, y_pred)
roc_auc_score(y_true, y_pred)
confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))


# 다중분류 데이터
y_true = pd.DataFrame([1, 2, 3, 3, 2, 1, 3, 3, 2, 1]) # 실제값
y_pred = pd.DataFrame([1, 2, 1, 3, 2, 1, 1, 2, 2, 1]) # 예측값
y_pred_proba = np.zeros((y_pred.shape[0],3))
y_pred_proba[np.arange(y_pred.shape[0]), y_pred.to_numpy().ravel()-1] = 1
roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')

y_true_str = pd.DataFrame(['A', 'B', 'C', 'C', 'B', 'A', 'C', 'C', 'B', 'A']) # 실제값
y_pred_str = pd.DataFrame(['A', 'B', 'A', 'C', 'B', 'A', 'A', 'B', 'B', 'A']) # 예측값


accuracy_score(y_true, y_pred)
# (average) 
#   None : 모든 class에 대해 계산
#   macro : class별로 똑같은 비중으로 평균
#   weighted : 데이터 갯수기반 가중치
#   micro : 전체 TP, FN을 한꺼번에 합쳐서 계산

recall_score(y_true, y_pred, average='macro')       
precision_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='macro')



# multi_class
#   ovo : One-vs-One (class 두개씩 짝지어서)
#   ovr : One-vs-Rest 
roc_auc_score(y_true.loc[:,0], y_pred_proba, multi_class='ovr', average='macro')
roc_auc_score(y_true.loc[:,0], y_pred_proba, multi_class='ovo', average='macro')
confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))


#########################################################################################
# 회귀 데이터
y_true = pd.DataFrame([1, 2, 5, 2, 4, 4, 7, 9]) # 실제값
y_pred = pd.DataFrame([1.14, 2.53, 4.87, 3.08, 4.21, 5.53, 7.51, 10.32]) # 예측값

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

mean_squared_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
r2_score(y_true, y_pred)
root_mean_squared_error(y_true, y_pred)
mean_absolute_percentage_error(y_true, y_pred)







#########################################################################################
X = train_set[cols_numeric]
y = train_set['income'].map(label_transform)

X_train = train_train_set[cols_numeric]
y_train = train_train_set['income'].map(label_transform)

X_valid = train_valid_set[cols_numeric]
y_valid = train_valid_set['income'].map(label_transform)
#########################################################################################
# Class Weight
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)

class_weight_dict = dict(zip(classes, class_weights))

print(class_weight_dict)

sample_weight = np.array([class_weight_dict[label] for label in y])
sample_weight_train = np.array([class_weight_dict[label] for label in y_train])
print(sample_weight)
print(sample_weight_train)


# random_forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=class_weight_dict
)

rf.fit(X_train, y_train)

pred = rf.predict(X_valid)


# gradientboosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    random_state=42
)

gb.fit(
    X_train,
    y_train,
    sample_weight=sample_weight_train
)

pred = gb.predict(X_valid)


# XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train, sample_weight=sample_weight_train)

pred = xgb.predict(X_valid)


# LightGBM
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    class_weight=class_weight_dict
)

lgbm.fit(X_train, y_train)

pred = lgbm.predict(X_valid)




#########################################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# KFold ----------------------------------------------------------------------
from sklearn.model_selection import KFold


# X: feature 데이터
# y: target 데이터

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

scores = []

for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model.fit(X_train, y_train)
    
    pred = model.predict(X_valid)
    
    score = accuracy_score(y_valid, pred)
    scores.append(score)

print("각 fold 점수:", scores)
print("평균 점수:", np.mean(scores))


# StratifiedKFold ----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

scores = []

for train_idx, valid_idx in skf.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model.fit(X_train, y_train)
    
    pred = model.predict(X_valid)
    
    score = accuracy_score(y_valid, pred)
    scores.append(score)

print("각 fold 점수:", scores)
print("평균 점수:", np.mean(scores))



# cross_val_score ----------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(random_state=42)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
# scoring='accuracy'
# scoring='f1'
# scoring='f1_macro'
# scoring='roc_auc'


print("각 fold 점수:", scores)
print("평균 점수:", np.mean(scores))


#########################################################################################
# GridSearchCV ----------------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    verbose=1
)

grid_search.fit(X, y)

print("Best score:", grid_search.best_score_)
print("Best params:", grid_search.best_params_)
print("Best estimator:", grid_search.best_estimator_)



# RandomizedSearchCV ----------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    verbose=1
)

random_search.fit(X, y)

print("Best score:", random_search.best_score_)
print("Best params:", random_search.best_params_)
print("Best estimator:", random_search.best_estimator_)


# BayesSearchCV ----------------------------------------------------------------------

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

bayes_search_spaces = {
    "n_estimators": Integer(100, 500),
    "max_depth": Categorical([3, 5, 7, 10, None]),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2", None])
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=bayes_search_spaces,
    n_iter=30,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

bayes_search.fit(X, y)

print("Best score:", bayes_search.best_score_)
print("Best params:", bayes_search.best_params_)
print("Best estimator:", bayes_search.best_estimator_)