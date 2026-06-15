import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


# dataset - laptop
df_train = pd.read_csv(f"{data_path}/part2/ch8/laptop_train.csv", encoding='utf-8-sig')
df_test = pd.read_csv(f"{data_path}/part2/ch8/laptop_test.csv", encoding='utf-8-sig')


df_train.shape, df_test.shape
df_train.convert_dtypes().dtypes

for df_name, df in zip(['train', 'test'], [df_train.convert_dtypes(), df_test.convert_dtypes()]):
    print(f"< {df_name} : {df.shape}>")
    df_summary = pd.concat([df.dtypes,
        df.nunique(),
        df.isna().sum(axis=0),
        df.agg(['min','max']).T],axis=1)
    
    print(df_summary)
    print()


cols_drop = []
col_y = 'Price'
cols_X_num = [c for c in df_train.select_dtypes(include=['int64','float64']).columns if c not in cols_drop + [col_y]]
cols_X_cat = [c for c in df_train.select_dtypes(include=['str','object']).columns if c not in cols_drop + [col_y]]


for c_cat in cols_X_cat:
    print(f"< {c_cat} >")
    df_cat_concat = pd.concat([df_train[c_cat].value_counts(dropna=False).sort_index(),
    df_test[c_cat].value_counts(dropna=False).sort_index()], axis=1)
    print(df_cat_concat)
    print()
    
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------------------
# Preprocessing : fillna + encoding
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FunctionTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

def fillna_num(X):
    return X.fillna(-1)

preprocess_num = Pipeline([
    ('fillna', FunctionTransformer(fillna_num)),
    ('encoder', StandardScaler())
])

def fillna_cat(X):
    return X.fillna('-')

preprocess_cat = Pipeline([
    ('fillna', FunctionTransformer(fillna_cat)),
    ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
])

# Preprocessing : numerical, categorical
from sklearn.compose import ColumnTransformer
preprocess = ColumnTransformer([
    ('num', preprocess_num, cols_X_num),
    ('cat', preprocess_cat, cols_X_cat)
])
# ----------------------------------------------------------------------------------------
# from sklearn.model_selection import train_test_split
# X_train, X_valid, y_train, y_valid = train_test_split(df_train[cols_X_num + cols_X_cat], df_train[col_y], test_size=0.2, st)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
models = {'RF': RandomForestRegressor(random_state=0),
          'GB': GradientBoostingRegressor(random_state=0),
          'XGB': XGBRegressor(random_state=0),
          'LGBM': LGBMRegressor(random_state=0, verbosity=-1)}


import pkg_resources
np.array([p.project_name for p in pkg_resources.working_set])
from sklearn.metrics import get_scorer_names
np.array(get_scorer_names())

params_grid = {}
params_grid['RF'] = {'model__n_estimators': [100, 500],
                     'model__max_depth': [3, 10],
                     'model__min_samples_leaf': [1, 5]}
params_grid['GB'] = {'model__n_estimators': [100, 500],
                     'model__learning_rate': [0.01, 0.1],
                     'model__max_depth': [3, 10],
                     'model__subsample': [0.5, 1.0]}
params_grid['XGB'] = {'model__n_estimators': [100, 500],
                     'model__learning_rate': [0.01, 0.1],
                     'model__max_depth': [3, 10],
                     'model__subsample': [0.5, 1.0]}
params_grid['LGBM'] = {'model__n_estimators': [100, 500],
                     'model__learning_rate': [0.01, 0.1],
                     'model__num_leaves': [31, 100]}
print(help(RandomForestRegressor))


# ----------------------------------------------------------------
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer_names

cv = KFold(n_splits=5, shuffle=True, random_state=0)
help(KFold)

results_best_estimators = {}
results_scores = {}
results_cv = {}
for model_name, model in models.items():
    # model_name = 'RF'
    print(model_name, end=' : ')

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    X = df_train[cols_X_num + cols_X_cat]
    y = df_train[col_y]

    model_pipe = Pipeline([
        ('preprocess', preprocess), 
        ('model', models[model_name])
    ])
    param_set = params_grid[model_name]

    searchCV = GridSearchCV(estimator=model_pipe,
                            param_grid=param_set,
                            scoring='r2',
                            return_train_score=True,
                            cv=cv)

    searchCV.fit(X, y)
    print(f"{searchCV.best_score_:.3f}")
    results_best_estimators[model_name] = searchCV.best_estimator_
    results_scores[model_name] = searchCV.best_score_
    results_cv[model_name] = pd.DataFrame(searchCV.cv_results_).T
    

pd.DataFrame({'pred':results_best_estimators['XGB'].predict(df_test[cols_X_num+cols_X_cat])})




########################################################################################


class SplitDisk(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
            
    def fit(self, X):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X['Hard_Disk_Capacity'] = X['Hard_Disk_Capacity'].str.replace(' TB', '024 GB').replace(' GB')

df_train['Hard_Disk_Capacity'].str.replace(' TB', '024 GB')
    
    
    


