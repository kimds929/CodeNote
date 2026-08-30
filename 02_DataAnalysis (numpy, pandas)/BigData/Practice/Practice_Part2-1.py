import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


########################################################################################################
# Regression
########################################################################################################

# # # dataset - flight 
# df_train = pd.read_csv(f"{data_path}/part2/ch8/flight_train.csv", encoding='utf-8-sig')
# df_test = pd.read_csv(f"{data_path}/part2/ch8/flight_test.csv", encoding='utf-8-sig')

# dataset - laptop
df_train = pd.read_csv(f"{data_path}/part2/ch8/laptop_train.csv", encoding='utf-8-sig')
df_test = pd.read_csv(f"{data_path}/part2/ch8/laptop_test.csv", encoding='utf-8-sig')

# # dataset - car
# df_train = pd.read_csv(f"{data_path}/part2/ch8/car_train.csv", encoding='utf-8-sig')
# df_test = pd.read_csv(f"{data_path}/part2/ch8/car_test.csv", encoding='utf-8-sig')



# -----------------------------------------------------------------------------------------------------

# EDA
print(df_train.shape, df_test.shape)
# df_train.info()
for df_name, df in zip(['train','test'], [df_train, df_test]):
    print(f"< {df_name} >")
    df_info = pd.concat([df.dtypes, 
                     df.nunique(), 
                     df.agg(['min','max']).T,
                     df.isna().sum(axis=0),
                    ], axis=1)
    print(df_info)
    print()



drop_cols = ['flight']
y_col = 'price'
X_cols_num = [c for c in df_train.select_dtypes(include=['float64', 'int64']).columns if c not in drop_cols + [y_col]]
X_cols_cat = [c for c in df_train.select_dtypes(include=['str','object']).columns if c not in drop_cols + [y_col]]

# -----------------------------------------------------------------------------
# Set PipeLine

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# numerical
num_pipe = Pipeline([
    ('encoder', StandardScaler())
])

# categorical
cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(sparse_output=False, drop='first'))
])

preprocess = ColumnTransformer([
    ('num', num_pipe, X_cols_num),
    ('cat', cat_pipe, X_cols_cat)
])


# ------------------------------------------------------------
from sklearn.model_selection import KFold, StratifiedKFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)


############################################################################################################
############################################################################################################

# CrossValidation
# ------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {'RF': RandomForestRegressor(random_state=0),
          'GB': GradientBoostingRegressor(random_state=0),
          'XGB': XGBRegressor(random_state=0),
          'LGBM': LGBMRegressor(verbosity=-1, random_state=0)}

# -----------------------------------------------------------
# 1개 모델만 실행
model_name = 'RF'

model_pipe = Pipeline([
    ('preprocess', preprocess),
    ('model', models[model_name])
])

from sklearn.metrics import get_scorer_names
print(np.array(get_scorer_names()))

train_X = df_train[X_cols_num+X_cols_cat]
train_y = df_train[y_col]
    
scores = cross_val_score(model_pipe, 
                X = train_X,
                y = train_y,
                cv = cv,
                scoring='neg_mean_squared_error')

scores.mean()

# -----------------------------------------------------------
# 여러 모델 실행 후 비교
from sklearn.model_selection import cross_val_score

results_scores = {}
results_models = {}
for model_name, model in models.items():
    print(model_name, end=' : ')
    
    model_pipe = Pipeline([
        ('preprocess', preprocess),
        ('model', models[model_name])
    ])
    scores = cross_val_score(estimator=model_pipe, 
                    X = train_X, 
                    y = train_y,
                    cv = cv,
                    scoring='neg_root_mean_squared_error')
    print(f"{scores.mean():.3f}")
    results_scores[model_name] = scores.mean()
    
    # model_pipe 재학습
    model_pipe.fit(train_X, train_y)
    results_models[model_name] = model_pipe

best_model_idx = np.argmax(np.array(list(results_scores.values())))
best_model_name = list(models.keys())[best_model_idx]
print(f"{best_model_name} → score : {results_scores[best_model_name]}")

pred = results_models[best_model_name].predict(df_test[X_cols_num+X_cols_cat])
pd.DataFrame({'pred':pred}, index=df_test.index)



############################################################################################################
############################################################################################################

# ----------------------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

params_range_grid = {}
params_range_grid['RF'] = {'model__n_estimators':[100, 500],
             'model__max_depth':[3, 10],
             'model__min_samples_leaf':[1, 5],
             }

params_range_grid['GB'] = {'model__n_estimators': [100, 500],
             'model__learning_rate': [0.01, 0.1],
             'model__max_depth': [3, 10],
             'model__subsample': [0.5, 1.0]}

params_range_grid['XGB'] = {'model__n_estimators': [100, 500],
             'model__learning_rate': [0.01, 0.1],
             'model__max_depth': [3, 10],
             'model__subsample': [0.5, 1.0]}

params_range_grid['LGBM'] = {'model__n_estimators': [100, 500],
             'model__learning_rate': [0.01, 0.1],
             'model__num_leaves': [31, 100]}

# ----------------------------------------------------------------------------------------
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
params_range_bayes = {}
params_range_bayes['RF'] = {'model__n_estimators': Integer(*[100, 500]),
             'model__max_depth':Integer(*[3, 10]),
             'model__min_samples_leaf':Integer(*[1, 3]),
             }

params_range_bayes['GB'] = {'model__n_estimators': Integer(*[100, 500]),
             'model__learning_rate': Real(*[0.01, 0.1], prior='log-uniform'),
             'model__max_depth': Integer(*[3, 10]),
             'model__subsample': Real(*[0.5, 1.0])}

params_range_bayes['XGB'] = {'model__n_estimators': Integer(*[100, 500]),
             'model__learning_rate': Real(*[0.01, 0.1], prior='log-uniform'),
             'model__max_depth': Integer(*[3, 10]),
             'model__subsample': Real(*[0.5, 1.0])}

params_range_bayes['LGBM'] = {'model__n_estimators': Integer(*[100, 500]),
             'model__learning_rate': Real(*[0.01, 0.1], prior='log-uniform'),
             'model__num_leaves': Integer(*[31, 100])}
# ----------------------------------------------------------------------------------------


from sklearn.metrics import get_scorer_names
print(np.array(get_scorer_names()))


# cv_method = 'grid'
cv_method = 'bayes'
print(cv_method)

train_X = df_train[X_cols_num+X_cols_cat]
train_y = df_train[y_col]

results_scores = {}
results_models = {}
results_cvs = {}
for model_name, model in models.items():
    
    print(model_name, end=" : ")
    model_pipe = Pipeline([
        ('preprocess', preprocess),
        ('model', models[model_name])
    ])
    if cv_method == 'grid':
        
        params = params_range_grid[model_name]
        searchCV = GridSearchCV(estimator=model_pipe,
                                param_grid=params,      # ☆
                                cv=cv,
                                scoring='neg_root_mean_squared_error',
                                return_train_score=True,
                                verbose=0)
    elif cv_method == 'bayes':
        params = params_range_bayes[model_name]
        searchCV = BayesSearchCV(estimator=model_pipe,
                                search_spaces=params,       # ☆
                                n_iter=30,                  # ☆
                                cv=cv,
                                scoring='neg_root_mean_squared_error',
                                return_train_score=True,
                                random_state = 0,
                                verbose=0)

    searchCV.fit(train_X, train_y)
    
    print(f"{searchCV.best_score_:.3f}")
    
    results_models[model_name] = searchCV.best_estimator_
    results_scores[model_name] = searchCV.best_score_
    results_cvs[model_name] = pd.DataFrame(searchCV.cv_results_)


best_model_idx = np.argmax(np.array(list(results_scores.values())))
best_model_name = list(models.keys())[best_model_idx]
print(f"{best_model_name} → score : {results_scores[best_model_name]}")

pred = results_models[best_model_name].predict(df_test[X_cols_num+X_cols_cat])
pd.DataFrame({'pred':pred}, index=df_test.index)
    
# best_estimator.predict(df_test[])


# ------------------------------------------------------------






########################################################################################################
# Classification
########################################################################################################

