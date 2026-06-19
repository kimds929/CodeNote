import numpy as np
import pandas as pd
# D:/DataScience/★GitHub_kimds929/CodeNote/"02_DataAnalysis (numpy, pandas)"/BigData
data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

pd.set_option('display.max_rows', 300)
import warnings
warnings.filterwarnings("ignore")
    
rs = 1

############################################################################################
# Part 1
############################################################################################
df = pd.read_csv(f"{data_path}/part4/ch3/members.csv")


print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1: 'nunique', 2: 'isna'})
# print(df_summary)

# < shape (100, 10) >
#              dtypes  nunique  isna         min         max
# id              str      100     0        id01        id99
# age         float64       71     0       -13.5       100.0
# city            str        4     0          경기          서울
# f1          float64       43    31        12.0       111.0
# f2            int64        3     0           0           2
# f3              str        3    28        gold         vip
# f4              str       16     0        ENFJ        ISTP
# f5          float64       20     0    9.796378   98.429899
# subscribed      str       93     0  2021-01-06  2021-12-30
# views       float64       96     4        42.0     21550.0


# 1. 
df_remove_na = df.dropna()
print(df_remove_na.shape)

use_row_limit = int(len(df_remove_na) * 0.7)
result = df_remove_na.iloc[:use_row_limit]['f1'].describe()['25%']

print(1)
print(f"P1. Ans : {int(result)}")
print('-'*100, end='\n')

# 2.
df = pd.read_csv(f"{data_path}/part4/ch3/year.csv", index_col='Unnamed: 0')

print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1: 'nunique', 2: 'isna'})
# print(df_summary)
# < shape (3, 200) >
#     dtypes  nunique  isna  min  max
# 0    int64        3     0  128  176
# 1    int64        3     0   74  132
# 2    int64        3     0   64  123
# 3    int64        3     0  110  140
# 4    int64        3     0   80  134
# ..     ...      ...   ...  ...  ...
# 195  int64        3     0   51  191
# 196  int64        3     0   81  137
# 197  int64        3     0  101  174
# 198  int64        3     0   56  194
# 199  int64        3     0   43  128

df_2000 = df.loc[2000]
print(2)
print(f"P2. Ans : {df_2000[df_2000 > df_2000.mean()].shape[0]}")
print('-'*100, end='\n')


# 3. 
df = pd.read_csv(f"{data_path}/part4/ch3/members.csv")

print(3)
print(f"P3. Ans : {df.isna().sum(axis=0).nlargest(1).index[0]}")
print('-'*100, end='\n')

print()
print('='*100)
print()
############################################################################################
# Part 2
############################################################################################

df_train = pd.read_csv(f"{data_path}/part4/ch3/train.csv", index_col='Unnamed: 0')
df_test = pd.read_csv(f"{data_path}/part4/ch3/test.csv", index_col='Unnamed: 0')

# for df_name, df in zip(['train','test'], [df_train, df_test]):
#     print(f"< {df_name} shape {df.shape} >")

#     df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
#                             df.agg(['min','max']).T], axis=1)\
#                                 .rename(columns={0:'dtypes', 1: 'nunique', 2: 'isna'})
#     print(df_summary)
#     print()
    
# < train shape (1490, 9) >
#                     dtypes  nunique  isna                min                            max  
# Age                  int64       11     0                 25                             35   
# Employment Type        str        2     0  Government Sector   Private Sector/Self Employed   
# GraduateOrNot          str        2     0                 No                            Yes   
# AnnualIncome         int64       30     0             300000                        1800000   
# FamilyMembers        int64        8     0                  2                              9   
# ChronicDiseases      int64        2     0                  0                              1   
# FrequentFlyer          str        2     0                 No                            Yes   
# EverTravelledAbroad    str        2     0                 No                            Yes   
# TravelInsurance      int64        2     0                  0                              1   

                    
# < test shape (497, 8) >
#                     dtypes  nunique  isna                min                            max  
# Age                  int64       11     0                 25                             35   
# Employment Type        str        2     0  Government Sector   Private Sector/Self Employed   
# GraduateOrNot          str        2     0                 No                            Yes   
# AnnualIncome         int64       27     0             300000                        1750000   
# FamilyMembers        int64        8     0                  2                              9   
# ChronicDiseases      int64        2     0                  0                              1   
# FrequentFlyer          str        2     0                 No                            Yes   
# EverTravelledAbroad    str        2     0                 No                            Yes   

                    
cols_drop = []
col_y = 'TravelInsurance'
cols_X_num = [k for k, v in df_train.nunique().items() if (v > 2) & (k != col_y)]
cols_X_cat = [k for k, v in df_train.nunique().items() if (v <= 2) & (k != col_y)]
cols_X = cols_X_num + cols_X_cat

print(f"cols_X_num : {cols_X_num}")
print(f"cols_X_cat : {cols_X_cat}")

# cols_X_num : ['Age', 'AnnualIncome', 'FamilyMembers']
# cols_X_cat : ['Employment Type', 'GraduateOrNot', 'ChronicDiseases', 'FrequentFlyer', 'EverTravelledAbroad', 'TravelInsurance']


# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,  MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder

def to_numpy(X):
    return np.array(X)

pipe_num = Pipeline([
    ('type_transform', FunctionTransformer(to_numpy)),
    ('encoding', StandardScaler())
])

def astype_str(X):
    return X.astype(str)

def label_encodeing(X):
    X_copy = X.copy()
    for c in X_copy.columns:
        le = LabelEncoder()
        X_copy[c] = le.fit_transform(X_copy[c])
    return X_copy
    

pipe_cat = Pipeline([
    ('astype_str', FunctionTransformer(astype_str)),
    # ('encoding', OneHotEncoder(sparse_output=False))
    ('encoding', FunctionTransformer(label_encodeing))
])

# print(pipe_num.fit_transform(df_train[cols_X_num]))
# print('-'*100, end='\n')

# print(pipe_cat.fit_transform(df_train[cols_X_cat]))
# print('-'*100, end='\n')

from sklearn.compose import ColumnTransformer

preprocess_X = ColumnTransformer([
    ('pipe_num', pipe_num, cols_X_num),
    ('pipe_Cat', pipe_cat, cols_X_cat)
])

# print( preprocess_X.fit_transform(df_train) )


#####################################################

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import compute_sample_weight

class WrapperClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, weight_mode='base'):
        self.estimator = estimator
        self.weight_mode = weight_mode
        
    def fit(self, X, y, **fit_params):
        self._estimator = clone(self.estimator)
        
        if self.weight_mode == 'weighted':
            fit_params['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y)

        self._estimator.fit(X, y, **fit_params)
        
        if hasattr(self._estimator, 'classes_'):
            self.classes_ = self._estimator.classes_
        return self
    
    def predict(self, X):
        return self._estimator.predict(X)
    
    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
    
    def score(self, X,y):
        return self._estimator.score(X)
 
models = {'RF': WrapperClassifier(RandomForestClassifier()),
          'GB': WrapperClassifier(GradientBoostingClassifier()),
          'XGB': WrapperClassifier(XGBClassifier()),
          'LGBM': WrapperClassifier(LGBMClassifier(verbosity=-1)),
          }


params = {}
params['RF'] = {'model__estimator__random_state':[0],
                'model__estimator__max_depth': [3, 5],
                # 'model__estimator__min_samples_leaf' : [1,3],
                # 'model__estimator__max_features':['sqrt',1],
                'model__weight_mode':['base','weighted']
                }
params['GB'] = {'model__estimator__random_state':[0],
                'model__estimator__learning_rate': [0.03, 0.1],
                # 'model__estimator__max_depth': [3,5],
                # 'model__estimator__n_estimators':[100, 300],
                'model__weight_mode':['base','weighted']
                }
params['XGB'] = {'model__estimator__random_state':[0],
                'model__estimator__learning_rate': [0.03, 0.1],
                # 'model__estimator__max_depth': [3,5],
                # 'model__estimator__n_estimators':[100, 300],
                'model__weight_mode':['base','weighted']
                }
params['LGBM'] = {'model__estimator__random_state':[0],
                  'model__estimator__num_leaves':[30,63],
                # 'model__estimator__learning_rate': [0.03, 0.1],
                # 'model__estimator__n_estimators':[100, 300],
                'model__weight_mode':['base','weighted']
                }

from sklearn.model_selection import StratifiedKFold, GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True)

########################################################################
from sklearn.metrics import get_scorer_names


X = df_train[cols_X]
y = df_train[col_y]
 
result_best_estimator = {}
result_best_score = {}
result_best_params = {}
result_cv = {}
for model_name, model in models.items():
    print(model_name, end=" : ")
    
    pipe_model = Pipeline([
        ('preproces', preprocess_X),
        ('model', models[model_name])
    ])
    
    grid_cv = GridSearchCV(estimator=pipe_model, cv=cv, 
                           param_grid = params[model_name],
                           return_train_score=True,
                           scoring='roc_auc',
                           verbose=1)
    grid_cv.fit(X,y)
    
    print(f"\t best_score : {grid_cv.best_score_}")
    result_best_estimator[model_name] = grid_cv.best_estimator_
    result_best_score[model_name] = grid_cv.best_score_
    result_best_params[model_name] = grid_cv.best_params_
    result_cv[model_name] = grid_cv.cv_results_


result_best_estimator
result_best_score
result_best_params
# pd.DataFrame(result_cv['RF'])


result_proba = result_best_estimator['XGB'].predict_proba(df_test[cols_X])[:,1]
# pd.DataFrame({'index':df_test.index, 'pred':result_proba}).to_csv("abc.csv", index=False)





# print(np.array(get_scorer_names()))
# ['accuracy' 'adjusted_mutual_info_score' 'adjusted_rand_score'
#  'average_precision' 'balanced_accuracy' 'completeness_score'
#  'd2_absolute_error_score' 'd2_brier_score' 'd2_log_loss_score'
#  'explained_variance' 'f1' 'f1_macro' 'f1_micro' 'f1_samples'
#  'f1_weighted' 'fowlkes_mallows_score' 'homogeneity_score' 'jaccard'
#  'jaccard_macro' 'jaccard_micro' 'jaccard_samples' 'jaccard_weighted'
#  'matthews_corrcoef' 'mutual_info_score' 'neg_brier_score' 'neg_log_loss'
#  'neg_max_error' 'neg_mean_absolute_error'
#  'neg_mean_absolute_percentage_error' 'neg_mean_gamma_deviance'
#  'neg_mean_poisson_deviance' 'neg_mean_squared_error'
#  'neg_mean_squared_log_error' 'neg_median_absolute_error'
#  'neg_negative_likelihood_ratio' 'neg_root_mean_squared_error'
#  'neg_root_mean_squared_log_error' 'normalized_mutual_info_score'
#  'positive_likelihood_ratio' 'precision' 'precision_macro'
#  'precision_micro' 'precision_samples' 'precision_weighted' 'r2'
#  'rand_score' 'recall' 'recall_macro' 'recall_micro' 'recall_samples'
#  'recall_weighted' 'roc_auc' 'roc_auc_ovo' 'roc_auc_ovo_weighted'
#  'roc_auc_ovr' 'roc_auc_ovr_weighted' 'top_k_accuracy' 'v_measure_score']
########################################################################
# print(help(GridSearchCV))



















# params['RF'] = {'model__estimator__random_state':[0],
#                 'model__estimator__max_depth': [3, 5],
#                 # 'model__estimator__min_samples_leaf' : [1,3],
#                 # 'model__estimator__max_features':['sqrt',1],
#                 'model__weight_mode':['base','weighted']
#                 }
# params['GB'] = {'model__estimator__random_state':[0],
#                 'model__estimator__learning_rate': [0.03, 0.1],
#                 # 'model__estimator__max_depth': [3,5],
#                 # 'model__estimator__n_estimators':[100, 300],
#                 'model__weight_mode':['base','weighted']
#                 }
# params['XGB'] = {'model__estimator__random_state':[0],
#                 'model__estimator__learning_rate': [0.03, 0.1],
#                 # 'model__estimator__max_depth': [3,5],
#                 # 'model__estimator__n_estimators':[100, 300],
#                 'model__weight_mode':['base','weighted']
#                 }
# params['LGBM'] = {'model__estimator__random_state':[0],
#                   'model__estimator__num_leaves':[30,63],
#                 # 'model__estimator__learning_rate': [0.03, 0.1],
#                 # 'model__estimator__n_estimators':[100, 300],
#                 'model__weight_mode':['base','weighted']
#                 }



# class WrapperClassifier(ClassifierMixin, BaseEstimator):
#     def __init__(self, estimator, weight_mode='base'):
#         self.estimator = estimator
#         self.weight_mode = weight_mode
    
#     def fit(self, X, y, **fit_params):
#         self._estimator = clone(self.estimator)
        
#         if self.weight_mode == 'weighted':
#             fit_params['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y)
        
#         self._estimator.fit(X,y, **fit_params)
        
#         if hasattr(self._estimator, 'classes_'):
#             self.classes_ = self._estimator.classes_
        
#         return self
    
#     def predict(self, X):
#         return self._estimator.predict(X)
    
#     def predict_proba(self, X):
#         return self._estimator.predict_proba(X)
    
#     def score(self, X,y):
#         return self._estimator.score(X, y)