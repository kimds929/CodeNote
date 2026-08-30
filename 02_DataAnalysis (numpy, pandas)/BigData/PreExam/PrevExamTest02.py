import numpy as np
import pandas as pd
# D:/DataScience/★GitHub_kimds929/CodeNote/"02_DataAnalysis (numpy, pandas)"/BigData
data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


import warnings
warnings.filterwarnings("ignore")
    
rs = 1

############################################################################################
# Part 1
############################################################################################
# df = pd.read_csv(f"{data_path}/part4/ch2/members.csv")

# df_suammry = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0), df.agg(['min','max']).T], axis=1).rename(columns={0:'dtypes', 1:'nuniuqe', 2:'isna'})

# print(f"< train: {df.shape} >")
# print(df_suammry)

# print('-'*100, end='\n')

# # 1
# df = df.sort_values('views', axis=0, ascending=False)
# print(df.head(11))
# print('-'*100, end='\n')

# df_view_large10 = df['views'].nlargest(10)

# print(df_view_large10)
# print('-'*100, end='\n')

# # df.iloc[:10]['views'] = df_view_large10.iloc[-1].item()
# df['views'] = np.where(df['views'] >= df_view_large10.iloc[-1], df_view_large10.iloc[-1], df['views'])
# print(df.head(11))

# print()
# print(f"Ans : {df.query('age >= 80')['views'].mean().round(2)}")
# print()


# # 2
# df_80 = df.iloc[:80]

# bef = df_80['f1'].dropna().std()
# print(f"bef : {bef:.3f} {df_80['f1'].dropna().shape}")
# print()
# mdian = df_80['f1'].describe()['50%']
# df_80['f1'] = df_80['f1'].fillna(mdian)

# print(df_80.isna().sum(axis=0))
# print()
# aft =  df_80['f1'].dropna().std()
# print(f"aft : {aft:.3f}  {df_80['f1'].dropna().shape}")


# print()
# print(f"Ans : {np.round(np.abs(aft-bef),2)}")


# # 3
# age_mean, age_std = df['age'].agg(['mean','std'])
# outlier_upper = age_mean + 1.5 * age_std
# outlier_lower = age_mean - 1.5 * age_std
# ans = df.query(f"(age > {outlier_upper}) | (age < {outlier_lower})")['age'].sum()

# print(f"Ans : {ans}")





############################################################################################
# Part 2
############################################################################################

X_train = pd.read_csv(f"{data_path}/part4/ch2/X_train.csv")
X_test = pd.read_csv(f"{data_path}/part4/ch2/X_test.csv")
y_train = pd.read_csv(f"{data_path}/part4/ch2/y_train.csv")


for df_name, df in zip(['X_train', 'X_test', 'y_train'], [X_train, X_test, y_train]):
    print(f"< {df_name}: {df.shape} >")
    df_suammry = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0), df.agg(['min','max']).T], axis=1).rename(columns={0:'dtypes', 1:'nuniuqe', 2:'isna'})
    print(df_suammry)
    print()

print('-'*100, end='\n')



# < X_train: (8799, 11) >
#                     dtypes  nuniuqe  isna     min     max
# ID                   int64     8799     0       1   10999
# Warehouse_block        str        5     0       A       F
# Mode_of_Shipment       str        3     0  Flight    Ship
# Customer_care_calls  int64        6     0       2       7
# Customer_rating      int64        5     0       1       5
# Cost_of_the_Product  int64      215     0      96     310
# Prior_purchases      int64        8     0       2      10
# Product_importance     str        3     0    high  medium
# Gender                 str        2     0       F       M
# Discount_offered     int64       65     0       1      65
# Weight_in_gms        int64     3761     0    1001    7846

# < X_test: (2200, 11) >
#                     dtypes  nuniuqe  isna     min     max
# ID                   int64     2200     0       5   10995
# Warehouse_block        str        5     0       A       F
# Mode_of_Shipment       str        3     0  Flight    Ship
# Customer_care_calls  int64        6     0       2       7
# Customer_rating      int64        5     0       1       5
# Cost_of_the_Product  int64      206     0      96     310
# Prior_purchases      int64        8     0       2      10
# Product_importance     str        3     0    high  medium
# Gender                 str        2     0       F       M
# Discount_offered     int64       65     0       1      65
# Weight_in_gms        int64     1741     0    1005    7588

# < y_train: (8799, 2) >
#                     dtypes  nuniuqe  isna  min    max
# ID                   int64     8799     0    1  10999
# Reached.on.Time_Y.N  int64        2     0    0      1



# ------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def change_dtype(X):
    X_copy = X.copy()
    for c, v in X_copy.nunique().items():
        if v < 10:
            X_copy[c] = X_copy[c].astype(str)
    return X_copy
    

pipe_change_dtype = Pipeline([
    ('change', FunctionTransformer(change_dtype))
])

X_train_transform = pipe_change_dtype.fit_transform(X_train)
X_test_transform = pipe_change_dtype.fit_transform(X_test)

cols_drop = ['ID']
col_y = 'Reached.on.Time_Y.N'
cols_X_num =[ c for c in X_train_transform.select_dtypes(exclude='str').columns if c not in cols_drop + [col_y]]
cols_X_cat = [ c for c in X_train_transform.select_dtypes(include='str').columns if c not in cols_drop + [col_y]]
cols_X = cols_X_num + cols_X_cat

print(f"cols_X_num : {cols_X_num}")
print(f"cols_X_cat : {cols_X_cat}")

print('-'*100, end='\n')
# ------------------------------------------------------------------------------
# for c in cols_X_cat:
#     print(f"< {c} >")
#     print(X_train_transform[c].value_counts())
#     print()

# print(f"< {col_y} >")
# print(y_train[col_y].value_counts())
# print()

# print('-'*100, end='\n')

# ------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

pipe_num = Pipeline([
    ('encoding', StandardScaler())
])
pipe_cat = Pipeline([
    ('encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

pipe_concat = ColumnTransformer([
    ('pipe_num', pipe_num, cols_X_num),
    ('pipe_cat', pipe_cat, cols_X_cat),
])


preprocessing = Pipeline([
    ('change_dtype', pipe_change_dtype),
    ('encoding', pipe_concat)
])

# print(preprocessing.fit_transform(X_train))
# print('-'*100, end='\n')

# ------------------------------------------------------------------------------
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import compute_class_weight, compute_sample_weight

y_classes = np.unique(y_train[col_y])
class_weight_list = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train[col_y])
class_weight = {k.item(): v.item() for k,v in zip(y_classes, class_weight_list)}
base_weight = {k.item(): 1 for k,v in zip(y_classes, class_weight_list)}


# sample_weight = compute_sample_weight(class_weight='balanced', y=y_train[col_y])


class WrapperClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, weight_mode=None):
        self.estimator = estimator
        self.weight_mode=weight_mode
        
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        
        if self.weight_mode == 'weight':
            fit_params['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y)
        
        self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, 'classes_'):
            self.classes_ = self.estimator_.classes_
        
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
    
    def score(self, X,y):
        return self.estimator_.score(X, y)
    


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {'RF' : WrapperClassifier(RandomForestClassifier()),
          'GB' : WrapperClassifier(estimator=GradientBoostingClassifier()),
          'XGB' : WrapperClassifier(estimator=XGBClassifier()),
          'LGBM' : WrapperClassifier(estimator=LGBMClassifier(verbosity=-1))
          }


print(help(RandomForestClassifier))


params = {}
params['RF'] = {'model__estimator__random_state': [0],
                'model__estimator__max_depth': [3, 10],
                'model__estimator__min_samples_leaf' : [1, 3],
                'model__estimator__max_features': [None, 'sqrt'],
                'model__estimator__class_weight':[base_weight, class_weight]
            }
params['GB'] = {'model__estimator__random_state': [0],
                'model__estimator__learning_rate':[0.03, 0.1],
                'model__estimator__n_estimators':[100,300],
                'model__estimator__max_depth': [3, 10],
                # 'model__estimator__subsample': [0.5, 1.0],
                'model__weight_mode': ['base', 'weight']
                }
params['XGB'] = {'model__estimator__random_state': [0],
                'model__estimator__learning_rate':[0.03, 0.1],
                'model__estimator__n_estimators':[100,300],
                'model__estimator__max_depth': [3, 10],
                # 'model__estimator__subsample': [0.5, 1.0],
                'model__weight_mode': ['base', 'weight']}
params['LGBM'] = {'model__estimator__random_state': [0],
                'model__estimator__learning_rate':[0.03, 0.1],
                'model__estimator__n_estimators':[100,300],
                # 'model__estimator__max_depth': [3, 10],
                'model__estimator__num_leaves' : [31, 63],
                'model__weight_mode': ['base', 'weight']}


# ------------------------------------------------------------------------------
from sklearn.metrics import get_scorer_names
# print(get_scorer_names())
# ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 
# 'balanced_accuracy', 'completeness_score', 'd2_absolute_error_score', 'd2_brier_score', 
# 'd2_log_loss_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 
# 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro',
# 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'matthews_corrcoef', 'mutual_info_score',
# 'neg_brier_score', 'neg_log_loss', 'neg_max_error', 'neg_mean_absolute_error',
# 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 
# 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
# 'neg_negative_likelihood_ratio', 'neg_root_mean_squared_error', 'neg_root_mean_squared_log_error',
# 'normalized_mutual_info_score', 'positive_likelihood_ratio', 'precision', 'precision_macro', 
# 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall',
# 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo',
# 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']



from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
# ------------------------------------------------------------------------------

X = X_train[cols_X]
y = y_train[col_y]

result_best_estimator = {}
result_best_score = {}
result_cv_results = {}

for model_name in models.keys():
# model_name = 'LGBM'
    print(model_name, end= " : ")
    pipe_model = Pipeline([
        ('preprocess', preprocessing),
        ('model', models[model_name])
    ])

    grid_cv = GridSearchCV(estimator=pipe_model, param_grid=params[model_name],
                        cv=cv, scoring='roc_auc', return_train_score=True,
                        verbose=1)
    # print(help(GridSearchCV))

    grid_cv.fit(X, y)

    best_estimator = grid_cv.best_estimator_
    best_score = grid_cv.best_score_
    result_cv = grid_cv.cv_results_

    result_best_estimator[model_name] = best_estimator
    result_best_score[model_name] = best_score
    result_cv_results[model_name] = result_cv
# break


print( np.array(list(result_best_score.values())) )
print( np.array(list(result_best_score.values())).argmax() )
print( np.array(list(result_best_score.values())).max() )
# print(f"AUC : {best_score}")        
# base [RF] AUC : 0.7375935704568033
# base [XGB] AUC : 0.7418999278630954 





# print(dir(sklearn.model_selection))


