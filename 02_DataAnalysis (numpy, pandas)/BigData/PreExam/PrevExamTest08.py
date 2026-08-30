import numpy as np
import pandas as pd
# D:/DataScience/★GitHub_kimds929/CodeNote/"02_DataAnalysis (numpy, pandas)"/BigData
data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

import warnings
warnings.filterwarnings("ignore")

############################################################################################
# Part 1
############################################################################################
df = pd.read_csv(f"{data_path}/part4/ch8/drinks.csv")

# print(f"< shape {df.shape} >")
df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)
# < shape (193, 6) >
#                                dtypes  nunique  isna          min            max
# country                           str      193     0  Afghanistan       Zimbabwe
# beer_servings                   int64      130     0            0            376
# spirit_servings                 int64      109     0            0            438
# wine_servings                   int64       79     0            0            370
# total_litres_of_pure_alcohol  float64       90     0          0.0           14.4
# continent                         str        6     0       Africa  South America


# Problem 1 ---------------------------------------------------------------------------
df_beer_groupc = df.groupby('continent')['beer_servings'].mean()


# P 1-1
print(1)
# print(df_beer_groupc.nlargest())
result =  df_beer_groupc.nlargest().index[0]
print(f"Ans :{result}")

print('-'*100)


# P 1-2
df_europe = df.query("continent == 'Europe'")

europ_top5 = df_europe['beer_servings'].nlargest(5)

print(2)
# print(europ_top5)

result = europ_top5.iloc[-1]
print(f"Ans :{result}")

print('-'*100)



# Problem 2 ---------------------------------------------------------------------------
df = pd.read_csv(f"{data_path}/part4/ch8/tourist.csv")

# print(f"< shape {df.shape} >")
df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)

# < shape (100, 5) >
#    dtypes  nunique  isna  min   max
# 나라    str      100     0  국가1  국가99
# 관광  int64       96     0  509  1499
# 공무  int64       80     0  100   297
# 사업  int64       78     0  202   399
# 기타  int64       67     0   50   149

df['방문객_합계'] = df[['관광','공무','사업','기타']].sum(axis=1)
df['관광객_비율'] = df['관광'] / df['방문객_합계']



# P1.
print(1)
p1_idx = df['관광객_비율'].nlargest(2).index[-1]
a = df.loc[p1_idx]['사업']

print(df['관광객_비율'].nlargest(2))
print(a)
print('-'*100)


# P2
print(2)
p2_idx = df['관광'].nlargest(2).index[-1]
b = df.loc[p2_idx]['공무']

print(df['관광'].nlargest(2))
print(b)
print('-'*100)

# P3
print(3)
print(f"Ans : {a+b}")
print('-'*100)


# Problem 3 ---------------------------------------------------------------------------
df = pd.read_csv(f"{data_path}/part4/ch8/chem.csv")

# print(f"< shape {df.shape} >")
df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)

# < shape (100, 4) >
#        dtypes  nunique  isna  min   max
# sample    str      100     0  샘플1  샘플99
# co      int64       63     0   15   109
# nmhc    int64       61     0   10    98
# etc     int64       64     0   20   117

# P1 / P2
print(1)
from sklearn.preprocessing import MinMaxScaler
std_dict = {}
for c in ['co','nmhc']:
    ms = MinMaxScaler()
    df[f"{c}_ms"] = ms.fit_transform(df[[c]]).ravel()
    std_dict[f"{c}_ms"] = df[f"{c}_ms"].std()
    
print(std_dict)
print('-'*100)

# P3
result = std_dict['co_ms'] - std_dict['nmhc_ms']
print(f"Ans : {result:.3f}")
print('-'*100)



############################################################################################
# Part 2
############################################################################################
df_train = pd.read_csv(f"{data_path}/part4/ch8/churn_train.csv")
df_test = pd.read_csv(f"{data_path}/part4/ch8/churn_test.csv")

# for df_name, df in zip(['train', 'test'], [df_train, df_test]):
#     print(f"< {df_name} shape {df.shape} >")
#     df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
#                             df.agg(['min','max']).T], axis=1)\
#                                 .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
#     print(df_summary)
#     print()
    
# < train shape (4116, 19) >
#                    dtypes  nunique  isna             min           max
# customerID            str     4116     0        CUST0000      CUST5879
# gender                str        2     0          Female          Male
# SeniorCitizen       int64        2     0               0             1
# Partner               str        2     0              No           Yes
# Dependents            str        2     0              No           Yes
# tenure              int64       72     0               1            72
# PhoneService          str        2     0              No           Yes
# MultipleLines         str        3     0              No           Yes
# InternetService       str        3     0             DSL            No
# OnlineSecurity        str        3     0              No           Yes
# OnlineBackup          str        3     0              No           Yes
# DeviceProtection      str        3     0              No           Yes
# TechSupport           str        3     0              No           Yes
# StreamingTV           str        3     0              No           Yes
# StreamingMovies       str        3     0              No           Yes
# Contract              str        3     0  Month-to-month      Two year
# PaperlessBilling      str        2     0              No           Yes
# PaymentMethod         str        4     0   Bank transfer  Mailed check
# TotalCharges      float64     4074     0           20.03        8589.6

# < test shape (1764, 18) >
#                  dtypes  nunique  isna             min           max
# customerID          str     1764     0        CUST0001      CUST5873
# gender              str        2     0          Female          Male
# SeniorCitizen     int64        2     0               0             1
# Partner             str        2     0              No           Yes
# Dependents          str        2     0              No           Yes
# tenure            int64       72     0               1            72
# PhoneService        str        2     0              No           Yes
# MultipleLines       str        3     0              No           Yes
# InternetService     str        3     0             DSL            No
# OnlineSecurity      str        3     0              No           Yes
# OnlineBackup        str        3     0              No           Yes
# DeviceProtection    str        3     0              No           Yes
# TechSupport         str        3     0              No           Yes
# StreamingTV         str        3     0              No           Yes
# StreamingMovies     str        3     0              No           Yes
# Contract            str        3     0  Month-to-month      Two year
# PaperlessBilling    str        2     0              No           Yes
# PaymentMethod       str        4     0   Bank transfer  Mailed check

cols_drop = ['customerID']
col_y = 'TotalCharges'
cols_X_num = [k for k, v in df_train.nunique().items() if (v >= 10) & (k not in cols_drop+[col_y])]
cols_X_cat = [k for k, v in df_train.nunique().items() if (v < 10) & (k not in cols_drop+[col_y])]
cols_X = cols_X_num + cols_X_cat

print(f"cols_X_num : {cols_X_num}" )
print(f"cols_X_cat : {cols_X_cat}" )



###################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def to_float_arr(X):
    return np.array(X.astype(float))

pipe_num = Pipeline([
    ('to_float_arr', FunctionTransformer(to_float_arr))
])

def to_str(X):
    return X.astype(str)

pipe_cat = Pipeline([
    ('to_str', FunctionTransformer(to_str)),
    ('encoding', OneHotEncoder(sparse_output=False))
])

from sklearn.compose import ColumnTransformer
pipe_X = ColumnTransformer([
    ('pipe_num', pipe_num, cols_X_num),
    ('pipe_cat', pipe_cat, cols_X_cat)
])

# print( pipe_X.fit_transform(df_train[cols_X]) )


##################################################
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {'RF': RandomForestRegressor(),
          'GB': GradientBoostingRegressor(),
          'XGB': XGBRegressor(),
          'LGBM': LGBMRegressor()
          }

params = {}
params['RF'] = {'model__random_state':[0],
                'model__min_samples_leaf':[1,3],
                'model__max_depth': [3, 5]
                # 'model__max_features': ['sqrt, 1]
    }
params['GB'] = {'model__random_state':[0],
                'model__learning_rate':[0.03, 0.1],
                # 'model__n_estimators':[100,300],
                'model__max_depth':[3,5],
    }
params['XGB'] = {'model__random_state':[0],
                'model__learning_rate':[0.03, 0.1],
                # 'model__n_estimators':[100,300],
                'model__max_depth':[3,5],
    }
params['LGBM'] = {'model__random_state':[0],
                'model__num_leaves': [31, 63],
                # 'model__n_estimators':[100,300],
                'model__learning_rate':[0.03, 0.1],
    }


print()
from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=0)

X = df_train[cols_X]
y = df_train[col_y]


r_be = {}
r_bs = {}
r_bp = {}
r_cr = {}
for model_name, model in models.items():
    print(model_name, end=' : ')
    
    model_pipe = Pipeline([
        ('preprocessing', pipe_X),
        ('model', model)
    ])
    
    grid_search_cv = GridSearchCV(estimator=model_pipe, cv=cv,
                                  param_grid=params[model_name],
                                  scoring='neg_mean_absolute_error',
                                  return_train_score=True,
                                  verbose=1
                                  )
    grid_search_cv.fit(X,y)
    
    print(grid_search_cv.best_score_)
    r_be[model_name] = grid_search_cv.best_estimator_
    r_bs[model_name] = grid_search_cv.best_score_
    r_bp[model_name] = grid_search_cv.best_params_
    r_cr[model_name] = grid_search_cv.cv_results_


best_idx = np.argmax(np.array(list(r_bs.values())))
best_model_name = list(r_bs.keys())[best_idx]

print()
print('< best estimator select > ')
print(best_model_name)
print(r_bs[best_model_name])

pred = r_be[best_model_name].predict(df_test[cols_X])
pd.DataFrame({'index':df_test.index, 'pred':pred}).to_csv("result.csv", index=False)



# print(help(GridSearchCV))
# estimator 
# param_grid 
# scoring 
# cv 
# verbose 
# return_train_score 

# cv_results_ 
# best_estimator_ 
# best_score_ 
# best_params_ 

# from sklearn.metrics import get_scorer_names
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







############################################################################################
# Part 3
############################################################################################

# Problem 1 ---------------------------------------------------------------------------

df = pd.read_csv(f"{data_path}/part4/ch8/churn.csv")

# print(f"< shape {df.shape} >")
# df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
#                         df.agg(['min','max']).T], axis=1)\
#                             .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)


# < shape (1000, 11) >
#                   dtypes  nunique  isna   min    max
# Churn              int64        2     0   0.0    1.0
# AccountWeeks       int64      184     0 -34.0  234.0
# ContractRenewal    int64        2     0   0.0    1.0
# DataPlan           int64        2     0   0.0    1.0
# DataUsage        float64       71     0  -3.5    4.5
# CustServCalls      int64        9     0  -3.0    5.0
# DayMins          float64      776     0   5.3  322.7
# DayCalls           int64      110     0  10.0  172.0
# MonthlyCharge    float64      524     0  -0.1  117.5
# OverageFee       float64      134     0   1.8   18.0
# RoamMins         float64      139     0   1.6   18.7


from statsmodels.formula.api import logit
cols_X = [c for c in df.columns if c != 'Churn']
cols_X_str = " + ".join(cols_X)

print(cols_X)
model = logit(f"Churn ~ {cols_X_str}", df).fit()


## P1.
print(1)

result = len([k for k, v in model.pvalues[1:].items() if v >= 0.05])
print(f"Ans : {result}")
print('-'*100)



## P2.
cols_X_new = [k for k, v in model.pvalues[1:].items() if v < 0.05]
cols_X_new_str = " + ".join(cols_X_new)
print(cols_X_new)

model_new = logit(f"Churn ~ {cols_X_new_str}", df).fit()

result = model_new.params[model_new.pvalues < 0.05].sum()


print(2)
print(f"Ans : {result:.3f}")
print('-'*100)

## P3.
result = np.exp(5*model_new.params['DataUsage'])

print(3)
print(f"Ans : {result:.3f}")
print('-'*100)





# Problem 2 ---------------------------------------------------------------------------

df = pd.read_csv(f"{data_path}/part4/ch8/piq.csv")

# print(f"< shape {df.shape} >")
# df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
#                         df.agg(['min','max']).T], axis=1)\
#                             .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)

# < shape (50, 4) >
#          dtypes  nunique  isna     min     max
# PIQ       int64       21     0   72.00  150.00
# Brain   float64       31     0   79.06  107.95
# Height  float64       22     0   62.00   77.00
# Weight    int64       28     0  106.00  191.00


from statsmodels.formula.api import ols
cols_X = [c for c in df.columns if c !='PIQ']
coss_X_str = " + ".join(cols_X)
print(cols_X)

model = ols(f"PIQ ~ {coss_X_str}",df).fit()

p1_idx = model.pvalues.nsmallest().index[0]
result = model.params.loc[p1_idx]

## 1.
print(1)
print(f"Ans : {result:.3f}")
print('-'*100)


## 2.
result = model.rsquared

print(2)
print(f"Ans : {result:.2f}")
print('-'*100)


## 3.
result = model.predict({'Brain':90, 'Height':70, 'Weight':150})

print(2)
print(f"Ans : {int(round(result.item(),0))}")
print('-'*100)

