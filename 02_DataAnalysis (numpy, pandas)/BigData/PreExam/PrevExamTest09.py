import numpy as np
import pandas as pd
# D:/DataScience/★GitHub_kimds929/CodeNote/"02_DataAnalysis (numpy, pandas)"/BigData
data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

import warnings
warnings.filterwarnings('ignore')


############################################################################################
# Part 1
############################################################################################
df = pd.read_csv(f"{data_path}/part4/ch9/loan.csv")

print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})

# print(df_summary)

# < shape (5000, 4) >
#      dtypes  nunique  isna         min         max
# 지역코드  int64      592     0  4100000000  4100000591
# 성별    int64        2     0           1           2
# 신용대출  int64     4988     0      100074      999904
# 담보대출  int64     4998     0      500619     9995836


# Problem 1 ---------------------------------------------------------------------------
# P1-1.
df['총대출액'] = df['신용대출'] + df['담보대출']

# P1-2.
df_group = df.groupby(['지역코드','성별'])['총대출액'].sum().unstack('성별')

# P1-3.
df_group['diff'] = (df_group[1] - df_group[2]).abs()
# print(df_group)
# print(df_group['diff'].nlargest(5))
print(f"Ans : {df_group['diff'].nlargest(5).index[0]}")

print('='*100)
print()


# Problem 2 ---------------------------------------------------------------------------

df = pd.read_csv(f"{data_path}/part4/ch9/crime.csv")

print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})

# print(df.head(3))
# print(df_summary)
# < shape (14, 16) >
#        dtypes  nunique  isna   min   max
# 연도      int64        7     0  2014  2020
# 구분        str        2     0  검거건수  발생건수
# 강력범죄    int64        8     0   500  1450
# 절도범죄    int64        6     0   500   900
# 폭력범죄    int64        7     0   400   880
# 지능범죄    int64        7     0   450  1500
# 풍속범죄    int64        3     0   400   750
# 특별경제범죄  int64        5     0   600  1034
# 마약범죄    int64        7     0   350  1400
# 보건범죄    int64        7     0   450  1400
# 환경범죄    int64        7     0   450  1000
# 교통범죄    int64        4     0   550   950
# 노동범죄    int64        4     0   400   800
# 선거범죄    int64        4     0   450   850
# 병역범죄    int64        3     0   400   750
# 기타범죄    int64        5     0   500  1000

df_stack = df.set_index(['연도','구분']).stack()
df_unstack1 = df_stack.unstack('구분')

# P2-1.
df_unstack1['검거율'] =  df_unstack1['검거건수'] / df_unstack1['발생건수']

# P2-2.
df_ratio = df_unstack1['검거율'].unstack(level=1)
columns = df_ratio.columns
# print(df_ratio.shape)
max_idx = np.argmax(df_ratio, axis=1)
# print(np.array(columns)[max_idx])

# P2-3.
df_value = df_unstack1['검거건수'].unstack(level=1)[columns]
# print(df_value)

results = np.array(df_value)[np.arange(len(df_value)), max_idx]
# print(results)

print(f"Ans: {results.sum()}")

print('='*100)
print()




# Problem 3 ---------------------------------------------------------------------------

df = pd.read_csv(f"{data_path}/part4/ch9/hr.csv")

print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})
                            
# print(df_summary)
# < shape (1000, 7) >
#          dtypes  nunique  isna       min        max
# 사원번호        str     1000     0     E0001      E1000
# 부서          str        6     0   Finance      Sales
# 성과등급        str        3     0         A          C
# 연봉        int64      643     0  40000000  149700000
# 근속연수    float64       20    48       1.0       20.0
# 교육참가횟수    int64        6     0         1          6
# 만족도     float64        9   120       1.0        9.0

# 3-1
df['만족도'] = df['만족도'].fillna(df['만족도'].mean())

# 3-2
year_mean = df.groupby(['부서', '성과등급'])['근속연수'].transform('mean').astype(int)
df['근속연수'] = df['근속연수'].fillna(year_mean)

# 3-3
df['연봉/근속연수'] = df['연봉'] / df['근속연수']
idx_3 = df['연봉/근속연수'].nlargest(3).index[-1]
ans_3_3 = df.loc[idx_3]['근속연수']

# 3-4
df['연봉/만족도'] = df['연봉'] / df['만족도']
idx_2 = df['연봉/만족도'].nlargest(2).index[-1]
ans_3_4 = df.loc[idx_2]['교육참가횟수']

# 3-5
ans_3_4 = df.loc[idx_2]['교육참가횟수']
print(f"Ans : {int(ans_3_3 + ans_3_4)}")


print('='*100)
print()




############################################################################################
# Part 2
############################################################################################

df_train = pd.read_csv(f"{data_path}/part4/ch9/farm_train.csv")
df_test = pd.read_csv(f"{data_path}/part4/ch9/farm_test.csv")

# for df_name, df in zip(['train', 'test'], [df_train, df_test]):
#     print(f"< shape {df.shape} >")

#     df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
#                             df.agg(['min','max']).T], axis=1)\
#                                 .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})
                                
#     print(df_summary)
#     print()


# < shape (4000, 9) >
#          dtypes  nunique  isna         min             max
# 농업면적    float64     4000     0  107.237719    99996.405452
# 연도        int64       24     0        2000            2023
# 지역          str       15     0          강원              충북
# 비료사용량   float64     3997     0         0.0       634.82716
# 비료잔여량   float64     4000     0    0.802506  1542162.909278
# 작물종류        str        3     0           밀               쌀
# 토양유형        str        3     0          모래              점토
# 등급          str        3     0           A               C
# 농약검출여부    int64        3     0           0               2

# < shape (1000, 8) >
#         dtypes  nunique  isna         min            max
# 농업면적   float64     1000     0  107.362573   99994.902124
# 연도       int64       24     0        2000           2023
# 지역         str       15     0          강원             충북
# 비료사용량  float64     1000     0    9.392367     679.814876
# 비료잔여량  float64     1000     0    0.703109  571258.416628
# 작물종류       str        3     0           밀              쌀
# 토양유형       str        3     0          모래             점토
# 등급         str        3     0           A              C


cols_drop = []
col_y = '농약검출여부'
cols_X_num = [k for k, v in df_train.nunique().items() if (v > 100) & (k not in cols_drop +[col_y])]
cols_X_cat = [k for k, v in df_train.nunique().items() if (v < 100) & (k not in cols_drop +[col_y])]
cols_X = cols_X_num + cols_X_cat

print(f"cols_X_num : {cols_X_num}" )
print(f"cols_X_cat : {cols_X_cat}" )

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder

def to_numpy(X):
    return np.array(X).astype(float)

pipe_num = Pipeline([
    ('to_float', FunctionTransformer(to_numpy))
]) 

def to_str(X):
    return X.astype(str)
pipe_cat = Pipeline([
    ('to_str', FunctionTransformer(to_str)),
    ('encoder', OneHotEncoder(sparse_output=False))
])


from sklearn.compose import ColumnTransformer
preprocess_X = ColumnTransformer([
    ('pipe_num', pipe_num, cols_X_num),
    ('pipe_cat', pipe_cat, cols_X_cat),
])

# print( preprocess_X.fit_transform(df_train[cols_X]) )

# ===================================================
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import compute_sample_weight
class WrapperClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, weight_mode='base'):
        self.estimator = estimator
        self.weight_mode = weight_mode
    
    def fit(self, X, y, **fit_params):
        self._estimator = clone(self.estimator)
        
        if self.weight_mode == 'balanced':
            fit_params['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y)
        
        self._estimator.fit(X, y, **fit_params)

        if hasattr(self._estimator, 'classes_'):
            self.classes_ = self._estimator.classes_
        
        return self

    def predict(self, X):
        return self._estimator.predict(X)
    
    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
    
    def score(self, X, y):
        return self._estimator.score(X)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


models = {'RF': WrapperClassifier(RandomForestClassifier()),
          'GB': WrapperClassifier(GradientBoostingClassifier()),
          'XGB': WrapperClassifier(XGBClassifier()),
          'LGBM': WrapperClassifier(LGBMClassifier(verbosity=-1))
          }


params = {}
params['RF'] = {'model__estimator__random_state':[0],
                # 'model__estimator__max_depth':[3,5],
                # 'model__estimator__min_samples_leaf':[1,3],
                # 'model__estimator__max_features':['sqrt',1],
                'model__weight_mode': ['base', 'balanced'],
                }
params['GB'] = {'model__estimator__random_state':[0],
                # 'model__estimator__learning_rate':[0.03, 0.1],
                # 'model__estimator__n_estimators':[100, 300],
                # 'model__estimator__max_depth':[3,5],
                'model__weight_mode': ['base', 'balanced'],
                }
params['XGB'] = {'model__estimator__random_state':[0],
                # 'model__estimator__learning_rate':[0.03, 0.1],
                # 'model__estimator__n_estimators':[100, 300],
                # 'model__estimator__max_depth':[3, 5],
                'model__weight_mode': ['base', 'balanced'],
                }
params['LGBM'] = {'model__estimator__random_state':[0],
                # 'model__estimator__num_leaves':[31, 63],
                # 'model__estimator__learning_rate':[0.03, 0.1],
                # 'model__estimator__n_estimators':[100, 300],
                'model__weight_mode': ['base', 'balanced'],
                }


from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

X = df_train[cols_X]
y = df_train[col_y]

results_best_estimator = {}
results_best_score = {}
results_best_params = {}
results_cv_results = {}

# print(help(GridSearchCV))

for model_name, model in models.items():
    print(model_name, end=" : ")
    
    model_pipe = Pipeline([
        ('preprocess', preprocess_X),
        ('model', model)
    ])
    
    grid_search_cv = GridSearchCV(estimator=model_pipe, cv=cv,
                                  param_grid=params[model_name],
                                  return_train_score=True,
                                  scoring='f1_macro',
                                  verbose=1
                                  )

    grid_search_cv.fit(X, y)
    
    print(grid_search_cv.best_score_)
    results_best_estimator[model_name] = grid_search_cv.best_estimator_ 
    results_best_score[model_name] = grid_search_cv.best_score_ 
    results_best_params[model_name] = grid_search_cv.best_params_
    results_cv_results[model_name] = grid_search_cv.cv_results_ 


best_idx = np.argmax(list(results_best_score.values()))
best_model_name = list(models.keys())[best_idx]
print(best_model_name)
print(results_best_score[best_model_name])
best_model = results_best_estimator[best_model_name]

pred = best_model.predict(df_test[cols_X])
pd.DataFrame({'index':df_test.index, 'pred':pred}).to_csv('result.csv', index=False)


# from sklearn.metrics import get_scorer_names
# print(np.array(get_scorer_names()))

# f1_micro

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
df = pd.read_csv(f"{data_path}/part4/ch9/design.csv")


print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})
                            
# print(df_summary)

# < shape (200, 7) >
#          dtypes  nunique  isna        min         max
# id        int64      200     0   1.000000  200.000000
# design  float64      200     0  36.260326   81.098586
# c1      float64      200     0   0.005522    0.986887
# c2      float64      200     0   0.005062    0.990505
# c3      float64      200     0   0.010838    0.999718
# c4      float64      200     0   0.018110    0.989960
# c5      float64      200     0   0.004632    0.996874

df_train = df.query("id <= 140")
df_test = df.query("id > 140")


# P 1-1
from statsmodels.formula.api import ols
model = ols("design ~ c1 + c2 + c3 + c4 + c5", df_train).fit()

pval_TF = (model.pvalues[1:] < 0.05)
# print(pval_TF)

print(1)
print(f"Ans : {pval_TF.sum()}")
print('-'*100)

# P 1-2
from statsmodels.formula.api import ols
model_new = ols("design ~ c1 + c2 + c4", df_train).fit()

pred_train = model_new.predict(df_train)
# df_train['pred_train'] = pred_train
# print(df_train[['design','pred_train']].corr())
from scipy.stats import pearsonr
result = pearsonr(pred_train, df_train['design'])

print(2)
print(f"Ans : {result.statistic:.3f}")
print('-'*100)

# P 1-3
from sklearn.metrics import root_mean_squared_error
pred_test = model_new.predict(df_test)
result = root_mean_squared_error(df_test['design'], pred_test)

print(3)
print(f"Ans : {result:.3f}")
print('-'*100)




# Problem 2 ---------------------------------------------------------------------------
df = pd.read_csv(f"{data_path}/part4/ch9/retention.csv")


print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0: 'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)

# < shape (80, 6) >
#                    dtypes  nunique  isna        min        max
# CustomerID          int64       80     0   1.000000  80.000000
# MonthlyCharges    float64       80     0  30.703823  97.784173
# CustomerTenure      int64       50     0   1.000000  72.000000
# HasPhoneService     int64        2     0   0.000000   1.000000
# HasTechInsurance    int64        2     0   0.000000   1.000000
# Churn               int64        2     0   0.000000   1.000000


# P 2-1
from statsmodels.formula.api import logit

model = logit("Churn ~ MonthlyCharges + CustomerTenure + C(HasPhoneService) + C(HasTechInsurance)",df).fit()


print(1)
# print(model.pvalues)
print(f"Ans : {model.pvalues['MonthlyCharges']:.3f}")
print('-'*100)


# P 2-2
print(2)
# print(model.params)
print(f"Ans : {np.exp(model.params['C(HasPhoneService)[T.1]']):.3f}")
print('-'*100)


# P 2-3
pred_proba = model.predict(df)
print(3)
print(f"Ans : {(pred_proba > 0.3).sum()}")
print('-'*100)


