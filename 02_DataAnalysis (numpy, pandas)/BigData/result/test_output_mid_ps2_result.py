import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

df = pd.read_csv(f"{data_path}/Practice/practice_mid.csv", encoding='utf-8-sig')

# print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                        .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})

# print(df_summary)
# < shape (72, 26) >
#             dtypes  nunique  isna     min      max
# id           int64       72     0       1       72
# split          str        2     0    test    train
# caff       float64       67     0   91.14    97.87
# charger        str        2     0     New      Old
# ch_t       float64       40     0    1.29     2.15
# pre        float64       67     0   57.99    62.96
# post       float64       70     0   57.13    62.73
# grp4           str        4     0      G1       G4
# score      float64       72     0   55.13    81.95
# fert           str        2     0      F1       F2
# water          str        3     0      W1       W3
# yield      float64       71     0   36.95     52.0
# acc            str        5     0       0       5p
# camp           str        2     0     BDA      ENG
# reg            str        2     0       N        Y
# disc         int64       29     0       0       30
# temp         int64       23     0      10       34
# ad           int64       71     0     217      937
# ord        float64       72     0  484.87  1247.24
# age          int64       35     0      22       65
# svc          int64        5     0       1        5
# soc          int64        5     0       1        5
# book         int64        5     0       1        5
# target       int64        2     0       0        1
# pred_prob  float64       72     0  0.0089   0.8949
# pred_cls     int64        2     0       0        1








print('-'*100, end='\n')

# 1.
# charger        str        2     0     New      Old
# ch_t       float64       40     0    1.29     2.15


from scipy.stats import shapiro, levene
x_new = df.query("charger == 'New'")['ch_t']
x_old = df.query("charger == 'Old'")['ch_t']

print(1)
print(f"shapiro : {shapiro(x_new).pvalue}, {shapiro(x_old).pvalue}") # pval1 >0.05, pval2 >0.05 : 둘다 정규성을 따른다.
print(f"levene : {levene(x_new, x_old).pvalue}")        # pval < 0.05 : 두 집단의 분산은 다르다.
print('-'*100, end='\n')


# 2. 다른집단 : 독립표본
from scipy.stats import ttest_ind
result = ttest_ind(x_new, x_old, alternative='less', equal_var=False)

print(2)
print(f"statistic : {result.statistic}")
print(f"p-value : {result.pvalue}")     # pval < 0.05 귀무가설 기각, 평균이 서로 다른 집단이다
print('-'*100, end='\n')


# 3. 같은 관측치 : 대응표본
from scipy.stats import ttest_rel
result = ttest_rel(x_new, x_old, alternative='less')

print(3)
print(f"statistic : {result.statistic}")
print(f"p-value : {result.pvalue}")     # pval < 0.05 귀무가설 기각, New가 Old보다 작다.
print('-'*100, end='\n')


# 4. 같은관측치 순위기반 : wilcoxon
from scipy.stats import wilcoxon
result = wilcoxon(x_new, x_old, alternative='less')

print(4)
print(f"statistic : {result.statistic}")
print(f"p-value : {result.pvalue}")     # pval <0.05, 순위기반으로도 두 집단은 다르다.
print('-'*100, end='\n')


# 5. 
# grp4           str        4     0      G1       G4
# score      float64       72     0   55.13    81.95

from scipy.stats import shapiro, levene

print(5)
group_data = []
for gi, gv in df.groupby('grp4'):
    print(gi, end=' : ')
    result = shapiro(gv['score'])
    print(result.pvalue)
    
    group_data.append(gv['score'])
    
print(levene(*group_data))
print('-'*100, end='\n')
# 결과 : 4그룹 모두 정규성 만족, 등분산 만족


# 6. grp4별 score 비교에서 집단간 제곱합(SSB)를 구해라
# 일원분산분석
# grp4           str        4     0      G1       G4
# score      float64       72     0   55.13    81.95

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols("score ~ C(grp4)", df).fit()
result = anova_lm(model)

print(6)
print(result)
print(f"SSB : {result.loc['C(grp4)']['sum_sq']:.4f}")
print('-'*100, end='\n')



# 7. grp4별 socre 비교에서 집단내 제곱합 (SSE)를 구해리.
print(7)
print(f"SSE : {result.loc['Residual']['sum_sq']:.4f}")
print('-'*100, end='\n')


## 8. p-value
print(8)
print(f"p-value : {result.loc['C(grp4)']['PR(>F)']:.4f}")
print('-'*100, end='\n')


## 9. 이원분산분석
# yield      float64       71     0   36.95     52.0
# fert           str        2     0      F1       F2
# water          str        3     0      W1       W3

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
df['yield_'] = df['yield'].copy()

model = ols("yield_ ~ C(fert) + C(water) + C(fert):C(water)", df).fit()
result = anova_lm(model)

print(9)
print(result)
print(f"p-value : {result.loc['C(water)']['PR(>F)']:.4f}")
print('-'*100, end='\n')

## 10.
print(10)
print(f"p-value : {result.loc['C(fert):C(water)']['PR(>F)']:.4f}")
print('-'*100, end='\n')


## 11. acc에서 5p범주가 차지하는 비율을 구해라.
# acc            str        5     0       0       5p

acc_vc = df['acc'].value_counts().sort_index()

print(11)
print(f"Ans. {acc_vc['5p'] / acc_vc.sum():.4f}")
print('-'*100, end='\n')



## 12. acc분포가 기대비율과 일치하는지 검정
acc_n = list(acc_vc)
expected_ratio = (0.60, 0.25, 0.08, 0.05, 0.02)
expected_n = np.array(expected_ratio) * len(df)

from scipy.stats import chisquare

result = chisquare(acc_n, expected_n)

print(12)
print(f"Ans. {result.pvalue:.4f}")      # p-val : 0.9610 > 0.05 → 기대비율과 일치
print('-'*100, end='\n')



## 13. camp, reg의 관련성(독립여부?) 검정 통계량을 구해라
# camp           str        2     0     BDA      ENG
# reg            str        2     0       N        Y

from scipy.stats import chi2_contingency

ct_mat = pd.crosstab(df['camp'], df['reg'])
result = chi2_contingency(ct_mat)

print(13)
print(f"Ans. {result.pvalue:.4f}")      # p-val : 0.8137 > 0.05 → 독립이다.
print('-'*100, end='\n')


## 14 : ord ~ disc + temp + add에서 temp의 회귀계수의 p-value
# ord        float64       72     0  484.87  1247.24
# disc         int64       29     0       0       30
# temp         int64       23     0      10       34
# ad           int64       71     0     217      937

from statsmodels.formula.api import ols

model = ols("ord ~ disc + temp + ad", df).fit()


print(14)
print(model.summary())

print(model.pvalues)
print(f"Ans1. {model.pvalues['temp']:.4f}")      # p-val : 0.9359 
print(f"Ans2. {model.pvalues['temp'] < 0.05}")      # p-val : 0.9359 > 0.05 유의성이 없음
print('-'*100, end='\n')



## 15.

print(15)
print(f"conf 90% lower bound : {model.conf_int(alpha=0.1).loc['temp'][0]}")
print(f"conf 90% upper bound : {model.conf_int(alpha=0.1).loc['temp'][1]}")
print('-'*100, end='\n')



## 16. ★

pred = model.get_prediction({'disc':15, 'temp':25, 'ad':300})
result = pred.summary_frame(alpha=0.1)

print(16)
print(f"mean 90% ci lower : {result.iloc[0]['mean_ci_lower']}")
print(f"mean 90% ci upper : {result.iloc[0]['mean_ci_upper']}")
# print(np.array(dir(pred)))
print('-'*100, end='\n')


## 17. 
print(17)
print(f"obs 90% ci lower : {result.iloc[0]['obs_ci_lower']}")
print(f"obs 90% ci upper : {result.iloc[0]['obs_ci_upper']}")
print('-'*100, end='\n')


## 18. split : train, target ~ age + svc + soc + book 이진분류모델 log-likelihood, deviance?
# split          str        2     0    test    train

# target       int64        2     0       0        1
# age          int64       35     0      22       65
# svc          int64        5     0       1        5
# soc          int64        5     0       1        5
# book         int64        5     0       1        5

from statsmodels.formula.api import logit

df_train = df.query("split == 'train'")
model = logit("target ~ age + svc + soc + book", df_train).fit()

print(18)
print(model.summary())
print(f"log-likelihood : {model.llf}")
print(f"deviance : {-2 * model.llf}")

print('-'*100, end='\n')



## 19. book 이 3증가 했을 때의 오즈비

print(19)
print(f"book 3 odds ratio : {np.exp(3 * model.params['book']):.4f}")
print('-'*100, end='\n')


## 20. 
print(20)
# print(model.pvalues)
# print(model.params)
print(f"Ans. {model.params[model.pvalues < 0.05].sum():.4f}")
print('-'*100, end='\n')


## 21. 
# from sklearn.metric import
# print(np.array(dir(sklearn.metrics)))
df_test = df.query("split == 'train'")

pred_proba = model.predict(df_test)
pred = (pred_proba > 0.5).astype(int)
df_test['pred'] = pred

cof_mat = df_test.groupby(['pred','target']).size().unstack(['target'])
# print(cof_mat)
print(21)
print(f' . TP : {cof_mat.iloc[0,0]}')
print(f' . TN : {cof_mat.iloc[1,1]}')
print(f' . FP : {cof_mat.iloc[0,1]}')
print(f' . FN : {cof_mat.iloc[1,0]}')
print('-'*100, end='\n')


## 22.
from sklearn.metrics import precision_score, recall_score, f1_score

print(22)
print(f"precision : {precision_score(df_test['target'], pred)}")
print(f"recall : {recall_score(df_test['target'], pred)}")
print(f"f1 : {f1_score(df_test['target'], pred)}")
print('-'*100, end='\n')


## 23.
df_test['pred_03'] = (pred_proba > 0.3).astype(int)
df_test['pred_07'] = (pred_proba > 0.7).astype(int)

print(23)
f1_03 = f1_score(df_test['target'], df_test['pred_03'])
f1_07 = f1_score(df_test['target'], df_test['pred_07'])
print(f"f1 0.3: {f1_03}")
print(f"f1 0.7: {f1_07}")
print(f"f1 compare : {f1_03 > f1_07}")
print()

precision_03 = precision_score(df_test['target'], df_test['pred_03'])
precision_07 = precision_score(df_test['target'], df_test['pred_07'])
print(f"precision 0.3: {precision_03}")
print(f"precision 0.7: {precision_07}")
print(f"precision compare : {precision_03 > precision_07}")
print()

recall_03 = recall_score(df_test['target'], df_test['pred_03'])
recall_07 = recall_score(df_test['target'], df_test['pred_07'])
print(f"recall 0.3: {recall_03}")
print(f"recall 0.7: {recall_07}")
print(f"recall compare : {recall_03 > recall_07}")

print()
print('threshold 0.3 is better.')
print('-'*100, end='\n')


## 24.

# ['ConfusionMatrixDisplay' 'DetCurveDisplay' 'DistanceMetric'
#  'PrecisionRecallDisplay' 'PredictionErrorDisplay' 'RocCurveDisplay'
#  '__all__' '__builtins__' '__cached__' '__doc__' '__file__' '__loader__'
#  '__name__' '__package__' '__path__' '__spec__' '_base' '_classification'
#  '_dist_metrics' '_pairwise_distances_reduction' '_pairwise_fast' '_plot'
#  '_ranking' '_regression' '_scorer' 'accuracy_score'
#  'adjusted_mutual_info_score' 'adjusted_rand_score' 'auc'
#  'average_precision_score' 'balanced_accuracy_score' 'brier_score_loss'
#  'calinski_harabasz_score' 'check_scoring' 'class_likelihood_ratios'
#  'classification_report' 'cluster' 'cohen_kappa_score'
#  'completeness_score' 'confusion_matrix' 'confusion_matrix_at_thresholds'
#  'consensus_score' 'coverage_error' 'd2_absolute_error_score'
#  'd2_brier_score' 'd2_log_loss_score' 'd2_pinball_score'
#  'd2_tweedie_score' 'davies_bouldin_score' 'dcg_score' 'det_curve'
#  'euclidean_distances' 'explained_variance_score' 'f1_score' 'fbeta_score'
#  'fowlkes_mallows_score' 'get_scorer' 'get_scorer_names' 'hamming_loss'
#  'hinge_loss' 'homogeneity_completeness_v_measure' 'homogeneity_score'
#  'jaccard_score' 'label_ranking_average_precision_score'
#  'label_ranking_loss' 'log_loss' 'make_scorer' 'matthews_corrcoef'
#  'max_error' 'mean_absolute_error' 'mean_absolute_percentage_error'
#  'mean_gamma_deviance' 'mean_pinball_loss' 'mean_poisson_deviance'
#  'mean_squared_error' 'mean_squared_log_error' 'mean_tweedie_deviance'
#  'median_absolute_error' 'multilabel_confusion_matrix' 'mutual_info_score'
#  'nan_euclidean_distances' 'ndcg_score' 'normalized_mutual_info_score'
#  'pair_confusion_matrix' 'pairwise' 'pairwise_distances'
#  'pairwise_distances_argmin' 'pairwise_distances_argmin_min'
#  'pairwise_distances_chunked' 'pairwise_kernels' 'precision_recall_curve'
#  'precision_recall_fscore_support' 'precision_score' 'r2_score'
#  'rand_score' 'recall_score' 'roc_auc_score' 'roc_curve'
#  'root_mean_squared_error' 'root_mean_squared_log_error'
#  'silhouette_samples' 'silhouette_score' 'top_k_accuracy_score'
#  'v_measure_score' 'zero_one_loss']
# -------------------------------------



# ['HC0_se' 'HC1_se' 'HC2_se' 'HC3_se' '_HCCM' '__class__' '__delattr__'
#  '__dict__' '__dir__' '__doc__' '__eq__' '__format__' '__ge__'
#  '__getattribute__' '__getstate__' '__gt__' '__hash__' '__init__'
#  '__init_subclass__' '__le__' '__lt__' '__module__' '__ne__' '__new__'
#  '__reduce__' '__reduce_ex__' '__repr__' '__setattr__' '__sizeof__'
#  '__str__' '__subclasshook__' '__weakref__' '_abat_diagonal' '_cache'
#  '_data_attr' '_data_in_cache' '_get_robustcov_results'
#  '_get_wald_nonlinear' '_is_nested' '_transform_predict_exog' '_use_t'
#  '_wexog_singular_values' 'aic' 'bic' 'bse' 'centered_tss'
#  'compare_f_test' 'compare_lm_test' 'compare_lr_test' 'condition_number'
#  'conf_int' 'conf_int_el' 'cov_HC0' 'cov_HC1' 'cov_HC2' 'cov_HC3'
#  'cov_kwds' 'cov_params' 'cov_type' 'df_model' 'df_resid' 'diagn'
#  'eigenvals' 'el_test' 'ess' 'f_pvalue' 'f_test' 'fittedvalues' 'fvalue'
#  'get_influence' 'get_prediction' 'get_robustcov_results' 'info_criteria'
#  'initialize' 'k_constant' 'llf' 'load' 'model' 'mse_model' 'mse_resid'
#  'mse_total' 'nobs' 'normalized_cov_params' 'outlier_test' 'params'
#  'predict' 'pvalues' 'remove_data' 'resid' 'resid_pearson' 'rsquared'
#  'rsquared_adj' 'save' 'scale' 'ssr' 'summary' 'summary2' 't_test'
#  't_test_pairwise' 'tvalues' 'uncentered_tss' 'use_t' 'wald_test'
#  'wald_test_terms' 'wresid']


# ['__class__' '__delattr__' '__dict__' '__dir__' '__doc__' '__eq__'
#  '__format__' '__ge__' '__getattribute__' '__getstate__' '__gt__'
#  '__hash__' '__init__' '__init_subclass__' '__le__' '__lt__' '__module__'
#  '__ne__' '__new__' '__reduce__' '__reduce_ex__' '__repr__' '__setattr__'
#  '__sizeof__' '__str__' '__subclasshook__' '__weakref__' 'conf_int' 'df'
#  'dist' 'dist_args' 'predicted' 'predicted_mean' 'row_labels' 'se'
#  'se_mean' 'se_obs' 'summary_frame' 'var_pred' 'var_pred_mean' 'var_resid']























































