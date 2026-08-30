import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

########################################################################################################
# 다중회귀
########################################################################################################


df_delivery = pd.DataFrame({
    '할인율': [28, 24, 13, 0, 27, 30, 10, 16, 6, 5, 7, 11, 11, 30, 25,
            4, 7, 24, 19, 21, 6, 10, 26, 13, 15, 6, 12, 6, 20, 2],
    '온도': [15, 34, 15, 22, 29, 30, 14, 17, 28, 29, 19, 19, 34, 10,
           29, 28, 12, 25, 32, 28, 22, 16, 30, 11, 16, 18, 16, 33, 12, 22],
    '광고비': [342, 666, 224, 764, 148, 499, 711, 596, 797, 484, 986, 347, 146, 362, 642,
            591, 846, 260, 560, 941, 469, 309, 730, 305, 892, 147, 887, 526, 525, 884],
    '주문량': [635, 958, 525, 25, 607, 872, 858, 732, 1082, 863, 904, 686, 699, 615, 893,
            830, 856, 679, 918, 951, 789, 583, 988, 631, 866, 549, 910, 946, 647, 943]
})

df_delivery.head()

from statsmodels.formula.api import ols

model = ols("주문량 ~ 할인율 + 온도 + 광고비", df_delivery).fit()
model.summary()

# 1. 할인율과 온도의 상관계수
df_delivery[['할인율', '온도']].corr()
round(df_delivery[['할인율', '온도']].corr().iloc[0, 1], 2)


# 2. 모델의 결정계수
round(model.rsquared, 2)

# 3. 각 변수의 회귀계수
model.params.iloc[1:].round(4)

# 4. 절편
model.params.iloc[0].round(4)

# 5. 온도 회귀계수가 통계적으로 유의한지 검정하시오.
model.pvalues['온도'].round(4)


# 6. 주문량 예측
pred = model.predict({'할인율':10, '온도':20, '광고비': 500}).item()
pred

# 7. 잔차제곱합
y = df_delivery['주문량']
y_pred = model.predict(df_delivery[['할인율', '온도', '광고비']])

((y-y_pred)**2).sum()

# 8. MSE
((y-y_pred)**2).mean()

# 9. 온도 회귀계수에 대한 90% 신뢰구간
model.conf_int(alpha=0.1).loc['온도']
# model.conf_int_el(alpha=0.1).loc['온도']

# 10. 신뢰구간 예측
pred = model.get_prediction({'할인율':15, '온도':25, '광고비': 300})
pred.summary_frame(alpha=0.1)
pred.summary_frame(alpha=0.1)[['mean_ci_lower', 'mean_ci_upper']]        # 90% 신뢰구간 : 평균 예측값의 범위**
pred.summary_frame(alpha=0.1)[['obs_ci_lower', 'obs_ci_upper']]         # 90% 예측구간 : 개별 관측값이 나올 범위**


# 11.
model.pvalues['광고비'] < 0.05      # 기각 : 영향을 준다.


model.summary()
model.params[['할인율', '온도']]



########################################################################################################
# 로지스틱회귀
########################################################################################################

import pandas as pd
# df = pd.read_csv("customer_travel.csv")
df_logistic = pd.read_csv("https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/refs/heads/main/part3/ch6/customer_travel.csv")


df_logistic.head()
df_logistic.describe()
df_logistic.nunique()

a = df_logistic.iloc[:400]
b = df_logistic.iloc[400:]
print(a.shape, b.shape)



# 1. 
from statsmodels.formula.api import logit
model = logit("target ~ age + service + social + booked", a).fit()
model.summary()

(model.pvalues[1:] >= 0.05)
(model.pvalues[1:] >= 0.05).sum()


# 2. p-value 값이 가장 큰 변수명
(model.pvalues[1:] < 0.05)

model2 = logit("target ~ age + booked", a).fit()
model2.summary()

model2.pvalues[1:].nlargest(1)
model2.pvalues[1:].nlargest(1).index

# 3. 회귀계수 절대값이 가장 큰 변수명
model2.params[1:].abs().nlargest(1)
model2.params[1:].abs().nlargest(1).index


# 4. log-likelihood
model2.llf

# 5. 잔차이탈도 (deviance)
-2 * model2.llf

# 6. 'booked 변수가 3증가시 오즈비
np.exp( model2.params['booked'] * 3 )

# 7. p-val < 0.05 인 변수들의 회귀계수 합
model2.params[model2.pvalues < 0.05].sum()

# 8.
pred_y = model2.predict(b)

from sklearn.metrics import accuracy_score
acc = accuracy_score(b['target'], (pred_y >0.5).astype(int))
# acc = ((pred_y >0.5).astype(int) == b['target']).sum() / b.shape[0]
acc


# 9
1 - acc






