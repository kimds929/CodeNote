import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"



########################################################################################################
########################################################################################################
# 회귀분석
########################################################################################################
########################################################################################################


########################################################################################################
# 상관계수
########################################################################################################

df_hw_simple = pd.DataFrame({
    '키': [150, 160, 170, 175, 165],
    '몸무게': [42, 50, 70, 64, 56]
})

df_hw_simple[['키','몸무게']].corr()
df_hw_simple[['키','몸무게']].corr(method='pearson')       # 선형 관계
df_hw_simple[['키','몸무게']].corr(method='spearman')      # 순위 기반 단조 관계
df_hw_simple[['키','몸무게']].corr(method='kendall')       # 순위쌍의 일치정도


########################################################################################################
# 단순 선형회귀
########################################################################################################

df_hw = pd.DataFrame({
    '키': [150, 160, 170, 175, 165, 155, 172, 168, 174, 158,
          162, 173, 156, 159, 167, 163, 171, 169, 176, 161],
    '몸무게': [42, 50, 70, 64, 56, 48, 68, 60, 65, 52,
            54, 67, 49, 51, 58, 55, 69, 61, 66, 53]
})


from statsmodels.formula.api import ols
model = ols("키 ~ 몸무게", df_hw).fit()
model.summary()


# SSE, SSR, SST
y = df_hw["키"]     # 실제값 y
y_pred = model.fittedvalues     # 예측값 y_hat
y_mean = y.mean()   # 평균값 y_bar

SSE = np.sum((y - y_pred) ** 2)             # SSE: 오차제곱합
SSR = np.sum((y_pred - y_mean) ** 2)        # SSR: 회귀제곱합   
SST = np.sum((y - y_mean) ** 2)             # SST: 총제곱합

print("SSR:", SSR)
print("SSE:", SSE)
print("SST:", SST)
print("SSR + SSE:", SSR + SSE)

# 예측
X_pred = pd.DataFrame([67], columns=['몸무게'])
model.predict(X_pred)

# 신뢰구간
model.conf_int(alpha=0.05)

# 몸무게 50일때의 예측키의 신뢰구간과 예측구간
X_pred = pd.DataFrame([50], columns=['몸무게'])
pred = model.get_prediction(X_pred)
pred.summary_frame(alpha=0.05)




########################################################################################################
# 다중 선형회귀
########################################################################################################

df_ad = pd.DataFrame({
    '매출액': [300, 320, 250, 360, 315, 328, 310, 335, 326, 280,
            290, 300, 315, 328, 310, 335, 300, 400, 500, 600],
    '광고비': [70, 75, 30, 80, 72, 77, 70, 82, 70, 80,
            68, 90, 72, 77, 70, 82, 40, 20, 75, 80],
    '직원수': [15, 16, 14, 20, 19, 17, 16, 19, 15, 20,
            14, 5, 16, 17, 16, 14, 30, 40, 10, 50]
    })

from statsmodels.formula.api import ols

model = ols("매출액 ~ 광고비 + 직원수", df_ad).fit()
model.summary()


# 광고비와 매출액의 상관계수를 구하시오
df_ad[['광고비', '매출액']].corr().iloc[0, 1]


# 광고비와 매출액의 t검정의 p-value를 구하시오 : 상관성여부
from scipy.stats import ttest_ind, pearsonr, shapiro, levene

pearsonr(df_ad['광고비'], df_ad['매출액'])


# 회귀모델의 결정계수를 구하시오
np.array(dir(model))
model.rsquared

# 회귀모델의 회귀계수를 구하시오
model.params

# 회귀모델에서 광고비의 회귀계수가 통계적으로 유의한지 검정했을 때의 p-value를 구하시오
model.pvalues

# 광고비 50, 직원수 20인 데이터가 있을 때 예상 매출액을 구하시오
# model.predict({"광고비": 50, "직원수":20})
model.predict({"광고비": [50], "직원수":[20]})


# 회귀모델의 잔차제곱합을 구하시오
y_pred = model.predict(df_ad[['광고비','직원수']])
y = df_ad['매출액']
sum((y-y_pred)**2)
# sum(model.resid**2)


# 회귀모델의 MSE
((y-y_pred)**2).mean()

# 각 변수별 95% 신뢰구간
model.conf_int(alpha=0.05)


# 광고비 45, 직원수 22일때 95% 신뢰구간과 예측구간을 구하시오
new_data = pd.DataFrame({'광고비':[45], '직원수':[22]})
pred = model.get_prediction(new_data)
result = pred.summary_frame(alpha=0.05)

result



########################################################################################################
# 범주형 변수 
########################################################################################################

df_study = pd.read_csv(f"{data_path}/part3/ch4/study.csv", encoding='utf-8-sig')

df_study.shape
df_study.head(5)

from statsmodels.formula.api import ols

model = ols("score ~ study_hours + C(material_type)", df_study).fit()
model.summary()





########################################################################################################
# 로지스틱 회귀
########################################################################################################


df_health = pd.read_csv(f"{data_path}/part3/ch5/health_survey.csv", encoding='utf-8-sig')

df_health.shape
df_health.head(5)

from statsmodels.formula.api import logit

model = logit("disease ~ age + bmi", df_health).fit()
model.summary()

# bmi변수의 계수값
model.params
model.params['bmi']

# bmi변수가 1단위 증가할때 desease의 오즈비(Odds ratio)?

np.exp(model.params['bmi'])


# log-likelihood 계산방법
model.llf

# 잔차이탈도(deviance)계산방법
-2 * model.llf 


from statsmodels.formula.api import glm
from statsmodels.api import families

# 방법 2 : smt.glm
model = glm("disease ~ age + bmi", data = df_health,
	family = families.Binomial()).fit()  

