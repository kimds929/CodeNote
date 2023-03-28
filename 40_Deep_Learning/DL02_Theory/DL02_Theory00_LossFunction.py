import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# 【 Probability & Likelihood 】------------------------------------------------------------------------------
# https://jinseob2kim.github.io/probability_likelihood.html
# https://jjangjjong.tistory.com/41
# https://www.youtube.com/watch?v=pYxNSUDSFH4&t=45s     # Probability & Likelihood
# https://www.youtube.com/watch?v=XepXtl9YKwc           # Likelihood
# https://www.youtube.com/watch?v=Dn6b9fCIUpM           # Maximum Likelihood

# Probability(확률) : 어떤 고정된 분포에서 해당 값이 관측될 확률
#   . P(범위 low~high |(given) 분포 mean=0.00, std=0.00)

# Likelihood (가능도, 우도추정) : 어떤 값이 관측되었을때, 어떤 확률 분포에서 왔을지에 대한 확률
#   . L(분포 mean=0.00, std=0.00 |(given) 어떤값)

# Maximum Likelihood (최대우도추정) : 관측값에 대한 총 가능도(모든 가능도의 곱)이 최대가 되는 분포(ex.정규분포에서는 평균, 편차)를 찾는 것


# 【 Entrophy 】------------------------------------------------------------------------------
# https://hyunw.kim/blog/2017/10/14/Entropy.html

# n개의 경우의수를 최적으로 찾기 위한 최적의 경우의수 : np.log2(n)   ※ 2진분류문제로 풀기

# 2개 분류문제에서 한가지 경우의 확률이 0 ~ 1.0까지 변할때 기대값
result =[]
for i in np.arange(0, 1.01, 0.01):
    result.append( -(i * np.log2(i) + (1-i) * np.log2(1-i)) )
plt.plot(result)

# n개 분류 문제에서 가질수 있는 최대 Entrophy값
entrophies =[]
for i in range(1,11):
    p = 1/i
    entrophies.append(-p * np.log2(p) * i)
plt.plot(range(1,11), entrophies, 'o-')
# plt.plot(range(1,11), [np.log2(p) for p in range(1,11)], 'o-')

# ★ (Entrophy)
#  . 최적화된 전략 하에서의 질문개수에 대한 기댓값 
#  . n개의 조건들이 다 동일한 확률(1/n)을 가질때 Entrophy는 최대가 된다.
#
#  Entrophy = - ∑ p * np.log2(p)


# 【 Cross Entrophy 】------------------------------------------------------------------------------
# https://hyunw.kim/blog/2017/10/26/Cross_Entropy.html



# ★ (Cross-Entrophy)
#  . 어떤 문제에 대해 특정 전략을 쓸 때 예상되는 질문개수에 대한 기댓값
#  . 최적의 전략을 사용할 때 cross entropy 값이 최소가 됩니다. 
#
# Cross-Entrophy = - ∑ p * np.log2(q)
#                      p : 특정 확률에 대한 참값 또는 목표 확률
#                      q : 우리 분류기가 현재 학습한 확률
#                       ※ q → p에 가까워질수록(최적의 전략에 가까워질수록) cross-entrophy값이 작아지게됨

# (Binary-Classification & Cross-Entrophy)
# CrossEntrophy = -y * np.log2(y_pred) - (1 - y) * (1 - y_pred)
p0 = 0
p1 = 1
qs = np.arange(0, 1.01, 0.01)

cross_entropy_p0 = []
cross_entropy_p1 = []
for q in qs:
    cross_entropy_p0.append(-(p0*np.log2(q) + (1-p0)*np.log2((1-q)) ))
    cross_entropy_p1.append(-(p1*np.log2(q) + (1-p1)*np.log2((1-q)) ))

plt.plot(qs, cross_entropy_p0, label='p : 0', color='steelblue')
plt.plot(qs, cross_entropy_p1, label='p : 1', color='orange')
plt.legend()
plt.axvline(0, color='steelblue', ls='--')
plt.axvline(1, color='orange', ls='--')
plt.axhline(0, color='black', linewidth=1)