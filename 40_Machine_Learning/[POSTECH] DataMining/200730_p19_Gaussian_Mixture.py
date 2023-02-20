import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

from sklearn import metrics

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# Set example dataset
x = np.c_[[4, 20, 3, 19, 17, 8, 19, 18, 12.5],
         [15, 13, 13, 4, 17, 11, 12, 6, 9]]
x_df = pd.DataFrame(x, columns=['experience', 'violation'])
x_df

x_df.iloc[:, 1].median()



# plot example dataset
plt.figure()
plt.scatter(x[:, 0], x[:, 1])
plt.show()


# ?GaussianMixture
# get_ipython().run_line_magic('pinfo', 'GaussianMixture')
# GaussianMixture(
#     n_components=1,
#     *,
#     covariance_type='full',   # spherical, diag, tied, full
#     tol=0.001,
#     reg_covar=1e-06,
#     max_iter=100,
#     n_init=1,
#     init_params='kmeans',
#     weights_init=None,
#     means_init=None,
#     precisions_init=None,
#     random_state=None,
#     warm_start=False,
#     verbose=0,
#     verbose_interval=10,
# )

# implement Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, 
                     covariance_type='full', random_state=1)
gmm.fit(x)


# clustering results (hard assignment)
labels = gmm.predict(x)
print(labels)


# clusterig result (soft assignment)
proba = gmm.predict_proba(x)        # 각 class에 속할 확률
proba_df = pd.DataFrame(proba, 
                       columns=['label_0', 'label_1', 'label_2'])
print(proba_df)


# result statistics
n_instances, n_features = x.shape

pi = gmm.weights_       # 사전확률 정보
pi_around = np.around(pi, decimals=3)
print('=== pi ===')
print('pi_0:', pi_around[0])
print('pi_1:', pi_around[1])
print('pi_2:', pi_around[2])

n = pi * n_instances
print('=== N ===')
print('N0:', n[0])
print('N1:', n[1])
print('N2:', n[2])
print('n의 합:', sum(n))

# mu = gmm.means_   # 중심값
mu = gmm.means_
mu

mu_0 = np.average(x, axis=0, weights=proba[:, 0])
mu_1 = np.average(x, axis=0, weights=proba[:, 1])
mu_2 = np.average(x, axis=0, weights=proba[:, 2])
print('=== mu ===')
print('mu_0:', mu_0)
print('mu_1:', mu_1)
print('mu_2:', mu_2)

# sigma             # 분산, 공분산값
sigma = gmm.covariances_
sigma

print('=== sigma ===')
print('sigma_0:')
print(sigma[0])
print('sigma_1:')
print(sigma[1])
print('sigma_2:')
print(sigma[2])

labels = gmm.predict(x)

    # Plotting
plt.figure(figsize = (8, 6))
plt.scatter(x[:, 0], x[:, 1] ,c = labels)





# ## [실습]
# ### Gausian mixture 모델의 covariance type을 다른 방법으로도 진행
# * covariance_type: spherical, diag, tied, full
# * covariance type에 따라 covariance가 어떻게 변하는지 확인
# * 클러스터링 결과가 어떻게 바뀌는지 확인

    # spherical : 원형태
gmm_spherical = GaussianMixture(n_components=3, 
                     covariance_type='spherical', random_state=1)
gmm_spherical.fit(x)
labels_spherical = gmm_spherical.predict(x)
sigma_spherical = gmm_spherical.covariances_
sigma_spherical     # 분산 == 반지름
    # Plotting
plt.figure(figsize = (8, 6))
plt.scatter(x[:, 0], x[:, 1] ,c = labels_spherical)
plt.grid(alpha=0.5)
plt.show()


    # diag : x축, y축방향으로만 펼처진 형태
gmm_diag = GaussianMixture(n_components=3, 
                     covariance_type='diag', random_state=1)
gmm_diag.fit(x)
labels_diag = gmm_diag.predict(x)
sigma_diag = gmm_diag.covariances_
sigma_diag
    # Plotting
plt.figure(figsize = (8, 6))
plt.scatter(x[:, 0], x[:, 1] ,c = labels_diag)
plt.grid(alpha=0.5)
plt.show()


    # tied : 모든 그룹의 공분산이 같다고 가정
gmm_tied = GaussianMixture(n_components=3, 
                     covariance_type='tied', random_state=1)
gmm_tied.fit(x)
labels_tied = gmm_tied.predict(x)
sigma_tied = gmm_tied.covariances_
sigma_tied      # 공분산행렬 1개
    # Plotting
plt.figure(figsize = (8, 6))
plt.scatter(x[:, 0], x[:, 1] ,c = labels_tied)
plt.grid(alpha=0.5)
plt.show()


    # full : 모든 그룹의 공분산이 같다고 가정
gmm_full = GaussianMixture(n_components=3, 
                     covariance_type='full', random_state=1)
gmm_full.fit(x)
labels_full = gmm_full.predict(x)
sigma_full = gmm_full.covariances_
sigma_full      # 공분산행렬 각 class별 도출
    # Plotting
plt.figure(figsize = (8, 6))
plt.scatter(x[:, 0], x[:, 1] ,c = labels_full)
plt.grid(alpha=0.5)
plt.show()


# 방법별 subplot
fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axe[0][0].set_title('spherical')
axe[0][0].scatter(x[:, 0], x[:, 1] ,c = labels_spherical)
axe[0][1].set_title('diag')
axe[0][1].scatter(x[:, 0], x[:, 1] ,c = labels_diag)
axe[1][0].set_title('tied')
axe[1][0].scatter(x[:, 0], x[:, 1] ,c = labels_tied)
axe[1][1].set_title('full')
axe[1][1].scatter(x[:, 0], x[:, 1] ,c = labels_full)
plt.show()


# -----------------------------------------------------------------------------
# ## Iris data set

# load datasets (Iris)
df = pd.read_csv(absolute_path + 'Iris.csv')
print(df)

# divide independent variables and label
X = df.iloc[:, 1:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()


# ![bic](그림/BIC.png)
bic_save = np.zeros(shape=(len(cv_types), len(n_components_range)))
bic_save        # column : covarience  /  row : n_components


# set parameter
n_components_range = range(2, 5)
cv_types = ['spherical', 'tied', 'diag', 'full']

bic_save = np.zeros(shape=(len(cv_types), len(n_components_range)))
for idx_cv, cv_type in enumerate(cv_types):
    for idx_n_comp, n_components in enumerate(n_components_range):
        gmm = GaussianMixture(n_components=n_components, 
                             covariance_type=cv_type)
        gmm.fit(X)
        bic = gmm.bic(X)
        print(f'{cv_type}|{n_components}: {bic}')
        bic_save[idx_cv, idx_n_comp] = bic

bic_save


# plot above results
from itertools import product

colors = ['navy', 'turquoise', 'cornflowerblue', 'darkorange']
# list(zip(cv_types, colors))



plt.figure()
bar_array = np.zeros(shape=(len(cv_types), len(n_components_range)), dtype='object')
for i, (cv_type, color) in enumerate(zip(cv_types, colors)):
    for j, n_components in enumerate(n_components_range):
        x_pos = n_components + (0.2 * i) - 0.4
        bar_array[i, j] = plt.bar(x_pos, bic_save[i, j], 
                                  width=0.2, color=color)

plt.xticks(n_components_range)
plt.title('BIC score per model')
plt.xlabel('Number of components')
plt.legend([b[0] for b in bar_array], cv_types)

plt.show()

