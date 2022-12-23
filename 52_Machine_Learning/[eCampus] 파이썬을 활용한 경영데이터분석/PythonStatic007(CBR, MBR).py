import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
# df = pd.read_csv('Database/supermarket_sales.csv')

# sklearn Dataset Load : iris, wine, breast_cancer
def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df

dir(datasets)

    # wine Dataset
df = Fun_LoadData('wine')
df.info()
df.head()
df.describe().T

df.groupby('Target').count()


# 【 사례기반추론(CBR, Case Based Reasoning or Memory Based Reasoning) 】 --------------------------------------------------------------------------------------------------------
# 사례기반추론 : 최근접이웃방안(Nearest Neighbor Approach)을 활용하여 유사한 사례를 찾아 현재의 상황에 적합한 값을 추론하는 방식
    # 과거 데이터를 기반으로 유사한 데이터 n개를 찾아 현재 데이터 값을 추론
# https://blog.naver.com/PostView.nhn?blogId=apr407&logNo=221251820335&parentCategoryNo=&categoryNo=59&viewDate=&isShowPopularPosts=true&from=search
# https://k02174.tistory.com/171


# 유사도
    # 변수간의 거리 : 일반적으로 두 값의 절대값으로 구함

# 최근접 이웃 방안 실행
df_NNA = df.drop('Target', axis=1)  # 최근접이웃거리 : 수치형변수만 사용

y = 'alcohol'
df_y = df_NNA[y]
df_x = df_NNA
df_x.info()
df_x.describe().T

# 학습데이터, 테스트 데이터 나누기
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2)

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_s_trainX = scaler.fit_transform(train_x)
df_s_testX = scaler.fit_transform(test_x)
# scaler.fit(train_x)
# df_s_trainX = scaler.transform(train_x)
# df_s_testX = scaler.transform(test_x)

# K-Nearest
from sklearn.neighbors import KNeighborsRegressor       # Predicting Numbers
from sklearn.neighbors import KNeighborsClassifier       # Predicting String
classifier = KNeighborsRegressor(n_neighbors=5)          # find 5 nearest neighbors
classifier.fit(train_x, train_y)

pred_y = classifier.predict(test_x)
pred_y
mape = np.mean(abs(pred_y - test_y) / test_y)
mape   # 평균절대퍼센티지오차

pd.DataFrame( [pred_y.tolist(), test_y.tolist()], index = ['pred', 'test']).T

error = []

# Calculating error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    error.append( np.mean(abs(pred_y - test_y) / test_y) )
error

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize = [8,6])
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=6)
plt.title('Error Rate K-Value')
plt.xlabel('K-Value')
plt.ylabel('MAPE')
plt.show()


# ---------------------------------------------------------------------------------------------------------------------------





# 【 협업필터링(Collaborative Filtering)) 】 ---------------------------------------------------------------------------------
# 최근접 이웃 알고리즘의 가장 대표적인 적용영역은 추천시스템이다. 
# 현재는 좀 더 정교한 다른 알고리즘이 많이 활용되지만 추천시스템의 기본적인 알고리즘인 협업필터링은 최근접이웃 알고리즘에 기반을 두고 있다.
# 협업필터링 : 최근접이웃 알고리즘 기반 알고리즘
# 최근접이웃 개념 : 본인의 평가 패턴과 가장 유사한 패턴을 보이는 사용자들의 데이터를 이용한 추천방식


# 유사도 구하기 : ① 피어슨 상관계수,  ② 코사인 시뮬레이터
    # 피어슨 상관계수를 더 많이 활용
# 유사 사용자들의 평점 예측 : (-) A의 평균보다 낮게 예측,  (+) A의 평균보다 높게 예측


# df = pd.read_csv('Database/movie_rating.csv')
import surprise
load_df = surprise.Dataset.load_builtin('ml-100k')
df = pd.DataFrame(load_df.raw_ratings, columns=['user', 'item', 'rate', 'id'])
df = df.drop('id', axis=1)

df.info()
df.head()
df.describe().T

# collaborative filtering
# 실습환경 : google colab
    # 런타임 유형변경 → GPU, TPU 설정 : 큰데이터 or 딥러닝 실행 가능
    # Ram, Disk 용량 제한, 12시간 이상 사용 불가
from surprise.model_selection import cross_validate

# 유사한 사용자 찾기 → 예측평점계산 → 오차계산

# [ KNN 최근접이웃 ]
sim_options1 = {'name': 'pearson'}
algo_KNN1 = surprise.KNNBasic(sim_options = sim_options1)       # KNNBasic : 최근접이웃 방법 적용
algo_KNN1_cv = cross_validate(algo_KNN1, load_df)['test_mae']
algo_KNN1_cv
algo_KNN1_cv.mean()


sim_options2 = {'name': 'cosine'}       # 코사인을 활용해 두 벡터각 내각의 크기를 활용해 유사 데이터 찾기
algo_KNN2 = surprise.KNNBasic(sim_options = sim_options2)       # KNNBasic : 최근접이웃 방법 적용
algo_KNN2_cv = cross_validate(algo_KNN2, load_df)['test_mae']
algo_KNN2_cv
algo_KNN2_cv.mean()


# 사용자 기반 필터링은 기본적 추천알고리즙이며, 이 기법이 제시된 이후에
# 아이템기반 협업필터링, Matrix Factorization, SVD 등의 기법이 제시되어 더 좋은 추천 성능을 보였음.


# [ SVD ]
# sim_options3 = {'name': 'pearson'}
algo_SVD = surprise.SVD(n_factors=100)
algo_SVD_cv = cross_validate(algo_SVD, load_df)['test_mae']
algo_SVD_cv
algo_SVD_cv.mean()


# [ Matrix Factorization ]
# sim_options4 = {'name': 'pearson'}
algo_NMF = surprise.NMF(n_factors=100)
algo_NMF_cv = cross_validate(algo_NMF, load_df)['test_mae']
algo_NMF_cv
algo_NMF_cv.mean()

# 추천시스템에 딥러닝도 사용가능
# https://github.com/cheungdaven/DeepRec

# ---------------------------------------------------------------------------------------------------------------------------