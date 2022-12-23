import pandas as pd
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import matplotlib
from plotnine import *
import seaborn as sns
from sklearn import datasets

# https://datascienceschool.net/notebook/ETC/    # 참고사이트 : 데이터 사이언스 스쿨

# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/supermarket_sales.csv')

pd.set_option('display.float_format', '{:.2f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None



# 【 군집화 (Cluster) 】  -------------------------------------------------------------------------------------------
# ○ k-평균 군집화
from sklearn.cluster import KMeans

x1 = 'Rating'
x2 = 'UnitPrice'

df_kmean = df[[x1,x2]]

    #산점도 그리기
plt.plot(df_kmean[x1], df_kmean[x2], 'bo')
ggplot(df_kmean, aes(x=x1, y=x2)) + geom_point(color='steelblue')

    # K-Mean : n개의 초기점을 임의로 정하고, 초점간의 유사도를 계산하여 군집으로 묶어줌 (데이터가 많을수록 유사한 군집화 결과가 나옴)
kmean = KMeans(n_clusters = 5, init='random')    # K-Mean Model 생성,  n_cluster : Cluster 갯수
kmean.fit(df_kmean)

predict = pd.DataFrame( kmean.predict(df_kmean) )   # K-Mean model에 의한 예측값

    # K-Mean model에 의한 예측값 기존 DataTable 에 적용
predict.columns = ['kmean']
df_kmean = pd.concat([df_kmean, predict], axis = 1)
# df_kmean['kmean'] = predict
df_kmean

df_kmean['kmean'] = 'k' + df_kmean['kmean'].astype('str')
# pd.Series(['k']*len(df_kmean)) + df_kmean['kmean'].astype('str')
# for i in df_kmean['kmean'].drop_duplicates().tolist():
#     df_kmean.loc[df_kmean['kmean']==i, 'kmean'] = 'k' + str(i)


# plt.scatter(df_kmean[x1], df_kmean[x2], c=df_kmean['kmean'], alpha=0.5)
# ggplot(df_kmean, aes(x=x1, y=x2)) + geom_point(aes(color='kmean'), alpha=0.5)
sns.scatterplot(x=x1, y=x2, hue='kmean', data=df_kmean)



    # K-Mean 결과해석 및 오류여부 확인
df_kmean.describe()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
    # https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/

    # Min-Max Scale (최대 최소값을 이용해 Scale을 변경)
scaleMinMax = MinMaxScaler()
df_kmean_MinMax = pd.DataFrame(scaleMinMax.fit_transform(df_kmean), columns=[x1, x2] )

kmean.fit(df_kmean_MinMax)
predict_MinMax = pd.DataFrame( kmean.predict(df_kmean_MinMax))   # K-Mean model에 의한 예측값
predict_MinMax.columns = ['kmean']
df_kmean_MinMax_result = pd.concat([df_kmean_MinMax, predict_MinMax], axis = 1)

ggplot(df_kmean_MinMax_result, aes(x=x1, y=x2)) + geom_point(aes(color='kmean'), alpha=0.5)
sns.scatterplot(x=df_kmean[x1], y=df_kmean[x2], data=df_kmean)

    # Standard Scale (평균, 편차를 이용해 정규표준화)
scaleStandard = StandardScaler()
df_kmean_Standard = pd.DataFrame(scaleStandard.fit_transform(df_kmean), columns=[x1, x2] )

kmean.fit(df_kmean_Standard)
predict_Standard = pd.DataFrame( kmean.predict(df_kmean_Standard))   # K-Mean model에 의한 예측값
predict_Standard.columns = ['kmean']
df_kmean_Standard_result = pd.concat([df_kmean_Standard, predict_Standard], axis = 1)

ggplot(df_kmean_Standard_result, aes(x=x1, y=x2)) + geom_point(aes(color='kmean'), alpha=0.5)


    # Scaler Inverse-Transform (standard)
df_kmean_Standard_origin = pd.DataFrame(scaleStandard.inverse_transform(df_kmean_Standard.iloc[:,:2]), columns=[x1, x2] )
df_kmean_Standard_origin_result = pd.concat([df_kmean_Standard_origin, predict_Standard], axis = 1)
df_kmean_Standard_origin_result

ggplot(df_kmean_Standard_origin_result, aes(x=x1, y=x2)) + geom_point(aes(color='kmean'), alpha=0.5)

    # k-mean Group별 평균값 (standard)
df_kmean_Standard_result.groupby('kmean').mean()    # k-mean 그룹별 평균값 (표준화)
df_kmean_Standard_origin_result.groupby('kmean').mean()    # k-mean 그룹별 평균값


# k평균 군집화 : 계층수를 2부터 하나씩 늘려가며 몇개의 계층이 분류가 잘 되었는지를 하나하나 확인해야하는 어려움이 있음
# 해결을위해 팔꿈치방식(Elbow)을 활용해야함   
# Total within-cluster sum of squared : 하나의 클러스터안의 개체간 거리의 합(작을수록 잘 뭉쳐있음) ← n이커질수록 작아짐
# 기울기가 크게 변화하는 구간 찾기
SumOfSquare = []
TotalK = range(2,10)

for k in TotalK:
    km = KMeans(n_clusters=k)
    km.fit(df_kmean_Standard)
    SumOfSquare.append(km.inertia_)    # inertia_ : Total within-cluster sum of squared Value

plt.plot(TotalK, SumOfSquare, 'bx-' )
plt.xlabel('k')
plt.ylabel('Sum of Sqared Distance')
plt.title('THe Elbow Method showing the potimal "k"')
#---------------------------------------------------------------------------------------------------------------------------------




# ○ 계층 군집화 --------------------------------------------------------------------------------------------------------------
# 모든점들간의 거리계산 → 유사한 두 점으로 집합생성 → 모든 집합간의 거리 계산 → 하나의 집합이 될때까지 계속 반복
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from sklearn.metrics import silhouette_score

df_cluster = df[[x1,x2]]
clustering = linkage(df_kmean, method='average')   # method(클러스터 사이거리 측정방식) average : 평균 / complete : 가장먼거리 / single : 가장가까운거리

    # clustering dendrogram plot
# plt.figure(figsize=(7,7))
# dendrogram(clustering,
#            leaf_rotation=90,
#            leaf_font_size=20,
# )

k = 3
fcluster_predict = pd.DataFrame(fcluster(clustering, k, criterion='maxclust'), columns=['predict'])   # criterion = 'distance'
df_cluster_result = pd.concat([df_cluster, fcluster_predict], axis =1 )
df_cluster_result

# silhouette_score로 그래프 그리기
# 군집의 숫자대로 나눴들때 군집화의 성과를 수치적으로 보여주는 지표 (1에가까울수록 군집화가 잘 되었다고 볼 수 있음)
clustering = linkage(df_kmean, method='average')   # method(클러스터 사이거리 측정방식) average : 평균 / complete : 가장먼거리 / single : 가장가까운거리
SilhouetteScore = []
TotalK = range(2,10)
for k in TotalK:
    cl = fcluster(clustering, k, criterion='maxclust')
    SilhouetteScore.append(silhouette_score( df_kmean, cl, metric='euclidean'))

plt.plot(TotalK, SilhouetteScore, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')

# ---------------------------------------------------------------------------------------------------------------------------