import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.decomposition import PCA
# http://scikit-learn.org

# sklearn Dataset Load
def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df

    # breast_cancer Dataset
df = Fun_LoadData('iris')

# iris Data Load
iris = datasets.load_iris()
dir(iris)

X = iris.data[:,[0,2]]  # 2개의 변수만 선택
y = iris.target

# Dimension
X.shape
y.shape

# data → DataFrame
feature_names = [iris.feature_names[0], iris.feature_names[2]]
feature_names
df_X = pd.DataFrame(X)
df_X.columns = feature_names

df_y = pd.DataFrame(y)
df_y.head()

# 결측치 파악
df_X.isnull().sum()
df_y.isnull().sum()

set(y)  # y에 대한 정보
iris.target_names


# 종속변수의 분포를 살핌
df_y[0].value_counts()
df_y[0].value_counts().plot(kind='bar')     # barPlot
df_y[0].hist()
plt.show()

# 독립변수의 분포를 살핌
for i in range(df_X.shape[1]):
    sns.distplot(df_X.iloc[:,i])        # distplot : histogram + area plot
    plt.title(df_X.columns[i])
    plt.show()


# PCA 실습 -----------------------------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X)
# pca.components_       # eigen vector
# pca.explained_variance_     # eigen value

eigen_v = pca.components_           # eigen vector (Column값)
eigen_v

# 새로운 PC Axis에 대한 좌표값
PC_score = pca.transform(X)
PC_score[0:5]

# Centering : 각각의 값에서 평균값을 빼주어 평균을 0으로 만들어주는 작업
mX = np.matrix(X)
for i in range(X.shape[1]):
    mX[:,i] = mX[:,i]-np.mean(X[:,i])
dfmX = pd.DataFrame(mX)
# dfmX.describe().apply(lambda x: round(x,2) if x.dtype==float else x, axis=1)

    # PC값과 Centering된 X값과 eigen_vector와의 값을 비교
# PC_score[0]
# np.dot(np.array(dfmX.loc[0]), eigen_v.transpose())

# 새로운 PC Axis에 대한 좌표값
(mX * eigen_v.transpose())[0:5]     # 행렬곱

# Centering(평균=0)된 데이터에 대한 좌표 및 각 eigen_vector 표시
plt.scatter(dfmX[0], dfmX[1])
origin = [0], [0]
plt.quiver(*origin, eigen_v[:,0], eigen_v[:,1], color=['tomato','mediumseagreen'], scale=3)
plt.show()

# PCA 에 따른 좌표변환 결과
eigen_v_transform = np.dot(eigen_v, eigen_v.transpose())    # eigen vector transform
plt.scatter(PC_score[:,0], PC_score[:,1])
origin = [0], [0]
plt.quiver(*origin, eigen_v_transform[:,0], eigen_v_transform[:,1], color=['tomato','mediumseagreen'], scale=3)
plt.show()




# PCA Model 활용 -----------------------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from DS_OLS_Tools import *

X2 = iris.data
X2.shape
pca2 = PCA(n_components=4)
pca2.fit(X2)
dir(pca2)

    # eigen value, eigen vector
pca2.explained_variance_     # eigen value

# eigen_value의 합 = 각 x값들의 분산의 합
pca2.explained_variance_.sum()      # eigen_value의 합
pd.DataFrame(X2).var().sum()        # 각 x값들의 분산의 합

pca2.explained_variance_ratio_          # eigen value ratio : eigen vector마다 의 설명하는 분산의 비율
pca2.explained_variance_ratio_.sum()    # eigen value ratio의 합 = 1
np.sqrt(pca2.explained_variance_ratio_)     # 각 eigen_value와 다른 변수들과의 상관관계

# eigen_vector ratio plot
plt.plot(range(len(pca2.explained_variance_ratio_ )), pca2.explained_variance_ratio_ , 'o-')
plt.ylim(bottom=0, top=1)
plt.show()

eigen_v2 = pca2.components_     # eigen vector


# 전체변수를 활용시: 모델의 복잡성으로 인해 기존자료를 이용한 분석은 수렴하지 않는 모습
clf = LogisticRegression(solver='sag', multi_class='multinomial').fit(X2, y)     # Y의 Level이 3개이상일때
# dir(clf)
pred = clf.predict(X2)

cfmat = confusion_matrix(y, pred)
cfmat
acc(cfmat)  # accuracy


# PC 2개만을 뽑아내여 분석한 경우 모델이 수렴
PC2 = pca2.transform(X2)[:,0:2]     #2 개의 변수를 선택

clf2 = LogisticRegression(solver='sag', multi_class='multinomial').fit(PC2, y)     # Y의 Level이 3개이상일때
pred2 = clf2.predict(PC2)

cfmat2 = confusion_matrix(y, pred2)
acc(cfmat2)  # accuracy


# 임의변수 2개로 fitting
clf3 = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=1000, random_state=0).fit(X2[:, 0:2], y)     # Y의 Level이 3개이상일때
pred3 = clf3.predict(X2[:,0:2])

cfmat3 = confusion_matrix(y, pred3)
cfmat3
acc(cfmat3)  # accuracy


print(f"[Accuracy] All: {round(acc(cfmat),3)} / PC2: {round(acc(cfmat2),3)} / 1,2Vals : {round(acc(cfmat3),3)}")

