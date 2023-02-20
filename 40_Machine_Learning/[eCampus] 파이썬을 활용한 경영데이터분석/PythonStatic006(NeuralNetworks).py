import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor     # Predicting Numbers
from sklearn.neural_network import MLPClassifier     # Predicting String
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

    # wine Dataset
df = Fun_LoadData('wine')
df.info()
df.head()

df.groupby('Target').count()


# 【 인공신경망 (Neural Network) 】 --------------------------------------------------------------------------------------------------------
# 범주형 변수 숫자화 : Dummy변수 생성
    # One Hot Encoding : 범주형 변수를 숫자로 변환
df_ohe = pd.get_dummies(df, columns=['Target'], drop_first=True) 

df_ohe.info()
df_ohe.describe().T
# df_ohe.describe().transpose()
df_ohe.head()


# 데이터 표준화 수행
    # Min-Max Scale (최대 최소값을 이용해 Scale을 변경)
scaler = MinMaxScaler()
df_s = pd.DataFrame(scaler.fit_transform(df_ohe))
df_s.columns = df_ohe.columns
df_s
df_s.describe().T

# x값, y값 선정
y = 'Target_class_1'        # MLPClassifier 의 y값은 범주형을 치환한 값
df_y = df_s[y]
df_x = df_s.drop([y], axis=1)

df_s.groupby(y).count()

# 학습데이터, 테스트 데이터 나누기
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3)

mlp = MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=[5]) 
    # activation = 'logistic'               # 함수
    # solver = 'lbfgs', 'sgd', 'adam'       # 최적의 가중치를 찾아주는 방법
    # hidden_layer_sizes = [4,10,2]         # 은닉층 노드수
mlp.fit(train_x, train_y)

print(f'Accuracy of Training : {mlp.score(train_x, train_y)}')
print(f'Accuracy of Test : {mlp.score(test_x, test_y)}')

predict_y = mlp.predict(df_x)       #예측결과
df_mlpPredict = pd.concat( [pd.Series( mlp.predict(df_x) ), df_x], axis = 1 )
df_mlpPredict = df_mlpPredict.rename(columns={0: 'predict'})


from sklearn.metrics import classification_report,confusion_matrix

predict_test_y = mlp.predict(test_x)       #예측결과
print(confusion_matrix(test_y, predict_test_y))     # confusion_matrix
print(classification_report(test_y,predict_test_y)) # classification_report
len(mlp.coefs_)
mlp.coefs_
    # coefs_ is list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.
len(mlp.intercepts_)
mlp.intercepts_
    # intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.

# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/



# 인공신경망 성과측정 : k겹 교차검증
from sklearn.model_selection import cross_val_score

auc = []
iterN = range(1,16)

for i in iterN:
    mlp = MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=[i]) 
    mlp.fit(train_x, train_y)
    scores = cross_val_score(mlp, df_x, df_y, cv=5, scoring='roc_auc')
    auc.append(scores.mean())

auc     # AUC value
np.mean(auc)


# 인공신경망 시각화
import matplotlib.pyplot as plt

plt.plot(iterN, auc, 'bx-')
plt.xlabel('Number of Nodes')
plt.ylabel('AUC')
plt.show()


# 파라미터 튜닝(parameter tuning) : 학습에 필요한 각종 파라미터의 최적값을 찾는 행위

# ---------------------------------------------------------------------------------------------------------------------------