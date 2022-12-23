import pandas as pd
import numpy as np
import scipy as sp
import math

from matplotlib import pyplot as plt
import matplotlib
from plotnine import *
import seaborn as sns
from sklearn import datasets

# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/supermarket_sales.csv')

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
df = Fun_LoadData('breast_cancer')
df.info()
df.head()

pd.set_option('display.float_format', '{:.3f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None



# 【 분류 기준 이론 】 -------------------------------------------------------------------------------------------------------
# ○ 순수도 : 데이터를 잘 나누었는지 잘 못 나누었는지를 판단하는 기준
    # 지니계수 : 불순도지표 (분류비율의 제곱합)
    # 자식노의 순수도는 두개의 노드가 있으므로 각 노드의 순수도를 구한 후에 두 값을 가중평균하여 구함.
    # 부모노드 갯수 : 15개, 순수도 0.5022
        # 1번 자식노드 : 8개, 순수도 0.5101
        # 2번 자식노드 : 7개, 순수도 0.5312
    # → 전체순수도 = 8/15 * 0.5101 + 7/15 * 0.5312 = 0.5213
    # 순수도가 가장 높은 Case를 선택


# ○ 정보이득 : 부모노드와 자식노드의 엔트로피 값 비교
    # 엔트로피 지표 : 정보이론에서 시스템이 얼마나 정리되지 않았는지를 측정하는 지표
    # 얼마나 많이 섞여있는지를 구함, 엔트로피 값이 감소할수록 순수도가 올라감 → 감소한 만큼을 정보이득으로 판단
    # 다양한 기준으로 나눠본 후 정보이득이 가장 높은 방법을 선택
# ---------------------------------------------------------------------------------------------------------------------------


# 【 Decision Tree 】 --------------------------------------------------------------------------------------------------------
y = 'Target'

from sklearn.tree import DecisionTreeClassifier #, export_graphviz
from sklearn.model_selection import train_test_split
from dtreeplt import dtreeplt

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:], df[y], test_size=0.3, random_state=101)

    # 의사결정나무 실행
dTreeAll = DecisionTreeClassifier(criterion='gini', random_state=0)       # 의사결정트리 선언  # criterion(분류방법) : gini
dTreeAll.fit(train_x, train_y)          # 의사결정나무 모형만들기
print('train_data 성과측정 : {:.3f}' .format(dTreeAll.score(train_x, train_y)))      # 학습집합의 성과측정
print('test_data 성과측정 : {:.3f}' .format(dTreeAll.score(test_x, test_y)))         # 테스트집합의 성과측정

    # 의사결정나무 시각화
dtreePlot = dtreeplt(model = dTreeAll, feature_names = df.columns[2:].tolist(), target_names = df[y].drop_duplicates().tolist())
dtreePlot.view()


    # 의사결정나무 실행 (Depth제한)
dTreeLimit = DecisionTreeClassifier(max_depth=3, random_state=0)    # 의사결정트리 선언: Depth제한
dTreeLimit.fit(train_x, train_y)
dTreeLimit.score(train_x, train_y)
print('train_data 성과측정 : {:.3f}' .format(dTreeLimit.score(train_x, train_y)))      # 학습집합의 성과측정
print('test_data 성과측정 : {:.3f}' .format(dTreeLimit.score(test_x, test_y)))         # 테스트집합의 성과측정

    # 의사결정나무 시각화 (Depth제한)
dtreePlotLimit = dtreeplt(model = dTreeLimit, feature_names = df.columns[2:].tolist(), target_names = df[y].drop_duplicates().tolist())
dtreePlotLimit.view()



    # 변수 중요도 계산
pd.DataFrame( [df.columns[2:].tolist(), dTreeAll.feature_importances_.tolist()], index=['variable','tree_importance']).T\
    .sort_values('tree_importance', ascending=False)        # 전체 모델
pd.DataFrame( [df.columns[2:].tolist(), dTreeLimit.feature_importances_.tolist()], index=['variable','tree_importance']).T\
    .sort_values('tree_importance', ascending=False)        # Depth 제한 모델

    # 변수 중요도 시각화
def Fun_plot_treeImportances(model, x_columns):
    n_features = model.feature_importances_.shape[0]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_columns)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

Fun_plot_treeImportances(dTreeAll, df.columns[2:].tolist())        # 전체 모델
Fun_plot_treeImportances(dTreeLimit, df.columns[2:].tolist())        # Depth 제한 모델


    # 범주형 변수 적용
# 범주형 변수는 연속형(숫자)로 변환 (0, 1)
# 변수의 범주가 3개인 경우, dummuy변수 두개가 필요
pd.get_dummies(df, columns=['Target'], drop_first=True)     # table, column: 연속형변수로 바꿀 열, drop_first: 첫번째 Level은 제거(n-1개)
# pd.get_dummies(df, drop_first=True)
# pd.get_dummies(df)


# 과소적합(underfitting) : 충분히 학습되지 않은경우
# 과적합(overfitting) : 일정 복잡도를 넘은 학습의 경우(학습 집합에 과적합됨)

# ---------------------------------------------------------------------------------------------------------------------------









