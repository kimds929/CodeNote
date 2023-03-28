#!/usr/bin/env python
# coding: utf-8

# import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, accuracy_score)
from sklearn.tree import DecisionTreeClassifier, plot_tree  #, export_graphviz

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# # 1. Hepatitis
# ## 1.1. Load dataset
df = pd.read_csv(absolute_path + 'Hepatitis_.csv')  # Hepatitis 데이터셋 불러오기
df.head()
df_info = DS_DF_Summary(df)


# 독립변수/종속변수 분리
X = df.iloc[:, :-1]
y = df.iloc[:, -1]      # y값 : class

X.head()
y.head()


# ## 1.2. Model learning
# ### 주요 파라미터 for pruning
# #### 1) 불순도 함수
# ##### criterion: {'gini', 'entropy'}
# 
# #### 2) 사전 가지 치기
# ##### max_depth
# ##### min_samples_split
# ##### min_samples_leaf
# 
# #### 3) 사후 가지 치기
# ##### ccp_alpha

# ?DecisionTreeClassifier
# get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')
    # 분지방법
    # 불순도 함수 : gini, entropy
    # 사전 가지치기 : 카이스퀘어 검정, 중단조건 적용  
    #               ※ n이 일정 갯수 이하면 추가로 쪼개지 마라 등
    # 사후 가지치기 : 비용-복잡도, 부정적 오차 가지치기

    # DecisionTreeClassifier(
    #     *,
    #     criterion='gini',         # 불순도 함수 : "gini", "entropy"
    #     splitter='best',          # (Ensemble)
    #     max_depth=None,           # 트리의 깊이(사전 가지치기)
    #     min_samples_split=2,      # 분지하기전에 추가로 쪼갤 조건 : 2개까지는 쪼개라(=다 쪼개라)
    #     min_samples_leaf=1,       # 분지후에 쪼갤 조건 : 분지에서 나오는 갯수가 몇개 이하는 쪼개지마라 : 1 (=다 쪼개라)
    #     min_weight_fraction_leaf=0.0,
    #     max_features=None,        # (Ensemble) : 몇개의 변수를 볼것인지
    #     random_state=None,
    #     max_leaf_nodes=None,
    #     min_impurity_decrease=0.0,
    #     min_impurity_split=None,
    #     class_weight=None,
    #     presort='deprecated',
    #     ccp_alpha=0.0,            # (사후 가지치기) cost-complexity-prunning alpha
    # )


# 모형 선언 후 학습
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# 예측 결과 (label)
y_pred = dtree.predict(X)
print('='*30)
print('predict')
print(y_pred)


# 예측 결과 (probability)
y_prob = dtree.predict_proba(X)
print('='*30)
print('predict_proba')
print(y_prob)

dir(dtree)
dtree.cost_complexity_pruning_path(X, y)    # 변화가 생기는 alpha값 list 및 그때의 불순도
dtree.feature_importances_

# ## 1.3. Plot trained tree
# ?plot_tree
# get_ipython().run_line_magic('pinfo', 'plot_tree')
# plot_tree(
#     decision_tree,
#     *,
#     max_depth=None,
#     feature_names=None,
#     class_names=None,
#     label='all',
#     filled=False,
#     impurity=True,
#     node_ids=False,
#     proportion=False,
#     rotate='deprecated',
#     rounded=False,
#     precision=3,
#     ax=None,
#     fontsize=None,
# )

# ### Basic plot
# plot tree (Basic)
plt.figure(figsize=(25, 15))
plot_tree(dtree)
plt.show()





# ### Plot with variable names
# 변수명 추출
var_names = X.columns
print('Type:', type(var_names))
print(var_names)


# Numpy array 형태로 통일 및 확인
var_names = X.columns.to_numpy()
print('Type:', type(var_names))
print(var_names)

# 각 노드에 분류 column display
plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names)
plt.show()


# ### Plot with class names
label_names = dtree.classes_
print(type(label_names))
print(label_names)


# class name추가
plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names, 
         class_names=True)
plt.show()

# class name string 형태로 지정해서 추가
plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names, 
         class_names=label_names.astype(str))
plt.show()


# ### Plot indicating majority class
plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names, 
         class_names=label_names.astype(str), 
         filled=True)   # class의 쏠림에 따라 색상을 부여
plt.show()


plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names, 
         class_names=label_names.astype(str), 
         filled=True,
         max_depth=2)   # max_depth부여
plt.show()




# ## 1.4. Evaluate model

# Confusion matrix 생성
# [[tn, fp], 
#  [fn, tp]]
con_mat = confusion_matrix(y, y_pred)
print(con_mat)


# precision = tp / (tp + fp)
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
# recall = sensitivity = tp / (tp + fn)
# The recall is intuitively the ability of the classifier to find all the positive samples.
# F1 = 2 * (precision * recall) / (precision + recall)
#The F1 score can be interpreted as a weighted average of the precision and recall,
# where an F1 score reaches its best value at 1 and worst score at 0
report = classification_report(y, y_pred)
print(report)


#define function to compute various measures
def classification_metrics(label, prediction):    
    (tn, fp, fn, tp) = confusion_matrix(label, prediction).reshape(-1)
    
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    auc = roc_auc_score(label, prediction)
    return sen, spe, auc


sensitivity, specificity, auc = classification_metrics(y, y_pred)
print('Tree Sensitivity:', sensitivity)
print('Tree Specificity:', specificity)





# # 2. Iris ------------------------------------------------
# ## 2.1. Load dataset
# Iris 데이터셋 불러오기
df = pd.read_csv(absolute_path + 'Iris.csv')
print(df.head())

# 독립변수와 종속변수 분리
X = df.loc[:,'Sepal.Length':'Petal.Width']
y = df.loc[:, 'Species']

X
y


# DataFrame으로 분석이 가능한 경우 그대로 사용,
# Numpy array 형태로 입력이 필요한 라이브러리 사용시 numpy로 변형 후 사용할 것.
# X = df.loc[:,'Sepal.Length':'Petal.Width'].to_numpy()
# y = df.loc[:, 'Species'].to_numpy()


# ## Model learning
# get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# 모형 선언 후 학습
dtree = DecisionTreeClassifier()
dtree.fit(X, y)


# 예측 결과 (label)
y_pred = dtree.predict(X)
print('='*30)
print('predict')
print(y_pred)


# 예측 결과 (확률)
y_prob = dtree.predict_proba(X)
print('='*30)
print('predict_proba')
print(y_prob)


# ### 학습된 트리 확인
var_names = X.columns
label_names = dtree.classes_


# 트리 그리기
plt.figure(figsize=(25, 15))
plot_tree(dtree, 
         feature_names=var_names, 
          class_names=label_names,
         filled=True)
plt.show()


# ## Evaluation
# Confusion Matrix 생성
con_mat = confusion_matrix(y, y_pred)
print(con_mat)

# precision = tp / (tp + fp)
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
# recall = sensitivity = tp / (tp + fn)
# The recall is intuitively the ability of the classifier to find all the positive samples.
# F1 = 2 * (precision * recall) / (precision + recall)
#The F1 score can be interpreted as a weighted average of the precision and recall,
# where an F1 score reaches its best value at 1 and worst score at 0
print(classification_report(y, y_pred))
    # class가 3개이상이므로, 하나의 class를 기준으로 잡고 계산하게 됨


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)


# ?roc_auc_score
# roc_auc_score(
#     y_true,
#     y_score,
#     *,
#     average='macro',
#     sample_weight=None,
#     max_fpr=None,
#     multi_class='raise',
#     labels=None,
# )
print('ROC AUC:', roc_auc_score(y_bin, y_prob))


lb = LabelBinarizer()
y_bin = lb.fit_transform(y)











# # 실습 - 사전 가지 치기 --------------------------------------------------------------------
# ## Dataset: Hepatitis

# ## Hyper-parameters
# ### - max_depth: (2, 3, 5)
# ### - min_samples_leaf: (10, 15, 20)
# 
# ## 문제: 5-fold cross validation으로 각 파라미터에 대한 평균 train 성능, 평균 test 성능(accuracy, auc) 도출

# ### - 참고사항
# ### cv = KFold(n_splits=5, shuffle=True)
# #### 한번에 작성이 어려울 경우, Holdout test 코드 작성하여 1회 train 성능 및 test 성능 도출
# #### n_instances = X.shape[0]
# #### idx_list = np.arange(n_instances)
# #### train_index, test_index = train_test_split(idx_list, test_size=0.2)
# 
# ## Output example
# #### --------------------------------------------------
# #### max_depth:2, min_samples_leaf:10
# #### train acc: 0.95, train auc: 0.7
# #### test acc: 0.85, test auc: 0.67
# #### --------------------------------------------------
# #### max_depth:2, min_samples_leaf:15
# #### train acc: 0.95, train auc: 0.7
# #### test acc: 0.85, test auc: 0.67
# #### ... (생략)


# print(df_info)


max_depth_list = [2, 3, 5]
min_samples_leaf_list = [10, 15, 20]

result_df = pd.DataFrame()

for md in max_depth_list:
    for msl in min_samples_leaf_list:
        # md = 2
        # msl = 10
        cv5 = KFold(n_splits=5, shuffle=True, random_state=37)

        case_result = {'max_depth' : md, 'min_samples_leaf': msl,

                'train_accuracy_mean': 0, 'train_auc_mean': 0,
                'train_accuracy': [], 'train_auc': [],

                'test_accuracy_mean': 0, 'test_auc_mean': 0,
                'test_accuracy': [], 'test_auc': []}

        for fold, (train_idx, test_idx) in enumerate(cv5.split(df)):
            train_X = df.iloc[train_idx, :-1]
            train_y = df.iloc[train_idx, -1]

            test_X = df.iloc[test_idx, :-1]
            test_y = df.iloc[test_idx, -1]

            dtree_cv = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl).fit(train_X, train_y)

            cv_train_pred_y = dtree_cv.predict(train_X)
            cv_test_pred_y = dtree_cv.predict(test_X)

            train_accuracy = accuracy_score(train_y, cv_train_pred_y)
            test_accuracy = accuracy_score(test_y, cv_test_pred_y)
            case_result['train_accuracy'].append(train_accuracy)
            case_result['test_accuracy'].append(test_accuracy)

            train_auc = roc_auc_score(train_y, cv_train_pred_y)
            test_auc = roc_auc_score(test_y, cv_test_pred_y)
            case_result['train_auc'].append(train_auc)
            case_result['test_auc'].append(test_auc)
            
        case_result['train_accuracy_mean'] = np.mean(case_result['train_accuracy'])
        case_result['test_accuracy_mean'] = np.mean(case_result['test_accuracy'])
        case_result['train_auc_mean'] = np.mean(case_result['train_auc'])
        case_result['test_auc_mean'] = np.mean(case_result['test_auc'])
        result_df = pd.concat([result_df, pd.Series(case_result)], axis=1)
result_summary = result_df.T.reset_index().drop(['index', 'train_accuracy', 'train_auc', 'test_accuracy', 'test_auc'], axis=1)

np.argmax(result_summary['test_accuracy_mean'])
np.argmax(result_summary['test_auc_mean'])















# # 실습 - 최적 트리 확인 --------------------------------------------------------------------------
# ## 위에서 학습되 트리 중 test acc, test auc가 높은 최적 트리를 적당히 정하고, 이를 plot_tree로 출력하라.
# ### 필요시 plot_tree에 max_depth 파라미터를 정수로 입력하여 출력 크기를 조정하라.

# 코드 작성
# ## Cost-complex pruning


# Hepatitis 데이터셋 불러오기
df = pd.read_csv(absolute_path + 'Hepatitis_.csv')
df.head()
df_info = DS_DF_Summary(df)


# 독립변수/종속변수 분리
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X.head()
y.head()


# Alpha 값에 따른 트리 모형의 변화 확인
dtree = DecisionTreeClassifier()
dtree_fit = dtree.fit(X,y)
path = dtree.cost_complexity_pruning_path(X, y)
path

# Alpha에 따른 Impurity array 형태로 도출
ccp_alphas, impurities = path.values()
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)
print(impurities)


# Plot alpha vs impurity
fig, ax = plt.subplots()
# 마지막 alpha와 impurity는 전혀 분할되지 않은 root node만 있는 tree이므로 제외
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()



# 전체 데이터에서 찾은 alpha를 기준으로 모형 비교를 진행
ccp_alphas

# 각 alpha에 대해 tree 학습
tree_list = []
for ccp_alpha in ccp_alphas:
    tree_alpha = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    tree_alpha.fit(X, y)
    tree_list.append(tree_alpha)


# subplots 개념
fig, ax = plt.subplots(nrows=2, ncols=1)
plt.show()


# alpha값이 커질수록 가지치기 갯수가 늘어남 : 조금더 간단한 tree모델이 됨
# alpha값과 depth를 통해 graph를 표현----------------
node_counts = [tree_alpha.tree_.node_count 
               for tree_alpha in tree_list]
depth = [tree_alpha.tree_.max_depth 
         for tree_alpha in tree_list]
fig, ax = plt.subplots(nrows=2, ncols=1)

ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")

ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.show()


# 3번째 트리 구조 확인
var_names = X.columns.to_numpy()
print(var_names)

class_names = tree_list[2].classes_
print(class_names)

plt.figure(figsize=(25, 15))
plot_tree(tree_list[2], 
         feature_names=var_names, 
         class_names=class_names.astype(str), 
         filled=True)
plt.show()




# ## Train-test split with cost-complexity pruning ----------------------------------
fig     # # alpha값과 depth에 따른 graph
# 0.01~0.04 사이에서 유의미한 트리가 있는것으로 보여 해당 구간에서 다시 모델링 실시
alpha_range = np.arange(0.01, 0.04, 0.003)
print(alpha_range)



from sklearn.model_selection import train_test_split

# X.shape로부터 관측치 수, 변수 수 추출
n_instances, n_features = X.shape
print('(n_instances, n_features)', (n_instances, n_features))

# 관측치 인덱스 생성 [0 ~ n_instances-1]
idx = np.arange(n_instances)

# 관측치 인덱스를 train과 test로 분할
train_index, test_index = train_test_split(idx, 
                                          train_size=0.8, 
                                          test_size=0.2,
                                          random_state=0)
# train, test 별 개수 출력
print('n_train:', len(train_index))
print('n_test:', len(test_index))


# 위에서 나눈 train index, test index를 이용해 데이터 분할
X_train, X_test= X.iloc[train_index,:], X.iloc[test_index, :]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# 각 알파에 대해 train을 이용해 훈련 후 train, test 성능(acc, auc)을 계산
acc_train_list = []
acc_test_list = []
auc_train_list = []
auc_test_list = []
for alpha in alpha_range:
    dtree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
    dtree.fit(X_train, y_train)
    
    y_train_pred = dtree.predict(X_train)
    y_test_pred = dtree.predict(X_test)
    
    y_train_prob = dtree.predict_proba(X_train)
    y_test_prob = dtree.predict_proba(X_test)
    
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    auc_train = roc_auc_score(y_train, y_train_prob[:, 1])
    auc_test = roc_auc_score(y_test, y_test_prob[:, 1])
    
    print('='*40)
    print('Alpha:', alpha)
    print('Acc (Train):', acc_train)
    print('ACC (Test):', acc_test)
    
    print('AUC (Train):', auc_train)
    print('AUC (Test):', auc_test)
    
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)
    auc_train_list.append(auc_train)
    auc_test_list.append(auc_test)

    # Accuracy
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alpha_range, acc_train_list, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(alpha_range, acc_test_list, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


    # F1 Score
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("f1_score")
ax.set_title("F1_score vs alpha for training and testing sets")
ax.plot(alpha_range, auc_train_list, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(alpha_range, auc_test_list, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# # 실습
# ## 위 과정을 5-fold cross validation을 이용해 평균 train acc, test acc, train auc, test auc를 계산하고 그래프를 그려 최적 알파를 확인하기.
# ## 5-fold cross validation으로도 확인이 어려울 경우 5-fold cross validation을 여러번 반복하여 평균 성능을 이용하기.
