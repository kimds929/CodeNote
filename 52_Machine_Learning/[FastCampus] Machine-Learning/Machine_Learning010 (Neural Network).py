import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.neural_network import MLPClassifier    # Neural Network
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification


# Neural Network 예제 -------------------------------------------------------------------------
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]

NN_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    # activation = 'logistic'               # 함수
    # alpha = 작을수록 제약을 두지 않음 (클수록 과적합 방지)
    # solver = 'lbfgs', 'sgd', 'adam'       # 최적의 가중치를 찾아주는 방법(알고리즘)
    # hidden_layer_sizes = [5,2]         # 은닉층 노드수

fit_NN = NN_model.fit(X, y)

fit_NN.predict([[2.,2.], [-1.,-2.]])
fit_NN.coefs_       # Hidden Layer 각각의 Weight
[coef.shape for coef in fit_NN.coefs_]



# 2. model의 복잡도에 따른 퍼포먼스 비교 ------------------------------------------------------------

# 설정할 parameter들을 입력.
h = .02      # h는 시각화를 얼마나 자세하게 할 것인가에 대한 위한 임의의 값
alphas = np.logspace(-5, 3, 5)  # 10^-5 ~ 10^3 까지 5개의 값(log-scale)
alphas
names = ['alpha ' + str(i) for i in alphas]
names

# Classifier List (alpha값에 따른 NN model)
classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(solver='lbfgs', alpha=i, random_state=1,
                                     hidden_layer_sizes=[100, 100]))
classifiers


# 데이터 생성
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)

pd.DataFrame(X).head()
pd.DataFrame(y).head()

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)  # 약간의 에러를 부여
linearly_separable = (X, y)



# 여러 모양의 추가 데이터셋 생성
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure = plt.figure(figsize=(17, 9))
i = 1


# DataSet시각화
for X, y in datasets:
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='black', s=25)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6, edgecolors='black', s=25)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()