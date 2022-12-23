import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold



iris = datasets.load_iris()
X=iris.data[:,:2]
y=iris.target

# SVM Model ---------------------------------------------------------------
C=1
svm_model =svm.SVC(kernel='linear',C=C)
fitted_svm = svm_model .fit(X,y)
fitted_svm.intercept_
dir(fitted_svm)

y_pred = fitted_svm.predict(X)
confusion_matrix(y, y_pred)


# kernel SVM 적합비교
    # Linear SVC
LinearSVC_model = svm.LinearSVC(C=C, max_iter=10000)
fitted_LinearSVC = LinearSVC_model.fit(X,y)

LinearSVC_y_pred = fitted_LinearSVC.predict(X)
confusion_matrix(y, LinearSVC_y_pred)

    # Radial basis SVC
rbfSVC_model = svm.SVC(kernel='rbf', gamma=0.7, C=C, max_iter=10000)
fitted_rbfSVC = rbfSVC_model.fit(X,y)
rbfSVC_y_pred = fitted_rbfSVC.predict(X)
confusion_matrix(y, rbfSVC_y_pred)

    # Polnomials SVC
# polySVC_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, max_iter=10000)
polySVC_model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
fitted_polySVC = polySVC_model.fit(X,y)
polySVC_y_pred = fitted_polySVC.predict(X)
confusion_matrix(y, polySVC_y_pred)


# 시각적 비교 -----------------------------------------------------------------
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

    # Model 정의
C = 1 #Regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)


    # Model 시각화
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()