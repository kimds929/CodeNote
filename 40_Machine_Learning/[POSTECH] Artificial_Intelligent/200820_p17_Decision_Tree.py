import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


# Basic Tree ---------------------------------------------------------------------------
x = np.linspace(0, 1, 100)
y = -x*np.log2(x) - (1-x)*np.log2(1-x)

# classify function plotting
plt.figure(figsize = (10, 8))
plt.plot(x, y, linewidth = 3)
plt.xlabel(r'$x$', fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.show()


# Quality of test
def D(x):
    y = -x*np.log2(x) - (1-x)*np.log2(1-x)
    return y


# module import 
from sklearn import tree
from sklearn.tree import export_graphviz, plot_tree
import pydotplus
from IPython.display import Image


data = np.array([[0, 0, 1, 0, 0],
                [1, 0, 2, 0, 0],
                [0, 1, 2, 0, 1],
                [2, 1, 0, 2, 1],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 2, 0],
                [1, 1, 0, 2, 0],
                [0, 0, 2, 1, 0]])      

x = data[:,0:4]
y = data[:,4]
print(x, '\n')
print(y)

# model
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state=0)
clf.fit(x,y)

clf.predict([[0, 0, 1, 0]])

# just run this cell to set the PATH variable.

import os, sys
PATH = 'graphviz-2.38\\release\\bin'
os.environ["PATH"] += os.pathsep + PATH


# Tree Display
dot_data = export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_data)

# decision Tree display
Image(graph.create_png())

# plt.figure(figsize=(10,10))
# plot_tree(clf)
# plt.show()



# 1.1. Nonlinear Classification ---------------------------------------------------------------------------

X1 = np.array([[-1.1,0],[-0.3,0.1],[-0.9,1],[0.8,0.4],[0.4,0.9],[0.3,-0.6],
               [-0.5,0.3],[-0.8,0.6],[-0.5,-0.5]])
     
X0 = np.array([[-1,-1.3], [-1.6,2.2],[0.9,-0.7],[1.6,0.5],[1.8,-1.1],[1.6,1.6],
               [-1.6,-1.7],[-1.4,1.8],[1.6,-0.9],[0,-1.6],[0.3,1.7],[-1.6,0],[-2.1,0.2]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

plt.figure(figsize=(10, 8))
plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
plt.title('SVM for Nonlinear Data', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.show()


N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N,1]), np.zeros([M,1])])


clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, random_state=0)
clf.fit(X,y)

clf.predict([[0, 1]])


# to plot
[X1gr, X2gr] = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

# q = clf.predict(Xp)
# q = np.asmatrix(q).reshape(-1,1)
# C1 = np.where(q == 1)[0]
cond = Xp[clf.predict(Xp) == 1]

plt.figure(figsize = (10, 8))
plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
# plt.plot(Xp[C1,0], Xp[C1,1], 'gs', markersize = 8, alpha = 0.1, label = 'Decison Tree')
plt.plot(cond[:,0], cond[:,1], 'gs', markersize = 8, alpha = 0.1, label = 'Decison Tree')
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.show()


# Tree Display
dot_data = export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# plt.figure(figsize=(10,10))
# plot_tree(clf)
# plt.show()




# 1.2. Multiclass Classification ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# generate three simulated clusters
mu1 = np.array([1, 7])
mu2 = np.array([3, 4])
mu3 = np.array([6, 5])

SIGMA1 = 0.8*np.array([[1, 1.5],
                       [1.5, 3]])
SIGMA2 = 0.5*np.array([[2, 0],
                       [0, 2]])
SIGMA3 = 0.5*np.array([[1, -1],
                       [-1, 2]])

X1 = np.random.multivariate_normal(mu1, SIGMA1, 100)
X2 = np.random.multivariate_normal(mu2, SIGMA2, 100)
X3 = np.random.multivariate_normal(mu3, SIGMA3, 100)

y1 = 1*np.ones([100,1])
y2 = 2*np.ones([100,1])
y3 = 3*np.ones([100,1])

plt.figure(figsize = (10, 8))
plt.title('Generated Data', fontsize = 15)
plt.plot(X1[:,0], X1[:,1], '.', label = 'C1')
plt.plot(X2[:,0], X2[:,1], '.', label = 'C2')
plt.plot(X3[:,0], X3[:,1], '.', label = 'C3')
plt.xlabel('$X_1$', fontsize = 15)
plt.ylabel('$X_2$', fontsize = 15)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.axis([-2, 10, 0, 12])
plt.show()



X = np.vstack([X1, X2, X3])
y = np.vstack([y1, y2, y3])

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
clf.fit(X,y)



res = 0.3
[X1gr, X2gr] = np.meshgrid(np.arange(-2,10,res), np.arange(0,12,res))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

# q = clf.predict(Xp)
# q = np.asmatrix(q).reshape(-1,1)
# C1 = np.where(q == 1)[0]
# C2 = np.where(q == 2)[0]
# C3 = np.where(q == 3)[0]

cond1 = Xp[clf.predict(Xp) == 1]
cond2 = Xp[clf.predict(Xp) == 2]
cond3 = Xp[clf.predict(Xp) == 3]

plt.figure(figsize = (10, 8))
plt.plot(X1[:,0], X1[:,1], '.', label = 'C1')
plt.plot(X2[:,0], X2[:,1], '.', label = 'C2')
plt.plot(X3[:,0], X3[:,1], '.', label = 'C3')
# plt.plot(Xp[C1,0], Xp[C1,1], 's', color = 'blue', markersize = 8, alpha = 0.1)
# plt.plot(Xp[C2,0], Xp[C2,1], 's', color = 'orange', markersize = 8, alpha = 0.1)
# plt.plot(Xp[C3,0], Xp[C3,1], 's', color = 'green', markersize = 8, alpha = 0.1)
plt.plot(cond1[:,0], cond1[:,1], 's', color = 'blue', markersize = 8, alpha = 0.1)
plt.plot(cond2[:,0], cond2[:,1], 's', color = 'orange', markersize = 8, alpha = 0.1)
plt.plot(cond3[:,0], cond3[:,1], 's', color = 'green', markersize = 8, alpha = 0.1)
plt.xlabel('$X_1$', fontsize = 15)
plt.ylabel('$X_2$', fontsize = 15)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.axis([-2, 10, 0, 12])
plt.show()

# Tree Display
dot_data = export_graphviz(clf, feature_names=['X1', 'X2'], class_names=['C1', 'C2', 'C3'])
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# plt.figure(figsize=(10,10))
# plot_tree(clf)
# plt.show()
















# 2. Random Forest ---------------------------------------------------------------------------

from sklearn import ensemble

clf = ensemble.RandomForestClassifier(n_estimators = 100, max_depth = 3, random_state= 0)
clf.fit(X,np.ravel(y))

res = 0.3
[X1gr, X2gr] = np.meshgrid(np.arange(-2,10,res), np.arange(0,12,res))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

# q = clf.predict(Xp)
# q = np.asmatrix(q).reshape(-1,1)
# C1 = np.where(q == 1)[0]
# C2 = np.where(q == 2)[0]
# C3 = np.where(q == 3)[0]

cond1 = Xp[clf.predict(Xp) == 1]
cond2 = Xp[clf.predict(Xp) == 2]
cond3 = Xp[clf.predict(Xp) == 3]

plt.figure(figsize = (10, 8))
plt.plot(X1[:,0], X1[:,1], '.', label = 'C1')
plt.plot(X2[:,0], X2[:,1], '.', label = 'C2')
plt.plot(X3[:,0], X3[:,1], '.', label = 'C3')
# plt.plot(Xp[C1,0], Xp[C1,1], 's', color = 'blue', markersize = 8, alpha = 0.1)
# plt.plot(Xp[C2,0], Xp[C2,1], 's', color = 'orange', markersize = 8, alpha = 0.1)
# plt.plot(Xp[C3,0], Xp[C3,1], 's', color = 'green', markersize = 8, alpha = 0.1)
plt.plot(cond1[:,0], cond1[:,1], 's', color = 'blue', markersize = 8, alpha = 0.1)
plt.plot(cond2[:,0], cond2[:,1], 's', color = 'orange', markersize = 8, alpha = 0.1)
plt.plot(cond3[:,0], cond3[:,1], 's', color = 'green', markersize = 8, alpha = 0.1)
plt.xlabel('$X_1$', fontsize = 15)
plt.ylabel('$X_2$', fontsize = 15)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.axis([-2, 10, 0, 12])
plt.show()














# Tutorial: Iris Classification ---------------------------------------------------------------------------
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import ensemble

import numpy as np

iris = datasets.load_iris()
print(iris.DESCR)

X = iris.data
y = iris.target
print(X.shape, y.shape)
print(np.min(X), np.max(X))

X = (X - np.mean(X))/np.std(X)
print(np.min(X), np.max(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
clf.fit(X_train, y_train)


dot_data = export_graphviz(clf, feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'], 
                                class_names=['Setosa', 'Versicolour', 'Virginica'])
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# plt.figure(figsize=(10,10))
# plot_tree(clf)
# plt.show()


clf.score(X_test, y_test)
f1_score(y_true=y_test, y_pred=clf.predict(X_test), average='macro')


# random_forest -----
clf_RF = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf_RF.fit(X_train, y_train)

clf_RF.score(X_test, y_test)
f1_score(y_true=y_test, y_pred=clf_RF.predict(X_test), average='macro')

