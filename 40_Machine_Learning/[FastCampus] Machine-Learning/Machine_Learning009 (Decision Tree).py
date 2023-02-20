from os import system

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
import graphviz
import pydotplus

# Decision_Tree format --------------------------------------------------------
X = [[0, 0], [1, 1]]
Y = [0, 1]

tree_model = tree.DecisionTreeClassifier()
fit_tree = tree_model.fit(X, Y)

fit_tree.predict([[1, 1]])

# Decision_Tree --------------------------------------------------------
iris = datasets.load_iris()
iris.data
iris.target


tree_model = tree.DecisionTreeClassifier(criterion='entropy')
fit_tree = tree_model.fit(iris.data,iris.target)


dot_data = tree.export_graphviz(fit_tree, out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph = graphviz.Source(dot_data)
graph

  # png 파일로 저장
# graph_png = pydotplus.graph_from_dot_data(dot_data)
# graph_png.write_png('tree.png')


  # 프루닝(가지치기) ----------
tree_model2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2) # max_depth 지정 
fit_tree2 = tree_model2.fit(iris.data,iris.target)


dot_data2 = tree.export_graphviz(fit_tree2, out_file=None,
                             feature_names=iris.feature_names,
                            class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True
                             )
graph2 = graphviz.Source(dot_data2)
graph2


confusion_matrix(iris.target, fit_tree.predict(iris.data))
confusion_matrix(iris.target, fit_tree2.predict(iris.data))



  # train-set, test-set (학습데이터, 테스트 데이터) 분리하여 Decision Tree 실행
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, stratify=iris.target, random_state=1)
  # stratify(층화추출) : y값의 class가 골고루 섞이게끔 하는 옵션

tree_model3 = tree.DecisionTreeClassifier(criterion="entropy")
fit_tree3 = tree_model3.fit(X_train,y_train)

confusion_matrix(y_test, fit_tree3.predict(X_test))



# Regression Tree (y값이 연속형) --------------------------------------------------------

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


regr1=tree.DecisionTreeRegressor(max_depth=2)
regr2=tree.DecisionTreeRegressor(max_depth=5)

regr1.fit(X,y)
regr2.fit(X,y)

X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]
X_test

y_1=regr1.predict(X_test)
y_2=regr2.predict(X_test)


plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()



dot_data_reg1 = tree.export_graphviz(regr1, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)
graph_reg1 = graphviz.Source(dot_data_reg1) 
graph_reg1


dot_data_reg2 = tree.export_graphviz(regr2, out_file=None, 
                                filled=True, rounded=True,  
                                special_characters=True)
graph_reg2 = graphviz.Source(dot_data_reg2) 
graph_reg2