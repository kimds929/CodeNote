# (Python) Decision_Tree 230102
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\Reference1) ★★ Python_정리자료(Git)\DS_Library')
from DS_DataFrame import *

database_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'

# Titanic Dataset
df = pd.read_csv(f"{database_path}/datasets_Titanic_Simple.csv", encoding='utf-8-sig')
df.shape

df['pclass'] = df['pclass'].astype(int).astype(str)
df['survived'] = df['survived'].astype(int).astype(str)
df.info()
# Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
# Survived : 0 = 사망, 1 = 생존
# Sex : male = 남성, female = 여성
# Age : 나이
# SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
# Parch : 타이타닉 호에 동승한 부모 / 자식의 수
# Fare : 승객 요금
# Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴




# Decision_Tree -------------------------------------------------------------------------

def decision_tree(data, y_name, x_name):
    gini = {}
    gini['gini'] = None
    gini['class'] = None

    gini_ = {}
    for sub_group in sorted(data[x_name].unique()):
        g1 = data.query(f"{x_name} == '{sub_group}'")
        g2 = data.query(f"{x_name} != '{sub_group}'")
        
        g1_0, g1_1 = g1[y_name].value_counts().sort_index()
        g2_0, g2_1 = g2[y_name].value_counts().sort_index()

        gini_g1 = 1 - (g1_0/len(g1))**2 - (g1_1/len(g1))**2
        gini_g2 = 1 - (g2_0/len(g2))**2 - (g2_1/len(g2))**2

        gini_value = len(g1)/len(data)*gini_g1 +  len(g2)/len(data)*gini_g2
        
        gini[sub_group] = {}
        gini[sub_group]['gini'] = gini_value
        gini[sub_group]['child_gini'] = (gini_g1, gini_g2)
        gini[sub_group]['child_value'] = ((g1_0, g1_1), (g2_0, g2_1))
        
        gini_[sub_group] = gini_value

    argmin_idx = np.argmin(list(gini_.values()))
    argmin_class = list(gini_.keys())[argmin_idx]
    gini['class'] = argmin_class
    gini['gini'] = gini_[argmin_class]
    return gini




df_tree = df[['survived','pclass', 'sex']]
df_tree['sex'] = (df_tree['sex'] == 'male').astype(int).astype(str)


y_col = 'survived'

# gini = {}
# for pclass in sorted(df['pclass'].unique()):
#     g1 = df.query(f"pclass == '{pclass}'")
#     g2 = df.query(f"pclass != '{pclass}'")
    
#     g1_0, g1_1 = gv['survived'].value_counts().sort_index()
#     g2_0, g2_1 = gv['survived'].value_counts().sort_index()

#     gini_g1 = 1 - (g1_0/len(g1))**2 - (g1_1/len(g1))**2
#     gini_g2 = 1 - (g2_0/len(g2))**2 - (g2_1/len(g2))**2

#     gini_value = len(g1)/len(df)*gini_g1 +  len(g2)/len(df)*gini_g2
    
#     gini[pclass] = gini_value

# argmin_idx = np.argmin(list(gini.values()))
# argmin_key = list(gini.keys())[argmin_idx]

decision_tree(df_tree, 'survived', 'pclass')
decision_tree(df_tree, 'survived', 'sex')



from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
X = df_tree[['pclass', 'sex']]
y = df_tree['survived']


DT = DecisionTreeClassifier()
# DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X, y)


tree.plot_tree(DT, feature_names=X.columns)
tree.plot_tree(DT, feature_names=X.columns, filled=True)   # class의 쏠림에 따라 색상을 부여
tree.plot_tree(DT, feature_names=X.columns, filled=True, max_depth=2)  # max_depth부여

DT.predict_proba(X)
DT.cost_complexity_pruning_path(X, y)   # 변화가 생기는 alpha값 list 및 그때의 불순도
DT.feature_importances_
