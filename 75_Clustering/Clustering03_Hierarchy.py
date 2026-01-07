import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

####################################################################################################
from sklearn.datasets import load_iris

iris = load_iris()
cols = ["_".join(c.split(" ")[:-1]) for c in iris['feature_names']]
df_iris = pd.DataFrame(iris['data'], columns=cols)
iris_target = pd.Series(iris['target']).apply(lambda x:iris['target_names'][x])
# df_iris.insert(0, 'target', iris_target)
df_iris['target'] = iris_target

# df_iris.to_csv("D:/DataScience/DataBase/Data_Tabular/datasets_iris.csv", index=False, encoding='utf-8-sig')
# df_iris = pd.read_csv("D:/DataScience/DataBase/Data_Tabular/datasets_iris.csv", encoding='utf-8-sig')

##############################################################################################################

# 5. 시각화 (꽃잎 길이와 너비 기준)
# plt.scatter(df_iris.iloc[:, 1], df_iris.iloc[:, 3], c=df_iris.iloc[:,0], cmap='viridis')
# plt.xlabel('Petal length (cm)')
# plt.ylabel('Petal width (cm)')
# plt.title('Iris dataset - K-means clustering')
# plt.show()


y = df_iris.iloc[:,-1]
y = pd.Categorical(y, categories=y.value_counts().index)
X = df_iris.iloc[:,:-1]

from sklearn.preprocessing import StandardScaler
X_norm = StandardScaler().fit_transform(X)


##################################################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_emb_pca = pca.fit_transform(X_norm)

plt.figure(figsize=(8, 6))
plt.title('PCA')
scatter = plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()


##################################################################################################
from sklearn.manifold import TSNE

# t-SNE
random_state = 1
tsne = TSNE(n_components=2, random_state=random_state, perplexity=10)
# perplexity (5~50)
#   작은 perplexity → 매우 국소적인 구조를 강조 (작은 클러스터가 잘 보임, 하지만 전체 구조 왜곡 가능)
#   큰 perplexity → 더 넓은 범위의 구조를 반영 (전체적인 분포를 잘 보존, 하지만 작은 구조가 뭉개질 수 있음)
X_emb_tsne = tsne.fit_transform(X_norm)


# 시각화
plt.figure(figsize=(8, 6))
plt.title('t-SNE')
scatter = plt.scatter(X_emb_tsne[:, 0], X_emb_tsne[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()


##################################################################################################

# from sklearn import cluster
#   'AffinityPropagation', 'AgglomerativeClustering', 'Birch',
#   'BisectingKMeans', 'DBSCAN', 'FeatureAgglomeration', 'HDBSCAN',
#   'KMeans', 'MeanShift', 'MiniBatchKMeans', 'OPTICS',
#   'SpectralBiclustering', 'SpectralClustering',
#   'SpectralCoclustering'
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import calinski_harabasz_score


# -----------------------------
# 덴드로그램 그리기 (SciPy)
# -----------------------------
# method: 'ward', 'single', 'complete', 'average'
Z = linkage(X_norm, method='ward')  # ward는 유클리드 기반에서 가장 흔함

plt.figure(figsize=(10, 6))
dend = dendrogram(Z, truncate_mode='lastp', p=30)  # 너무 길면 일부만
plt.title("Dendrogram (truncated)")
plt.xlabel("Cluster (truncated)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()



# -----------------------------
# 실제 클러스터 라벨링 (Scikit-learn)
# -----------------------------
K = 3
clust_model = AgglomerativeClustering(
    n_clusters=K,
    linkage='ward',     # 'ward', 'complete', 'average', 'single'
    metric='euclidean'  # ward면 사실상 euclidean 고정으로 쓰는 경우가 일반적
)
pred_labels = clust_model.fit_predict(X)

# -----------------------------
# 간단 평가 : silhouette_score
# -----------------------------
sil = silhouette_score(X, pred_labels, metric='euclidean')
print("Silhouette score:", sil)


calinski = calinski_harabasz_score(X, pred_labels)
print("Calinski Harabasz score:", calinski)




####################################################################################
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from scipy.optimize import linear_sum_assignment

def hungarian_relabel(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)

    # contingency (true x pred)
    t_index = {t:i for i,t in enumerate(true_ids)}
    p_index = {p:j for j,p in enumerate(pred_ids)}
    cont = np.zeros((len(true_ids), len(pred_ids)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cont[t_index[t], p_index[p]] += 1

    # maximize matches -> minimize negative
    row_ind, col_ind = linear_sum_assignment(-cont)
    mapping = {pred_ids[j]: true_ids[i] for i, j in zip(row_ind, col_ind)}

    y_pred_mapped = np.array([mapping.get(p, -1) for p in y_pred])
    return y_pred_mapped, mapping, cont

pred_labels_match, pred_mapping, pred_contingency =  hungarian_relabel(y.codes, pred_labels)


# visualize
colors = ['steelblue', 'mediumseagreen', 'coral']
plt.figure(figsize=(8, 6))
plt.title('PCA')
for latent_p, y_true, y_pred in zip(x_emb_pca, y.codes, pred_labels_match):
    plt.scatter(latent_p[0], latent_p[1], color=colors[y_true], label=y, s=20)
    plt.scatter(latent_p[0], latent_p[1], facecolors='none', edgecolors=colors[y_pred], s=25, linewidths=1)

true_handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[i], markersize=8, label=f'True {i}')
    for i in range(len(colors))
    ]

pred_handles = [
    Line2D([0], [0], marker='o', color=colors[i], markerfacecolor='none', markersize=8, linewidth=1.5, label=f'Pred {i}')
    for i in range(len(colors))
    ]
# legend 두 개로 분리 
leg1 = plt.legend(handles=true_handles, title='True label', loc='upper right', bbox_to_anchor=(1.2,1))
plt.gca().add_artist(leg1)
plt.legend(handles=pred_handles, title='Pred cluster', loc='upper right', bbox_to_anchor=(1.2,0.8))
plt.show()



# ---------------------------------------------------------------------------------------------------------
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

def external_scores(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
    }

external_scores(y.codes, pred_labels)



####################################################################################
# Explainable Clustering (post-hoc : Decision Tree)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

DT_clf = DecisionTreeClassifier(criterion='entropy')
DT_clf.fit(X, pred_labels)

# (predict)
DT_clf_pred = DT_clf.predict(X)
DT_clf_pred

DT_clf_pred_proba = DT_clf.predict_proba(X)
DT_clf_pred_proba

# (plot)
tree.plot_tree(DT_clf, feature_names=X.columns, filled=True, max_depth=5)  # max_depth부여
plt.show()

# (Feature Importance)
DT_clf.feature_importances_
plt.barh(X.columns, DT_clf.feature_importances_)
plt.show()


# (Evaluate Decision Tree Classifier)
from sklearn.metrics import confusion_matrix
confusion_matrix(pred_labels, DT_clf_pred)
