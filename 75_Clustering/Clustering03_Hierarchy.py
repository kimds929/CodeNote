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
# plt.axhline(y=10, linestyle='--', color='red')  # 높이선
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
pred_labels = clust_model.fit_predict(X_norm)

# -----------------------------
# 간단 평가 : silhouette_score
# -----------------------------
sil = silhouette_score(X_norm, pred_labels, metric='euclidean')
print("Silhouette score:", sil)


calinski = calinski_harabasz_score(X_norm, pred_labels)
print("Calinski Harabasz score:", calinski)



Ks = range(2, 11)
sil_list = []
for k in Ks:
    m = AgglomerativeClustering(n_clusters=k, linkage='ward')
    lab = m.fit_predict(X_norm)
    sil_list.append(silhouette_score(X_norm, lab))
plt.plot(Ks, sil_list, 'o-')
plt.show()

best_k = Ks[int(np.argmax(sil_list))]
best_k, max(sil_list)





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

# confusion matrix
pd.crosstab(y, pred_labels_match, rownames=['True'], colnames=['Cluster'])

# visualize
colors = np.array(['steelblue', 'mediumseagreen', 'coral'])
plt.figure(figsize=(8, 6))
plt.title('PCA')
plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=colors[y.codes], s=20, marker='o', linewidths=0)
K = len(np.unique(pred_labels_match))
for k in range(K):
    idx = (pred_labels_match == k)
    plt.scatter(
        x_emb_pca[:, 0][idx], x_emb_pca[:, 1][idx], facecolors='none', edgecolors=colors[k],
        s=25, marker='o', linewidths=1)
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
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn import tree


DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
DT_clf.fit(X_norm, pred_labels_match)

# (predict)
DT_clf_pred = DT_clf.predict(X_norm)
DT_clf_pred

DT_clf_pred_proba = DT_clf.predict_proba(X_norm)
DT_clf_pred_proba

# (plot)
tree.plot_tree(DT_clf, feature_names=X.columns, filled=True, max_depth=5)  # max_depth부여
plt.show()



# (Feature Importance)
DT_clf.feature_importances_
plt.barh(X.columns, DT_clf.feature_importances_)
plt.show()

from sklearn.inspection import permutation_importance
r = permutation_importance(
    DT_clf, X_norm, pred_labels_match,
    n_repeats=30,
    random_state=0,
    scoring='accuracy'   # fidelity 측면. 필요하면 f1_macro도 가능
)
imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(6,4))
plt.barh(imp.index, imp.values)
plt.title("Permutation Importance (DecisionTree surrogate)")
plt.show()


# (Evaluate Decision Tree Classifier)
from sklearn.metrics import confusion_matrix
confusion_matrix(pred_labels_match, DT_clf_pred)




# ---------------------------------------------------------------------------------
# (text_plot)
rules = export_text(DT_clf, feature_names=list(X.columns))
print(rules)

# leaf별 “클러스터 분포/샘플수”까지 포함
def extract_leaf_rules_with_stats(clf, feature_names, X_data, y_cluster):
    """
    clf: 학습된 DecisionTreeClassifier (surrogate)
    X_data: (n, d)
    y_cluster: (n,) surrogate가 설명하려는 클러스터 라벨(정답처럼 취급)
    """
    tree_ = clf.tree_
    fn = feature_names
    paths = []

    # 각 샘플이 어느 leaf로 가는지
    leaf_id = clf.apply(X_data)

    def recurse(node, rule_parts):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = fn[tree_.feature[node]]
            thr = tree_.threshold[node]

            # left
            recurse(tree_.children_left[node], rule_parts + [f"({name} <= {thr:.3f})"])
            # right
            recurse(tree_.children_right[node], rule_parts + [f"({name} > {thr:.3f})"])
        else:
            # leaf node
            paths.append((node, rule_parts))

    recurse(0, [])

    out = []
    for node, parts in paths:
        # 이 leaf로 떨어지는 샘플들
        idx = (leaf_id == node)
        n = idx.sum()
        if n == 0:
            continue
        vals, cnts = np.unique(y_cluster[idx], return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, cnts)}
        pred_class = int(np.argmax(clf.tree_.value[node][0]))  # leaf에서의 예측 class
        rule = " AND ".join(parts) if parts else "(ROOT)"
        out.append({
            "leaf": int(node),
            "n_samples": int(n),
            "pred_cluster": pred_class,
            "cluster_dist": dist,
            "rule": rule
        })

    # 샘플 많은 순 정렬
    out = sorted(out, key=lambda x: x["n_samples"], reverse=True)
    return out

rules_info = extract_leaf_rules_with_stats(DT_clf, list(X.columns), X_norm, pred_labels_match)

for r in rules_info[:10]:
    print(f"[leaf {r['leaf']}] n={r['n_samples']} pred={r['pred_cluster']} dist={r['cluster_dist']}")
    print("  IF", r["rule"])
    print()    


# -----------------------------------------------------------------------------
# Local explanation: 특정 샘플이 왜 그 클러스터인지 “경로(path)”로 설명

def explain_one_by_path(clf, x_row, feature_names):
    """
    x_row: shape (d,) 또는 (1,d)
    """
    x_row = np.asarray(x_row)
    if x_row.ndim == 1:
        x_row = x_row.reshape(1, -1)

    node_indicator = clf.decision_path(x_row)  # sparse
    leaf_id = clf.apply(x_row)[0]

    tree_ = clf.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node_index = node_indicator.indices  # 방문 노드들

    conditions = []
    for node_id in node_index:
        if node_id == leaf_id:
            continue
        if feature[node_id] < 0:
            continue

        fname = feature_names[feature[node_id]]
        thr = threshold[node_id]
        val = x_row[0, feature[node_id]]

        # 다음 노드가 left인지 right인지로 조건 결정
        go_left = (val <= thr)
        sign = "<=" if go_left else ">"
        conditions.append(f"{fname} (= {val:.3f}) {sign} {thr:.3f}")

    pred = clf.predict(x_row)[0]
    proba = clf.predict_proba(x_row)[0]
    return {
        "pred_cluster": int(pred),
        "proba": proba,
        "leaf_id": int(leaf_id),
        "path_conditions": conditions
    }

# 예: 0번 샘플 설명
i = 100
info = explain_one_by_path(DT_clf, X_norm[i], list(X.columns))
print("Pred cluster:", info["pred_cluster"])
print("Prob:", info["proba"])
print("Leaf:", info["leaf_id"])
print("Path:")
for c in info["path_conditions"]:
    print(" -", c)
