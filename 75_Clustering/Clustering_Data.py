import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ============================================================
# Plot utils (2D/3D 데이터만 시각화)
# ============================================================
def plot_2d(X, y=None, title="2D dataset"):
    plt.figure()
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], s=10)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

def plot_3d(X, y=None, title="3D dataset"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=8)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.tight_layout()
    plt.show()


# ============================================================
# Helpers
# ============================================================
def _rng(seed):
    return np.random.default_rng(seed)

def _standardize(X, standardize=True):
    return StandardScaler().fit_transform(X) if standardize else X

def _add_gaussian_noise(X, noise_std=0.0, seed=0):
    if noise_std <= 0:
        return X
    r = _rng(seed)
    return X + r.normal(0, noise_std, size=X.shape)


# ============================================================
# 2D / 3D Generators (모두 y_true 포함)
# ============================================================
def gen_blobs_isotropic(n=2000, k=4, cluster_std=1.0, seed=0, standardize=True):
    """[2D] 구형 blob 군집 (KMeans에 유리)"""
    X, y = make_blobs(n_samples=n, centers=k, cluster_std=cluster_std, random_state=seed)
    return _standardize(X, standardize), y.astype(int)

def gen_blobs_anisotropic(n=2000, k=4, cluster_std=1.0, seed=0, standardize=True):
    """[2D] 타원형(비등방) blob 군집 (KMeans는 가능하나 난이도↑)"""
    X, y = make_blobs(n_samples=n, centers=k, cluster_std=cluster_std, random_state=seed)
    A = np.array([[0.6, -0.8],
                  [0.8,  0.6]])
    S = np.array([[2.0, 0.0],
                  [0.0, 0.5]])
    M = A @ S
    X = X @ M
    return _standardize(X, standardize), y.astype(int)

def gen_blobs_varied_density(n=2000, seed=0, standardize=True):
    """[2D] 클러스터별 밀도/분산이 다름 (DBSCAN/HDBSCAN 강점 확인용)"""
    r = _rng(seed)
    centers = np.array([[-6, -2], [-2, 5], [4, -1], [7, 6]])
    stds = np.array([0.4, 1.2, 0.7, 2.2])
    k = len(centers)
    ns = r.multinomial(n, [1.0/k] * k)

    Xs, ys = [], []
    for i, (c, s, ni) in enumerate(zip(centers, stds, ns)):
        Xi = r.normal(loc=c, scale=s, size=(ni, 2))
        Xs.append(Xi)
        ys.append(np.full(ni, i, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _standardize(X, standardize), y

def gen_moons(n=2000, noise=0.08, seed=0, standardize=True):
    """[2D] 두 개의 초승달 모양 (Spectral/DBSCAN에서 강함)"""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return _standardize(X, standardize), y.astype(int)

def gen_circles(n=2000, noise=0.05, factor=0.5, seed=0, standardize=True):
    """[2D] 동심원 (Spectral/DBSCAN에서 강함)"""
    X, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
    return _standardize(X, standardize), y.astype(int)

def gen_spiral(n=2000, turns=3.0, noise_std=0.12, seed=0, standardize=True):
    """[2D] 스파이럴 2개 (비선형 구조, density/graph 계열 테스트)"""
    r = _rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = r.uniform(0, turns * np.pi, n1)
    t2 = r.uniform(0, turns * np.pi, n2)

    x1 = np.vstack([t1 * np.cos(t1), t1 * np.sin(t1)]).T
    x2 = np.vstack([t2 * np.cos(t2 + np.pi), t2 * np.sin(t2 + np.pi)]).T
    X = np.vstack([x1, x2])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    X = _add_gaussian_noise(X, noise_std=noise_std, seed=seed + 7)
    return _standardize(X, standardize), y

def gen_ring_plus_blob(n=2000, seed=0, standardize=True):
    """[2D] 링(원형) + 중앙 blob (형태+밀도 혼재 케이스)"""
    r = _rng(seed)
    n1 = n // 2
    n2 = n - n1

    theta = r.uniform(0, 2*np.pi, n1)
    rad = r.normal(5.0, 0.3, n1)
    ring = np.vstack([rad*np.cos(theta), rad*np.sin(theta)]).T

    blob = r.normal(loc=[0, 0], scale=[1.0, 1.0], size=(n2, 2))

    X = np.vstack([ring, blob])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    return _standardize(X, standardize), y

def gen_swiss_roll_3d(n=3000, noise=0.15, n_bins=3, seed=0, standardize=True):
    """
    [3D] Swiss Roll 매니폴드.
    - 원래는 연속변수 t로 생성되므로 '정답 클러스터'가 애매함.
    - 여기서는 t를 구간(bin)으로 나눠 y_true를 인위적으로 만든다.
    - n_bins=3이면 3개 클러스터(구간) 라벨 생성.
    """
    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    # t 분위수로 구간화 → y_true 생성
    qs = np.quantile(t, np.linspace(0, 1, n_bins + 1)[1:-1])
    y = np.digitize(t, bins=qs).astype(int)
    return _standardize(X, standardize), y


# ============================================================
# High-dimensional Generators (모두 y_true 포함)
# ============================================================
def gen_highdim_gaussian_mixture(n=5000, d=50, k=6, sep=4.0, cov_scale=1.0, seed=0, standardize=True):
    """[고차원] Gaussian mixture (분리 난이도 조절: sep↑ 쉬움 / cov_scale↑ 어려움)"""
    r = _rng(seed)
    ns = r.multinomial(n, [1.0/k] * k)
    centers = r.normal(0, 1, size=(k, d)) * sep

    Xs, ys = [], []
    for i, ni in enumerate(ns):
        std = cov_scale * r.uniform(0.6, 1.4, size=d)
        Xi = r.normal(loc=centers[i], scale=std, size=(ni, d))
        Xs.append(Xi)
        ys.append(np.full(ni, i, dtype=int))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _standardize(X, standardize), y

def gen_subspace_clusters(n=6000, d=100, k=5, sub_dim=10, sep=6.0, noise_std=1.0, seed=0, standardize=True):
    """
    [고차원] 서브스페이스 클러스터링:
    - 각 클러스터가 서로 다른 feature subset(sub_dim)에서만 잘 분리.
    - 전체 차원 d로 보면 거리 기반은 어려워질 수 있음.
    """
    r = _rng(seed)
    ns = r.multinomial(n, [1.0/k] * k)

    X = r.normal(0, noise_std, size=(n, d))
    y = np.empty(n, dtype=int)

    start = 0
    for i, ni in enumerate(ns):
        end = start + ni
        y[start:end] = i
        dims = r.choice(d, size=sub_dim, replace=False)
        shift = r.normal(0, 1, size=sub_dim) * sep
        X[start:end, dims] += shift
        start = end

    return _standardize(X, standardize), y

def gen_overlapping_clusters(n=5000, d=30, k=4, seed=0, standardize=True):
    """[고차원] 의도적으로 많이 겹치는 Gaussian mixture"""
    return gen_highdim_gaussian_mixture(n=n, d=d, k=k, sep=1.2, cov_scale=2.2, seed=seed, standardize=standardize)

def gen_imbalanced(n=6000, d=20, k=5, major_ratio=0.7, sep=4.0, seed=0, standardize=True):
    """[고차원] 클러스터 크기 불균형 (1개가 major_ratio 비중)"""
    r = _rng(seed)
    probs = [major_ratio] + [(1-major_ratio)/(k-1)]*(k-1)
    ns = r.multinomial(n, probs)
    centers = r.normal(0, 1, size=(k, d)) * sep

    Xs, ys = [], []
    for i, ni in enumerate(ns):
        std = r.uniform(0.8, 1.4, size=d)
        Xi = r.normal(centers[i], std, size=(ni, d))
        Xs.append(Xi)
        ys.append(np.full(ni, i, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _standardize(X, standardize), y

def gen_with_outliers(n=5000, d=20, k=4, outlier_ratio=0.05, seed=0, standardize=True):
    """
    [고차원] 정상 k개 클러스터 + 이상치(outlier)
    - 정상 y_true: 0..k-1
    - outlier y_true: -1  (평가 시 제외 옵션 제공)
    """
    r = _rng(seed)
    n_out = int(n * outlier_ratio)
    n_in = n - n_out

    Xin, yin = gen_highdim_gaussian_mixture(n=n_in, d=d, k=k, sep=4.0, cov_scale=1.0, seed=seed, standardize=False)
    lo = np.percentile(Xin, 1, axis=0) - 8
    hi = np.percentile(Xin, 99, axis=0) + 8
    Xout = r.uniform(lo, hi, size=(n_out, d))
    yout = np.full(n_out, -1, dtype=int)

    X = np.vstack([Xin, Xout])
    y = np.concatenate([yin, yout])
    return _standardize(X, standardize), y


def gen_double_helix_3d(
    n=4000,
    turns=4.0,
    radius=5.0,
    pitch=1.0,
    noise_std=0.15,
    phase=np.pi,
    seed=0,
    standardize=True
):
    """
    [3D] Double Helix (3차원 나선 2개)
    - X.shape = (n, 3)
    - y_true: 0 또는 1 (각 helix 소속)
    
    파라미터:
    - turns: 나선이 몇 바퀴 도는지
    - radius: 반지름
    - pitch: z축으로 올라가는 속도(1회전당 z 증가량에 영향)
    - noise_std: 가우시안 노이즈 강도
    - phase: 두 번째 helix의 위상 차이(기본 pi → 반대편에서 시작)
    """
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    # t: 각 helix의 진행 파라미터
    t1 = rng.uniform(0, 2*np.pi*turns, n1)
    t2 = rng.uniform(0, 2*np.pi*turns, n2)

    # helix 1
    x1 = radius * np.cos(t1)
    y1 = radius * np.sin(t1)
    z1 = pitch * t1

    # helix 2 (위상 이동)
    x2 = radius * np.cos(t2 + phase)
    y2 = radius * np.sin(t2 + phase)
    z2 = pitch * t2

    X1 = np.vstack([x1, y1, z1]).T
    X2 = np.vstack([x2, y2, z2]).T

    X = np.vstack([X1, X2])
    y_true = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    # noise
    if noise_std > 0:
        X = X + rng.normal(0, noise_std, size=X.shape)

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y_true

# ============================================================
# Main API: generate_dataset
# ============================================================
def generate_dataset(
    name: str,
    n: int = 2000,
    d: int = 50,
    k: int = 4,
    seed: int = 0,
    standardize: bool = True,
    **kwargs
):
    """
    클러스터링 실험용 synthetic dataset 생성기.
    모든 경우에 (X, y_true)를 반환한다.

    -------------------------
    [지원 유형 / 차원]
    -------------------------
    1) 2D (X.shape=(n, 2))
      - "blobs"           : 구형 Gaussian blob (KMeans baseline)
      - "anisotropic"     : 타원형(선형변환) blob
      - "varied_density"  : 클러스터별 밀도/분산 다름 (DBSCAN/HDBSCAN 테스트)
      - "moons"           : 두 초승달 (Spectral/DBSCAN 강점)
      - "circles"         : 동심원 (Spectral/DBSCAN 강점)
      - "spiral"          : 스파이럴 2개 (강한 비선형)
      - "ring_blob"       : 링 + 중앙 blob (형태+밀도 혼재)

    2) 3D (X.shape=(n, 3))
      - "swiss_roll_3d"   : Swiss roll 매니폴드
                            * y_true는 연속 파라미터 t를 n_bins 구간으로 나눈 인위적 라벨
                            * kwargs: noise, n_bins
      - "double_helix_3d" : 3D 두 나선(helix) → y_true가 자연스럽고 명확

    3) 고차원 (X.shape=(n, d), d 사용)
      - "hd_gmm"          : 고차원 Gaussian mixture (sep/cov_scale로 난이도 조절)
                            * kwargs: sep, cov_scale
      - "subspace"        : 서브스페이스 클러스터 (각 클러스터가 다른 feature subset에서만 분리)
                            * kwargs: sub_dim, sep, noise_std
      - "overlap"         : 많이 겹치는 고차원 mixture (hard case)
      - "imbalanced"      : 클러스터 크기 불균형
                            * kwargs: major_ratio, sep
      - "outliers"        : 정상 클러스터 + outlier (y_true=-1이 outlier)
                            * kwargs: outlier_ratio

    -------------------------
    [반환]
    -------------------------
    X: np.ndarray
    y_true: np.ndarray (정답 라벨)
      - 대부분 0..K-1
      - outliers만 outlier=-1 포함 가능

    -------------------------
    [주의]
    -------------------------
    - standardize=True이면 StandardScaler 적용
    - 일부 데이터는 "정답 라벨"이 자연 정의가 애매해서(예: swiss_roll)
      '구간화' 같은 방식으로 y_true를 구성했다.
    """
    name = name.lower().strip()

    if name == "blobs":
        return gen_blobs_isotropic(
            n=n, k=k, cluster_std=kwargs.get("cluster_std", 1.0),
            seed=seed, standardize=standardize
        )

    if name == "anisotropic":
        return gen_blobs_anisotropic(
            n=n, k=k, cluster_std=kwargs.get("cluster_std", 1.0),
            seed=seed, standardize=standardize
        )

    if name == "varied_density":
        return gen_blobs_varied_density(n=n, seed=seed, standardize=standardize)

    if name == "moons":
        return gen_moons(n=n, noise=kwargs.get("noise", 0.08), seed=seed, standardize=standardize)

    if name == "circles":
        return gen_circles(
            n=n, noise=kwargs.get("noise", 0.05), factor=kwargs.get("factor", 0.5),
            seed=seed, standardize=standardize
        )

    if name == "spiral":
        return gen_spiral(
            n=n, turns=kwargs.get("turns", 3.0), noise_std=kwargs.get("noise_std", 0.12),
            seed=seed, standardize=standardize
        )

    if name == "ring_blob":
        return gen_ring_plus_blob(n=n, seed=seed, standardize=standardize)

    if name == "swiss_roll_3d":
        return gen_swiss_roll_3d(
            n=kwargs.get("n3d", max(3000, n)),
            noise=kwargs.get("noise", 0.15),
            n_bins=kwargs.get("n_bins", 3),
            seed=seed, standardize=standardize
        )

    if name == "hd_gmm":
        return gen_highdim_gaussian_mixture(
            n=kwargs.get("n_hd", max(5000, n)),
            d=d, k=k,
            sep=kwargs.get("sep", 4.0),
            cov_scale=kwargs.get("cov_scale", 1.0),
            seed=seed, standardize=standardize
        )

    if name == "subspace":
        return gen_subspace_clusters(
            n=kwargs.get("n_hd", max(6000, n)),
            d=d, k=k,
            sub_dim=kwargs.get("sub_dim", max(5, d // 10)),
            sep=kwargs.get("sep", 6.0),
            noise_std=kwargs.get("noise_std", 1.0),
            seed=seed, standardize=standardize
        )

    if name == "overlap":
        return gen_overlapping_clusters(
            n=kwargs.get("n_hd", max(5000, n)),
            d=d, k=k, seed=seed, standardize=standardize
        )

    if name == "imbalanced":
        return gen_imbalanced(
            n=kwargs.get("n_hd", max(6000, n)),
            d=d, k=k,
            major_ratio=kwargs.get("major_ratio", 0.7),
            sep=kwargs.get("sep", 4.0),
            seed=seed, standardize=standardize
        )

    if name == "outliers":
        return gen_with_outliers(
            n=kwargs.get("n_hd", max(5000, n)),
            d=d, k=k,
            outlier_ratio=kwargs.get("outlier_ratio", 0.05),
            seed=seed, standardize=standardize
        )
        
    if name == "double_helix_3d":
        return gen_double_helix_3d(
            n=kwargs.get("n3d", max(3000, n)),
            turns=kwargs.get("turns", 4.0),
            radius=kwargs.get("radius", 5.0),
            pitch=kwargs.get("pitch", 1.0),
            noise_std=kwargs.get("noise_std", 0.15),
            phase=kwargs.get("phase", np.pi),
            seed=seed,
            standardize=standardize
        )

    raise ValueError(f"Unknown dataset name: {name}")


# ============================================================
# Clustering + Evaluation (내부지표 + 외부지표)
# ============================================================
def cluster_kmeans(X, k, seed=0):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return km.fit_predict(X)

def cluster_gmm(X, k, seed=0):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    y_pred = gmm.fit_predict(X)
    return y_pred, gmm

def evaluate_clustering(X, y_true, y_pred, ignore_label=-1):
    """
    내부지표: silhouette, DBI, CH  (X와 y_pred만 필요)
    외부지표(정답 필요): ARI, AMI, NMI, Homogeneity, Completeness, V-measure

    - ignore_label=-1: y_true가 -1인 샘플(예: outlier)을 평가에서 제외
      (내부지표는 y_pred만으로 계산하므로 제외 여부는 선택 사항인데,
       일반적으로 outlier를 포함하면 silhouette 등이 왜곡될 수 있어 함께 제외하도록 구현)
    """
    X_eval = X
    yt = y_true
    yp = y_pred

    if ignore_label is not None:
        mask = (yt != ignore_label)
        X_eval = X[mask]
        yt = yt[mask]
        yp = yp[mask]

    metrics = {}

    # 내부지표(라벨이 1개면 silhouette/CH가 의미 없을 수 있음)
    uniq = np.unique(yp)
    if len(uniq) > 1:
        metrics["silhouette"] = float(silhouette_score(X_eval, yp))
        metrics["ch"] = float(calinski_harabasz_score(X_eval, yp))
    else:
        metrics["silhouette"] = np.nan
        metrics["ch"] = np.nan
    metrics["dbi"] = float(davies_bouldin_score(X_eval, yp))

    # 외부지표
    metrics["ARI"] = float(adjusted_rand_score(yt, yp))
    metrics["AMI"] = float(adjusted_mutual_info_score(yt, yp))
    metrics["NMI"] = float(normalized_mutual_info_score(yt, yp))
    metrics["homogeneity"] = float(homogeneity_score(yt, yp))
    metrics["completeness"] = float(completeness_score(yt, yp))
    metrics["v_measure"] = float(v_measure_score(yt, yp))

    return metrics


# ============================================================
# Quick Demo
# ============================================================
def basic_unsupervised_report(X, labels):
    # 노이즈(-1) 라벨이 있으면 실루엣은 제외하거나 노이즈 제거 후 계산하는 게 안전
    unique = set(labels)
    has_noise = (-1 in unique)
    n_clusters = len(unique) - (1 if has_noise else 0)

    out = {"n_clusters": n_clusters, "has_noise": has_noise}
    if n_clusters >= 2:
        # 노이즈가 있으면 노이즈 제외 버전도 같이 계산
        if has_noise:
            mask = labels != -1
            X2, y2 = X[mask], labels[mask]
            if len(set(y2)) >= 2:
                out["silhouette(noise_excluded)"] = silhouette_score(X2, y2)
        else:
            out["silhouette"] = silhouette_score(X, labels)

        out["davies_bouldin"] = davies_bouldin_score(X, labels)
        out["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    return out


# 예시 1) 2D moons ----------------------------------------------------------
X, y = generate_dataset("moons", n=2500, seed=42, noise=0.08)
# y_pred = cluster_kmeans(X, k=2, seed=0)  # moons는 KMeans가 약한 편
# m = evaluate_clustering(X, y_true, y_pred)
# print("[moons] metrics:", m)
plot_2d(X, y, "moons: y_true")
# plot_2d(X, y_pred, "moons: y_pred (KMeans)")

# ---------------------------------------------------------------------------
# 2D varied_density
X, y = generate_dataset("varied_density", n=2500, seed=42)
plot_2d(X, y, "varied_density (2D)")


X, y = generate_dataset("spiral", n=2500, seed=42, turns=3.0, noise_std=0.12)
plot_2d(X, y, "spiral (2D)")

# 3D demo
X3, y3 = generate_dataset("swiss_roll_3d", seed=0, noise=0.2)
plot_3d(X3, y3, "swiss_roll_3d")

X3, y3 = generate_dataset("double_helix_3d", seed=0, noise=0.2)
plot_3d(X3, y3, "swiss_roll_3d")

# high-d demo
Xh, yh = generate_dataset("subspace", d=100, k=6, seed=0, sub_dim=10, sep=7.0)
print("high-d subspace:", Xh.shape, "labels:", np.unique(yh).shape)


# 예시 2) 고차원 subspace (표현학습/서브스페이스 난이도)
Xh, yh = generate_dataset("subspace", n=6000, d=100, k=6, seed=0, sub_dim=10, sep=7.0)
y_pred_h = cluster_kmeans(Xh, k=6, seed=0)
mh = evaluate_clustering(Xh, yh, y_pred_h)
print("[subspace] metrics:", mh)

# 예시 3) outliers 포함 (y_true=-1 제외하고 평가)
Xo, yo = generate_dataset("outliers", n=5000, d=20, k=4, seed=0, outlier_ratio=0.07)
y_pred_o = cluster_kmeans(Xo, k=4, seed=0)
mo = evaluate_clustering(Xo, yo, y_pred_o, ignore_label=-1)
print("[outliers] metrics(ignore -1):", mo)
#################################################################################

#--------------------------------------------------------------------------------
# K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, n_init="auto").fit(X)

pred_y = kmeans.predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)


#--------------------------------------------------------------------------------
# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6).fit(X)

pred_y = gmm.predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)
# plot_3d(X, pred_y)


#--------------------------------------------------------------------------------
# Hierarchical
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X, method='single')  # ward는 유클리드 기반에서 가장 흔함
dend = dendrogram(Z, truncate_mode='lastp', p=30)  

hierarchy = AgglomerativeClustering(n_clusters=2, linkage="ward")   # 합칠 때, 군집 내 분산(SSE) 증가량을 최소화
pred_y = hierarchy.fit_predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)


hierarchy = AgglomerativeClustering(n_clusters=2, linkage="single")     # 두 군집의 점들 중 가장 가까운 점끼리 거리
pred_y = hierarchy.fit_predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)

hierarchy = AgglomerativeClustering(n_clusters=2, linkage="complete")   # 두 군집 점들 중 가장 먼 점끼리 거리
pred_y = hierarchy.fit_predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)


hierarchy = AgglomerativeClustering(n_clusters=2, linkage="average")   # 두 군집의 모든 쌍 거리 평균
pred_y = hierarchy.fit_predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)


#--------------------------------------------------------------------------------
# DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.1, min_samples=10)
pred_y = dbscan.fit_predict(X)  # 노이즈는 -1
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)

#--------------------------------------------------------------------------------
# # HDBSCAN
# import hdbscan
# hdbscan_cluster = hdbscan.HDBSCAN(
#     min_cluster_size=30,
#     min_samples=10
# )
# pred_y = hdbscan_cluster.fit_predict(X)

# basic_unsupervised_report(X, pred_y)

# plot_2d(X, pred_y)



#--------------------------------------------------------------------------------
# Spectral Clustering
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=2,
    affinity="nearest_neighbors",  # 또는 "rbf"
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=0
)

pred_y = spectral.fit_predict(X)
basic_unsupervised_report(X, pred_y)

plot_2d(X, pred_y)


#################################################################################
