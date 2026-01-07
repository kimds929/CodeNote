import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


#################################################################################
rng = np.random.RandomState(2)
# rng = np.random.RandomState(4)
n = 30
scale = (3,8)
points = rng.rand(n,2) *scale

# --------------------------------------------------------------------------------
plt.figure(figsize=(3,8))
plt.scatter(*list(points.T), c="black", s=5)
plt.xlim(-0.05,scale[0]+0.05)
plt.ylim(-0.05,scale[1]-0.05)

# --------------------------------------------------------------------------------
xx, yy = np.meshgrid(
    np.linspace(0, scale[0], 50),
    np.linspace(0, scale[1], 50)
)

grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)
# --------------------------------------------------------------------------------





#################################################################################
class Kernel:
    """
    Unified param interface:
      - kind == "gaussian": param means Sigma (scalar sigma OR (d,d) covariance)
      - kind != "gaussian": param means radius boundary
          * param scalar: spherical boundary (Euclidean)
          * param matrix (d,d): ellipsoidal boundary via Mahalanobis distance using param as Sigma
            (dist = sqrt((x-y)^T Sigma^{-1} (x-y)), boundary at dist <= 1)
    """

    def __init__(self, default_Sigma=1, default_radius=1, kind='gaussian'):
        self.kind = kind
        self.default_Sigma = default_Sigma
        self.default_radius = default_radius

    @staticmethod
    def _as_array(x):
        return np.asarray(x)

    @staticmethod
    def _is_matrix_param(param):
        return isinstance(param, np.ndarray) and param.ndim == 2

    def pairwise_distance(self, X, Y=None, Sigma=None):
        """
        dist_ij = sqrt( (x_i - y_j)^T Sigma^{-1} (x_i - y_j) )   if Sigma is (d,d)
               = ||x_i - y_j||                                  if Sigma is None
               = ||x_i - y_j|| / Sigma                          if Sigma is scalar
        """
        X = self._as_array(X)
        Y = X if Y is None else self._as_array(Y)

        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)

        if self._is_matrix_param(Sigma):
            Sigma_inv = np.linalg.inv(Sigma)
            sq = ((diff @ Sigma_inv) * diff).sum(axis=-1)   # (n, m)
            sq = np.maximum(sq, 0.0)                        # numerical safety
            return np.sqrt(sq)

        # Euclidean
        sq = (diff ** 2).sum(axis=-1)
        sq = np.maximum(sq, 0.0)
        dist = np.sqrt(sq)

        # scalar scaling (sigma)
        if Sigma is not None:
            dist = dist / float(Sigma)

        return dist

    def transform(self, X, Y=None, param=None, kind=None):
        """
        Unified API:
          - gaussian: param == Sigma (scalar or matrix). If None => default_Sigma.
                     K = exp(-0.5 * dist^2) where dist uses Sigma.
          - uniform/linear/epanechnikov/quartic:
                     param == radius boundary (scalar) OR ellipsoid matrix.
                     If param is scalar r: dist = Euclidean, u = dist/r.
                     If param is matrix B (d,d): use Sigma=B to compute Mahalanobis dist,
                        then boundary is dist <= 1 (so param encodes the ellipsoid).
                        i.e. u = dist (already normalized).
        """
        kind = self.kind if kind is None else kind
        
        # --- gaussian: param is Sigma (scalar or matrix) ---
        if kind == "gaussian":
            Sigma = self.default_Sigma if param is None else param
            dist = self.pairwise_distance(X, Y, Sigma=Sigma)
            return np.exp(-0.5 * dist**2)

        else:
            # --- non-gaussian: param is boundary ---
            boundary = self.default_radius if param is None else param

            # Case 1) scalar radius -> Euclidean dist normalized by r
            if not self._is_matrix_param(boundary):
                r = float(boundary)
                dist = self.pairwise_distance(X, Y, Sigma=None)   # metric fixed to Euclidean
                u = dist / r                                      # normalized distance
            # Case 2) matrix boundary -> ellipsoid, normalized so boundary at u=1
            else:
                # 여기서 boundary는 'ellipsoid를 정의하는 Sigma'로 사용
                # u = sqrt((x-y)^T Sigma^{-1} (x-y))  (already normalized)
                u = self.pairwise_distance(X, Y, Sigma=boundary)

            # Now apply compact-support falloff with boundary at u=1
            if kind == "uniform":
                return (u <= 1.0).astype(float)

            if kind == "linear":
                return np.clip(1.0 - u, 0.0, 1.0)

            if kind == "epanechnikov":
                return np.clip(1.0 - u**2, 0.0, 1.0)

            if kind == "quartic":
                t = np.clip(1.0 - u**2, 0.0, 1.0)
                return t**2

            raise ValueError("kind must be one of: 'gaussian', 'uniform', 'linear', 'epanechnikov', 'quartic'")

    def __call__(self, X, Y=None, param=None, kind=None):
        self.transform(X, Y, param, kind)

# def gaussian_kernel(X, Y=None, Sigma=None):
#     """
#     X: (n, d)       # 평가 points
#     Y: (m, d) or None (=> Y = X)        # 기준 points
#     sigma: float, if None => sqrt(d)
#     """
#     X = np.asarray(X)
#     Y = X if Y is None else np.asarray(Y)
    
#     # X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1) # (n, 1) 
#     # Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1) # (1, m) 
#     # sq_dist = sq_dist = X_norm - 2 * X @ Y.T + Y_norm # X^2 -2 X Y + Y^2
#     # K = np.exp(-1/(2*Sigma**2) * sq_dist)

#     diff = (X[:, None, :] - Y[None, :, :])    # (n, m, d)
    
#     if type(Sigma) == np.ndarray:
#         K = np.exp( -0.5 * ((diff @ np.linalg.inv(Sigma))* diff).sum(-1) )
#     else:
#         if Sigma is None:
#             Sigma = np.sqrt(X.shape[1])
#         K = np.exp( -0.5 * (diff **2).sum(-1)/Sigma )
#     return K
# ################################################################################

# (visualize kernels) normalized distance u = d / r
base_point = np.zeros([1,1])
u = np.linspace(0, 2.0, 100).reshape(-1,1)

kernel = Kernel()
K_gaussian = kernel.transform(u, base_point, kind='gaussian').sum(-1)           # exp(-1/2 u^2) : np.exp(-0.5 * u**2)
K_uniform = kernel.transform(u, base_point, kind='uniform').sum(-1)             # 1 (u≤1), 0 (u>1) : (u <= 1.0).astype(float)
K_linear = kernel.transform(u, base_point, kind='linear').sum(-1)               # max(0, 1-u) : np.clip(1.0 - u, 0.0, 1.0)
K_epanechnikov = kernel.transform(u, base_point, kind='epanechnikov').sum(-1)   # max(0, 1-u^2) : np.clip(1.0 - u**2, 0.0, 1.0)
K_quartic = kernel.transform(u, base_point, kind='quartic').sum(-1)             # max(0, (1-u^2)^2) : np.clip(1.0 - u**2, 0.0, 1.0)**2

# visualize plot
plt.figure(figsize=(7,5))
plt.plot(u, K_gaussian, label="Gaussian", linewidth=2)          # exp(-1/2 u^2) : np.exp(-0.5 * u**2)
plt.plot(u, K_uniform, label="Uniform", linewidth=2)            # 1 (u≤1), 0 (u>1) : (u <= 1.0).astype(float)
plt.plot(u, K_linear, label="Linear", linewidth=2)              # max(0, 1-u) : np.clip(1.0 - u, 0.0, 1.0)
plt.plot(u, K_epanechnikov, label="Epanechnikov", linewidth=2)  # max(0, 1-u^2) : np.clip(1.0 - u**2, 0.0, 1.0)
plt.plot(u, K_quartic, label="Quartic", linewidth=2)            # max(0, (1-u^2)^2) : np.clip(1.0 - u**2, 0.0, 1.0)**2

plt.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
plt.text(1.02, 0.5, "boundary (u=1)", color="gray")

plt.xlabel("normalized distance  u = d / r")
plt.ylabel("kernel value")
plt.title("1D influence profiles of kernels")
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


#################################################################################
# kind = gaussian/uniform/linear/epanechnikov/quartic
kernel = Kernel(kind='gaussian')


# param = 1 * np.eye(2)
# param = 0.15 * np.eye(2)
param_vec = np.array([1,100]) 
param_vec_norm = param_vec/np.linalg.norm(param_vec)
param = np.diag(param_vec_norm) *1


K_grid = kernel.transform(grid, points, param=param)
intensive_val = kernel.transform(points, points, param=param).sum(axis=1)

# --------------------------------------------------------------------------------
i = 22  # 보고 싶은 기준 point inde
Z = K_grid[:, i].reshape(xx.shape)
pd.DataFrame(Z).to_clipboard()

plt.figure(figsize=scale)
cont = plt.contourf(xx, yy, Z, cmap="Reds", alpha=1, levels=np.linspace(0,1,11))
plt.scatter(*list(points.T), c="black", s=5)
for ei, (p, iv) in enumerate(zip(points, intensive_val)):
    if iv < 3:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, alpha=0.2)
    else:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, color='red')
    
plt.colorbar(cont)
plt.show()












##########################################################################################
kernel = Kernel(kind='gaussian')

# param = 0.15 * np.eye(2)
param_vec = np.array([1,100]) 
param_vec_norm = param_vec/np.linalg.norm(param_vec)
param = np.diag(param_vec_norm) *1


# -------------------------------------------------------------------------------------
K_grid = kernel.transform(grid, points, param=param)
intensive_val = kernel.transform(points, points, param=param).sum(axis=1)

Z = K_grid.sum(axis=-1).reshape(xx.shape)   # 전체

# -------------------------------------------------------------------------------------
# visualize all
plt.figure(figsize=scale)
cont = plt.contourf(xx, yy, Z, cmap="Reds", alpha=1, levels=np.round(np.sqrt(np.linspace(0, 50,11)),1) )
# cont = plt.contourf(xx, yy, Z, cmap="Reds", alpha=1, levels=np.linspace(0,1,11))
plt.scatter(*list(points.T), c="black", s=5)
for ei, (p, iv) in enumerate(zip(points, intensive_val)):
    # plt.text(p[0], p[1], ei)
    if iv < 3:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, alpha=0.2)
    else:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, color='red')
    
plt.colorbar(cont)
plt.show()

# (Z > 3).sum() / np.prod(Z.shape)      # 면적율


# -------------------------------------------------------------------------------------
# visualize abnormal
plt.figure(figsize=scale)
scat = plt.scatter(*list(points.T), c=intensive_val, s=8, cmap='Reds', vmin=3, vmax=4)
plt.scatter(*list(points.T), edgecolors='gray', linewidths=1, facecolors='none', s=8, alpha=0.5)
plt.colorbar(scat)
for ei, (p, iv) in enumerate(zip(points, intensive_val)):
    # plt.text(p[0], p[1], ei)
    if iv < 3:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, alpha=0.2)
    else:
        plt.text(p[0], p[1], f"{iv:.1f}", fontsize=7, color='red')
##########################################################################################

