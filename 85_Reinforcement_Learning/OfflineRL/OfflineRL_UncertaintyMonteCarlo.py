import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from scipy.stats import beta


########################################################################################################
# (Beta Distribution)

alpha_ = 20
beta_ = 20
beta.ppf(1-0.05, alpha_, beta_) - beta.ppf(0.05, alpha_, beta_)
(beta.ppf(1-0.05, alpha_, beta_) + beta.ppf(0.05, alpha_, beta_))/2


def plot_beta_distributions(alpha_, beta_, color=None):
    x = np.linspace(0, 1, 100)
    
    # plt.figure(figsize=(8, 5))
    
    y = beta.pdf(x, alpha_, beta_)
    plt.plot(x, y, label=f'alpha={alpha_}, beta={beta_}', color=color)
    
    plt.title('Beta Distributions')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()


alpha_list = [1, 4, 5, 10]
beta_list = [1, 2, 8, 10]


for alpha_, beta_ in zip(alpha_list, beta_list):
    plot_beta_distributions(alpha_, beta_, color=(1-alpha_/10, 1-beta_/10, 0))
plt.legend(loc='upper right', bbox_to_anchor=(1.4,1))
plt.show()

########################################################################################################


# from IPython.display import clear_output
# import time
# rng = np.random.RandomState()    


# p = 0.6

# alpha_ = 0
# beta_ = 0


# x = np.linspace(0, 1, 100)
# for i in range(30):
#     sample_p = rng.rand()
#     # np.random.binomial(n=1, p=0.5)
    
#     print(int(sample_p < p))
#     if sample_p < p:
#         alpha_ += 1
#     else:
#         beta_ += 1
#     y = beta.pdf(x, alpha_, beta_)
    
#     plt.plot(x, y, label=f'alpha={alpha_}, beta={beta_}')
#     plt.axvline(p, color='red', ls='--', alpha=0.5)
#     plt.show()
#     clear_output(wait=True)
#     time.sleep(0.1)




###########################################################################################################
# (Beta Distributino Class)
from scipy.stats import beta

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.beta = beta

    def add_alpha(self, inc=1):
        self.a += inc

    def add_beta(self, inc=1):
        self.b += inc

    def init_params(self, a=None, b=None):
        a = self.a if a is None else a
        b = self.b if b is None else b
        return a, b
    
    def __repr__(self):
        return f"Beta(a={self.a}, b={self.b})"

    # ----------------------------
    # sampling / density
    # ----------------------------
    def rvs(self, a=None, b=None, loc=0, scale=1, size=1, random_state=None):
        a, b = self.init_params(a, b)
        return self.beta.rvs(a=a, b=b, loc=loc, scale=scale, size=size, random_state=random_state)

    def pdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.pdf(x, a=a, b=b, loc=loc, scale=scale)

    def logpdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logpdf(x, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # CDF / SF / inverse
    # ----------------------------
    def cdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.cdf(x, a=a, b=b, loc=loc, scale=scale)

    def logcdf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logcdf(x, a=a, b=b, loc=loc, scale=scale)

    def sf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.sf(x, a=a, b=b, loc=loc, scale=scale)

    def logsf(self, x, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.logsf(x, a=a, b=b, loc=loc, scale=scale)

    def ppf(self, q, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.ppf(q, a=a, b=b, loc=loc, scale=scale)

    def isf(self, q, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.isf(q, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # moments / stats
    # ----------------------------
    def moment(self, order, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.moment(order, a=a, b=b, loc=loc, scale=scale)

    def stats(self, a=None, b=None, loc=0, scale=1, moments='mv'):
        a, b = self.init_params(a, b)
        return self.beta.stats(a=a, b=b, loc=loc, scale=scale, moments=moments)

    def entropy(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.entropy(a=a, b=b, loc=loc, scale=scale)

    def expect(self, func, a=None, b=None, loc=0, scale=1,
               lb=None, ub=None, conditional=False, **kwds):
        a, b = self.init_params(a, b)
        # SciPy expect는 args=(a,b) 형태로 shape를 전달
        return self.beta.expect(func, args=(a, b), loc=loc, scale=scale,
                                lb=lb, ub=ub, conditional=conditional, **kwds)

    # ----------------------------
    # point estimates
    # ----------------------------
    def median(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.median(a=a, b=b, loc=loc, scale=scale)

    def mean(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.mean(a=a, b=b, loc=loc, scale=scale)

    def var(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.var(a=a, b=b, loc=loc, scale=scale)

    def std(self, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.std(a=a, b=b, loc=loc, scale=scale)

    def interval(self, confidence, a=None, b=None, loc=0, scale=1):
        a, b = self.init_params(a, b)
        return self.beta.interval(confidence, a=a, b=b, loc=loc, scale=scale)

    # ----------------------------
    # fit
    # ----------------------------
    def fit(self, data, update=False, **kwargs):
        """
        SciPy beta.fit(data)는 (a, b, loc, scale)를 반환.
        - update=False: 추정치만 반환
        - update=True : self.a, self.b도 업데이트 (loc/scale은 별도 관리 안 함)
        """
        a_hat, b_hat, loc_hat, scale_hat = self.beta.fit(data, **kwargs)
        if update:
            self.a = a_hat
            self.b = b_hat
        return a_hat, b_hat, loc_hat, scale_hat







########################################################################################################
########################################################################################################
########################################################################################################
# (Environment Setting)
rng = np.random.RandomState(6)

def random_noise(scale=0.2):
    return rng.rand()*scale - scale/2


n_process = rng.randint(3,8)
n_fac_in_process = rng.randint(1, 5, size=n_process)
max_fac = max(n_fac_in_process)

# ---------------------------------------------------------------------------------------------------
# make position
def make_pos(n_fac_in_process):
    x_pos = [[np.nan] + [ei+1]*f  for ei, f in enumerate(n_fac_in_process)]
    y_pos = [[np.nan] + (np.arange(1, f+1)/(f+1) * (max_fac+2) + random_noise() ).tolist()  for f in n_fac_in_process]
    return x_pos, y_pos

x_pos, y_pos = make_pos(n_fac_in_process)
print(x_pos)
print(y_pos)



# ---------------------------------------------------------------------------------------------------
def process_plot(n_fac_in_process, pathes=[], label=[], abnormal=[], beta_dist=None,
            box_kwargs=dict(conf=0.90, box_width=0.2, box_height=0.5, box_dx=0.1, text_fs=5) ):
    conf      = box_kwargs["conf"]
    box_width = box_kwargs["box_width"]
    box_height= box_kwargs["box_height"]
    box_dx    = box_kwargs["box_dx"]
    text_fs   = box_kwargs["text_fs"]

    # -------------------------
    x_pos, y_pos = make_pos(n_fac_in_process)

    # -------------------------
    # point
    for x_list, y_list in zip(x_pos, y_pos):
        for yi, (x, y) in enumerate(zip(x_list, y_list)):
            if not np.isnan(x):
                plt.scatter(x, y, color='gray')
                plt.text(x-0.3, y+0.1, f"x{x}_y{yi}", fontsize=8, alpha=0.7)

    # -------------------------
    # abnormal
    for (axi, ayi, p) in abnormal:
        scatter = plt.scatter(x_pos[axi-1][ayi], y_pos[axi-1][ayi], color=((p/2), 0, 0))
        scateter_color = scatter.get_facecolor()
        plt.text(x_pos[axi-1][ayi]-0.075, y_pos[axi-1][ayi]-0.1, f"{p:.2f}", fontsize=8, color=scateter_color)

    # -------------------------
    # path
    for pi, path in enumerate(pathes):
        path_pos = [(x[p], y[p]) for x, y, p in zip(x_pos, y_pos, path) if not np.isnan(x[p])]
        path_pos_T = np.array(path_pos).T
        
        try:
            if len(label) == len(pathes):
                line_color = 'red' if label[pi] > 0.5 else 'mediumseagreen'
                line, = plt.plot(*path_pos_T, alpha=max(0.1, 1/len(pathes)), color=line_color)
                plt.text(*path_pos_T[:, -1], f"MTL_{pi} ({label[pi]:.1f})", fontsize=8, color=line_color)
            else:
                line, = plt.plot(*path_pos_T, alpha=max(0.3, 1/len(pathes)))
                line_color = line.get_color()
                plt.text(*path_pos_T[:, -1], f"MTL_{pi}", fontsize=8, color=line_color)
        except:
            pass
    
    
    # -------------------------
    # beta box per point
    # beta_dist[x_index][y_index] = Beta(a,b)
    # y_index=0은 표시하지 않음
    if beta_dist is not None:
        # 네가 제시한 방식에 맞춰서 "미리 mean/lcb/ucb를 리스트로 계산"해둠
        mean_list = [[float(y.mean()) for y in x] for x in beta_dist]
        lcb_list  = [[float(y.interval(conf)[0]) for y in x] for x in beta_dist]
        ucb_list  = [[float(y.interval(conf)[1]) for y in x] for x in beta_dist]

        for xi, f in enumerate(n_fac_in_process, start=1):  # xi=1..n_process
            f = int(f)
            for yi in range(1, f+1):  # ★ y=0 제외
                # 좌표
                x0 = x_pos[xi-1][yi]
                y0 = y_pos[xi-1][yi]

                # beta_dist 길이 부족시 스킵(안전)
                if (xi-1) >= len(beta_dist) or yi >= len(beta_dist[xi-1]):
                    continue

                bobj = beta_dist[xi-1][yi]
                mean = mean_list[xi-1][yi]
                lcb  = lcb_list[xi-1][yi]
                ucb  = ucb_list[xi-1][yi]

                # a,b 표기용(가능하면 표시)
                a = getattr(bobj, "a", None)
                b = getattr(bobj, "b", None)
                if (a is None or b is None) and hasattr(bobj, "args"):
                    try:
                        a, b = bobj.args[:2]
                    except Exception:
                        pass

                # 박스 위치(포인트 오른쪽)
                bx = x0 + box_dx
                by = y0 - box_height/2

                # 박스 테두리
                rect = Rectangle((bx, by), box_width, box_height, fill=False, linewidth=0.6, alpha=0.8)
                plt.gca().add_patch(rect)

                # 0~1 스케일 표시
                plt.text(bx - 0.03, by, "0", fontsize=text_fs, ha='right', va='center', alpha=0.8)
                plt.text(bx - 0.03, by + box_height, "1", fontsize=text_fs, ha='right', va='center', alpha=0.8)

                # 값 -> 박스 y좌표 변환 (클램프)
                def v_to_y(v):
                    v = 0.0 if v < 0 else (1.0 if v > 1 else v)
                    return by + v * box_height

                y_l = v_to_y(lcb)
                y_m = v_to_y(mean)
                y_u = v_to_y(ucb)

                # L/M/U 가로선 표시
                plt.plot([bx, bx + box_width], [y_l, y_l], color='mediumseagreen' , linewidth=0.6, alpha=0.9)
                plt.plot([bx, bx + box_width], [y_m, y_m], color='coral', linewidth=0.9, alpha=0.9)
                plt.plot([bx, bx + box_width], [y_u, y_u], color='steelblue' , linewidth=0.6, alpha=0.9)

                # 텍스트 표시(fontsize=5)
                plt.text(bx + box_width + 0.02, y_u, f"U {ucb:.2f}", fontsize=text_fs, va='center')
                plt.text(bx + box_width + 0.02, y_m, f"M {mean:.2f}", fontsize=text_fs, va='center')
                plt.text(bx + box_width + 0.02, y_l, f"L {lcb:.2f}", fontsize=text_fs, va='center')

                # (a,b) 박스 위 표시
                if a is not None and b is not None:
                    plt.text(
                        bx + box_width/2, by + box_height + 0.03,
                        f"a={float(a):.0f}, b={float(b):.0f}",
                        fontsize=text_fs, ha='center', va='bottom', alpha=0.85
                    )

    plt.xlim(0.5, len(n_fac_in_process)+1)

# process_plot(n_fac_in_process)
# plt.show()

# ---------------------------------------------------------------------------------------------------
# generate random path
def random_path_generator(n=1):
    return np.stack([[int(rng.choice(np.arange(f+1))) for f in n_fac_in_process] for _ in range(n)])

random_paths = random_path_generator(n=3)
print(random_paths)
# process_plot(n_fac_in_process, random_paths)


process_plot(n_fac_in_process, random_paths)
plt.show()





###################################################################################################
abnormal_p = 0.7
# ---------------------------------------------------------------------------------------------------
# # random label generate
# label = (rng.random( (len(X_random_paths),1) ) > abnormal_p).astype(int)
# print(label.flatten())


# designated label generate
def generate_abnormal_xy(n_fac_in_process, n=1):
    abnormal_ps = np.round((rng.random(size=n) /2 + 0.5), 2)
    abnormal_xyp = []
    for i, p in zip(range(n), abnormal_ps):
        abnormal_idx = rng.rand() * sum(n_fac_in_process)
        abnormal_x = np.where(np.cumsum(n_fac_in_process) > abnormal_idx)[0][0]+1
        abnormal_y = (n_fac_in_process - (np.cumsum(n_fac_in_process) - abnormal_idx)).astype(int)[abnormal_x-1]+1
        abnormal_xyp.append( (abnormal_x, abnormal_y, p))
    return abnormal_xyp


abnormal_xyp = generate_abnormal_xy(n_fac_in_process, n=1)
process_plot(n_fac_in_process, abnormal=abnormal_xyp)
plt.show()
# ---------------------------------------------------------------------------------------------------



# ★ random path generate
X_random_paths = random_path_generator(n=100)
print(X_random_paths)
# ---------------------------------------------------------------------------------------------------



def generate_path_probs_and_labels(
    random_paths: np.ndarray,
    n_fac_in_process: np.ndarray,
    abnormal_xyp: list,   # [(abnormal_x, abnormal_y, p0), ...]
    base_abnormal_p: float = 0.03,
    scale: float = 100.0
):
    """
    Returns
    -------
    p_path : (N,) float
        path별 샘플링된 불량확률
    y      : (N,) int
        0=양호, 1=불량
    """
    yp_true = []
    yp_obs = []

    # 안전장치: abnormal_xyp가 공정/시설 범위 밖이면 에러 대신 무시하고 싶다면 continue로 바꿔도 됨
    for i, path in enumerate(random_paths):
        p_total = base_abnormal_p
        for (ax, ay, p0) in abnormal_xyp:
            if not (1 <= ax <= len(n_fac_in_process)):
                raise ValueError(f"abnormal_x={ax} is out of range (1..{len(n_fac_in_process)})")
            if not (1 <= ay <= n_fac_in_process[ax-1]):
                raise ValueError(f"abnormal_y={ay} is out of range (1..{n_fac_in_process[ax-1]}) for process {ax}")

            if int(path[ax-1]) == int(ay):
                p_total = float(1 - (1-p0)* (1-p_total))
        
        yp_sample = rng.beta(p_total*scale, (1-p_total)*scale)
        
        yp_true.append(p_total)
        yp_obs.append(yp_sample)

    return np.array(yp_true), np.array(yp_obs)



yp_true, yp_obs = generate_path_probs_and_labels(X_random_paths, n_fac_in_process, abnormal_xyp)


# -------------------------------------------------------------------------------------------
# total path
plt.figure(figsize=(12,12))
plt.subplot(3,2,1)
process_plot(n_fac_in_process, X_random_paths, label=yp_true, abnormal=abnormal_xyp)

# normal vs. abnormal path
plt.subplot(3,2,3)
process_plot(n_fac_in_process, X_random_paths[yp_true < 0.5], label=yp_true[yp_true < 0.5], abnormal=abnormal_xyp)
plt.subplot(3,2,4)
process_plot(n_fac_in_process, X_random_paths[yp_true > 0.5], label=yp_true[yp_true > 0.5], abnormal=abnormal_xyp)

# # normal vs. abnormal path
# plt.subplot(3,2,5)
# process_plot(n_fac_in_process, X_random_paths[yp_obs < 0.5], label=yp_true[yp_true < 0.5], abnormal=abnormal_xyp)
# plt.subplot(3,2,6)
# process_plot(n_fac_in_process, X_random_paths[yp_obs > 0.5], label=yp_true[yp_true > 0.5], abnormal=abnormal_xyp)
# plt.show()
plt.show()


# -------------------------------------------------------------------------------------------




# ★ y_observation (0/1)
y_obs = rng.binomial(1, yp_obs)



# visualize
idx = rng.randint(len(X_random_paths))
process_plot(n_fac_in_process, X_random_paths[[idx]], label=y_obs[[idx]], abnormal=abnormal_xyp)



# beta(0,0).interval(0.95)

###########################################################################################################
# Learning
process_param_alpha = [np.bincount(xg, minlength=n_fac_in_process[i]+1)+1 for i, xg in enumerate(X_random_paths[np.where(y_obs==1)[0]].T)]
process_param_beta = [np.bincount(xg, minlength=n_fac_in_process[i]+1)+1 for i, xg in enumerate(X_random_paths[np.where(y_obs==0)[0]].T)]

# beta distributions for each process
process_beta_dist = []
for x_alpha_, x_beta_ in zip(process_param_alpha, process_param_beta):
    beta_dist = [Beta(alpha_, beta_) for alpha_, beta_ in zip(x_alpha_, x_beta_)]
    process_beta_dist.append(beta_dist)


# [[y.mean() for y in x] for x in process_beta_dist]      # mean
# [[float(y.interval(0.90)[0]) > 0.5 for y in x] for x in process_beta_dist]    # lcb
# [[float(y.interval(0.90)[1]) < 0.5 for y in x] for x in process_beta_dist]    # ucb


plt.figure(figsize=(10,8))
process_plot(n_fac_in_process,  label=y_obs[[idx]], abnormal=abnormal_xyp, beta_dist=process_beta_dist)
plt.show()






# ---------------------------------------------------------------------------------------------------
def node_to_edge(paths):
    paths_shifted = np.roll(paths, shift=-1, axis=1)
    return np.stack([paths[:,:-1], paths_shifted[:,:-1]], axis=-1)


X_random_edges = node_to_edge(X_random_paths)       # B, F, E 
X_random_edges_T = X_random_edges.transpose(1, 0, 2)   # F, B, E

n_count_edges = []
for xi in range(X_random_edges_T.shape[0]):
    edges, count = np.unique(X_random_edges.transpose(1, 0, 2)[xi], axis=0, return_counts=True)
    edge_count = np.concatenate([edges, count.reshape(-1,1)], axis=-1)
    n_count_edges.append(edge_count)

print(n_count_edges)





from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

def gradient_lineplot(x, y, color=['steelblue', 'steelblue'], alpha=1.0, linewidth=None, ls='-', dpi=30, ax=None):
    """
    x, y 좌표를 따라 여러 색상으로 그라데이션 선을 그립니다.
    색상 변화는 실제 선의 길이에 비례합니다.
    """
    x = np.array(x)
    y = np.array(y)
    
    # 좌표 세분화
    x_fine = []
    y_fine = []
    for i in range(len(x)-1):
        x_fine.extend(np.linspace(x[i], x[i+1], dpi, endpoint=False))
        y_fine.extend(np.linspace(y[i], y[i+1], dpi, endpoint=False))
    x_fine.append(x[-1])
    y_fine.append(y[-1])
    
    x_fine = np.array(x_fine)
    y_fine = np.array(y_fine)
    
    # n개의 색상으로 colormap 생성
    cmap = LinearSegmentedColormap.from_list("custom_gradient", list(color))
    
    # 색상 값: 선의 실제 길이에 비례
    distance = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)
    cumdist = np.insert(np.cumsum(distance), 0, 0)
    t = cumdist / cumdist[-1]  # 0~1 범위로 정규화
    
    colors = cmap(t)
    colors[:, -1] = alpha  # alpha 적용

    # segment 생성
    points = np.array([x_fine, y_fine]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # LineCollection 생성
    lc = LineCollection(segments, colors=colors[:-1], linewidth=linewidth, linestyles=ls)
    
    # ax 지정 (없으면 현재 활성 Axes 사용)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    
    ax.autoscale()  # 또는 ax.autoscale_view()
    return lc

# x = np.linspace(0,20,500)
# gradient_lineplot(x, np.sin(x), color=[(1,1,1), (0.5,0.5,0.5)], linewidth=np.sqrt(100))





# Learning Plot
plt.figure(figsize=(10,8))
# [[b.mean() for b in bb]for bb in process_beta_dist]
for nce, xs, xt, ys, yt, bs, bt in zip(n_count_edges, x_pos[:-1], x_pos[1:], y_pos[:-1], y_pos[1:], process_beta_dist[:-1], process_beta_dist[1:]):
    # break
    edges_info = [([xs[s], xt[t]], [ys[s], yt[t]], w, [bs[s].mean(), bt[t].mean()]) for s, t, w in nce if not np.isnan(sum([xs[s], xt[t], ys[s], yt[t]]))]
    
    for es, et, ew, eb in edges_info:
        lc = gradient_lineplot(es, et, linewidth=np.sqrt(ew*5), color=[(eb[0],1-eb[0],0), (eb[1], 1-eb[1], 0)], alpha=0.2)
process_plot(n_fac_in_process,  label=y_obs[[idx]], abnormal=abnormal_xyp, beta_dist=process_beta_dist)
plt.show()
        
        
















# # ----------------------------------------------------------------------------------------------------------------
# # beta distribution을 Deep Neural Network가 학습해서 binary classification의 uncertainty도 학습할 수 있을까? loss funciton을 beta distribution기반으로 해야할까?
# # 방금 질문한 beta distribution을 Deep Neural Network가 학습해서 binary classification의 uncertainty도 학습하는 과정을 
# # 간단한 예제 데이터 기반으로 torch 코드를 보여줘.

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# # ===== 1. 예제 데이터 생성 =====
# torch.manual_seed(42)
# n_samples = 200
# X = torch.rand(n_samples, 1) * 2 - 1  # [-1, 1] 범위
# y = (X[:, 0] > 0).float()  # 0 또는 1 라벨

# # ===== 2. 모델 정의 =====
# class BetaNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(1, 16)
#         self.fc2 = nn.Linear(16, 16)
#         self.fc_alpha = nn.Linear(16, 1)
#         self.fc_beta = nn.Linear(16, 1)
    
#     def forward(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         alpha = F.softplus(self.fc_alpha(h)) + 1e-6  # 양수 제약
#         beta = F.softplus(self.fc_beta(h)) + 1e-6
#         return alpha, beta

# # ===== 3. Loss 함수 (Beta-Bernoulli NLL) =====
# def beta_bernoulli_nll(alpha, beta, target):
#     p = alpha / (alpha + beta)
#     nll = - (target * torch.log(p + 1e-8) + (1 - target) * torch.log(1 - p + 1e-8))
#     return nll.mean()

# # ===== 4. 학습 =====
# model = BetaNet()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(200):
#     alpha_pred, beta_pred = model(X)
#     loss = beta_bernoulli_nll(alpha_pred, beta_pred, y)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if (epoch+1) % 50 == 0:
#         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# # ===== 5. 예측 및 불확실성 계산 =====
# with torch.no_grad():
#     alpha_pred, beta_pred = model(X)
#     mean_pred = alpha_pred / (alpha_pred + beta_pred)
#     var_pred = (alpha_pred * beta_pred) / ((alpha_pred + beta_pred)**2 * (alpha_pred + beta_pred + 1))
    
# # 일부 샘플 출력
# for i in range(5):
#     print(f"X={X[i].item():.2f}, y={y[i].item()}, "
#           f"mean={mean_pred[i].item():.3f}, var={var_pred[i].item():.4f}")

    