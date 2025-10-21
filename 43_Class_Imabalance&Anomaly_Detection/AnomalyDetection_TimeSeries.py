import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 기본 유틸
# ---------------------------
def _sample_length(mean_len=250, std_len=15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    L = int(rng.normal(mean_len, std_len))
    return max(L, 32)  # 안정성 확보

def _first_order_response(L, rng=None, target=1.0):
    """
    1차 시스템(지연 없는) 목표값 수렴 곡선 + 약한 감쇠진동 + 저잡음
    정상 케이스의 '대세 패턴'을 만듭니다.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.linspace(0, 1.0, L)
    k = rng.uniform(5.0, 7.0)                # 수렴 속도
    y0 = 0  #rng.uniform(-0.15, 0.15)              # 초기 편차
    base = target - (target - y0) * np.exp(-k * t)

    # 약한 감쇠 진동(공정의 미세 온도 흔들림 등)
    freq = rng.uniform(0.5, 1.0)
    damp = rng.uniform(2.0, 4.0)
    oscill = 0.05 * np.sin(2 * np.pi * freq * t) * np.exp(-damp * t)

    # 저잡음 + AR(1) 형태의 상관 잡음 소량
    eps = rng.normal(0, 0.02, size=L)
    ar = np.zeros(L)
    phi = rng.uniform(0.2, 0.5)
    for i in range(1, L):
        ar[i] = phi * ar[i-1] + eps[i]

    return base + oscill + ar

def _clip(y, lo=-5, hi=5):
    return np.clip(y, lo, hi)




# ---------------------------
# 보조 유틸
# ---------------------------

def _smooth_step_vector(L: int, start: int, width: int, k: float = 6.0) -> np.ndarray:
    """
    길이 L에서, start 시점부터 width 동안 0→1로 부드럽게 올라가는 시그모이드 스텝.
    - k: 커질수록 경사가 급해짐. 6이면 꽤 자연스러움.
    """
    t = np.arange(L)
    x = (t - start) / max(1, width)
    z = 1 / (1 + np.exp(-k * (x - 0.5)))          # 0~1로 전이
    z[t < start] = 0.0
    z[t > start + width] = 1.0
    return z
# ---------------------------
# 정상/이상 시계열 생성기
# ---------------------------
def generate_normal_series(mean_len=250, std_len=15, rng=None):
    """
    정상 패턴: 목표값으로 점진 수렴 + 약한 감쇠진동 + 저잡음
    """
    rng = np.random.default_rng() if rng is None else rng
    L = _sample_length(mean_len, std_len, rng)
    target = rng.uniform(0.8, 1.2)
    y = _first_order_response(L, rng, target=target)
    return _clip(y)

# ---------------------------
# 이상 생성기
# ---------------------------


# ---- (보조) 동일 파라미터로 t 벡터를 받아 수렴곡선 계산 ----
def _first_order_response_with_t(t: np.ndarray, target: float, k: float, y0: float = 0.0):
    # y(t) = target - (target - y0) * exp(-k * t)
    return target - (target - y0) * np.exp(-k * t)

# ---- (교체) 이상 시계열: 시간 워핑으로 속도만 다르게 ----
def generate_anomalous_series(
    mean_len=250, std_len=15, rng=None,
    anomaly_type='early_surge',
    p_range_early=(0.65, 0.85),   # p<1  → 초반 빠름
    p_range_late=(1.4, 2.0),      # p>1  → 후반 빠름
    noise_scale=1.0               # 노이즈 강도(정상 대비 배율)
):
    """
    - early_surge : 초반만 더 빠르게 수렴 (t -> t^p, p<1)
    - late_surge  : 후반에 더 빠르게 수렴 (t -> t^p, p>1)
    최종 도달 위치는 normal_series와 동일(target 동일, w(1)=1 보장).
    """
    rng = np.random.default_rng() if rng is None else rng
    L = _sample_length(mean_len, std_len, rng)

    # 정상과 '동일한' 파라미터 범위로 베이스 정의(최종 레벨 동일하게)
    t = np.linspace(0.0, 1.0, L)
    k = rng.uniform(5.0, 7.0)          # 정상과 동일 대역
    y0 = 0.0
    target = rng.uniform(0.8, 1.2)

    # 시간 워핑 지수 p 선택
    if anomaly_type == 'early_surge':
        p = float(rng.uniform(*p_range_early))
    elif anomaly_type == 'late_surge':
        p = float(rng.uniform(*p_range_late))
    else:
        raise ValueError("anomaly_type은 'early_surge' 또는 'late_surge'만 지원합니다.")

    # 단조(time-warp): w(0)=0, w(1)=1 → 최종 레벨 동일
    t_warp = np.power(t, p)

    # 속도만 변경된 수렴곡선
    y = _first_order_response_with_t(t_warp, target=target, k=k, y0=y0)

    # (정상과 같은 결로의) 아주 약한 감쇠진동 + 상관잡음 추가
    #   - 정상 대비 noise_scale 배로 조절 가능(=1이면 동일 수준)
    freq = rng.uniform(0.5, 1.0)
    damp = rng.uniform(2.0, 4.0)
    oscill = 0.05 * np.sin(2 * np.pi * freq * t) * np.exp(-damp * t)

    eps = rng.normal(0, 0.02 * noise_scale, size=L)
    ar = np.zeros(L); phi = rng.uniform(0.2, 0.5)
    for i in range(1, L):
        ar[i] = phi * ar[i-1] + eps[i]

    y = y + oscill + ar
    return _clip(y), anomaly_type, {"p": p}

# ---------------------------
# 데이터셋 대량 생성기
# ---------------------------
def generate_dataset(n_normal=200, n_anom=200, mean_len=250, std_len=15, seed=None, anomaly_type=None):
    """
    반환:
      series_list: 길이가 제각각인 1D ndarray들의 리스트
      labels     : 0(정상) / 1(이상)
      meta       : 각 샘플의 메타정보(dict, anomaly_type 포함)
    """
    rng = np.random.default_rng(seed)
    series_list, labels, meta = [], [], []

    for _ in range(n_normal):
        y = generate_normal_series(mean_len, std_len, rng)
        series_list.append(y)
        labels.append(0)
        meta.append({"type": "normal"})

    for _ in range(n_anom):
        y, atype, info = generate_anomalous_series(mean_len, std_len, rng, anomaly_type=anomaly_type)
        series_list.append(y)
        labels.append(1)
        meta.append({"type": atype})

    return series_list, np.array(labels, dtype=int), meta

# early_surge, late_surge
X, y, info = generate_dataset(n_normal=500, n_anom=20, anomaly_type='late_surge')
print(len(X), "series generated.")
print("첫 샘플 길이:", len(X[0]), "라벨:", y[0], "메타:", info[0])




# 정상/이상 분리
normal_series, anomaly_series, meta = [], [], []
for i, label in enumerate(y):
    if label == 0:
        normal_series.append(X[i])
    else:
        anomaly_series.append(X[i])


# 시각화
plt.figure()
# 정상패턴들
for i in range(len(normal_series)):
    plt.plot(normal_series[i], alpha=0.5, color='steelblue')

# 이상 패턴들
for i in range( len(anomaly_series)):
    plt.plot(anomaly_series[i], alpha=0.5, color='coral')
plt.xlabel("Time Step")
plt.ylabel("Value")
# plt.legend(loc=('upper right'), bbox_to_anchor=(1,1))
plt.tight_layout()

plt.show()





def pad_series_list(series_list, pad_value=np.nan):
    """
    서로 다른 길이의 시계열을 최대 길이에 맞춰 zero padding.
    Args:
        series_list (list[np.ndarray]): 각 시계열 (길이 다름)
        pad_value (float): 패딩값 (기본 0)
    Returns:
        np.ndarray: shape = (N, max_len)
    """
    max_len = max(len(s) for s in series_list)
    padded = np.full((len(series_list), max_len), pad_value, dtype=float)
    for i, s in enumerate(series_list):
        padded[i, :len(s)] = s
    return padded




y
X_Series = pad_series_list(normal_series + anomaly_series)








# ===== Control-VAE for Time-Series Anomaly Detection (Transformer) =====
# 기대 입력:
#   X_Series : (N, Lmax) numpy array, NaN padding
#   y        : (N,) numpy array, 0=normal, 1=anomaly
# 위 두 변수는 네 코드에서 이미 만들어짐.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# ---------------------------
# 0) 데이터 유틸
# ---------------------------
def make_mask_from_nan(x_np: np.ndarray):
    """NaN이 아닌 위치 True, NaN인 패딩 False (for attention mask & loss mask)"""
    mask = ~np.isnan(x_np)
    x_filled = np.nan_to_num(x_np, nan=0.0)
    return x_filled, mask

def compute_control_target_tail_mean(x_np: np.ndarray, mask_np: np.ndarray, tail_frac=0.10):
    """각 시계열 뒤 10% 유효구간의 평균 → setpoint 근사"""
    N, L = x_np.shape
    c = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        valid_idx = np.where(mask_np[i])[0]
        if len(valid_idx) == 0:
            c[i, 0] = 0.0
            continue
        tail_len = max(5, int(len(valid_idx) * tail_frac))
        tail_idx = valid_idx[-tail_len:]
        c[i, 0] = float(np.nanmean(x_np[i, tail_idx]))
    return c

def train_val_split_idx(y, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    # 정상만 학습하므로, train은 정상에서만 샘플링
    normal_idx = idx[y == 0]
    rng.shuffle(normal_idx)
    val_n = max(1, int(len(normal_idx) * val_ratio))
    val_idx = normal_idx[:val_n]
    train_idx = normal_idx[val_n:]
    return train_idx, val_idx

class SeriesDataset(Dataset):
    def __init__(self, x_np, mask_np, c_np, y_np=None, idx=None, zscore=True, fit_stats=None):
        """
        x_np: (N, L), mask_np: (N, L), c_np: (N,1)
        zscore=True면 정상 train 통계로 표준화하여 모델에 입력
        """
        if idx is None:
            idx = np.arange(len(x_np))
        self.x = x_np[idx].astype(np.float32)
        self.mask = mask_np[idx].astype(bool)
        self.c = c_np[idx].astype(np.float32)
        self.y = None if y_np is None else y_np[idx].astype(np.int64)
        # 표준화 통계
        if fit_stats is None:
            valid = self.mask
            mu = (self.x * valid).sum() / valid.sum()
            var = ( ((self.x - mu)**2) * valid ).sum() / valid.sum()
            std = np.sqrt(var + 1e-8)
            self.mu, self.std = float(mu), float(std)
        else:
            self.mu, self.std = fit_stats
        if zscore:
            self.x = (self.x - self.mu) / max(self.std, 1e-6)
            self.c = (self.c - self.mu) / max(self.std, 1e-6)
        self.fit_stats = (self.mu, self.std)

    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i]),           # (L,)
            torch.from_numpy(self.mask[i]),        # (L,)
            torch.from_numpy(self.c[i]),           # (1,)
            torch.tensor(-1 if self.y is None else self.y[i])
        )


# ---------------------------
# 1) 모델: Control-VAE (Transformer)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d_model)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096, sinusoidal_init=True):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))  # learnable
        if sinusoidal_init:
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            with torch.no_grad():
                self.pe[0].copy_(pe)

    def forward(self, x):       # x: (B,L,d)
        return x + self.pe[:, :x.size(1)]


class ControlVAE(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_ff=256, z_dim=16):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        # self.pos = PositionalEncoding(d_model)
        self.pos = LearnablePositionalEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_logvar = nn.Linear(d_model, z_dim)

        # conditioning: c(1) -> emb, z -> emb, 합쳐서 context token
        self.c_emb = nn.Linear(1, d_model)
        self.z_emb = nn.Linear(z_dim, d_model)
        self.ctx_norm = nn.LayerNorm(d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

        # latent control head: z -> c_hat
        self.to_c_hat = nn.Linear(z_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask, c, teacher_force=True):
        """
        x: (B, L), mask: (B, L) bool, c: (B,1)
        """
        B, L = x.shape
        x_in = self.inp(x.unsqueeze(-1))          # (B,L,d)
        x_in = self.pos(x_in)
        src_key_padding = ~mask                   # True=PAD
        h = self.encoder(x_in, src_key_padding_mask=src_key_padding)  # (B,L,d)
        # 가변길이 mean pooling (마스크 적용)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B,1,1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1, keepdim=False) / denom.squeeze(2)  # (B,d)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        # context token: z + c
        ctx = self.ctx_norm(self.z_emb(z) + self.c_emb(c))              # (B,d)
        ctx = ctx.unsqueeze(1)                                          # (B,1,d)

        # Decoder 입력: teacher forcing로 x를 한칸 오른쪽 시프트(+ ctx 토큰을 맨 앞에)
        # tgt: (B, 1 + L, d)
        tgt_seq = torch.cat([torch.zeros(B, 1, 1, device=x.device), x.unsqueeze(-1)], dim=1)  # 0 + x
        tgt = self.inp(tgt_seq)                                         # (B,1+L,d)
        tgt = self.pos(tgt)

        # tgt 마스크/패딩
        tgt_key_padding = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=x.device), ~mask], dim=1)

        # memory = encoder hidden 전체 + context를 앞에 붙일 수도 있으나,
        # 여기서는 memory로 '컨텍스트 토큰'만 사용해도 충분한 프로토타입이 됨.
        # memory = ctx
        # dec = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding)  # (B,1+L,d)
        
        memory = torch.cat([ctx, h], dim=1)                       # (B, 1+L, d)
        mem_key_padding = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=x.device),
                                    ~mask], dim=1)
        dec = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding,
                        memory_key_padding_mask=mem_key_padding)

        yhat = self.out(dec)[:, 1:, 0]                                         # (B,L)

        c_hat = self.to_c_hat(z)                                              # (B,1)
        return yhat, mu, logvar, c_hat

# ---------------------------
# 2) 손실/학습 루프
# ---------------------------
def kl_anneal_factor(step, warmup=1500, max_beta=1.0):
    # 선형 warmup
    return float(min(max_beta, max_beta * step / max(1, warmup)))

def masked_mse(pred, target, mask):
    diff = (pred - target)**2
    diff = diff * mask
    denom = mask.sum(dim=1).clamp(min=1)
    return (diff.sum(dim=1) / denom).mean()

def masked_mse_timeweighted(pred, target, mask):
    B, L = pred.shape
    t = torch.linspace(0, 1, L, device=pred.device)
    # early/late 모두 포착: 양끝 가중 ↑ (초기·후기 강조)
    w = 1.0 + 0.5*(torch.cos(2*np.pi*(t-0.5))**2)   # 양끝>중간
    diff = (pred - target)**2 * w
    diff = diff * mask
    denom = (mask*w).sum(dim=1).clamp(min=1e-6)
    return (diff.sum(dim=1) / denom).mean()


def train_control_vae(
    model, optimizer, train_loader, val_loader, global_step, device='cpu',
    epochs=20, lambda_ctrl=0.2, kl_warmup=1500
):
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x, mask, c, _ in train_loader:
            x, mask, c = x.to(device), mask.to(device), c.to(device)
            yhat, mu, logvar, c_hat = model(x, mask, c, teacher_force=True)

            recon = masked_mse(yhat, x, mask)
            # recon = masked_mse_timeweighted(yhat, x, mask)
            
            kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
            ctrl = F.mse_loss(c_hat, c)

            beta = kl_anneal_factor(global_step, warmup=kl_warmup, max_beta=1.0)
            loss = recon + beta*kl + lambda_ctrl*ctrl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            global_step += 1

        # 간단한 검증
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x, mask, c, _ in val_loader:
                x, mask, c = x.to(device), mask.to(device), c.to(device)
                yhat, mu, logvar, c_hat = model(x, mask, c)
                recon = masked_mse(yhat, x, mask)
                kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
                ctrl = F.mse_loss(c_hat, c)
                loss = recon + kl + lambda_ctrl*ctrl
                val_loss += loss.item()
        print(f"[{ep:02d}] train {tr_loss/len(train_loader):.4f} | val {val_loss/len(val_loader):.4f}")
    return model

# ---------------------------
# 3) 스코어링 & 임계값
# ---------------------------
def series_score(model, x, mask, c, device):
    model.eval()
    with torch.no_grad():
        yhat, mu, logvar, c_hat = model(x, mask, c)
        recon = ((yhat - x)**2 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        ctrl = torch.abs(c_hat - c).squeeze(1)
        # 필요하면 + gamma*KL 추가 가능
        score = recon + 0.5 * ctrl      # α=0.5 가중
    return score.cpu()

def fit_threshold(scores_train, q=0.95):
    thr = float(torch.quantile(scores_train, q))
    return thr

# ---------------------------
# 4) 파이프라인 실행
# ---------------------------
# 4-1) 입력에서 마스크/컨트롤/분할 만들기
X_filled, M = make_mask_from_nan(X_Series)       # (N,L), (N,L)
C = compute_control_target_tail_mean(X_Series, M, tail_frac=0.10)  # (N,1)

train_idx, val_idx = train_val_split_idx(y, val_ratio=0.15, seed=42)
test_idx = np.setdiff1d(np.arange(len(y)), train_idx)  # (정상/이상 혼합)

# train 표준화 통계로 학습/검증/테스트 모두 정규화
train_ds_tmp = SeriesDataset(X_filled, M, C, y, idx=train_idx, zscore=True, fit_stats=None)
fit_stats = train_ds_tmp.fit_stats
train_ds = SeriesDataset(X_filled, M, C, y, idx=train_idx, zscore=True, fit_stats=fit_stats)
val_ds   = SeriesDataset(X_filled, M, C, y, idx=val_idx,   zscore=True, fit_stats=fit_stats)
test_ds  = SeriesDataset(X_filled, M, C, y, idx=test_idx,  zscore=True, fit_stats=fit_stats)

BATCH = 64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False)


# 4-2) 모델 학습
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = ControlVAE(d_model=128, nhead=4, num_layers=1, dim_ff=128, z_dim=16).to(device)
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"

opt = torch.optim.Adam(model.parameters(), lr=5e-4)
global_step = 0

model = train_control_vae(model, opt, train_loader, val_loader, global_step, device,
                          epochs=100, lambda_ctrl=0.2, kl_warmup=2000)



# 4-3) 스코어 계산 & 임계값 설정(정상 train 기반)
device = next(model.parameters()).device
def collect_scores(dl):
    all_s, all_y = [], []
    for x, mask, c, yi in dl:
        s = series_score(model, x.to(device), mask.to(device), c.to(device), device)
        all_s.append(s); all_y.append(yi)
    return torch.cat(all_s), torch.cat(all_y)

train_scores, _ = collect_scores(train_loader)
thr = fit_threshold(train_scores, q=0.90)

test_scores, test_y = collect_scores(test_loader)
pred = (test_scores.numpy() > thr).astype(int)
gt = test_y.numpy().clip(min=0)  # -1 → 0으로



# plt.plot(X_filled[test_idx][:-20].T, color='steelblue', alpha=0.3)
# plt.plot(X_filled[test_idx][-20:].T, color='coral', alpha=0.3)
# np.unique(gt, return_counts=True)
# np.unique(pred, return_counts=True)


# 간단 지표
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

print( confusion_matrix(pred, gt))

print("\n[Threshold @ 95% train-normal]")
print(classification_report(gt, pred, target_names=["Normal","Anomaly"]))

try:
    auc = roc_auc_score(gt, test_scores.numpy())
    print(f"AUC: {auc:.4f}")
except Exception:
    pass



