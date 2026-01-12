import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


###################################################################################################
rng = np.random.RandomState()

# --------------------------------------------------------------------------------------------------
# scale = 1/50
# n_log_mean = 1.8
# n_log_std = 1
# n_sample = 20
# n_defect = (10 ** rng.normal(loc=n_log_mean, scale=n_log_std, size=n_sample) *scale ).astype(int)

n_mean = 25
n_std = 7
n_sample = 20
n_defect = rng.normal(loc=n_mean, scale=n_std, size=n_sample).astype(int)

# --------------------------------------------------------------------------------------------------

w_mean = 1250
w_std = 150
l_log_mean = 3
l_log_std = 0.2

size_w = rng.normal(loc=w_mean, scale=w_std, size=(n_sample,1)).round(1) *scale
size_l = (10**rng.normal(loc=l_log_mean, scale=l_log_std, size=(n_sample,1))).round(1) *scale
sizes = np.concatenate([size_w, size_l], axis=1)

# --------------------------------------------------------------------------------------------------

df = pd.DataFrame()
mtl_list = []
for i in range(n_sample):
    n_defect_i = n_defect[i]
    if n_defect_i > 0:
        defect_locs = (np.random.rand(n_defect_i,2) * sizes[i]).round(1)
        size_broadcast = np.ones_like(defect_locs) * sizes[i]
        mtl_mat = np.concatenate([size_broadcast, defect_locs], axis=1)
        mtl_list.append(mtl_mat)
        df_sub = pd.DataFrame(mtl_mat, columns=['W', 'L', 'loc_w', 'loc_l'])
        df_sub.insert(0, 'mtl_idx', f"mtl_{i:02d}")
        df = pd.concat([df, df_sub], axis=0)

    
df.groupby(['mtl_idx']).size()

# --------------------------------------------------------------------------------
i = 2
mtl_sample = mtl_list[i]
plt.figure(figsize=[mtl_sample[0][0]/6 , mtl_sample[0][1]/3])
plt.scatter(mtl_sample[:,2], mtl_sample[:,3], c="black", s=5)


true


