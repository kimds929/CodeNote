import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CLT (Central Limit Theorem) 중심극한정리
mo = np.array([0,100])
# size=10

result = {}
# [1, 30, 100]
for size in [5, 30, 1000]:        # 집단내 sample의 갯수
    result[size] = {}
    for n_counts in [5, 30, 1000]:  # 표본의 갯수 (시행횟수)
        result[size][n_counts] = []
        samples = []
        for _ in range(n_counts):
            samples = np.random.choice(mo, size, replace=True)
            result[size][n_counts].append(np.mean(samples))
        result[size][n_counts] = np.stack(result[size][n_counts])


figs_nr = len(result.keys())

print(f"μ: {mo.mean()},  σ: {mo.std()}")
plt.figure(figsize=(18,figs_nr*5))
for es, (size, c_dict) in enumerate(result.items()):
    for ec, (n_counts, values) in enumerate(c_dict.items()):
        plt.subplot(figs_nr, 3, es*3+ec+1)
        plt.title(f"(sample_size in group) n:{size}\n(gruop_size) k:{n_counts}\
            \nmean: {round(values.mean(),2)},   std: {round(values.std(),2)},   estimated_std: {round(values.std()*np.sqrt(size),2)}")
        plt.hist(values, bins=30, edgecolor='gray', color='skyblue')
        # plt.xlim(-1,101)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
