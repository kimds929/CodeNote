# Central Limit Theorem (중심극한정리) ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# g1 = stats.norm(10, 1)
# g2 = stats.norm(5, 3)
# g3 = stats.norm(-10, 5)

class GaussMix():
    def __init__(self, random_state=None):
        self.gaussian = np.array([])
        self.weights = np.array([]).astype(float)
        self.norms = []
        self.random_state = random_state
        
        self.mean = None
        self.std = None
    
    def add(self, mean, std, weight=1):
        gaussian = np.array([mean, std, weight]).astype(float)
        self.weights = np.hstack([self.weights, weight])
        self.norms.append(stats.norm(mean, std))
        
        if len(self.gaussian):
            self.gaussian = np.vstack([self.gaussian, gaussian])
        else:
            self.gaussian = gaussian.reshape(1,-1)
        self.gaussian[:,-1] = self.weights / self.weights.sum()
        self.mean = self.gaussian[:,0] @ self.gaussian[:,-1]
        self.std = self.sample(10**7).std()
        
    def pdf(self, x):
        p = 0
        for norm, (mean, std, weight) in zip(self.norms, self.gaussian):
            p += norm.pdf(x)*weight
        return p
    
    def sample(self, size, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        rng = np.random.RandomState(random_state)
        x = np.zeros(size)
        for (mean, std, weight) in self.gaussian:
            x += weight * rng.normal(mean, std, size=size)
        return x

    def pdf_plot(self):
        x_ = np.linspace(self.mean-30*self.std, self.mean+30*self.std, 1000)
        x_pdf = self.pdf(x_)
        x_minmax = np.where(x_pdf > 1e-4)[0][[0,-1]]
        x = np.linspace(*list(x_[x_minmax]), 1000)
        f = plt.figure()
        plt.title(f"mean: {self.mean}\nstd: {self.std}")
        plt.plot(x, self.pdf(x))
        plt.show()
        return f


# gm = GaussMix()
# gm.add(10, 1, 0.1)
# gm.add(5, 3, 0.2)
# gm.add(-10, 5, 0.7)

# gm.norms

# gm.sample(100000).mean()
# gm.sample(100000).std()

# x1 = np.linspace(-20,15, 10**6)
# xp = gm.pdf(x1)
# xp_ = xp / xp.sum()

# plt.figure()
# plt.title(f"mean: {gm.mean:.2f}, std: {gm.std:.2f}")
# plt.plot(x1, xp)
# plt.show()
# --------------------------------------------------------------------------------------------------


class Bernoulli():
    def __init__(self, x, p=None, random_state=None):
        self.x = np.array(x)
        self.p = np.ones_like(x)*1/len(x) if p is None else (np.array(p) / np.array(p).sum())
        self.mean = self.x @ self.p
        
        self.random_state = random_state
        self.std = self.sample(10**7).std()
        
    def sample(self, size, replace=True, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        rng = np.random.RandomState(random_state)
        return rng.choice(self.x, size, replace=replace, p=self.p)

    def pdf_plot(self):
        f = plt.figure()
        plt.title(f"mean: {self.mean}\nstd: {self.std}")
        plt.bar(self.x, self.p)
        plt.show()
        return f

# bernoulli = Bernoulli([50, 100])
# bernoulli = Bernoulli([50, 100 ,150])
# bernoulli = Bernoulli([50, 100], p=[0.3, 0.7])
# bernoulli = Bernoulli([50, 100 ,150], p=[0.5, 0.2, 0.5])
# --------------------------------------------------------------------------------------------------





bernoulli = Bernoulli([50, 100])
# bernoulli = Bernoulli([50, 100], p=[0.3, 0.7])
# bernoulli = Bernoulli([50, 100, 150], p=[0.1, 0.3, 0.6])
bernoulli.pdf_plot()


gaussian = GaussMix()
# gaussian.add(10, 5)
gaussian.add(10, 1, 0.3)
gaussian.add(5, 3, 0.2)
gaussian.add(-10, 5, 0.5)
gaussian.pdf_plot()



 # CLT (Central Limit Theorem) 중심극한정리
distribution = bernoulli            # bernoulli
distribution = gaussian             # gaussian


random_state = None
n_samples_in_group = [5, 30, 1000]
k_samples_of_group = [5, 30, 1000]

result = {}

for size in n_samples_in_group:        # 집단내 sample의 갯수
    result[size] = {}
    for n_counts in k_samples_of_group:  # 표본의 갯수 (시행횟수)
        result[size][n_counts] = []
        samples = []
        for _ in range(n_counts):
            samples = distribution.sample(size, random_state=random_state)
            
            result[size][n_counts].append(np.mean(samples))
        result[size][n_counts] = np.stack(result[size][n_counts])



figs_nr = len(result.keys())
print(f"μ: {distribution.mean},  σ: {distribution.std}")


plt.figure(figsize=(18,figs_nr*5))
for es, (size, c_dict) in enumerate(result.items()):
    for ec, (n_counts, values) in enumerate(c_dict.items()):
        plt.subplot(figs_nr, 3, es*3+ec+1)
        plt.title(f"(sample_size in group) n:{size}\n(gruop_size) k:{n_counts}\
            \nmean: {round(values.mean(),2)},   std: {round(values.std(),2)},   estimated_std: {round(values.std()*np.sqrt(size),2)}")
        plt.hist(values, bins=30, edgecolor='gray', color='skyblue')
        # plt.xlim(-1,101)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

