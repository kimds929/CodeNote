
a = np.arange(12)
b = np.random.randint(0,3, size=12)

s = np.stack([a,b]).T
s

x = s[:,0]
g = s[:,1]

group_counts = np.bincount(g)
group_sums = np.bincount(g, weights=x)
group_means = group_sums / group_counts

group_sq_diffs = np.bincount(g, weights=(x-group_means[g])**2)
group_stds = np.sqrt(group_sq_diffs / group_counts)

normalized = (x - group_means[g]) / (group_stds[g] + 1e-8)
