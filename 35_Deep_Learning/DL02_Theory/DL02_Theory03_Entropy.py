
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot 그려주는 함수 구현
def plot_graph(X, y, X_hat=None, y_hat=None, str_title=None):
    num_X = X.shape[0]
    
    fig = plt.figure(figsize=(8,6))

    if str_title is not None:
        plt.title(str_title, fontsize=20, pad=20)

    plt.plot(X, y, ls='none', marker='o', markeredgecolor='white')
    
    if X_hat is not None and y_hat is not None:
        plt.plot(X_hat, y_hat)

    plt.tick_params(axis='both', labelsize=14)
    plt.show()

x_0 = 3 + np.random.randn(5)
y_0 = np.zeros(5)

x_1 = 5 + np.random.randn(5)
y_1 = np.ones(5)

x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0, y_1))

print(x)
print(y)

plot_graph(x, y, str_title='dataset')


sigmoid(x)


# entropy
p1_set = np.linspace(0,1,11)
p2_set = np.linspace(0,1,11)

p_sets = np.array(np.meshgrid(p1_set, p2_set)).T.reshape(-1,2)

entropies = []
for p_set in p_sets:
    entropy = p_set[0]*np.log2(1/p_set[1]) + (1-p_set[0])*np.log2(1/(1-p_set[1]))
    entropies.append(entropy)

p_result = pd.DataFrame(np.hstack([p_sets, np.array(entropies).reshape(-1,1)]))
p_result.columns = ['p1','p2','entropy']
p_contour = p_result.groupby(['p1','p2'])['entropy'].mean().unstack('p1')
p_contour
p_contour.to_clipboard()

# cross_entropy
plt.contourf(p_contour.index, p_contour.columns, p_contour, cmap='jet')
plt.colorbar()
plt.show()

# entropy graph
p_contour.apply(lambda x: np.argmax(x),axis=1)
plt.plot(p1_set, np.array(p_contour).diagonal())
plt.show()

