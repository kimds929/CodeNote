import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
n=30
points = rng.rand(n,2)
labels = rng.randint(0, 3, size=(n)) 

colors = ['steelblue', 'mediumseagreen', 'coral']
plt.figure()
for label in range(labels.max()+1):
    filtered_points = points[labels==label]
    plt.scatter(filtered_points[:,0], filtered_points[:,1], color=colors[label], label=label)
plt.legend(loc='upper right', bbox_to_anchor=(1.2,1))
plt.show()

dist_map = np.sqrt(((points[None,:,:] - points[:,None,:])**2).sum(axis=-1))


k = 5
knn_points = np.argsort(dist_map, axis=1)[:,1:(1+k)]
from scipy.stats import mode
knn_pred = mode(labels[knn_points], axis=1).mode



colors = ['steelblue', 'mediumseagreen', 'coral']

plt.figure()
for label in range(labels.max()+1):
    filtered_points = points[labels==label]
    plt.scatter(filtered_points[:,0], filtered_points[:,1], color=colors[label], label=label)
    
for label in range(knn_pred.max()+1):
    filtered_points = points[knn_pred==label]
    plt.scatter(filtered_points[:,0], filtered_points[:,1], 
                color='none', edgecolors=colors[label], label=label)

plt.legend(loc='upper right', bbox_to_anchor=(1.2,1))
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(labels, knn_pred)
print(classification_report(labels, knn_pred))