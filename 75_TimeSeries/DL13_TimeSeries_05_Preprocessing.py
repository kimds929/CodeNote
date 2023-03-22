
import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DS_DeepLearning import shape
from DS_TimeSeries import series_plot, sequential_transform, sequential_filter_index_from_src_index


# (example) ---------------------------------------------------------------------
# https://engineer-mole.tistory.com/239
# np.put()
# np.place()
# np.putmask()

df1 = pd.DataFrame()
df1['date'] = pd.date_range('2021-01-01', '2021-01-31')
# df1['date'] = pd.date_range('2021-01-01', '2021-01-10')
df1['value'] = range(len(df1['date']))
df1['rand'] = np.random.rand(len(df1['date']))
df1['target'] = np.arange(len(df1['date']))+100
# d1['date'] = d1['date'].astype('str').apply(lambda x: x.replace('-',''))
# ------------------------------------------------------------------------------

y_col = 'target'
# X_cols = ['value','rand']
X_cols = ['target']

df_anal = df1.set_index('date')

# series_plot
series_plot(df_anal)    # Graph

# sequential_transform: Split stacked data ***
idx1, X_stack = sequential_transform( df_anal[X_cols], window=10)
           
print(shape(idx1), shape(X_stack))   ## (22, 10), (22, 10, 2)


# sequential_filter_index_from_src_index
idx2, src_idx, trg_idx = sequential_filter_index_from_src_index(idx1)
# idx2, src_idx, trg_idx = sequential_filter_index_from_src_index(idx1, trg_window=(-2,2))
print(idx2.shape, src_idx.shape, trg_idx.shape)

# data filter
X = np.array(df_anal[X_cols])[src_idx]
y = np.array(df_anal[y_col])[trg_idx]

time_index = df_anal.index[idx2]
time_index_Xmatrix = df_anal.index[src_idx]
time_index_ymatrix = df_anal.index[trg_idx]

print(X.shape, y.shape, time_index.shape, time_index_Xmatrix.shape, time_index_ymatrix.shape)
# ((20, 10, 2), (20, 5), (20, 1), (20, 10), (20, 5))
# pd.DataFrame(X[:,:,0]).to_clipboard(index=False,header=False)
# pd.DataFrame(y).to_clipboard(index=False,header=False)
# pd.DataFrame(time_index_ymatrix).to_clipboard(index=False,header=False)


# Graph ***
text_dict = {'0_start':(0,0), '0_end':(0,-1), '-1_start':(-1,0), '-1_end':(-1,-1)}

plt.figure(figsize=(15,6))
plt.subplot(3,1,1)
plt.plot(df_anal.index, df_anal[y_col], color='mediumseagreen', marker='o')
plt.plot(time_index_ymatrix[:,0], y[:,0], color='red',alpha=0.5)
plt.plot(time_index_ymatrix[:,-1], y[:,-1], color='red',alpha=0.5)
for name, point in text_dict.items():
    plt.text(time_index_ymatrix[point[0],point[1]], y[point[0], point[1]], f"↓ {name}")
for e,c in enumerate(df_anal[X_cols].columns):
    plt.subplot(3,1,e+2)
    plt.plot(df_anal.index, df_anal[c], color='steelblue', marker='o')
    plt.plot(time_index, X[:,-1,e], color='gold')
plt.show()


# Train_Test_Split
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, time_index, test_size=test_size, shuffle=False)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, train_index.shape, test_index.shape)
print(train_index[[0,-1]], test_index[[0, -1]])
