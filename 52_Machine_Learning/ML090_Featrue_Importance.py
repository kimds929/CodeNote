import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = r'D:\Python\Dataset'
df = pd.read_csv(path + '/wine_aroma.csv')
df


from sklearn.model_selection import train_test_split, KFold

trainvalid_set, test_set = train_test_split(df, test_size=0.2)
# train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2)

cv = KFold(3, shuffle=True, random_state=1)
# list(cv.split(trainvalid_set))
for cv_idx, (train_idx, valid_idx) in enumerate(cv.split(trainvalid_set), 1):
    train_set = trainvalid_set.iloc[train_idx]
    valid_set = trainvalid_set.iloc[valid_idx]
    break

trainvalid_X = trainvalid_set.iloc[:, :-1]
trainvalid_y = trainvalid_set.iloc[:, -1]

train_X = train_set.iloc[:,:-1]
train_y = train_set.iloc[:,-1]

valid_X = valid_set.iloc[:,:-1]
valid_y = valid_set.iloc[:,-1]

test_X = test_set.iloc[:,:-1]
test_y = test_set.iloc[:,-1]



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score



import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

# LinearRegression
LR = LinearRegression(fit_intercept=True)
LR.fit(X=trainvalid_X, y=trainvalid_y)

LR_pred = LR.predict(test_X)
mean_squared_error(y_true=test_y, y_pred=LR_pred)


# DecisionTree
DT = DecisionTreeRegressor()
DT.fit(X=trainvalid_X, y=trainvalid_y)

DT_pred = LR.predict(test_X)
mean_squared_error(y_true=test_y, y_pred=DT_pred)


# RandomForest
RF = RandomForestRegressor(random_state=1)
RF.fit(X=trainvalid_X, y=trainvalid_y)

RF_pred = RF.predict(test_X)
# test_y
mean_squared_error(y_true=test_y, y_pred=RF_pred)


# GradientBoosting
GB = GradientBoostingRegressor(random_state=1)
GB.fit(X=trainvalid_X, y=trainvalid_y)
GB_pred = GB.predict(test_X)
mean_squared_error(y_true=test_y, y_pred=GB_pred)



# tensorflow
class tf_regressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(5)
        self.fc2 = tf.keras.layers.Dense(10)
        self.fc3 = tf.keras.layers.Dense(1)
    
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

tf_model = tf_regressor()
tf_loss = tf.keras.losses.MeanSquaredError()
tf_optimizer = tf.keras.optimizers.Adam(0.0003)

tf_dataset = tf.data.Dataset.from_tensor_slices( (trainvalid_X.to_numpy().astype(np.float32), trainvalid_y.to_numpy()) )
tf_dataset = tf_dataset.batch(5).shuffle(True)

epochs = 100
for epoch in range(1, epochs+1):
    for idx, (batch_x, batch_y) in enumerate(tf_dataset):
        with tf.GradientTape() as Tape:
            pred = tf_model(batch_x)
            loss = tf_loss(y_true=batch_y, y_pred=pred)
        gradients = Tape.gradient(loss, tf_model.trainable_variables)
        tf_optimizer.apply_gradients(zip(gradients, tf_model.trainable_variables))
    
    if epoch%10 == 0:
        print(f'\r {epoch} epoch, loss: {loss}', end='')
        print('')
    
tf_pred = tf_model( tf.constant(test_X.to_numpy(), dtype=tf.float32) )
tf_loss(y_true=tf.constant(test_y.to_numpy(), dtype=tf.float32), y_pred= tf_pred)









#【 Permutation feature importance 】 --------------------------------------------------------------------------
#   The permutation feature importance is defined to be the decrease in a model score
#   when a single feature value is randomly shuffled.
#   This procedure breaks the relationship between the feature and the target, 
#   thus the drop in the model score is indicative of how much the model depends on the feature
# https://scikit-learn.org/stable/modules/permutation_importance.html
# https://hong-yp-ml-records.tistory.com/51     # kor
# https://eat-toast.tistory.com/10              # kor

from sklearn.inspection import permutation_importance
# ?permutation_importance
RF_pi = permutation_importance(estimator=RF, X=test_X, y=test_y, scoring='neg_mean_squared_error', n_repeats=10, random_state=1)
RF_pi
# test_X.shape               # (7, 9)
# RF_pi.importances.shape     # (9, 10)

def pi_order(pi, feature_names):
    for i in pi.importances_mean.argsort()[::-1]:
        # if pi.importances_mean[i] - 2 * pi.importances_std[i] > 0:
        print(f"{feature_names[i]:<8}"
            f"{pi.importances_mean[i]:.3f}"
            f" +/- {pi.importances_std[i]:.3f}")

    imp_summary = pd.DataFrame([pi['importances_mean'], pi['importances_std']]).T
    imp_summary.columns = ['mean', 'std']
    imp_summary = pd.concat([imp_summary, pd.DataFrame(pi['importances'])], axis=1)
    imp_summary.index = feature_names
    imp_summary.sort_values('mean', ascending=False, inplace=True)
    return imp_summary

pi_order(pi=RF_pi, feature_names=test_X.columns)

# # plotting
# for i, k in enumerate(RF_pi.importances):
#     sns.distplot(k, hist=False, label=test_X.columns[i])
# plt.legend()
# plt.show()


tf_pi = permutation_importance(estimator=tf_model,
    X=tf.constant(test_X.to_numpy(), dtype=tf.float32),
    y=tf.constant(test_y.to_numpy(), dtype=tf.float32),
    scoring='neg_mean_squared_error', n_repeats=10, random_state=1)

pi_order(pi=tf_pi, feature_names=test_X.columns)




# [ eli5 library ] ---------------
# conda install -c conda-forge eli5
import eli5
from eli5.sklearn import PermutationImportance

RF_eli5pi = PermutationImportance(estimator=RF, random_state=1, scoring='neg_mean_squared_error')
RF_eli5pi.fit(test_X, test_y)

# RF_eli5pi.feature_importances_
# RF_eli5pi.feature_importances_std_
# RF_eli5pi.results_

eli5.show_weights(RF_eli5pi, top=50, feature_names =test_X.columns.tolist())






#【 Partial_Dependence Plot 】 --------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/partial_dependence.html

