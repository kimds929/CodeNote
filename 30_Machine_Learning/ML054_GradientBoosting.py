import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# check step funciton
def check_step(n_iter):
    iter_list = []
    if n_iter < 5:
        iter_list = range(1, n_iter)
    else:
        iter_list.append(0)
        order = int(np.log10(n_iter))
        for i in range(0, order):
            if order - i > 1:
                
                iter_list.append(int(10**(i+1)))
            else:
                for j in [4, 2, 1]:
                    iter_list.append(int(10**(i+1)/j))
                    
        if n_iter != iter_list[-1]:
            for k in [4,2]:
                if 10**(order+1)// k < n_iter :
                    iter_list.append(int(10**(order+1)// k))
            iter_list.append(n_iter)
            
    return iter_list


# data load
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/Data_Tabular'
train_df = pd.read_csv(f'{url_path}/wine_aroma.csv', encoding='utf-8-sig')


train_X = train_df.sort_values('Aroma', axis=0).iloc[:,:-1]
train_y = train_df.sort_values('Aroma', axis=0).iloc[:,-1]


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

GB = GradientBoostingRegressor(random_state=2)
GB.fit(train_X,train_y)
GB.predict(train_X)


plt.figure()
plt.scatter(train_y, train_y, alpha=0.5)
plt.plot(train_y, GB.predict(train_X))
plt.show()

# feature_importance
plt.figure()
plt.barh(train_X.columns, GB.feature_importances_)
plt.show()



# (Original_Ver) Graident Boosting -------------------------------------------------------------
n_iter = 500
max_depth = 1
lr = 0.01

DT = DecisionTreeRegressor(max_depth=max_depth)
DT.fit(train_X,train_y)

shrink_list = [1]
model_list = [DT]
pred_list = [DT.predict(train_X)]
resid_list = [train_y - pred_list[-1]]
feature_importance = DT.feature_importances_


for _ in range(n_iter):
    mdl = DecisionTreeRegressor(max_depth=max_depth)
    mdl.fit(train_X, resid_list[-1])
    pred = pred_list[-1] + lr * mdl.predict(train_X)
    
    shrink_list.append(lr)
    model_list.append(mdl)
    pred_list.append(np.array(pred))
    resid_list.append(np.array(train_y - pred))

    feature_importance += lr*mdl.feature_importances_

feature_importance = feature_importance / feature_importance.sum()


# predict
np.stack([lr*mdl.predict(train_X) for mdl, lr in zip(model_list, shrink_list)]).sum(0)
pred_list[-1]



# predict_plot
step = check_step(n_iter)

plt.figure()
plt.scatter(train_y, train_y, alpha=0.5)
plt.plot(train_y, pred_list[-1], color='coral')
for i in step:
    plt.plot(train_y, pred_list[i], color='coral',label=f'{i}', alpha=np.log(i+1)/np.log(np.max(step)+1)*0.7)
    plt.text(np.array(train_y)[-1], pred_list[i][-1], f"{i} step")
plt.show()


# feature_importance
plt.figure()
plt.barh(train_X.columns, feature_importance)
plt.show()






# (Prevent Overfitting) Graident Boosting  -------------------------------------------------------------
max_iter = 500
max_depth = 1
lr = 0.01
subsample = 1   # Bagging
replace = True
alpha = 0.999   # Shrinkage
tol = 1e-4      # Early Stop

rng = np.random.RandomState(None)


DT = DecisionTreeRegressor(max_depth=max_depth)
DT.fit(train_X,train_y)

shrink_list = [1]
model_list = [DT]
pred_list = [DT.predict(train_X)]
resid_list = [train_y - pred_list[-1]]
feature_importance = DT.feature_importances_

shrink_factor = 1
for i in range(max_iter):
    mdl = DecisionTreeRegressor(max_depth=max_depth)
    choice_idx = rng.choice(range(len(train_X)), size=int(len(train_X)*subsample), replace=replace)
    
    mdl.fit(np.array(train_X)[choice_idx], np.array(resid_list[-1])[choice_idx])
    pred = pred_list[-1] + lr*shrink_factor * mdl.predict(train_X)
    
    shrink_list.append(lr*shrink_factor)
    model_list.append(mdl)
    pred_list.append(np.array(pred))
    resid_list.append(np.array(train_y - pred))

    feature_importance += lr*shrink_factor * mdl.feature_importances_
    
    shrink_factor *= alpha
    
    # resid_n-1 - resid_n < tol
    if np.abs((resid_list[-2].mean() - resid_list[-1].mean())/ resid_list[-1].mean()) < tol:
        break
    
    # y - y_pred < tol
    # if np.abs(1 - pred_list[-1].mean()) / y.mean() < tol:
    #     break
n_iter = i
feature_importance = feature_importance / feature_importance.sum()


# predict
np.stack([lr*mdl.predict(train_X) for mdl, lr in zip(model_list, shrink_list)]).sum(0)
pred_list[-1]


# predict_plot
step = check_step(n_iter)

plt.figure()
plt.scatter(train_y, train_y, alpha=0.5)
plt.plot(train_y, pred_list[-1], color='coral')
for i in step:
    plt.plot(train_y, pred_list[i], color='coral',label=f'{i}', alpha=np.log(i+1)/np.log(np.max(step)+1)*0.7)
    plt.text(np.array(train_y)[-1], pred_list[i][-1], f"{i} step")
plt.show()


# feature_importance
plt.figure()
plt.barh(train_X.columns, feature_importance)
plt.show()


####################################################################################################



# Partial Dependence --------------------------------------------------------------------------------
# X_mean
train_X.mean()

# Make Grid Table
X_col = 'Mo'
grid = pd.DataFrame(np.repeat(train_X.mean().to_numpy().reshape(1,-1),30, axis=0)
                    ,columns=train_X.columns)
grid[X_col] = np.linspace(train_X[X_col].min(), train_X[X_col].max(), 30)

pred = np.stack([lr*mdl.predict(grid) for mdl, lr in zip(model_list, shrink_list)]).sum(0)

# Partial Dependence Plot
plt.figure()
plt.title(f"{X_col}")
plt.plot(grid[X_col], pred, 'o-', alpha=0.5)
plt.ylim(3.8,6.2)
plt.show()
####################################################################################################