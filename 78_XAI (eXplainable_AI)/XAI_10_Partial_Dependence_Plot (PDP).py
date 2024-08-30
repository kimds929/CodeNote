import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# modeling
X = data_wine.iloc[:,:-1]
y = data_wine.iloc[:,-1]

RF = RandomForestRegressor()
RF.fit(X, y)

# RF.feature_importances_
plt.figure()
plt.barh(X.columns[::-1], RF.feature_importances_[::-1])
plt.show()


# Partial Dependent
x1_name = 'Sr'
# x2_name = None
x2_name = 'Mg' 
n_point = 10


# data_range
x1 = np.linspace(X[x1_name].min(), X[x1_name].max(), n_point)
if x2_name is not None:
    x2 = np.linspace(X[x2_name].min(), X[x2_name].max(), n_point)

# grid_data
if x2_name is None:
    grid_element = pd.DataFrame(x1, columns=[x1_name])
else:
    grid_temp = pd.DataFrame(0, index=x1, columns=x2)
    grid_element = grid_temp.stack().reset_index()
    grid_element.drop(0, axis=1, inplace=True)
    grid_element.columns = [x1_name, x2_name]
    # pd.DataFrame(np.stack(np.meshgrid(x1, x2)).reshape(2,-1).T, columns=[x1_name, x2_name])

# mesh_grid
grid_mesh = np.zeros((grid_element.shape[0], X.shape[1]))
grid = pd.DataFrame(grid_mesh, columns=X.columns)
for i in X.columns:
    if i in grid_element.columns:
        grid[i] = grid_element[i]
    else:
        grid[i] = X[i].mean()

# pred_fream, visualizatioin
if x2_name is None:
    grid_pred = grid[[x1_name]]
    grid_pred['pred'] = RF.predict(grid)
    
    plt.figure()
    plt.plot(grid_pred[x1_name], grid_pred['pred'], 'o-', alpha=0.5)
    plt.show()
else:
    # RF.predict(grid)
    grid_pred = grid[[x1_name, x2_name]]
    grid_pred['pred'] = RF.predict(grid)
    contour = grid_pred.set_index([x1_name, x2_name]).unstack(x1_name)
    contour.columns = x1

    plt.figure()
    plt.contourf(contour.index, contour.columns, contour, cmap='jet')
    plt.colorbar()
    plt.show()