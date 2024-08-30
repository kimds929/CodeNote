import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *


# 1. Open Data Set --------------------------------------------------------------------------------------
# We will use open dataset to apply what we have learned.
# The open dataset which we will use for regression is 'Weather in Szeged 2006 - 2016' from kaggle.
# Is there a relationship between humidity and temperature?
# What about between humidity and apparent temperature?
# Can you predict the maximum temperature given the minimum temperature?
# You can download the dataset from below url but already included in 'data_files/weatherHistory.csv'.

# https://www.kaggle.com/budincsevity/szeged-weather/version/1#weatherHistory.csv
# There are another datasets for regression in the below URL.

# https://www.kaggle.com/smid80/weatherww2/data
# https://www.kaggle.com/sohier/calcofi



# 2. Data load --------------------------------------------------------------------------------------
xlsx_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
weather_data_csv = pd.read_csv(xlsx_path + 'weatherHistory.csv')
weather_data_csv_info = DS_DF_Summary(weather_data_csv)

print(weather_data_csv_info)
weather_data_csv.head()





# 3. Preprocessing --------------------------------------------------------------------------------------
    # 3.1. Data Drop and Column Rearrange
# Data Drop - 'Formatted Date', 'Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'
weather_data = weather_data_csv.drop(['Formatted Date', 'Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'], axis=1)

# Column rearrange
cols = list(weather_data.columns.values)
print("cols : ", cols)


output_columns = cols[1]        # y값 정의
cols.pop(1)        
cols.append(output_columns)

weather_data = weather_data[cols]   # DataFrame column 재정의
weather_data.head()
weather_data_info = DS_DF_Summary(weather_data)



    # 3.2. Data Sorting
x_axis = np.arange(0, len(weather_data[output_columns]), 1)
print(x_axis)

plt.figure(figsize = (10, 8))
plt.title("Apparent Temp", fontsize = 15)
plt.plot(x_axis, weather_data['Apparent Temperature (C)'])


    # Data sort
weather_data = weather_data.sort_values(by=output_columns, axis=0)
weather_data.head()

plt.figure(figsize = (10, 8))
plt.title("Apparent Temp", fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(x_axis, weather_data['Apparent Temperature (C)'], 'o', markersize = 1)



    # 3.3. pandas to numpy array
inout_data = weather_data.values
# inout_data = weather_data.to_numpy()
print(f'numpy_data_shape : {inout_data.shape}')


    # 3.4. define input and output
input_data = inout_data[:,:-1]
output_data = inout_data[:,-1].reshape(-1,1)
print(input_data.shape)
print(output_data.shape)









# 4. Predict Apparent Temperature --------------------------------------------------------------------------------------
    # 4.1. Least Square Solution
input_data_matrix = np.asmatrix(input_data)

# (A.T @ A).I @ A.T @ y

theta = (input_data_matrix.T @ input_data_matrix).I @ input_data_matrix.T @ output_data
y_hat = input_data_matrix @ theta


plt.figure(figsize = (10, 8))
plt.title('Least Square Solution', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(y_hat, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()



    # 4.2. Gradient Descent
input_data_matrix = np.asmatrix(input_data)

theta = np.random.randn(6, 1)
theta = np.asmatrix(theta)

print("initial theta : ", theta.reshape([1, -1]))

# step size
alpha_init = 0.00000000001


for i in range(200000) :
    if i < 10000 :
        alpha = alpha_init - alpha_init * 0.00001 * i
    
    # ▽f : df = 2 * (A.T @ A @ theta - A.T @ y)
    df = 2 * (input_data_matrix.T @ input_data_matrix @ theta - input_data_matrix.T @ output_data)
    theta = theta - alpha * df 
    
print("fitted theta : ", theta.reshape([1, -1]))


y_hat = input_data_matrix @ theta

plt.figure(figsize = (10, 8))
plt.title('Gradient Descent', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(y_hat, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()




    # 4.3. CVXPY  ****
import cvxpy as cvx

input_data_matrix = np.asmatrix(input_data)

theta1 = cvx.Variable([6,1])
obj = cvx.Minimize(cvx.norm(input_data_matrix @ theta1 - output_data, 2))
prob = cvx.Problem(obj)
result = prob.solve(solver = 'SCS')

print("theta1 : ", theta1.value.reshape([1, -1]))

y_hat1 =  input_data_matrix @ theta1.value

    # plot
plt.figure(figsize = (10, 8))
plt.title('CVXPY', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(y_hat1, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()




    # 4.3.1. Ridge --------------------------------------------------
lamb = 1

theta2 = cvx.Variable([6,1])
obj = cvx.Minimize(cvx.norm(input_data_matrix @ theta2 - output_data, 2) + lamb * cvx.norm(theta2, 2)**2)
prob = cvx.Problem(obj)
result = prob.solve(solver = 'SCS')

print(theta2.value)



    # Ridge plot-show
input_data_matrix = np.asmatrix(input_data)

y_hat2 = input_data_matrix @ theta2.value

plt.figure(figsize = (10, 8))
plt.title('CVXPY Ridge', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(y_hat2, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()





    # 4.3.2. Lasso --------------------------------------------------
lamb = 1

theta3 = cvx.Variable([6,1])
obj = cvx.Minimize(cvx.norm(input_data_matrix @ theta3 - output_data, 2) + lamb * cvx.norm(theta3, 1))
prob = cvx.Problem(obj)
result = prob.solve(solver = 'SCS')

print(theta3.value)


    # Lasso plot-show
input_data_matrix = np.asmatrix(input_data)

y_hat3 = input_data_matrix @ theta3.value

plt.figure(figsize = (10, 8))
plt.title('CVXPY Lasso', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(y_hat3, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()




    # 4.3.3. Plot
plt.figure(figsize = (15, 12))
plt.title('Comparison', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(y_hat1, 'o', markersize = 1, alpha = 0.5, label = 'CVXPY')
plt.plot(y_hat2, 'o', markersize = 1, alpha = 0.5, label = 'Ridge')
plt.plot(y_hat3, 'o', markersize = 1, alpha = 0.5, label = 'Lasso')
plt.plot(output_data, 'o', markersize = 1, alpha = 0.5, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()



    # 데이터가 너무 많아서 100개만 랜덤으로 추출해서 plotting
index = np.random.randint(0, len(weather_data['Apparent Temperature (C)']), 100)
index = np.sort(index)

plt.figure(figsize = (15, 12))
plt.title('Comparison', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Apparent Temperature', fontsize = 15)
plt.plot(y_hat1[index], 'o', markersize = 3, alpha = 0.5, label = 'CVXPY')
plt.plot(y_hat2[index], 'o', markersize = 3, alpha = 0.5, label = 'Ridge')
plt.plot(y_hat3[index], 'o', markersize = 3, alpha = 0.5, label = 'Lasso')
plt.plot(output_data[index], 'o', markersize = 3, alpha = 0.5, label = 'Real')
plt.legend(fontsize = 15, markerscale=5)
plt.show()




# 변수 계수 축소 확인
plt.figure(figsize = (15, 6))

plt.subplot(1, 2, 1)
plt.title(r'Ridge: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(theta2.value)

plt.subplot(1, 2, 2)
plt.title(r'Lasso: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(theta3.value)

plt.show()






    # 4.4. Sklearn Module --------------------------------------------------
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(input_data, output_data)


    # regression plot
plt.figure(figsize = (10, 8))
plt.title('Sklearn', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(reg.predict(input_data), 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()

    # Regression 계수 확인
reg_coef = np.vstack([reg.intercept_.reshape([-1, 1]), reg.coef_.reshape([-1, 1])])

plt.figure(figsize = (10, 8))
plt.title(r'Regression: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(reg_coef)
plt.show()




    # 4.4.1. Ridge
ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(input_data, output_data)

plt.figure(figsize = (10, 8))
plt.title('Ridge', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(ridge.predict(input_data), 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()

    # Ridge 계수축소 확인
ridge_coef = np.vstack([ridge.intercept_.reshape([-1, 1]), ridge.coef_.reshape([-1, 1])])

plt.figure(figsize = (10, 8))
plt.title(r'Ridge: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(ridge_coef)
plt.show()



    # 4.4.2. Lasso
lasso = linear_model.Lasso(alpha=1.0)
lasso.fit(input_data, output_data)

plt.figure(figsize = (10, 8))
plt.title('Lasso', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(lasso.predict(input_data), 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()


    # 계수축소 확인
lasso_coef = np.vstack([lasso.intercept_.reshape([-1, 1]), lasso.coef_.reshape([-1, 1])])

plt.figure(figsize = (10, 8))
plt.title(r'Lasso: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(lasso_coef)
plt.show()





    # 4.4.3. Plot (Lasso, Ridge 계수축소)
ridge_coef = np.vstack([ridge.intercept_.reshape([-1, 1]), ridge.coef_.reshape([-1, 1])])
lasso_coef = np.vstack([lasso.intercept_.reshape([-1, 1]), lasso.coef_.reshape([-1, 1])])

plt.figure(figsize = (15, 6))

plt.subplot(1, 2, 1)
plt.title(r'Ridge: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(ridge_coef)

plt.subplot(1, 2, 2)
plt.title(r'Lasso: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(lasso_coef)

plt.show()





    # 4.5. Polynomial   
# 4.5.1. Polynomial basis
xp_1, xp_2 = np.meshgrid(np.arange(-1, 1, 0.01), 
                         np.arange(-1, 1, 0.01))
print(xp_1)
print(xp_2)

xp = np.hstack([xp_1.reshape([-1, 1]), xp_2.reshape([-1, 1])])


# the number of basis of each axis
d = 4

polybasis = np.hstack([xp**i for i in range(d)])

polybasis.shape
xp.shape


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (10, 8))

for i in range(d) :
    ax = fig.add_subplot(2, 2, i+1, projection = '3d')
    ax.set_title('Generated Data', fontsize = 15)
    ax.set_xlabel('$X_1$', fontsize = 15)
    ax.set_ylabel('$X_2$', fontsize = 15)
    ax.set_zlabel('Y', fontsize = 15)
    ax.scatter(xp[:, 0], xp[:, 1], polybasis[:, 2*i], alpha = 0.9, marker = '.', label = 'Data')
    ax.scatter(xp[:, 0], xp[:, 1], polybasis[:, 2*i + 1], alpha = 0.9, marker = '.', label = 'Data')


    # 4.5.2. Fit
# the number of basis of each axis
d = 10

polybasis = np.hstack([input_data ** i for i in range(3)])
polybasis = np.asmatrix(polybasis)
polybasis.shape



theta1 = cvx.Variable([polybasis.shape[1],1])
obj = cvx.Minimize(cvx.norm(polybasis @ theta1 - output_data, 2))
prob = cvx.Problem(obj)
result = prob.solve(solver = 'SCS')

print("theta1 : ", theta1.value.reshape([1, -1]))


# fit 
y_hat = polybasis @ theta1.value

# plot
plt.figure(figsize = (10, 8))
plt.title('Polynomial Function', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(y_hat, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()




# 4.5.3. Normalize and fit ------------------------------------------
input_data_normal = input_data

for i in range(input_data.shape[1]) :
    input_data_normal[:, i] = input_data[:, i] - np.min(input_data[:, i])
    input_data_normal[:, i] = input_data[:, i] / np.max(input_data[:, i])
    
input_data_normal


# the number of basis of each axis
d = 10

polybasis = np.hstack([input_data_normal**i for i in range(3)])
polybasis = np.asmatrix(polybasis)
polybasis.shape



theta2 =
obj = 
prob = cvx.Problem(obj)
result = prob.solve(solver = 'SCS')

print("theta2 : ", theta2.value.reshape([1, -1]))



# fit 
y_hat =

# plot
plt.figure(figsize = (10, 8))
plt.title('Polynomial Function - Normalized', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(y_hat, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()



# 4.5.4. Compare
plt.figure(figsize = (15, 6))

plt.subplot(1, 2, 1)
plt.title(r'Origin data: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(theta1.value)

plt.subplot(1, 2, 2)
plt.title(r'Normalized data: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(theta2.value)

plt.show()


# 4.6. RBF --------------------------------------------------
    # 4.6.1. RBF basis
from mpl_toolkits.mplot3d import Axes3D

xp_1, xp_2 = np.meshgrid(np.arange(-1, 1, 0.01), 
                         np.arange(-1, 1, 0.01))

xp = np.hstack([xp_1.reshape([-1, 1]), xp_2.reshape([-1, 1])])

# the number of basis of each axis
d = 4

# center of basis 
u_1, u_2 = np.meshgrid(np.linspace(-1, 1, d), 
                       np.linspace(-1, 1, d))

u = np.hstack([u_1.reshape([-1, 1]), u_2.reshape([-1, 1])])

sigma = 0.2

rbfbasis = np.hstack([np.exp(-np.linalg.norm((xp-u[i])**2, 2, axis=1).reshape(-1,1)/(2 *sigma**2)) for i in range(d**2)])

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(1, 1, 1, projection = '3d')

ax.set_title('Generated Data', fontsize = 15)
ax.set_xlabel('$X_1$', fontsize = 15)
ax.set_ylabel('$X_2$', fontsize = 15)
ax.set_zlabel('Y', fontsize = 15)
for i in range(d**2) :
    ax.scatter(xp[:, 0], xp[:, 1], rbfbasis[:, i], marker = '.', label = 'Data')
plt.show()




    # 4.6.2. Correlation Matrix
correlation_matrix = weather_data.corr()
correlation_matrix

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()




    # 4.6.3. Data normalize
weather_data = weather_data_csv.loc[:, ['Temperature (C)', 'Humidity', 'Visibility (km)', 'Apparent Temperature (C)']]

# data sort
weather_data = weather_data.sort_values(by = 'Apparent Temperature (C)', axis = 0)
weather_data



inout_data = weather_data.values

input_data = inout_data[:, 0:-1]
output_data = inout_data[:, -1].reshape([-1, 1])



for i in range(input_data.shape[1]) :
    input_data[:, i] = input_data[:, i] - np.min(input_data[:, i])
    input_data[:, i] = input_data[:, i] / np.max(input_data[:, i])
    
input_data



    # 4.6.4. Fit
# the number of basis of each axis
d = 10

# center of basis 
u_1, u_2, u_3 = np.meshgrid(np.linspace(np.min(input_data[:, 0]), np.max(input_data[:, 0]), d), 
                            np.linspace(np.min(input_data[:, 1]), np.max(input_data[:, 1]), d),
                            np.linspace(np.min(input_data[:, 2]), np.max(input_data[:, 2]), d))

u = np.hstack([u_1.reshape([-1, 1]), 
               u_2.reshape([-1, 1]), 
               u_3.reshape([-1, 1])])

u.shape




sigma = 0.2

rbfbasis = 
rbfbasis = np.asmatrix(rbfbasis)
rbfbasis




theta = 
y_hat =

plt.figure(figsize = (10, 8))
plt.title('RBF Function', fontsize = 15)
plt.xlabel('Data index', fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.plot(y_hat, 'o', markersize = 1, label = 'Prediction')
plt.plot(output_data, 'o', markersize = 1, label = 'Real')
plt.legend(fontsize = 15, markerscale=10)
plt.show()




