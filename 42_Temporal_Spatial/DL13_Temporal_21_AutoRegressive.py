
# (Python) ARIMA TimeSeries 230508
# 전통적으로 시계열 데이터 분석은 
# AR(Autoregressive), 
# MA(Moving average), 
# ARMA(Autoregressive Moving average), 
# ARIMA(Autoregressive Integrated Moving average) 
import pandas as pd

url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/Data_TimeSeries'

df_temp = pd.read_csv(f'{url_path}/12-22YR_Seoul_Temperature.csv', encoding='utf-8-sig')
df_temp.sample(5)
df_temp.shape

time_col = '일시'
y_col = '평균기온(℃)'
X_cols = []

df_temp = pd.read_csv(f'{url_path}/AUS_Weather_DataSet.csv', encoding='utf-8-sig')
df_temp.sample(5)
df_temp.shape
time_col = 'Date'
y_col = 'Temp3pm'
X_cols = []

cols = [time_col, y_col] + X_cols

df_temp = df_temp[cols].dropna()
df_temp.isna().sum(0)

df_temp[time_col] = pd.to_datetime(df_temp[time_col], format='%Y-%m-%d')

print(df_temp[time_col].min(), df_temp[time_col].max())
df_temp_5D = df_temp.set_index(time_col).resample('5D')[[y_col]+X_cols].mean()
df_temp_10D = df_temp.set_index(time_col).resample('10D')[[y_col]+X_cols].mean()
df_temp_20D = df_temp.set_index(time_col).resample('20D')[[y_col]+X_cols].mean()


# df_target = df_temp_5D.reset_index().query(f"'2015-01-01' <= {time_col} < '2022-01-01'")
df_target = df_temp_20D.reset_index().query(f"'2013-01-01' <= {time_col} < '2016-01-01'").dropna()
# df_target = df_temp.set_index(time_col)[[cols]]
# df_target = df_temp_10D.to_frame()

plt.figure(figsize=(10,3))
plt.plot(df_target[time_col], df_target[y_col])
plt.show()



# (AR Model)
#   Autoregressive 모델은 자기 회귀 모델이라고 불린다. 
#   과거 시점의 자기 자신의 데이터가 현 시점의 자기 자신에게 영향을 미치는 모델이라는 뜻
# X(t) = w*X(t-1) + b + u*e(t)
# 시점 t에 대한 데이터는 이전 시점의 자기 데이터에 가중치 w를 곱하고 상수 b를 더하고(회귀식), 
# error term인 e(t)에 가중치 u를 곱한 것을 더해서 표현할 수 있다
# df_target.shape



# https://zephyrus1111.tistory.com/102
import numpy as np 
import statsmodels.tsa as tsa
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR, AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



X = np.array(df_target[y_col]).reshape(-1,1)
# pd.Series(X.ravel()).isna().sum()

class AR():
    def __init__(self):
        pass
    
    def fit(self, X, p_order=1):
        """ 
        X: 1D numpy 배열, 시계열 데이터 
        p_order: int, AR 모델 차수 
        반환값: residual
        """ 
        X_arr = np.array(X).ravel()
        len_arr = len(X_arr) - p_order
        
        Xmat = np.roll(X, p_order-p_order)[:len_arr].reshape(-1,1)
        Ymat = np.roll(X, -p_order)[:len_arr].reshape(-1,1)
        
        for p in range(p_order-1):
            Xmat = np.hstack([np.roll(X_arr, -p-1)[:len_arr].reshape(-1,1), Xmat])
        
        self.beta = np.linalg.inv(Xmat.T @ Xmat) @ Xmat.T @ Ymat     # Linear 회귀계수
        self.pred = Xmat @ self.beta 
        self.res = Ymat - self.pred
        
        self.X_mat = Xmat
        self.Y_mat = Ymat
        
        return self.res

ar = AR()
ar.fit(X, 3)


plt.figure()
plt.plot(X)
plt.plot(ar.X_mat[:,-1])
plt.plot(ar.pred)
# plt.plot(ar.res)
plt.show()
