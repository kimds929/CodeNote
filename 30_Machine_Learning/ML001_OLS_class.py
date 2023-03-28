import numpy as np
import pandas as pd





from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
    # StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
    # RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
    # MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
    # MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
# scaler = eval('StandardScaler' + '()')
# scaler_result = scaler.fit(a6)
# scaler_result.transform(a6)
# scaler_result = scaler.fit_transform(a6)
# scaler.inverse_transform(scaler_result)

def fun_convert_ndarray(x):
    if type(x) == pd.DataFrame:
        return x.to_numpy()
    elif type(x) == pd.Series:
        return x.to_frame().to_numpy()
    elif type(x) == np.ndarray:
        return x



class DS_OLS:
    def __init__(self, y, x, const=True, method='OLS', tun_params=False, scaler=False):
        # method : OLS, ridge, lasso, elastic
        # tun_params : 
        # scaler : StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

        self.y_type = type(y)
        self.x_type = type(x)

        if self.y_type == pd.DataFrame:
            self.y_columns = y.columns
        elif self.y_type == pd.Series:
            self.y_columns = [y.name]

        if self.x_type == pd.DataFrame:
            self.x_columns = x.columns
        elif self.x_type == pd.Series:
            self.x_columns = [x.name]

        self.y = fun_convert_ndarray(y)
        self.x = fun_convert_ndarray(x)

        self.scaler = scaler
        if scaler:
            self.scaler_class = eval(self.scaler + '()')
            self.y = self.scaler_class.fit_transform(self.y)
            self.x = self.scaler_class.fit_transform(self.x)

        self.intercept = const
        if self.intercept:
            self.x = self.add_constant(self.x)
            self.x_columns = ['const'] + self.x_columns

    def fit(self):
        # b = np.linalg.inv(x.T @ x) @ x.T @ y
        self.c_matrix = np.linalg.inv(self.x.T @ self.x)
        self.c_diag = np.diag(self.c_matrix, k=0)
        self.params = self.c_matrix @ self.x.T @ self.y
        self.y_pred = self.predict(X=self.x)
        # self.y_pred = (self.x * self.params.reshape((1,-1))).sum(1).reshape((-1,1))

        evaluate_result = fun_ols_evalueate_model(y_true=self.y, y_pred=self.y_pred, X=self.x, const=self.intercept, resid=True)
        self.df_total = evaluate_result['df_total']
        self.df_resid = evaluate_result['df_resid']
        self.df_model = evaluate_result['df_model']

        self.resid = evaluate_result['resid']

        self.sse = evaluate_result['sse']
        self.ssr = evaluate_result['ssr']
        self.sst = evaluate_result['sst']

        self.mse = evaluate_result['mse']
        self.msr = evaluate_result['msr']
        self.mst = evaluate_result['mst']
        self.rmse = np.sqrt(evaluate_result['mse'])

        self.fvalue = evaluate_result['fvalue']
        self.rsquared = evaluate_result['rsquared']
        self.rsquared_adj = evaluate_result['rsquared_adj']

        self.varience_params = (self.mse * self.c_diag).reshape((-1,1))
        self.std_params = np.sqrt(self.varience_params)
        self.tvalues = self.params / self.std_params

        if self.x_type == pd.DataFrame or self.x_type == pd.Series:
            self.resid = pd.Series(self.resid.ravel())
            self.varience_params = pd.Series(self.varience_params.ravel())
            self.std_params = pd.Series(self.std_params.ravel())
            self.tvalues = pd.Series(self.tvalues.ravel())
        
        return self
    
    def predict(self, X):
        y_pred = (fun_convert_ndarray(X) * self.params.reshape((1,-1))).sum(1).reshape((-1,1))
        if self.x_type == pd.DataFrame or self.x_type == pd.Series:
            y_pred = pd.Series(y_pred.ravel())
        return y_pred

    # def __call__(self):
    #     if scaler:
            
    def add_constant(self, ndarray):
        return np.concatenate((np.ones(len(ndarray)).reshape((-1,1)), ndarray), axis=1)


a = DS_OLS(y=df_y, x=df_x, scaler='StandardScaler')
a = DS_OLS(y=df_y, x=df_x, scaler=False).fit()



df_test
a1 = df_test[['y']]         # (DataFrame) single-column
a2 = df_test['y']           # (Series)
a3 = a1.to_numpy()          # (ndarray) column vector
a4 = a2.to_numpy()          # (ndarray) row vector
a5 = df_test[['x1','x2']]   # (DataFrame) DataFrame multi-column
a6 = a5.to_numpy()          # (ndarray) column vector




