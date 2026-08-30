



############################################################################################
# Part 2
############################################################################################
# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")




# < train (3500, 10) >
#           dtypes  nunique  isna   min        max
# 회원ID       int64     3500     0     0       3499

# 최대구매액      int64      699     0   -30       7066
# 환불금액     float64      469  2300   1.0     5638.0
# 방문당구매건수  float64     1107     0   1.0  22.083333
# 주말방문비율   float64     1142     0   0.0        1.0

# 방문일수       int64      147     0     1        285
# 구매주기       int64      135     0     0        166

# 주구매상품     object       42     0  가공식품        화장품
# 주구매지점     object       24     0   강남점        포항점

# 총구매액       int64     1537     0  -524      23232


# < test (2482, 9) >
#           dtypes  nunique  isna   min     max
# 회원ID       int64     2482     0  3500    5981
# 최대구매액      int64      654     0  -374    5932
# 환불금액     float64      389  1615   1.0  8715.0
# 주구매상품     object       41     0  가공식품     화장품
# 주구매지점     object       24     0   강남점     포항점
# 방문일수       int64      134     0     1     222
# 방문당구매건수  float64      900     0   1.0  15.875
# 주말방문비율   float64      940     0   0.0     1.0
# 구매주기       int64      136     0     0     177




cols_drop = ['회원ID']
col_y = '총구매액'
cols_X_num = ['최대구매액','환불금액', '방문당구매건수','주말방문비율', '방문일수','구매주기']
cols_X_cat = ['주구매상품', '주구매지점']

cols_X = cols_X_num + cols_X_cat

# for c in cols_X_cat:
# 	print(c)
# 	df_cat_categories = pd.concat([train[c].value_counts().sort_index(), test[c].value_counts().sort_index()], axis=1)
# 	print(df_cat_categories)
# 	print()

print('#'*100)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder

# train['환불금액'] =train['환불금액'].fillna(0)
# test['환불금액'] =test['환불금액'].fillna(0)
def preprocess_num(X):
	X_copy = X.copy()
	X_copy['환불금액'] = X_copy['환불금액'].fillna(0)
	return X_copy.astype(float)
	
pipe_num = Pipeline([
	('preprocess_num', FunctionTransformer(preprocess_num))
])


def preprocess_cat(X):
	return X.astype(str)

pipe_cat = Pipeline([
	('preprocess_cat', FunctionTransformer(preprocess_cat)),
	('encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])


from sklearn.compose import ColumnTransformer
pipe_X = ColumnTransformer([
	('pipe_num', pipe_num, cols_X_num),
	('pipe_cat', pipe_cat, cols_X_cat)
])

# print( pipe_X.fit_transform(train[cols_X]).shape )
# print( pipe_X.fit_transform(train[cols_X]) )


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {}
models['RF'] = RandomForestRegressor()
models['GB'] = GradientBoostingRegressor()
models['XGB'] = XGBRegressor()
models['LGBM'] = LGBMRegressor(verbosity=-1)


params = {}
params['RF'] = {'model__random_state': [0],
								'model__max_depth': [3,5],
								'model__min_samples_leaf': [1,3],
							 }
params['GB'] = {'model__random_state': [0],
								'model__learning_rate': [0.03,  0.1],
								'model__n_estimators': [100, 300],
							 }
params['XGB'] = {'model__random_state': [0],
								 'model__learning_rate': [0.03, 0.1],
								'model__n_estimators': [100,300],
							 }
params['LGBM'] = {'model__random_state': [0],
									'model__num_leaves': [31, 63],
									'model__learning_rate': [0.03, 0.1],
							 }


from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.metrics import get_scorer_names
# print(get_scorer_names())
cv = KFold(n_splits=3, shuffle=True, random_state=0)

X = train[cols_X]
y = train[col_y].astype(float)

r_be = {}
r_bs = {}
r_bp = {}
r_cv = {}

# model_name = 'LGBM'
for model_name, model in models.items():
	print(model_name, end=' : ')
	model_pipe = Pipeline([
		('pipe_X', pipe_X),
		('model', models[model_name])
	])

	grid_search_cv = GridSearchCV(estimator=model_pipe, cv=cv,
															 param_grid=params[model_name],
															 scoring='neg_root_mean_squared_error',
															 return_train_score=True,
															 verbose=1)
	grid_search_cv.fit(X,y)

	print(grid_search_cv.best_score_)
	r_be[model_name] = grid_search_cv.best_estimator_
	r_bs[model_name] = grid_search_cv.best_score_
	r_bp[model_name] = grid_search_cv.best_params_
	r_cv[model_name] = grid_search_cv.cv_results_
	# break
	

best_estimator_idx = np.argmax(np.array(list(r_bs.values())))
best_estimaor_name = list(r_bs.keys())[best_estimator_idx]

print(best_estimaor_name)
print(-r_bs[best_estimaor_name])
pred = r_be[best_estimaor_name].predict(test[cols_X])

pd.DataFrame({'pred':pred}, index=test.index).to_csv('result.csv',index=False)































############################################################################################
# Part 3
############################################################################################


# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
import numpy as np
import pandas as pd

df = pd.read_csv("data/bcc.csv")

# 사용자 코딩
print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                        .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
print(df_summary)
# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출
# < shape (116, 2) >
#                  dtypes  nunique  isna   min   max
# Resistin        float64      116     0  3.21  82.1
# Classification    int64        2     0  1.00   2.0
print('-'*100)

df['log_Resistin'] = np.log(df['Resistin'])
x1 = df.query("Classification==1")['log_Resistin']
x2 = df.query("Classification==2")['log_Resistin']

v1, v2 = x1.var(), x2.var()
n1, n2 = x1.shape[0], x2.shape[0]

result = None
if n1 > n2:
	result = v1/v2
else:
	result = v2/v1


# 1.
print(1)
print(f"Ans : {result:.3f}")
print('-'*100)

# 2. 


result = ((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2)
print(2)
print(f"Ans : {result:.3f}")
print('-'*100)

# 3
from scipy.stats import ttest_ind, levene
print(levene(x1, x2))		# pvalue=0.180 >0.05

result = ttest_ind(x1, x2, equal_var=True)

print(3)
print(f"Ans : {result.pvalue:.3f}")  # p-value=0.003 <0.05 : 귀무가설 기각
print('-'*100)


















