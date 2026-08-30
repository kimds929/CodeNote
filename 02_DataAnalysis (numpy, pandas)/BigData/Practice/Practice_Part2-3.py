# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
import numpy as np
import pandas as pd

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

# Set Params ###################################################################################################

rs = 0
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# EDA ###################################################################################################
# 사용자 코딩
for df_name, df in zip(['train', 'test'], [train, test]):
	print(f"< {df_name} : {df.shape} >")
	df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0), df.agg(['min','max']).T], axis=1)
	df_summary.columns = ['dtype', 'nunique', 'isna', 'min', 'max']
	print(df_summary)
	print()


cols_drop = ['회원ID']
col_y = '총구매액'
cols_X_num = [c for c in train.select_dtypes(include=['int64', 'float64']).columns if c not in cols_drop + [col_y]]
cols_X_cat = [c for c in train.select_dtypes(include=['object']).columns if c not in cols_drop + [col_y]]
cols_X = cols_X_num + cols_X_cat

print(f"cols_X_num : {cols_X_num}")
print(f"cols_X_cat : {cols_X_cat}")

print('#' * 100, end='\n')
# dtype ###################################################################################################
# # Numerical
# qs = np.arange(0, 1.1, 0.1)
# print(train[cols_X_num].describe(percentiles=qs).map(lambda x: np.round(x, 2)))

# # Categorical
# for c in cols_X_cat:
# 	print(f"< {c} >")
# 	df_concat = pd.DataFrame()
# 	for df_name, df in zip(['train', 'test'], [train, test]):
# 		df_temp = df[c].value_counts().sort_index()
# 		df_temp.name = df_name
# 		df_concat = pd.concat([df_concat, df_temp], axis=1)
# 	print(df_concat.T)
# 	print()

# print('#' * 100, end='\n')
# Preprocessing ################################################################################################


from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin

# pipe num
def fillna_num(X):
	X_copy = X.copy()
	X_copy['환불금액'] = X_copy['환불금액'].fillna(0)
	# print(X_copy.isna().sum())
	return X_copy

pipe_num = Pipeline([
	('fillna', FunctionTransformer(fillna_num).set_output(transform='pandas')),
	('encoder', StandardScaler())
])

# print(pipe_num.fit_transform(train[cols_X_num]))

# pipe cat
# df_concat = pd.concat([train[cols_X_cat], test[cols_X_cat]], axis=0)
# oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# oh_encoder.fit(df_concat)

pipe_cat = Pipeline([
	('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
	# ('encoder', FunctionTransformer(oh_encoder.transform))
])


# pipe num + cat
preprocess = ColumnTransformer([
	('pipe_num', pipe_num, cols_X_num),
	('pipe_cat', pipe_cat, cols_X_cat)
])
print('#' * 100, end='\n')


# Preprocessing ################################################################################################
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {'RF': RandomForestRegressor(random_state=rs),
				 'GB': GradientBoostingRegressor(random_state=rs),
				 'XGB': XGBRegressor(random_state=rs),
				 'LGBM': LGBMRegressor(random_state=rs, verbosity=-1)}


# ----------------------------------------------------------------------
# for mdl_name, mdl in models.items():
# 	print(mdl_name)
# 	print(mdl.get_params().keys())
# 	print('-'*50)

params = {}
params['RF'] = {
							# 'model__n_estimators': [100, 300],
							'model__max_depth': [None, 10],
							'model__max_features': ['sqrt', 1.0],
							# 'model__min_sample_leaf':[1, 3],
							 }
# params['RF'] = {'model__n_estimators': [100, 300],
# 							'model__max_depth': [None, 5, 10],
# 							'model__max_features': ['sqrt', 1.0],
# 							'model__min_sample_leaf':[1, 3],
# 							 }
# params['GB'] = {'model__n_estimators': [100, 300],
# 							'model__learning_rate': [0.05, 0.1],
# 							'model__max_depth': [3, 5],
# 							'model__subsample':[0.8, 1.0],
# 							 }
# params['GB'] = {'model__n_estimators': [100, 300],
# 							'model__learning_rate': [0.05, 0.1],
# 							'model__max_depth': [3, 5],
# 							'model__subsample':[0.8, 1.0],
# 							 }
# params['LGBM'] = {'model__n_estimators': [100, 300],
# 							'model__max_depth': [3, 5],
# 							'model__learning_rate': [0.05, 0.1],
# 							'model__num_leaves':[31, 60],
# 							 }

# ----------------------------------------------------------------------

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import get_scorer_names
print(get_scorer_names())

cv = KFold(n_splits=4, shuffle=True, random_state=rs)

# ----------------------------------------------------------------------
# X = train[cols_X]
# y = train[col_y]
X = train[cols_X]
y = train[col_y]


model_name = 'RF'
model = models[model_name]
param_grid = params[model_name]

model_pipe = Pipeline([
	('preprocess', preprocess),
	('model', model)
])

search_cv = GridSearchCV(estimator=model_pipe, cv=cv, param_grid=param_grid,
												scoring='neg_root_mean_squared_error', return_train_score=True, verbose=0)
search_cv.fit(X,y)

# print('#' * 100, end='\n')
# # print(help(GridSearchCV))
# print(dir(search_cv))
# print('#' * 100, end='\n')

best_estimator = search_cv.best_estimator_
best_score = search_cv.best_score_
print(f"best_score : {best_score}")
# print(pd.DataFrame(search_cv.cv_results_).T)

# print('-'* 100, end='\n')
# df_pred_test = pd.DataFrame({'pred':best_estimator.predict(test[cols_X])})
# print(df_pred_test)
# print('-'* 100, end='\n')



# print('#' * 100, end='\n')
# print(help(cross_val_score))
# print('#' * 100, end='\n')
# import sklearn.model_selection
# print(dir(sklearn.model_selection))
# print(help(FunctionTransformer(fillna_num).set_output))
# print(help(cross_val_score))
# print(np.array(dir(sklearn.base)))
# from sklearn.pipeline import PipeLine

	
	
# 답안 제출 참고
# 아래 코드는 예시이며 변수명 등 개인별로 변경하여 활용
# pd.DataFrame변수.to_csv("result.csv", index=False)
