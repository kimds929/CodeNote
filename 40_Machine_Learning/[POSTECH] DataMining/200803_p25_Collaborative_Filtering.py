#!/usr/bin/env python
# coding: utf-8

# import libraries
import numpy as np
import pandas as pd

from surprise import KNNBasic, KNNWithMeans
from surprise import SVD, NMF
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split


import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# create example data for collaborative filtering
UserItemMatrix = np.array([np.array([5, np.nan, 4, np.nan, 1, np.nan, 3]),
                           np.array([4, 4, 4, np.nan, np.nan, np.nan, 1]),
                           np.array([5, 4, np.nan, 1, 2, np.nan, 3]),
                           np.array([1, 2, 1, 4, 3, 5, 2]),
                           np.array([np.nan, 1, np.nan, 3, 5, 5, np.nan]),
                           np.array([np.nan, 2, np.nan, np.nan, 4, 4, 2]),
                           np.array([5, np.nan, np.nan, 1, np.nan, np.nan, 2])
                          ])

UserItemMatrix


# explore User-Item matrix
df = pd.DataFrame(UserItemMatrix, 
                  columns=['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7'])
df['user_id'] = list(df.index)
df



# transform data into appropriate form for library
 
df_melt = df.melt(id_vars='user_id', value_name='rating')        # 세로로 늘려주기
df_melt.dropna(inplace=True)
df_melt.variable = df_melt.variable.str.replace('item', '')
df_melt


# df.melt(value_vars=['item1','item2'],id_vars=['user_id'])




# transform data into appropriate form for library (cont'd)
ratings_dict = {'itemID': df_melt['variable'].astype(np.int),
               'userID': df_melt['user_id'].astype(np.int),
               'rating': df_melt['rating'].astype(np.float)}
df_melt_df = pd.DataFrame(ratings_dict)
df_melt_df
reader = Reader(rating_scale=(1, 5))

    # user, item, rating 세개의 column만 허용, 위치 변경하면 안됨
data = Dataset.load_from_df(df_melt_df[['userID', 'itemID', 'rating']], reader)


# build trainset
train_set = data.build_full_trainset()      # 전체데이터 = 학습데이터
# train_set


# learn KNN-based algorithm model (user-based similarities)
sim_options = {'name': 'pearson'}
# other similarity option: 'msd', 'cosine'

algo = KNNBasic(sim_options=sim_options)
algo.fit(train_set)



# predict unknown ratings from learned model
    # 개개값 예측하기
target_u = [6]
target_i = [2, 3, 5, 6]

predictions = []
for user in target_u:
    for item in target_i:
        predictions.append(algo.predict(user, item))
predictions


    # build testset      # 값이 있었던 object들
test_set = train_set.build_testset()
test_set

    # build testset      # 값이 없었던 object들 : global 평균으로 대체
anti_test_set = train_set.build_anti_testset()
anti_test_set

    # 한꺼번에 예측하기
algo.test(test_set)
algo.test(anti_test_set)


    # 다른 방법론을 적용하여 실습하기
algo2 = KNNWithMeans(sim_options={'name': 'pearson'})
algo2.fit(train_set)
algo2.test(anti_test_set)

algo3 = SVD(n_factors=3)
algo3.fit(train_set)
algo3.test(anti_test_set)

algo4 = NMF(n_factors=100)
algo4.fit(train_set)
algo4.test(anti_test_set)


# # Practice ---------------------------------------------------------
# ### Open 'movielens-1m.dat'. Refer to the file 'read_dataset2(ref)' to open this file.
# ### Practice data processing in this file and try 4 different algorithms.

df2 = pd.read_csv(absolute_path + 'movielens-1m.dat', sep='::',
                 names=['user', 'item', 'rating', 'timestamp'], engine='python')
df2.head()
df2.describe()

reader_movie = Reader(rating_scale=(1, 5))
data_movie = Dataset.load_from_df(df2[['user', 'item', 'rating']], reader_movie)

train_movie = data_movie.build_full_trainset()


sim_options = {'name': 'pearson'}
# other similarity option: 'msd', 'cosine'

algo_movie = KNNBasic(sim_options=sim_options)
algo_movie.fit(train_movie)


test_movie = train_movie.build_testset()
anti_test_movie = train_movie.build_anti_testset()

    # 한꺼번에 예측하기
algo_movie.test(test_movie)
algo_movie.test(anti_test_movie)
























#!/usr/bin/env python
# coding: utf-8

# import libraries
import numpy as np
import pandas as pd

from surprise import KNNBasic, KNNWithMeans
from surprise import SVD, NMF
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split, KFold


# create example data for collaborative filtering
# UserItemMatrix = np.array([np.array([5, np.nan, 4, np.nan, 1, np.nan, 3]),
#                            np.array([4, 4, 4, np.nan, np.nan, np.nan, 1]),
#                            np.array([5, 4, np.nan, 1, 2, np.nan, 3]),
#                            np.array([1, 2, 1, 4, 3, 5, 2]),
#                            np.array([np.nan, 1, np.nan, 3, 5, 5, np.nan]),
#                            np.array([np.nan, 2, np.nan, np.nan, 4, 4, 2]),
#                            np.array([5, np.nan, np.nan, 1, np.nan, np.nan, 2])
#                           ])

# UserItemMatrix


# # explore User-Item matrix
# df = pd.DataFrame(UserItemMatrix, 
#                   columns=['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7'])
# df['user_id'] = list(df.index)
# df


# # transform data into appropriate form for library
# df = df.melt(id_vars='user_id', value_name='rating')
# df.dropna(inplace=True)
# df.variable = df.variable.str.replace('item', '')
# df


# # transform data into appropriate form for library (cont'd)
# ratings_dict = {'itemID': df['variable'].astype(np.int),
#                'userID': df['user_id'].astype(np.int),
#                'rating': df['rating'].astype(np.float)}
# df = pd.DataFrame(ratings_dict)

# reader = Reader(rating_scale=(1, 5))

# data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


# # learn KNN-based algorithm model (user-based similarities)
# sim_options = {'name': 'pearson'}
# # other similarity option: 'msd', 'cosine'

# algo = KNNBasic(sim_options=sim_options)


# KFold evaluation
n_splits=5
cv = KFold(n_splits)        # surpriese KFold
acc = np.zeros(shape=(n_splits,))
fcp = np.zeros(shape=(n_splits,))

for i, (trainset, testset) in enumerate(cv.split(data)):
    algo.fit(train_set)
    
    predictions = algo.test(testset)
    
    acc[i] = accuracy.rmse(predictions)
    fcp[i] = accuracy.fcp(predictions)

print('Average performance')
print(np.mean(acc))
print(np.mean(fcp))     # 경향성이 얼마나 되는지를 맞추는 지표 (경향성만을 기준으로 판단)




# cross_validate 
from surprise.model_selection import cross_validate
result = cross_validate(algo, data)

# ?cross_validate
    # 다른 방법론을 적용하여 실습하기
algo2 = KNNWithMeans(sim_options={'name': 'pearson'})
algo3 = SVD(n_factors=3)
algo4 = NMF(n_factors=100)

result2 = cross_validate(algo2, data)
result3 = cross_validate(algo3, data)
result4 = cross_validate(algo4, data)

result2
result3
result4



# # Practice ------------------------------------------------------------
# ### Read Joke dataset (pd.read_csv('UserRatings1.csv'))
# ### Compare KNNWithMeans with pearson, SVD with 50 factors
# ##### metric = 'rmse'

joke_df = pd.read_csv(absolute_path + 'UserRatings1.csv')
joke_df.head(3)

joke_melt = joke_df.melt(id_vars='JokeId', var_name='item_id', value_name='rating')
joke_melt.head(10)

joke_melt['item_id'] = joke_melt['item_id'].str.replace('User','')



