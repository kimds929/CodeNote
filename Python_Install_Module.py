# 【 python coding style 】-----------------------------------------------------------------------
# https://www.python.org/dev/peps/pep-0008/     # 공식문서
# https://wayhome25.github.io/python/2017/05/04/pep8/
# https://kongdols-room.tistory.com/18


# 【 Library 설치 업데이트 】---------------------------------------------------------------------
# pip install --upgrade scikit-learn

# 【 데이터 분석 기본 Library 】---------------------------------------------------------------------
# conda install numpy
# conda install pandas
# conda install matplotlib
# conda install seaborn
# conda install ipykernel
# conda install notebook
# conda install pylint


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy


# 통계
# conda install scipy
# conda install statsmodels



# 크롤링
# crawling -------------------------
# beautifulsoup
# selenium









# 데이터 탐색
#【 pandas-profiling 】 ------------------------------------------------------------------------
# conda uninstall -c conda-forge pandas-profiling

# import pandas_profiling
# from pandas_profiling import ProfileReport

# profile = df.profile_report()
# profile = ProfileReport(train,title="bio train data set") #profile 파일 만들기
# profile.to_file(output_file = "bio_train_profile.html")   #html 파일로 꺼내기


# 결측치 
#【 missingno 】 ----------------------------------------------------
# conda install -c conda-forge missingno
# import missingno as msno

# msno.bar()
# msno.matrix()
# msno.heatmap()
# msno.dendrogram()


# 【 DASK 】병렬처리 -------------------------------------------------
# https://datascienceschool.net/01%20python/04.09%20Dask%20%EC%82%AC%EC%9A%A9%EB%B2%95%20%EA%B8%B0%EC%B4%88.html
# https://mindscale.kr/course/python-ds/5/

# conda install -c conda-forge dask  



# 반복작업
# 【 itertools 】 Functions creating iterators for efficient looping-----------------------------------------------
# https://docs.python.org/3/library/itertools.html
# https://hamait.tistory.com/803

# from itertools import combinations, product


# 콘솔 클리어
# 【 IPython 】Interactive Kernel console 
# from IPython.display import clear_output
# clear_output(wait=True)     # display clear
# Jupyter Notebook
# ipykernel


# 【 time 】
# import time
# time.sleep(2)

# 【 progress_bar 】
# conda install -c conda-forge tqdm
# from tqdm.notebook import tqdm      # Progress_bar

# 【 클립보드 】
# conda install -c conda-forge pyperclip
# clipboard, win32clipboard, pyperclip
# from copy import deepcopy










# 머신러닝
# 【 scikit-learn 】--------------------------------------------------------------------------------
# conda install scikit-learn
# pip install sklearn
# import sklearn
    # from sklearn.preprocessing import *           # OneHotEncoder, OrdinalEncoder, StandardScaler
    # from sklearn.model_selection import *         # rain_test_split, KFold, GridSearchCV, RandomizedSearchCV
    # from sklearn.linear_model import *            # LinearRegression, Lasso, Ridge
    # from sklearn.cross_decomposition import *     # PLSRegression
    # from sklearn.neighbors import *               # KNeighborsClassifier
    # from sklearn.naive_bayes import *             # CategoricalNB, GaussianNB
    # from sklearn.discriminant_analysis import *   # LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    # from sklearn.mixture import *                 # GaussianMixture
    # from sklearn.svm import *                     # SVC
    # from sklearn.tree import *                    # DecisionTreeClassifier, plot_tree  #, export_graphviz
    # from sklearn.ensemble import *                # RandomForestClassifier, GradientBoostingClassifier
    # from sklearn.metrics import *                 # mean_squared_error
    # from sklearn.cluster import *                 # KMeans

    # from sklearn.compose import ColumnTransformer ???


# 【 sklearn-optimization 】
# https://scikit-optimize.github.io/stable/getting_started.html
# conda install -c conda-forge scikit-optimize
# pip install scikit-optimize

# https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
# from skopt import BayesSearchCV
# from skopt import gp_minimize
# from skopt import Optimizer

# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram


# 【 bayesian-optimization 】
# conda install -c conda-forge bayesian-optimization
# from bayes_opt import BayesianOptimization, UtilityFunction

# https://notebook.community/zzsza/TIL/python/bayesian-optimization



# conda install -c conda-forge eli5

# conda install -c anaconda scikit-image
# import skimage
# conda install -c conda-forge shap
# import shap

# conda install -c conda-forge imbalanced-learn
# from imblearn.over_sampling import SMOTE

# conda install -c conda-forge xgboost
# conda install -c conda-forge lightgbm
# conda install -c conda-forge catboost





# 딥러닝
# 【 tensorflow 】-------------------------------------------------------------------------------
# conda install tensorflow
# conda install -c anaconda tensorflow-gpu

# pip install --upgrade tensorflow
# pip install --upgrade tensorflow-gpu

# import tensorflow as tf

# pip install tensorflow_probability
# pip install tfp-nightly 
# conda install cloudpickle
# conda install -c conda-forge tensorflow-probability
# import tensorflow_probability as tfp

# conda install -c anaconda tensorflow-estimator
# import tensorflow_estimator

# pip install tensorflow_hub
# conda install -c conda-forge tensorflow-hub
# import tensorflow_hub as hub


# 【 pytorch 】
# https://pytorch.org/

# conda install -c pytorch pytorch
# conda install -c pytorch torchvision
# conda install -c conda-forge pycocotools
# pip install pycocotools 


# pip install pytorchtools
# from pytorchtools import EarlyStopping
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb


# pip install facenet_pytorch

# pip install pycocotools           # Computer Vision 



# 【 GPU-option 】
# Terminal GPU 사용 확인하기 ***
# nvidia-smi            # 1번만보기
# nvidia-smi -l 5       # 5초단위
# nvidia-smi -lms 5000  # 5초단위



# 【 Image Control 】
# import os
# from glob import glob     # 다중 path
# from PIL import Image     # 이미지 불러오기, 내보내기 등 제공

# conda install -c conda-forge opencv
# conda install opencv
# pip install opencv-python
# import cv2 as cv                # 이미지 resize 등 제공
# https://076923.github.io/posts/Python-opencv-1/           # OpenCV Blog

# pip install imageio
# import imageio

# pip install scikit-image==0.16.2
# pip install scipy==1.4.1
# import skimage





# [ graph ]
# conda install -c conda-forge node2vec
# import node2vec

# conda install networkx
# import networkx as nx




# 자연어처리
# pip install konlpy
# sudo apt-get update
# sudo apt install default-jdk        # Java Error가 날경우

# conda install -c conda-forge jpype1


# pip install gensim
# conda install -c pytorch torchtext
# conda install -c conda-forge mosestokenizer

# from torchtext.data import Field, BucketIterator, interleave_keys
# from torchtext.datasets import TranslationDataset
# from torchtext.data import Example
# from mosestokenizer import *



# 강화학습
# 【 gym 】-------------------------------------------------------------------------------
# conda install -c conda-forge gym
# pip install gym   # 강화학습 예제 Environment
# import gym

# pip install cmake
# pip install gym-super-mario-bros
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py


# conda install swig # needed to build Box2D in the pip install
# pip install box2d-py # a repackaged version of pybox2d
# pip install Box2D










# 최적화 ------------------
#【 cvxpy 】 ----------------------------------------------------
# import cvxpy
# conda install -c conda-forge cvxpy




# 아래에서 사용된 cPickle 은 Python 자료형으로 데이터를 저장하고 불러오는 패키지이다.
# from six.moves import cPickle
# input_image = cPickle.load(open('./image_files/lena.pkl', 'rb'))


# import pydotplus      # Decision Tree Graph








# Interactive -------------------------------------------------------------------
# pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
# from ipywidgets import interact

# https://junpyopark.github.io/interactive_jupyter/
# cf.go_offline(connected=True)

# def f(x):
#     return x*x

# def f2(x, y):
#     return x*x + y*y
    
# interact(f, x=(-30, 30, 1))
# interact(f2, x=(-30, 30, 1), y=(-30,30,1))



# df = pd.DataFrame(np.random.random([100,3]) * 10)
# df.columns = ['Feature1','Feature2','Feature3']
# df.head()

# @interact
# def show_data_more_than(column=['Feature2','Feature3'], 
#                         x=(0,10,1)):
#     return df.loc[df[column] > x]


# @interact
# def greeting(text="World"):
#     print("Hello {}".format(text))

