##############################################################################################################################
## 【 Bayesian Optimization 】 ###############################################################################################
##############################################################################################################################
# (git) bayes_opt : https://github.com/fmfn/BayesianOptimization 
# (git_advance) bayes_opt : https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
# (install) conda install -c conda-forge bayesian-optimization



##### Hyper-Parameter Tuning ###############################################################
# #  bayesian-optimization 라이브러리의 BayesianOptimization 클래스 import
# from bayes_opt import BayesianOptimization
# import numpy as np

# # 실험해보고자하는 hyperparameter 집합
# pbounds = {'max_depth': (3, 7),
#             'learning_rate': (0.01, 0.2),
#             'n_estimators': (5000, 10000),
#             'gamma': (0, 100),
#             'min_child_weight': (0, 3),
#             'subsample': (0.5, 1),
#             'colsample_bytree' :(0.2, 1)
#             }

# # Bayesian optimization 객체 생성
# # f : 탐색 대상 함수, pbounds : hyperparameter 집합
# # verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함
# # random_state : Bayesian Optimization 상의 랜덤성이 존재하는 부분을 통제 
# bo=BayesianOptimization(f=XGB_cv, pbounds=pbounds, verbose=2, random_state=1 )    

# # 메소드를 이용해 최대화 과정 수행
# # init_points :  초기 Random Search 갯수
# # n_iter : 반복 횟수 (몇개의 입력값-함숫값 점들을 확인할지! 많을 수록 정확한 값을 얻을 수 있다.)
# # acq : Acquisition Function들 중 Expected Improvement(EI) 를 사용
# # xi : exploration 강도 (기본값은 0.0)
# bo.maximize(init_points=2, n_iter=10, acq='ei', xi=0.01)

# # ‘iter’는 반복 회차, ‘target’은 목적 함수의 값, 나머지는 입력값을 나타냅니다. 
# # 현재 회차 이전까지 조사된 함숫값들과 비교하여, 현재 회차에 최댓값이 얻어진 경우, 
# # bayesian-optimization 라이브러리는 이를 자동으로 다른 색 글자로 표시하는 것을 확인할 수 있습니다


from bayes_opt import BayesianOptimization, UtilityFunction

# black box function
def object_function(x):
    if type(x) == list:
        x = x[0]
    return -(x-2)*(x-17)*(x+5)*(x+15)+3

# real data
x = np.linspace(-20,20, 100)
plt.plot(x, object_function(x))
plt.show()


# finding points (constraints)
pbounds = {'x':(-20,20)}     

# init ***
bo = BayesianOptimization(f=object_function, pbounds=pbounds, verbose=2, random_state=1)
#   f : object_function to optimize
#   pbounds : params range to optimize
#   random_state : fixed random_state
#   verbose : display option (0: noting display, 1: show simple, 2: show detail)

# Applicable methods/attribute in class ***
# (method)  maximize(), 
# (attribute) max, res
# 'dispatch', 'get_subscribers',
# 'probe', 'register', 'set_bounds',
# 'set_gp_params', 'space', 'subscribe', 'suggest', 'unsubscribe'

# maximize  ***
bo.maximize()
#   init_points=5,      # How many steps of random exploration you want to perform.
#   n_iter=25,          # How many steps of bayesian optimization you want to perform.
#   acq='ucb',
#   kappa=2.576,
#   xi=0.0,
#   **gp_params,


# maximum points  ***
bo.max      # maximum point / conditions of params
# {'target': 22956.597421882052, 'params': {'x': 11.928020087844509}}       # max points: {value, parameters}






from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import acq_max

def func(x):
    y =(x+13)*(x+8)*(x-3)*(x-10)
    return -y 

# def func(x):
#     def logic(xp):
#         if xp < -10:
#             return 5
#         elif xp < -5:
#             return 3
#         elif xp < 0:
#             return 10
#         elif xp < 5:
#             return 6
#         elif xp < 10:
#             return 2
#         elif xp < 15:
#             return 8
#         else:
#             return 1
#     return np.vectorize(logic)(x)

x = np.linspace(-15,15,50)
plt.plot(x, func(x))

bo = BayesianOptimization(f=None, pbounds={'x':(-15,15)}, random_state=1, verbose=2)
bo.maximize()


bo0 = BayesianOptimization(f=func, pbounds={'x':(-15,15)}, random_state=1, verbose=2)
bo0._space.random_sample()
bo0._prime_queue(2)
bo0._queue._queue
bo0._space._target
bo0._space._params
bo0._space._keys
bo0._space.target_func

bo0.maximize(init_points=0, n_iter=0)
bo0.maximize(init_points=0, n_iter=10)
bo0.probe(params=6.804, lazy=True)
bo0.probe(params=6.804, lazy=False)   # register
bo0.max


# acq_max(ac=bayes_utils.utility, gp=bo0._gp, y_max=3564.3153521, bounds=bo0._space.bounds, random_state=bo0._random_state)


# Bayesian Optimization Advance
bo0= BayesianOptimization(f=func, pbounds={'x':(-15,15)}, random_state=1, verbose=2)
bayes_utils = UtilityFunction(kind='ucb', kappa=2.576, xi=0.0)
print(bo0._space.params, bo0._space.target)

for i in range(5):
    print(f'-- iter: {i+1} ----------------------------')
    next_points = bo0.suggest(bayes_utils)
    next_target = func(**next_points)
    print(' . next point')
    print(next_points, next_target)
    print(bo0._space.params.ravel(), bo0._space.target)
    print(' . register_result')
    bo0.register(params=next_points, target=next_target)
    print(bo0._space.params.ravel(), bo0._space.target)
    print('\n')


##############################################################################################################################
##############################################################################################################################



# https://machinelearningmastery.com/what-is-bayesian-optimization/
# https://sonsnotation.blogspot.com/2020/11/11-2-gaussian-progress-regression.html

# example of the test problem
import math
import numpy as np
from matplotlib import pyplot as plt
 
# objective function
def objective(x, noise=0.1):
	noise = np.random.normal(loc=0, scale=noise)
	return (x**2 * math.sin(5 * math.pi * x)**6.0) + noise
 
# grid-based sample of the domain [0,1]
X = np.arange(0, 1, 0.01)
# sample the domain without noise
y = [objective(x, 0) for x in X]


# sample the domain with noise
ynoise = [objective(x) for x in X]
# find best result
ix = np.argmax(y)
print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))


plt.scatter(X, ynoise)  # plot the points with noise
plt.plot(X, y)  # plot the points without noise
plt.show()  # show the plot


from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)
 
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	plt.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = np.asarray(np.arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	plt.plot(Xsamples, ysamples)
	# show the plot
	plt.show()
 
# sample the domain sparsely with noise
X = np.random.random(100)
y = np.asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot the surrogate function
plot(X, y, model)












#################################################################################################
# (git) scikit-optimize : https://github.com/scikit-optimize/scikit-optimize
# (install) conda install -c conda-forge scikit-optimize
# scikit-opt do not perform gradient-based optimization. For gradient-based optimization algorithms look at 'scipy.optimize'
from skopt import gp_minimize, Optimizer

opt = Optimizer([(-2.0, 2.0)])

for i in range(20):
    suggested = opt.ask()
    y = f(suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)
gp_minimize(object_function, [(-20, 20)], random_state=1, verbose=2, n_calls=10)
res = gp_minimize(object_function, [(-20, 20)], random_state=1, verbose=0, n_calls=20)
# ?gp_minimize
res.keys()
res.models
res.fun
res.func_vals
res.specs

opt = Optimizer([(-20, 20)])

for i in range(20):
    suggested = opt.ask()
    y = -object_function(suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)

#################################################################################################










##############################################################################################################################
#### 【Customizing Function】 ##################################################################################################
##############################################################################################################################





# HyperParameter Tunning
from bayes_opt import BayesianOptimization, UtilityFunction

# (git) bayes_opt : https://github.com/fmfn/BayesianOptimization 
# (git_advance) bayes_opt : https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
# (install) conda install -c conda-forge bayesian-optimization


# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
class BayesOpt:
    """
     【required (Library)】 bayes_opt.BayesianOptimization, bayes_opt.UtilityFunction
     【required (Custom Module)】 EarlyStopping
    """
    def __init__(self, f, pbounds, random_state=None, verbose=2):
        self.verbose = verbose
        self.f = f
        self.pbounds = pbounds
        self.random_state = random_state
        self.random_generate = np.random.RandomState(self.random_state)
        
        self.bayes_opt = BayesianOptimization(f=f, pbounds=pbounds, random_state=random_state, verbose=verbose)
        self._space = self.bayes_opt._space
        
        self.res = []
        self.max = {'target':-np.inf, 'params':{}}
        self.repr_max = {}
        
        self.last_state = ''
    
    def decimal(self, x, rev=0):
        return 2 if x == 0 else int(-1*(np.floor(np.log10(abs(x)))-3-rev))
    
    def auto_decimal(self, x, rev=0):
        if np.isnan(x):
            return np.nan
        else:
            decimals = self.decimal(x, rev=rev)
            if decimals < 0:
                return x
            else:
                return round(x, decimals)

    def print_result(self):
        epoch = len(self.bayes_opt._space.target)
        last_target = self.auto_decimal(self.bayes_opt._space.target[-1])
        last_params = {k: self.auto_decimal(v) for k, v in zip(self.bayes_opt._space.keys, self.bayes_opt._space.params[-1])}
        last_state = '**Maximum' if epoch == np.argmax(self.bayes_opt._space.target) + 1 else self.last_state
        
        if self.verbose > 0:
            if self.verbose > 1 or last_state == '**Maximum':
                print(f"{epoch} epoch) target: {last_target}, params: {str(last_params)[:50]} {last_state}")
        self.last_state = ''
    
    def maximize(self, init_points=5, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, patience=None, **gp_params):
        
        if patience is not None:
            bayes_utils = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
            n = 1
            
            # init_points bayesian
            for i in range(init_points):
                self.bayes_opt.probe(self.bayes_opt._space.random_sample(), lazy=False)
                self.print_result()
                n += 1
            
            # EarlyStop
            early_stop_instance = EarlyStopping(patience=patience, optimize='maximize')
            early_stop_instance.early_stop(score=self.bayes_opt.max['target'], save=self.bayes_opt.max['params'])
            
            last_state = 'break' if patience == 0 else None
            while last_state != 'break' or n < n_iter:
                # Bayesian Step
                next_points = self.bayes_opt.suggest(bayes_utils)
                next_target = func(**next_points)
                self.bayes_opt.register(params=next_points, target=next_target)
                
                if n >= n_iter:
                    last_state = early_stop_instance.early_stop(score=next_target, save=next_points)
                    self.last_state = '' if last_state == 'None' else last_state

                self.print_result()
                n += 1
            
        else:
            self.bayes_opt.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi, **gp_params)
        
        # result            
        target_auto_format = self.auto_decimal(self.bayes_opt.max['target'])
        parmas_auto_format = {k: self.auto_decimal(v) for k, v in self.bayes_opt.max['params'].items()}
        self.repr_max = {'target':target_auto_format, 'params': parmas_auto_format}

        self.res = self.bayes_opt.res
        self.max = self.bayes_opt.max

    def __repr__(self):
        if len(self.repr_max) > 0:
            return f"(bayes_opt) BayesianOptimization: {self.repr_max}"
        else:
            return f"(bayes_opt) BayesianOptimization: undefined"

