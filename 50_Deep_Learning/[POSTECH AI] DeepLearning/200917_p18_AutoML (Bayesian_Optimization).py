import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import tensorflow as tf
# import tensorflow_probability as tfp

# conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization, UtilityFunction

# from IPython.display import clear_output
# import time



# Baysian Optimization : Exploitation / Exploration
def target(x):      # real_function
    return np.exp(-(x-2)**2) + np.exp(-(x-6)**2/10) + 1 / (x**2+1)

x_lp = np.linspace(-2, 10, 10000).reshape(-1,1)
y_lp = target(x_lp)

plt.plot(x_lp, y_lp)
plt.show()

# x_lp[np.argmax(y_lp)]


bayes_opt = BayesianOptimization(f=target, pbounds={'x': (-2,10)}, random_state=27)
# ?BayesianOptimization
# BayesianOptimization(f, pbounds, random_state=None, verbose=2)

bayes_opt.maximize(init_points=2, n_iter=12, acq='ucb', kappa=5)
# acq: acquisition_function('ucb...')이 내부적으로 돌아가서 최대가 되는 지점을 찾아줌
# kappa: Exploration 정도 지정      # ucb = mean + kappa * std
    # kappa ↑ → Exploration에 집중하겠다
# ?BayesianOptimization.maximize
# bayes_opt.maximize(
#     init_points=5,
#     n_iter=25,
#     acq='ucb',
#     kappa=2.576,
#     xi=0.0,
#     **gp_params,
# )

# dir(bayes_opt)
bayes_opt.max           # max_point (best_case)
bayes_opt._gp           # setting parameter
bayes_opt.res           # history
len(bayes_opt.space)    # iter횟수


x_res = np.array([res['params']['x'] for res in bayes_opt.res])
y_res = np.array([res['target'] for res in bayes_opt.res])

plt.plot(x_lp, y_lp, label='target')
plt.plot(x_res, y_res, 'r--', marker='D', alpha=0.5, label='Observations')
plt.show()


fitted = bayes_opt._gp.fit(x_res[...,np.newaxis], y_res)
fitted
bayes_opt._gp.predict(x_lp, return_std=True)




def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)    # 현재 몇번 iteration 했는지?
    fig.suptitle(f'Gaussian Process and Utility Function After {steps} steps')

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # bayesian optimization으로 현재까지 관측된 x, y값
    x_obs = np.array([[res['params']['x']] for res in optimizer.res])
    y_obs = np.array([res['target'] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    
    # plot function
    axis.plot(x, y, linewidth=3, label='target')        # 실제 함수
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='r')   # 현재 관측된 값 ◆
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=0.6, fc='c', ec='None', label='95% confidence interval'
        )
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_xlabel('x', fontdict={'size':20})
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    # acquition function
    util_func = UtilityFunction(kind='ucb', kappa=5, xi=0)
    utility = util_func.utility(x=x, gp=optimizer._gp, y_max=0)
    acq.plot(x, utility, color='purple', label='Utility Function')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', 
        markersize=15, markerfacecolor='gold', 
        markeredgecolor='k', markeredgewidth=1, label='NExt Best Gauess')
    acq.set_xlim((-2,10))
    acq.set_ylim((0, np.max(utility)+0.5))
    acq.set_xlabel('x', fontdict={'size': 20})
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.show()


bayes_opt = BayesianOptimization(f=target, pbounds={'x': (-2,10)}, random_state=27)
bayes_opt.maximize(init_points=0, n_iter=1, acq='ucb', kappa=5)
plot_gp(bayes_opt, x_lp, y_lp)






# Cifar10 Data에 HyperParameter Optimize
(x_trainval, y_trainval), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_trainval.shape, y_trainval.shape, x_test.shape, y_test.shape)

x_trainval = x_trainval.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

y_trainval = tf.one_hot(y_trainval.squeeze(), 10)
y_test = tf.one_hot(y_test.squeeze(), 10)
print(x_trainval.shape, y_trainval.shape, x_test.shape, y_test.shape)

x_train = x_trainval[:1000]
y_train = y_trainval[:1000]

x_valid = x_trainval[49000:]
y_valid = y_trainval[49000:]

print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)



train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(2000)
# valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(2000)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2000)



def build_network(num_layers_conv, 
                num_layers_dense, 
                num_channels_cnn, 
                num_nodes_dense):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32,32,3)))

    # Conv_layers
    for i in range(num_layers_conv):
        model.add(tf.keras.layers.Conv2D(num_channels_cnn, kernel_size=(3,3), 
                strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())

    # Dense_layers
    for i in range(num_layers_dense):
        model.add(tf.keras.layers.Dense(num_nodes_dense, activation='relu'))
    
    # output_layers
    model.add(tf.keras.layers.Dense(10))
    return model

def build_and_optimize(num_layers_conv, 
                num_layers_dense, 
                num_channels_cnn, 
                num_nodes_dense,
                log_learning_rate):
    num_layers_conv = int(num_layers_conv)
    num_layers_dense = int(num_layers_dense)
    num_channels_cnn = int(num_channels_cnn)
    num_nodes_dense = int(num_nodes_dense)
    learning_rate = np.exp(log_learning_rate)

    # print(f'num_layers_conv: {num_layers_conv}')
    # print(f'num_layers_dense: {num_layers_dense}')
    # print(f'num_channels_cnn: {num_channels_cnn}')
    # print(f'num_nodes_dense: {num_nodes_dense}')
    # print(f'learning_rate: {learning_rate}')


    model = build_network(num_layers_conv, num_layers_dense, 
                num_channels_cnn, num_nodes_dense)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate), 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])
    model.fit(train_dataset, epochs=10, verbose=0)
    loss, accuracy = model.evaluate(x_valid, y_valid, verbose=0)
    # print(f'accuracy: {accuracy}')
    return accuracy


bound = {'num_layers_conv': (2, 5.9),
        'num_layers_dense': (2, 4.9),
        'num_channels_cnn': (32, 128.9),
        'num_nodes_dense': (32, 256.9),
        'log_learning_rate': (-5,-1)}


bayes_optimizer = BayesianOptimization(f=build_and_optimize, pbounds=bound, random_state=7)
bayes_optimizer.maximize(init_points=2, n_iter=10, kappa=5)

bayes_optimizer.max # best_case

acc_frame = pd.Series([res['target'] for res in bayes_optimizer.res], name='accuracy').to_frame()
params_frame = pd.DataFrame([res['params'] for res in bayes_optimizer.res])
bayesian_frame = pd.concat([acc_frame, params_frame], axis=1)
bayesian_frame


# models = build_network(2, 4, 3, 3)
# models.compile(optimizer=tf.keras.optimizers.SGD(0.001), 
#                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
#                 metrics=['accuracy'])
# models.fit(train_dataset, epochs=10, verbose=2)
# models.evaluate(x_valid, y_valid)


