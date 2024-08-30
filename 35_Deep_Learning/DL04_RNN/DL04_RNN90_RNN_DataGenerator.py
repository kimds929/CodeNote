
# (Python) Random Sequence Generator 230307
import numpy as np
import scipy as sp
from scipy import stats
rng = np.random.RandomState(1)
rn_prob = rng.rand(4,5)

np.random.normal()
stats.multivariate_normal.pdf()

# Generating Sequence Data ------------------------------------------------------
states = [0,1,2,3,4,5,6]

init_prob = [0, 0.015, 0.015, 0.12, 0.25, 0.3, 0.3]

prob_dict = {
 0:[1, 0, 0, 0, 0, 0, 0],
 1:[0.8, 0, 0.17, 0.02, 0.01, 0, 0],
 2:[0.7, 0.22, 0, 0.05, 0.02, 0.01, 0],
 3:[0, 0.35, 0.4, 0, 0.15, 0.05, 0.05],
 4:[0, 0.25, 0.25, 0.3, 0, 0.1, 0.1],
 5:[0, 0.15, 0.15, 0.3, 0.2, 0, 0.2],
 6:[0, 0.15, 0.15, 0.2, 0.3, 0.2,0]
 }

# vmin = 0
# vmax = 5
# mean_ = 4
# std_ = 3
def normal_generator(v_mean, v_std, size, v_min=0, v_max=5):
    values = np.random.normal(v_mean,v_std, size=size)
    values[values < v_min] = v_min
    values[values > v_max] = v_max
    return np.round(values,0).astype(int)


def gen_seq_data(states, init_prob, prob_dict, size=100, max_len=100, random_states=None):
    # max_len = 100
    
    assert sum(init_prob) == 1,  "Sum of init staes's probability is 1"
    assert (pd.DataFrame(prob_dict).sum(0) == 1).all(),  "Sums of each staes's probability are 1"

    rng = np.random.RandomState(random_states)
    states_df = pd.Series(rng.choice(states, p=init_prob, size=size)).to_frame()
    next_series = np.array([True])
    
    n = 1
    while next_series.any():
        next_series = states_df.iloc[:,-1].apply(lambda x: rng.choice(states, p=prob_dict[x]) )
        next_series.name = n
        states_df = pd.concat([states_df, next_series], axis=1)
        
        if n >= max_len:
            break
        n += 1
    return states_df
# ------------------------------------------------------

seq_example = gen_seq_data(states, init_prob, prob_dict, size=10000, random_states=0)
seq_example
