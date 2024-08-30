#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From the wikipedia page with slight modification
https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
"""

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]

    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}

    # Run Viterbi when t > 0

    for t in range(1, len(obs)):
        V.append({})

        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)

            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break

    for line in dptable(V):
        print(line)

    opt = []

    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None

    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

# The function viterbi takes the following arguments: obs is the sequence of
# observations, e.g. ['normal', 'cold', 'dizzy']; states is the set of hidden
# states; start_p is the start probability; trans_p are the transition
# probabilities; and emit_p are the emission probabilities. For simplicity of
# code, we assume that the observation sequence obs is non-empty and that
# trans_p[i][j] and emit_p[i][j] is defined for all states i,j.

# In the running example, the forward/Viterbi algorithm is used as follows:


obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')


start_p = {'Healthy': 0.6, 'Fever': 0.4}

trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
   
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }
   
viterbi(obs,states,start_p,trans_p,emit_p)











# --------------------------------------------------------------------------------------------------------------



# https://datascienceschool.net/03%20machine%20learning/20.01%20%ED%9E%88%EB%93%A0%20%EB%A7%88%EC%BD%94%ED%94%84%20%EB%AA%A8%ED%98%95.html
# https://github.com/hmmlearn/hmmlearn
# pip install hmmlearn --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org

from hmmlearn.hmm import GaussianHMM, CategoricalHMM, MultinomialHMM
from hmmlearn import hmm
import sklearn
np.array(dir(sklearn.decomposition))
np.array(dir(sklearn.gaussian_process))

np.random.seed(3)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
model.startprob_ = np.array([0.9, 0.1])
model.transmat_ = np.array([[0.95, 0.05], [0.15, 0.85]])
model.means_ = np.array([[1.0], [-3.0]])
model.covars_ = np.array([[15.0], [40.0]])
X, Z = model.sample(500)

plt.figure(figsize=(10,10))
plt.subplot(311)
plt.plot(X)
plt.title("random variable")
plt.subplot(312)
plt.plot(Z)
plt.title("discrete state")
plt.subplot(313)
plt.plot((1 + 0.01*X).cumprod())
plt.title("X cumulated")
plt.tight_layout()
plt.show()






states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

# parameter
start_probability = np.array([0.6, 0.4])
transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])
emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob = start_probability
model.transmat = transition_probability
model.emissionprob = emission_probability

# predict a sequence of hidden states based on visible states
bob_says = [0, 2, 1, 1, 2, 0]
model = model.fit(bob_says)
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")