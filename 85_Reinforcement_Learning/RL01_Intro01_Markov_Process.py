import numpy as np
import pandas as pd

# MC Ep_isode -----------------------------------------------------------
state1 = ['s1', 's2', 's3']
P1 = np.array([[0,   0,   1],
              [1/2, 1/2, 0],
              [1/3, 2/3, 0]])

# eigen vector
P_eig1 = np.linalg.eig(P1.T)
P_eig_value1 = P_eig1[0]       # eigen vector
print(P_eig_value1)

where1 = list(map(lambda x: np.allclose(x, 1), P_eig_value1)).index(True)

P_eig_vector1 = P_eig1[1][:,where1]       # eigen vector
print(P_eig_vector1)

p_eig1 = P_eig_vector1/np.sum(P_eig_vector1)
print(p_eig1)  # result
pd.Series(p_eig1, index=state1)    # result


# fixed iteration
p_i1 = np.array([1/3, 1/3, 1/3])
# p_i1 = np.array([1, 0, 0])
# p_i1 = np.array([0, 1, 0])
# p_i1 = np.array([0, 0, 1])

for _ in range(100):
    p_i1 = p_i1 @ P1

print(p_i1)

pd.Series(p_i1, index=state1)    # result





# Studnet Markov Chain Episode -----------------------------------------------------------
state2 = ['C1','C2','C3','Pass','Pub','FB','Sleep']
P2 = np.array([[0, 0.5, 0, 0, 0, 0.5, 0],
               [0, 0, 0.8, 0, 0, 0, 0.2],
               [0, 0, 0, 0.6, 0.4, 0, 0],
               [0, 0, 0, 0, 0, 0, 1],
               [0.2, 0.4, 0.4, 0, 0, 0, 0],
               [0.1, 0, 0, 0, 0, 0.9, 0],
               [0, 0, 0, 0, 0, 0, 1]])

# eigen vector
P_eig2 = np.linalg.eig(P2.T)
P_eig_value2 = P_eig2[0]       # eigen vector
print(P_eig_value2)

where2 = list(map(lambda x: np.allclose(x, 1), P_eig_value2)).index(True)

P_eig_vector2 = P_eig2[1][:,where2]       # eigen vector
print(P_eig_vector2)

p_eig2 = P_eig_vector2/np.sum(P_eig_vector2)
print(p_eig2)  # result
pd.Series(p_eig2, index=state2)    # result

# fixed iteration
p_i2_random = np.random.randint(1,100, 7)
p_i2 = p_i2_random / p_i2_random.sum()
print(p_i2)

for _ in range(100):
    p_i2 = p_i2 @ P2

print(np.round(p_i2,5))
pd.Series(p_i2, index=state2)    # result : i번째 time step에서 해당 state에 있을 확률

