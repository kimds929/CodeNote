import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sigmoid function --------------------------------------------------------------------------------
def sigmoid(x, b1=1, b0=0):
    # return np.exp(x) / (1+np.exp(x))
    return 1/ (1+np.exp(-1*(b1*x + b0)))

x = np.linspace(-10,10,50)

f = plt.figure()
plt.plot(x, sigmoid(x))
plt.show()


# b1
b1_sets = np.array([ 10. ,   5. ,   1. ,   0.5,   0.2,   
        0. ,  -0.2,  -0.5,  -1. , -5. , -10. ])

fb1 = plt.figure(figsize=(9,9))
for e, b1 in enumerate(b1_sets):
    b1_rep = str(int(b1)) if np.abs(b1) >=1 else str(float(b1))

    plt.subplot(4,3,e+1)
    plt.plot(x, sigmoid(x, b1=b1), label=f"β1: {b1_rep}", alpha=0.5)
    if b1 > 0:
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='upper right')
    plt.ylim(-0.1,1.1)
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

# b0
b0_sets =  np.array([ 5. ,   3. ,   1. ,   0.5,   0.2,   
        0. ,  -0.2,  -0.5,  -1. , -3. , -5. ])

fb0 = plt.figure(figsize=(9,9))
for e, b0 in enumerate(b0_sets):
    b0_rep = str(int(b0)) if np.abs(b0) >=1 else str(float(b0))

    plt.subplot(4,3,e+1)
    plt.plot(x, sigmoid(x, b1=1, b0=b0), label=f"β0: {b0_rep}", alpha=0.5, color='orange')
    plt.legend(loc='upper left')
    plt.axvline(-b0, alpha=0.3, color='gray', ls='--')
    plt.ylim(-0.1,1.1)
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()



