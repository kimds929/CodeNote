from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1, 2], [5, 8], [1.5 ,1.8], [8 , 8], [1, 0.6], [9, 11]]).astype('float32')
y = np.array([-1, 1, -1, 1, -1, 1]).astype('float32')

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# SVM -------------------------------------
clf_svm = SVC(kernel='linear', C=1.0)
clf_svm.fit(X, y)

clf_svm.support_vectors_    # support vector

clf_svm.predict([[0.58, 0.76]])
clf_svm.predict([[10.58, 10.76]])
w = clf_svm.coef_[0]
b = clf_svm.intercept_

# z = w0 * x0 + w1 + x1 + b
# 0 = w0 * x0 + w1 + x1 + b
# x1 = -w0/w1 * x0 - b/w1

a = -w[0]/w[1]
c = -b / w[1]
xs = np.linspace(0, 12, 50)
ys = a * xs + c

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(xs, ys, 'r--', alpha=0.5)
plt.plot(clf_svm.support_vectors_[:,0], clf_svm.support_vectors_[:,1], marker='o', markeredgecolor='red', color='none', ls='')
plt.show()



# SVM by tensorflow ----------------------------------------------------------------
import tensorflow as tf

def soft_margin_loss(W, true, pred, C=1):
    return (0.5 * tf.reduce_sum(tf.square(W)) 
            + C * tf.reduce_sum(tf.maximum(tf.zeros_like(true), 1 - true * pred)) )

class TensorflowSVM(tf.keras.Model):
    def __init__(self, dim_x, C=1):
        super(TensorflowSVM, self).__init__()
        self.W = tf.Variable(tf.zeros(shape=(dim_x, 1)))
        self.b = tf.Variable(tf.zeros(shape=(1,)))

        self.C = C
    
    def call(self, x):
        return tf.squeeze(x @ self.W + self.b)

    def train_step(self, data): # to use model.fit
        x, y = data
        with tf.GradientTape() as Tape:
            pred = tf.squeeze(x @ self.W + self.b)
            loss = soft_margin_loss(self.W, y, pred, self.C)
        gradients = Tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss}

# X = np.array([[1, 2], [5, 8], [1.5 ,1.8], [8 , 8], [1, 0.6], [9, 11]]).astype('float32')
# y = np.array([-1, 1, -1, 1, -1, 1]).astype('float32')

tf_svm = TensorflowSVM(2)
tf_svm.compile(optimizer=tf.keras.optimizers.SGD(0.01))
tf_svm.fit(X, y, batch_size=6, epochs=5000, verbose=0)

tf_w = tf_svm.W.numpy()
tf_b = tf_svm.b.numpy()

tf_a = - tf_w[0,0] / tf_w[1,0]
tf_c = - tf_b[0] / tf_w[1,0]

# xs = np.linspace(0, 12, 50)
tf_ys = tf_a * xs + tf_c



plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(xs, ys, 'r--', alpha=0.5, label='sklearn')
plt.plot(xs, tf_ys, color='orange', ls='--', alpha=0.5, label='tensorflow')
plt.legend(loc=1)
plt.show()




# ---------------------------------------------------------------------------
import sklearn
from sklearn import datasets

ds_X, ds_y = datasets.make_moons(n_samples=300, noise=0.16, random_state=42)
plt.scatter(ds_X[:,0], ds_X[:,1], c=ds_y)

# Linear Kernel
clf_linear = SVC(kernel='linear', random_state=100)
clf_linear.fit(ds_X, ds_y)


linear_w = clf_linear.coef_[0]
linear_b = clf_linear.intercept_

linear_a = -linear_w[0]/linear_w[1]
linear_c = -linear_b / linear_w[1]

linear_xs = np.linspace(-1.5, 2.5, 50)
linear_ys = linear_a * linear_xs + linear_c


plt.scatter(ds_X[:, 0], ds_X[:, 1], c=ds_y)
plt.plot(linear_xs, linear_ys, 'r--', alpha=0.5, label='sklearn_linear')
plt.legend(loc=1)
plt.show()


# RBF Kernel
clf_rbf = SVC(kernel='rbf', random_state=100)
clf_rbf.fit(ds_X, ds_y)

def plot_decision_boundary(clf, X, y, bound):
    x0s = np.linspace(bound[0], bound[1], 100)
    x1s = np.linspace(bound[2], bound[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    print(x0.shape, x1.shape)

    z = clf.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    
    # plt.contourf 등고선 함수
    # plt.contourf(x0s, x1s, z, cmap=plt.cm.coolwarm, alpha=0.8)
    # plt.contourf(x0s, x1s, z, cmap='RdBu', alpha=0.8)
    plt.contourf(x0s, x1s, z, cmap='coolwarm', alpha=0.8)
    plt.scatter(X[:,0], X[:, 1], c=y)
    plt.show()

plot_decision_boundary(clf_rbf, ds_X, ds_y, (-1.5, 2.5, -1, 1.5))

# np.c_[np.array([1,2,3]), np.array([4,5,6])]
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
# np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
# array([[1, 2, 3, ..., 4, 5, 6]])









# Mulit-Class_SVM ---------------------------------------------------------------
X_mc = np.array([[0,1], [1,1], [2,4], [3,5]]).astype('float32')
y_mc = np.array([0, 1, 2, 3])

plt.scatter(X_mc[:,0], X_mc[:,1], c=y_mc)
plt.show()

# decision_function: ovr
clf_mc = SVC(kernel='linear', decision_function_shape='ovr')
# (decision_function_shape) ova : one-vs-all  /  ovr : one-vs-rest  /  ovo = one-vs-one
clf_mc.fit(X_mc, y_mc)

# clf.predict([[2,1.1]])
plot_decision_boundary(clf_mc, X_mc, y_mc, (-1,10, -1,10))


# decision_function: ovo
clf_mc2 = SVC(kernel='linear', decision_function_shape='ovo')
clf_mc2.fit(X_mc, y_mc)
plot_decision_boundary(clf_mc2, X_mc, y_mc, (-1,10, -1,10))


# sigmoid kernel
clf_mc_sigmoid = SVC(kernel='sigmoid', decision_function_shape='ovo')
clf_mc_sigmoid.fit(X_mc, y_mc)
plot_decision_boundary(clf_mc_sigmoid, X_mc, y_mc, (-1,10, -1,10))

# rbf kernel
clf_mc_rbf = SVC(kernel='rbf', decision_function_shape='ovo')
clf_mc_rbf.fit(X_mc, y_mc)
plot_decision_boundary(clf_mc_rbf, X_mc, y_mc, (-1,10, -1,10))

