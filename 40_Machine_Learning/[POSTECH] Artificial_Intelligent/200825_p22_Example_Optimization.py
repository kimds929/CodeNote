import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import cvxpy as cvx

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'


# Problem 1: Optimization -------------------------------------------------------------------------------------
# load images
imag1 = plt.imread(absolute_path + 'postech_map.PNG')
imag1.shape

plt.figure(figsize=(20,20))
plt.imshow(imag1)


# location information
postech = np.array([[200], [800]], dtype=np.float32)
emart = np.array([[635], [15]], dtype=np.float32)
terminal = np.array([[948], [675]], dtype=np.float32)
market = np.array([[413],[860]], dtype=np.float32)

# a) [2pts] Visualize locations of each facility on the map.
# (marker shape = 'o', markersize = 25)
plt.figure(figsize=(20,20))
plt.imshow(imag1)
plt.plot(postech[0], postech[1], 'X', markersize=25)
plt.plot(emart[0], emart[1], 'X', markersize=25)
plt.plot(terminal[0], terminal[1], 'X', markersize=25)
plt.plot(market[0], market[1], 'X', markersize=25)


# b) [5pts] Visualize the optimal location minimizing the distance to each facility. (Using cvxpy)
# (marker shape = 'o', markersize = 25)
x = cvx.Variable([2,1])

obj = cvx.Minimize(cvx.norm(postech-x, 2) + 2* cvx.norm(emart-x, 2) + 
                   cvx.norm(terminal-x, 2) + cvx.norm(market-x, 2))

prob = cvx.Problem(obj)
result = prob.solve()



plt.figure(figsize=(20,20))
plt.imshow(imag1)
plt.plot(postech[0], postech[1], 'X' ,markersize=25)
plt.plot(emart[0], emart[1], 'X', markersize=25)
plt.plot(terminal[0], terminal[1], 'X', markersize=25)
plt.plot(market[0], market[1], 'X', markersize=25)
plt.plot(x.value[0], x.value[1], 'kX', markersize=25)




# c) [5pts] Find the optimal location where e-mart is within 400 and terminal is within 500.
# (400 and 500 are pixel distance.)
x = cvx.Variable([2, 1])
obj = cvx.Minimize(cvx.norm(postech-x, 2) + cvx.norm(emart-x, 2) + 
                   cvx.norm(terminal-x, 2) + cvx.norm(market-x, 2))
const = [cvx.norm(terminal-x, 2) <= 500,
        cvx.norm(emart-x, 2) <= 400]
prob = cvx.Problem(obj, const)
result = prob.solve()

plt.figure(figsize=(20,20))
plt.imshow(imag1)
plt.plot(postech[0], postech[1], 'X' ,markersize=25)
plt.plot(emart[0], emart[1], 'X', markersize=25)
plt.plot(terminal[0], terminal[1], 'X', markersize=25)
plt.plot(market[0], market[1], 'X', markersize=25)
plt.plot(x.value[0], x.value[1], 'kX', markersize=25)






# Problem 2: Digit Classification & Clustering --------------------------------------------------------------------------
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


data = datasets.load_digits(n_class=3)
print(data.data.shape, data.target.shape)


data0 = data.images[data.target == 0]
data1 = data.images[data.target == 1]
data2 = data.images[data.target == 2]
print(data0.shape, data1.shape, data2.shape)


for temp in [data0, data1, data2]:
    plt.figure(figsize=(2, 2))
    plt.imshow(temp[np.random.randint(len(temp))], 'gray_r');
    plt.show()



def featureA(data):
    return np.mean(data, axis=(1, 2)).reshape(-1, 1)

def featureB(data):
    return np.mean(data[:,3:5,3:5], axis=(1, 2)).reshape(-1, 1)

feature0_A = featureA(data0)
feature1_A = featureA(data1)
feature2_A = featureA(data2)

feature0_B = featureB(data0)
feature1_B = featureB(data1)
feature2_B = featureB(data2)


plt.figure(figsize = (6, 6))
plt.scatter(feature0_A, feature0_B, s=8, label='digit 0')
plt.scatter(feature1_A, feature1_B, s=8, label='digit 1')
plt.scatter(feature2_A, feature2_B, s=8, label='digit 2')
plt.legend()
plt.title('Feature Space')
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.show()










from sklearn import linear_model
from sklearn import svm

def my_perceptron(d0, d1):
    X = np.vstack([d0, d1])
    y = np.vstack([np.ones([d0.shape[0], 1]), -np.ones([d1.shape[0], 1])])
    
    clf = linear_model.Perceptron(tol=1e-3)
    clf.fit(X, np.ravel(y))

    w0 = clf.intercept_[0]
    w1 = clf.coef_[0,0]
    w2 = clf.coef_[0,1]
    
    x1p = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape(-1,1)
    x2p = - w1/w2*x1p - w0/w2
    plt.figure(figsize = (6, 6))
    plt.scatter(d0[:,0], d0[:,1], s=8, label='C1')
    plt.scatter(d1[:,0], d1[:,1], s=8, label='C0')
    plt.plot(x1p, x2p, c = 'k', linewidth = 1, label = 'Perceptron')
    plt.legend()
    plt.title('Feature Space')
    plt.xlabel('Feature A')
    plt.ylabel('Feature B')
    plt.show()
    w = [w0, w1, w2]
    return w

def my_svm(d0, d1):
    X = np.vstack([d0, d1])
    y = np.vstack([np.ones([d0.shape[0], 1]), -np.ones([d1.shape[0], 1])])
    
    clf = svm.SVC(kernel = 'linear', tol=1e-3)
    clf.fit(X, np.ravel(y))

    w0 = clf.intercept_[0]
    w1 = clf.coef_[0,0]
    w2 = clf.coef_[0,1]
    
    x1p = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).reshape(-1,1)
    x2p = - w1/w2*x1p - w0/w2
    plt.figure(figsize = (6, 6))
    plt.scatter(d0[:,0], d0[:,1], s=8, label='C1')
    plt.scatter(d1[:,0], d1[:,1], s=8, label='C0')
    plt.plot(x1p, x2p, c = 'k', linewidth = 1, label = 'Perceptron')
    plt.legend()
    plt.title('Feature Space')
    plt.xlabel('Feature A')
    plt.ylabel('Feature B')
    plt.show()
    w = [w0, w1, w2]
    return w



X0 = np.hstack([feature0_A, feature0_B])
X1 = np.hstack([feature1_A, feature1_B])
X2 = np.hstack([feature2_A, feature2_B])



w_1 = my_perceptron(X0, X1)
w_3 = my_perceptron(X1, X2)
w_2 = my_perceptron(X2, X0)


w_1 = my_svm(X0, X1)
w_2 = my_svm(X1, X2)
w_3 = my_svm(X2, X0)



w_1 = my_perceptron(X0, np.vstack([X1, X2]))
w_2 = my_perceptron(X1, np.vstack([X2, X0]))
w_3 = my_perceptron(X2, np.vstack([X0, X1]))



w_1 = my_svm(X0, np.vstack([X1, X2]))
w_2 = my_svm(X1, np.vstack([X2, X0]))
w_3 = my_svm(X2, np.vstack([X0, X1]))




feature_A = np.vstack([feature0_A, feature1_A, feature2_A])
feature_B = np.vstack([feature0_B, feature1_B, feature2_B])


# result plotting
xx, yy = np.meshgrid(np.arange(np.min(feature_A), np.max(feature_A), 0.1), np.arange(np.min(feature_B), np.max(feature_B), 0.1))
grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])


z_1 = np.empty(len(grid_X))
z_2 = np.empty(len(grid_X))
for i in range(len(grid_X)):
    w = grid_X[i]
    z_1[i] = (1*w_1[0] + grid_X[i][0]*w_1[1] + grid_X[i][1]*w_1[2]) > 0
    z_2[i] = (1*w_2[0] + grid_X[i][0]*w_2[1] + grid_X[i][1]*w_2[2]) > 0
z_1 = z_1.reshape(xx.shape)
z_2 = z_2.reshape(xx.shape)
z_3 = (z_1 == z_2).astype(np.float32)
z = 0*z_1 + 1*z_2 + 2*z_3


plt.figure(figsize = (10, 10))
plt.scatter(feature2_A, feature2_B, s=8, label='C2')
plt.scatter(feature1_A, feature1_B, s=8, label='C1')
plt.scatter(feature0_A, feature0_B, s=8, label='C0')
plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.2)
plt.legend()
plt.title('Feature Space')
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.show()


feature = np.hstack([feature_A, feature_B])
feature.shape















from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(feature)

plt.figure(figsize = (10,8))
plt.plot(feature[kmeans.labels_ == 0,0], feature[kmeans.labels_ == 0,1], 'b.', label = 'C0')
plt.plot(feature[kmeans.labels_ == 1,0], feature[kmeans.labels_ == 1,1], 'g.', label = 'C1')
plt.plot(feature[kmeans.labels_ == 2,0], feature[kmeans.labels_ == 2,1], 'r.', label = 'C2')
plt.legend(fontsize = 12)
plt.show()




data0 = data.data[data.target == 0]
data1 = data.data[data.target == 1]
data2 = data.data[data.target == 2]
data_total = np.vstack([data0, data1, data2])
data_total = (data_total - np.mean(data_total))/np.std(data_total)










# Autoencoder
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(64,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation=None),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(64, activation=None)
])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
              loss='mean_squared_error',
              metrics=['mse'])

training = model.fit(data_total, data_total, batch_size=5, epochs=250, verbose=0)

plt.figure(figsize=(10,8))
plt.plot(training.history['loss'], label = 'training')
plt.xlabel('Epochs', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.legend(fontsize = 12)
plt.ylim([np.min(training.history['loss']), np.max(training.history['loss'])])
plt.show()

scores = model.evaluate(data_total, data_total, verbose=2)
print('loss: {}'.format(scores[0]))



model_latent = tf.keras.models.Model(model.input, model.layers[1].output)
latent0 = model_latent.predict(data0)
latent1 = model_latent.predict(data1)
latent2 = model_latent.predict(data2)



plt.figure(figsize = (10,8))
plt.scatter(latent0[:,0], latent0[:,1], label='digit 0')
plt.scatter(latent1[:,0], latent1[:,1], label='digit 1')
plt.scatter(latent2[:,0], latent2[:,1], label='digit 2')
plt.legend()
plt.show()









from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(data_total)


latent0 = pca.transform(data0)
latent1 = pca.transform(data1)
latent2 = pca.transform(data2)


plt.figure(figsize = (10,8))
plt.scatter(latent0[:,0], latent0[:,1], label='digit 0')
plt.scatter(latent1[:,0], latent1[:,1], label='digit 1')
plt.scatter(latent2[:,0], latent2[:,1], label='digit 2')
plt.legend()
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(data_total)

plt.figure(figsize = (10,8))
plt.plot(feature[kmeans.labels_ == 0,0], feature[kmeans.labels_ == 0,1], 'b.', label = 'C0')
plt.plot(feature[kmeans.labels_ == 1,0], feature[kmeans.labels_ == 1,1], 'g.', label = 'C1')
plt.plot(feature[kmeans.labels_ == 2,0], feature[kmeans.labels_ == 2,1], 'r.', label = 'C2')
plt.legend(fontsize = 12)
plt.show()








