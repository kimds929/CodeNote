

# 4. Looking at Parameters 
# To understand a network's behavior
# 4.1. Multi-Layers with Keras -------------------------------------------------------------------

# 【 Keras 】
    # model
    # model.compile() → 어떻게 학습할지, option 설정(optimizer, loss, metric)
    # model.fit(x, y)
    # model.evaluate() → Test 데이터 셋에 대해 전체 결과를 보고 싶을때
    # model.predict() → 하나의 입력으로 결과를 보고 싶을때

# training data gerneration

m = 1000
x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1-1)**2 + 2*x2 + 5

C1 = np.where(g >= 0)[0]
C0 = np.where(g < 0)[0]
N = C1.shape[0]
M = C0.shape[0]

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

train_X = np.vstack([X1, X0])
train_X = np.asmatrix(train_X).astype(np.float32)

train_y = np.vstack([np.ones([N,1]), np.zeros([M,1])])
train_y = tf.keras.utils.to_categorical(train_y).astype(np.float32)
print(train_X.shape, train_y.shape)

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.title('Nonlinearly Distributed Data', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 12)
plt.show()




# keras ---------------------------------------------------------------
n_input = 2
n_hidden = 2
n_output = 2

def ANN(n_input, n_hidden, n_output):
    L0 = tf.keras.layers.Input(shape=[n_input])     # input 만들기
    L = tf.keras.layers.Dense(units=n_hidden, activation='sigmoid')(L0) # node
    L = tf.keras.layers.Dense(units=n_output, activation='sigmoid')(L)
    model = tf.keras.Model(L0, L)
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

model = ANN(n_input, n_hidden, n_output)
model.summary()


n_batch = 50 
n_epoch = 500
n_prt = 250

history = model.fit(train_X, train_y, verbose=0, batch_size=n_batch, epochs=n_epoch)
# verbose : 학습시키는 과정을 얼마나 보여줄 것인지? 0 : 안보여줌, 1 : 각 출력결과 확인

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('iteration', fontsize = 15)
plt.legend()
plt.show()


# 학습된 weight 가져오기
for layer in model.layers:
    print(layer.name , '-' *50)
    print(layer.get_weights())


weights, biases = {}, {}
weights['hidden'], biases['hidden'] = model.layers[1].get_weights()
weights['output'], biases['output'] = model.layers[2].get_weights()


# predict
p_hidden = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
H = p_hidden.predict(train_X)
model.evaluate(train_X, train_y)

p_hidden.summary()


# plotting : Kernel
plt.figure(figsize=(10, 8))

xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).astype(np.float32)
Z = 1/(1 + np.exp(-(grid_X @ weights['output'] + biases['output'])))
Z = np.argmax(Z, axis=1).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
plt.plot(H[:N,0], H[:N,1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(H[N:,0], H[N:,1], 'bo', alpha = 0.4, label = 'C0')
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()



# plotting : Original
x1p = np.arange(-5, 5, 0.01).reshape(-1, 1)
x2p = - weights['hidden'][0,0]/weights['hidden'][1,0]*x1p - biases['hidden'][0]/weights['hidden'][1,0]
x3p = - weights['hidden'][0,1]/weights['hidden'][1,1]*x1p - biases['hidden'][1]/weights['hidden'][1,1]

plt.figure(figsize=(10, 8))
xx, yy = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-4, 4, 0.01))
grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).astype(np.float32)
Z = np.argmax(model.predict(grid_X), axis=1).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
plt.title('Accuracy: %.1f%%' % (100*model.evaluate(train_X, train_y, verbose=0)[1]))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(x1p, x2p, 'k', linewidth = 3, label = '')
plt.plot(x1p, x3p, 'g', linewidth = 3, label = '')
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([-5, 5])
plt.ylim([-4, 4])
plt.show()








# 4.2. Multi-Neurons  ----------------------------------------------------------------
model = ANN(n_input=2, n_hidden=3, n_output=2)
model.summary()

history = model.fit(train_X, train_y, verbose=0, batch_size=n_batch, epochs=n_epoch)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('iteration', fontsize = 15)
plt.legend()
plt.show()






weights, biases = {}, {}
weights['hidden'], biases['hidden'] = model.layers[1].get_weights()
weights['output'], biases['output'] = model.layers[2].get_weights()

def plotSpace():
    x1p = np.arange(-5, 5, 0.01).reshape(-1, 1)
    yps = [-weights['hidden'][0,i]/weights['hidden'][1,i]*x1p -biases['hidden'][i]/weights['hidden'][1,i] for i in range(weights['hidden'].shape[1])]

    plt.figure(figsize=(10, 8))
    xx, yy = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-4, 4, 0.01))
    grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).astype(np.float32)
    Z = np.argmax(model.predict(grid_X), axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.title('Accuracy: %.1f%%' % (100*model.evaluate(train_X, train_y, verbose=0)[1]))
    plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
    plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
    for yp in yps:
        plt.plot(x1p, yp, linewidth=3, label='')
    plt.xlabel('$x_1$', fontsize = 15)
    plt.ylabel('$x_2$', fontsize = 15)
    plt.legend(loc = 1, fontsize = 12)
    plt.axis('equal')
    plt.xlim([-5, 5])
    plt.ylim([-4, 4])
    plt.show()


plotSpace()









# -----------------------------------------------------------------------------------
# training data gerneration

m = 1000
x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1*x2-1)**2 + 2*x2 + 5

C1 = np.where(g >= 0)[0]
C0 = np.where(g < 0)[0]
N = C1.shape[0]
M = C0.shape[0]
m = N + M

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

train_X = np.vstack([X1, X0])
train_X = np.asmatrix(train_X)

train_y = np.vstack([np.ones([N,1]), np.zeros([M,1])])
train_y = tf.keras.utils.to_categorical(train_y)

plt.figure(figsize=(10, 8))
plt.title('Accuracy: %.1f%%' % (100*model.evaluate(train_X, train_y, verbose=0)[1]))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.title('Nonlinearly Distributed Data', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 12)
plt.show()



# tensorflow ****
model = ANN(n_input=2, n_hidden=4, n_output=2)
model.summary()

history = model.fit(train_X, train_y, verbose=1, batch_size=n_batch, epochs=n_epoch)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('iteration', fontsize = 15)
plt.legend()
plt.show()

weights, biases = {}, {}
weights['hidden'], biases['hidden'] = model.layers[1].get_weights()
weights['output'], biases['output'] = model.layers[2].get_weights()

plotSpace()





# -----------------------------------------------------------------------------------
# training data gerneration

m = 1000
x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1-1)**2 + 2*x2*x1 + 5

C1 = np.where(g >= 0)[0]
C0 = np.where(g < 0)[0]
N = C1.shape[0]
M = C0.shape[0]
m = N + M

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

train_X = np.vstack([X1, X0])
train_X = np.asmatrix(train_X)

train_y = np.vstack([np.ones([N,1]), np.zeros([M,1])])
train_y = tf.keras.utils.to_categorical(train_y)

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.title('Nonlinearly Distributed Data', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 12)
plt.show()




# tensorflow ****
model = ANN(n_input=2, n_hidden=4, n_output=2)
model.summary()

history = model.fit(train_X, train_y, verbose=0, batch_size=n_batch, epochs=n_epoch)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('iteration', fontsize = 15)
plt.legend()
plt.show()

weights, biases = {}, {}
weights['hidden'], biases['hidden'] = model.layers[1].get_weights()
weights['output'], biases['output'] = model.layers[2].get_weights()

plotSpace()

# https://www.youtube.com/embed/BR9h47Jtqyw?rel=0
