import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf
import tensorflow.keras as keras

import math
# absolute_path = r'D:\Python\★★Python_POSTECH_AI\Dataset'



# RNN by Numpy **** ------------------------------------------------------------------

batch_size = 5      # batch  
seq = 3      # timesteps
embed_dim = 4            # features

rng = np.random.RandomState(1)

X = rng.random((batch_size, seq, embed_dim)).astype('float32')    # (Batch, Seq, Embedding)
print(f'X_shape: {X.shape}')
print(X)


units = 2       # Dim
activation_fuc = lambda x: np.tanh(x)

X0 = X[0,0]
X1 = X[0,1]
X0.shape    # (Embedding)

# Weight Initailize
rng = np.random.RandomState(1)
Wx = rng.normal(size=(X.shape[-1], units)).astype('float32')  # (Embedding, Dim)
Wh = rng.normal(size=(units, units)).astype('float32')    # *(Dim, Dim)
b = np.zeros((units,), dtype='float32')       # bias: (1, Dim)
h0 = np.zeros((units,), dtype='float32')      # h0 = hidden state : (1, Dim)


# RNN forward for each input element
h1 = activation_fuc(h0 @ Wh +  X0 @ Wx + b)     # (1, Dim)
h1
h2 = activation_fuc(h1 @ Wh +  X1 @ Wx + b)     # (1, Dim)
h2

# # RNN forward for entire input
# h_output = activation_fuc(h0 @ Wh +  X @ Wx + b)   # (Batch, Seq, Dim)
# h_output.shape
# h_output


# RNN all steps -----------------------------------------------------------------------------
# Hidden State Initialize
stateful = False
hidden_state = np.zeros((units,), dtype='float32')
print(f'hidden_state: {hidden_state.shape}, Wx: {Wx.shape}, Wh: {Wh.shape}, b: {b.shape}')

outputs = []
for batch in X:
    if not stateful:
        hidden_state = np.zeros((units,), dtype='float32')
    batch_output = []
    
    for sequence in batch:
        output_seq = activation_fuc(sequence[np.newaxis,...] @ Wx + hidden_state @ Wh + b).ravel()
        batch_output.append(output_seq)
        hidden_state = output_seq
    outputs.append(batch_output)

sequence_outputs = np.array(outputs)
sequence_outputs           # all sequence output
print(sequence_outputs.shape)
sequence_outputs[:,-1,:]   # last_output



# Torch RNN -------------------------------------------------------------------

rnn = torch.nn.RNN(embed_dim, units, batch_first=True)

state_dict = rnn.state_dict()
weight_dict = [torch.tensor(e, requires_grad=True) for e in [Wx.T, Wh.T, b, h0]]
for k, vn in zip(state_dict, weight_dict):
    state_dict[k] = vn
rnn.load_state_dict(state_dict)


# RNN forward for each input element
r1, r2 = rnn(torch.tensor(X0).reshape(1,1,-1))
r1

# # RNN forward for entire input
r1, r2 = rnn(torch.tensor(X))
r1              # all sequence output
# r1[:,-1,:]    # last_output
r2              # last_output = r1[:,-1,:]


rnn_ouput_torch = []
for batch_x in X:
    result = rnn(torch.tensor(batch_x[np.newaxis, ...]))
    rnn_ouput_torch.append(result[0].detach().numpy()[0])
rnn_ouput_torch = np.array(rnn_ouput_torch)
rnn_ouput_torch             # r1 : all sequence output
rnn_ouput_torch[:,-1,:]     # r2 : last_output = r1[:,-1,:]







# RNN by Tensorflow  **** ------------------------------------------------------------------
units = 2
print(f'X_shape: {X.shape}')
# tf.keras.layers.SimpleRNN?
#   . return_sequences: Boolean. Whether to return the last output
#             in the output sequence, or the full sequence. Default: `False`.
#   . return_state: Boolean. Whether to return the last state
#            in addition to the output. Default: `False`
#   . stateful: Boolean (default False). If True, the last state
#             for each sample at index i in a batch will be used as initial
#           state for the sample of index i in the following batch.

# rnn = tf.keras.layers.SimpleRNN(units=units, return_sequences=True, stateful=True)
# # rnn = tf.keras.layers.SimpleRNN(units=units, return_sequences=True, stateful=False)
# rnn.reset_states()
# rnn.set_weights([tf.constant(Wx), tf.constant(Wh), tf.constant(b)])
# # rnn.weights
# rnn_result = rnn(X)
# rnn_result


# Tensorflow ***
rnn = tf.keras.layers.SimpleRNN(units=units, return_sequences=True, stateful=True)
result = rnn(X[:1])
rnn.reset_states()
rnn.set_weights([tf.constant(Wx), tf.constant(Wh), tf.constant(b)])

# rnn.weights
rnn_result = []
for batch_x in X:
    result = rnn(batch_x[np.newaxis, ...])
    rnn_result.append(result[0])
rnn_result = np.array(rnn_result)
rnn_result

rnn_result.shape


# Pytorch ***
rnn_torch = torch.nn.RNNCell(X.shape[0], hidden_size=units, bias=True)
# rnn_torch(torch.FloatTensor(X[[0]]))

# --------------------------------------------------------------------------------------




# RNN Weight 확인 ------------------------------------------------------------------
class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(units=2, return_sequences=True,)
    
    def call(self, x):
        x = self.rnn(x)
        return x

rnn0 = RNN()
r = rnn0(X)
r
rnn0.weights
wx, wh, b = rnn0.weights


# r[0, 0].numpy()
# r[0, 0].numpy() - (X[0][0] @ wx.numpy() + b.numpy())

# h0 = 0
# h1 = r[0,0].numpy()
# np.math.tanh(X[0][1] @ wx.numpy() + b.numpy() + wh.numpy() @ h1 )
# h2 = r[0,1].numpy()
# np.math.tanh(X[0][2] @ wx.numpy() + b.numpy() + wh.numpy() @ h2 )

# --------------------------------------------------------------------------------------






### RNN Tensorflow vs Torch ####################################################################
# Random X, weights --------------------------------------------------------------------
x = np.random.random((2,3,4)).astype('float32')
x.shape      # (2,3,4) (batch, seq, features) (batch, len_sentence, voca_embedding)

w1 = np.random.random((4,5))
w2 = np.random.random((5,5))
w3 = np.zeros(5)

# w1 = np.arange(20).reshape(4,5)/20
# w2 = np.arange(25).reshape(5,5)/25
# w3 = np.zeros(5)


# Tensorflow.keras RNN ------------------------------------------------------------------
rnn1 = tf.keras.layers.SimpleRNN(5)
rnn1_1 = tf.keras.layers.SimpleRNN(5, return_sequences=True)
rnn1_2 = tf.keras.layers.SimpleRNN(5, return_state=True)
rnn1_3 = tf.keras.layers.SimpleRNN(5, return_sequences=True, return_state=True)

rnn1.set_weights([tf.constant(w1),tf.constant(w2),tf.constant(w3)])
rnn1_1.set_weights([tf.constant(w1),tf.constant(w2),tf.constant(w3)])
rnn1_2.set_weights([tf.constant(w1),tf.constant(w2),tf.constant(w3)])
rnn1_3.set_weights([tf.constant(w1),tf.constant(w2),tf.constant(w3)])


rnn1(x).shape   # (2, 5) (batch, last_state)
rnn1_1(x)
rnn1_1(x).shape   # (2, 3, 5) (batch, seq, last_state)

rnn1_2(x)
rnn1_2(x)[0].shape   # (2, 5) (batch, last_seq)
rnn1_2(x)[1].shape   # (2, 5) (batch, last_state)

rnn1_3(x)
rnn1_3(x)[0].shape   # (2, 3, 5)   (batch, seq, last_state)
rnn1_3(x)[1].shape   # (2, 5) (batch, last_state)



# torch RNN ------------------------------------------------------------------

# x = np.random.random((2,3,4)).astype('float32')
# x.shape      # (2,3,4) (batch, seq, features) (batch, len_sentence, voca_embedding)

# w1 = np.random.random((4,5))
# w2 = np.random.random((5,5))
# w3 = np.zeros(5)

# w1 = np.arange(20).reshape(4,5)/20
# w2 = np.arange(25).reshape(5,5)/25
# w3 = np.zeros(5)

rnn2 = torch.nn.RNN(4, 5)  # (2,3,4) (batch, seq, features) (batch, len_sentence, voca_embedding)
rnn2_1 = torch.nn.RNN(4, 5,  batch_first=True)  # (3,2,4) (seq, batch, features) (len_sentence, batch, voca_embedding)
rnn2_2 = torch.nn.RNN(4, 5,  batch_first=True)  # (3,2,4) (seq, batch, features) (len_sentence, batch, voca_embedding)

rnn3 = torch.nn.RNN(4, 5,  batch_first=True, bidirectional=True)  # (3,2,4) (seq, batch, features) (len_sentence, batch, voca_embedding)
rnn3 = torch.nn.RNN(4, 5,  batch_first=True, bidirectional=True)  # (3,2,4) (seq, batch, features) (len_sentence, batch, voca_embedding)



# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) 
#           instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
#           See the Inputs/Outputs sections below for details. Default: False
t_x = torch.tensor(x, dtype=torch.float32)
t_x_tp = torch.tensor(x.transpose(1,0,2), dtype=torch.float32)

# mono0direcional
set_params=  [torch.tensor(w1.T, requires_grad=True),
            torch.tensor(w2.T, requires_grad=True),
            torch.tensor(w3, requires_grad=True),
            torch.tensor(w3, requires_grad=True),
            ]

state_dict = rnn2_1.state_dict()
state_dict2 = rnn2_2.state_dict()
for k, vn in zip(state_dict, set_params):
    state_dict[k] = vn
    state_dict2[k] = -vn

rnn2.load_state_dict(state_dict)
rnn2_1.load_state_dict(state_dict)
rnn2_2.load_state_dict(state_dict2)

# bi0directional
set_params3=  set_params + [-p for p in set_params]

state_dict3 = rnn3.state_dict()
for k, vn in zip(state_dict3, set_params3):
    state_dict3[k] = vn
state_dict3
rnn3.load_state_dict(state_dict3)


r2 = rnn2(torch.Tensor(x))
r21 = rnn2_1(torch.Tensor(x))
r22 = rnn2_2(torch.Tensor(x[:,::-1,:].copy()))
r31 = rnn3(torch.Tensor(x))


# B: Batch, S: Sequence, E: Embedding, D: Bidirectional, H: Hidden, L: num_layers
# *(rnn2) batch_first = False
x.shape     # (S,B,E) (2,3,4)
r2[0]       # (S, B, D∗H) (2,3,5)
r2[1]       # (D*L, B, H) (1,3,5)

# *(rnn2_1) batch_first = True
x.shape     # (B,S,E) (2,3,4)
r21[0]      # (B, S, D∗H) (2,3,5)
r21[1]      # (D*L, B, H) (1,2,5)

# *(rnn2_2) batch_first = False, reverse_input
x.shape     # (B,S,E) (2,3,4)
r22[0]      # (B, S, D∗H) (2,3,5)
r22[1]      # (D*L, B, H) (1,2,5)

# *(rnn3) batch_first = False, bidirectional=True
x.shape     # (B,S,E) (2,3,4)
r31[0]      # (B, S, D∗H) (2,3,10)    
#  . forward:  r31[0][:,:,:5] == r21[0] 
#  . backward: r31[0][:,:,5:] == r22[0][:,::-1,:]
r31[1]      # (D*L, B, H) (2,2,5)
#   . forward:  r31[1][0,:,:] == r21[1]
#   . backward: r31[1][1,:,:] == r22[1]
#   [ r31[0][:,-1,:5],  # (batch, last, forward)    == r21[1]
#     r31[0][:,0,5:] ]  # (batch, first, backward)  == r22[1]



# (einsum rnn) --------------------------------------------------
# c2_1 = np.tanh(np.einsum('se,eh->sh',x[0,:,:],w1)+w3@w2)
# c2_2 = np.tanh(np.einsum('se,eh->sh',x[1,:,:],w1)+c2_1@w2)

# c21_1 = np.tanh(np.einsum('se,eh->sh',x[:,0,:],w1)+w3@w2)
# c21_2 = np.tanh(np.einsum('se,eh->sh',x[:,1,:],w1)+c21_1@w2)
# c21_3 = np.tanh(np.einsum('se,eh->sh',x[:,2,:],w1)+c21_2@w2)

# c22_1 = np.tanh(np.einsum('se,eh->sh',x[:,2,:],w1)+w3@w2)
# c22_2 = np.tanh(np.einsum('se,eh->sh',x[:,1,:],w1)+c22_1@w2)
# c22_3 = np.tanh(np.einsum('se,eh->sh',x[:,0,:],w1)+c22_2@w2)
# ----------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
####################################################################################################






# Tensorflow.keras RNN ------------------------------------------------------------------
# tf.keras.metrics.CategoricalAccuracy?
# tf.keras.metrics.SparseCategoricalAccuracy
x = np.array([1,2])[np.newaxis,...].astype('float32')
x.shape

embed = tf.keras.layers.Embedding(input_dim=10, output_dim=4)

# return_sequences: False: 아웃풋 시퀀스의 마지막 아웃풋을 반환  / True: 시퀀스 전체를 반환할지 여부.
# return_state: False : 아웃풋에 더해 마지막 hidden_state도 미반환 / True : 아웃풋에 더해 마지막 hidden_state도 반환 
# 일반적으로 output = final_hidden_state
rnn1 = tf.keras.layers.SimpleRNN(1)
rnn1_1 = tf.keras.layers.SimpleRNN(1, return_sequences=False, return_state=False)
rnn1_2 = tf.keras.layers.SimpleRNN(1, return_sequences=True, return_state=True)
rnn1_3 = tf.keras.layers.SimpleRNN(1, return_sequences=False, return_state=True)
rnn1_4 = tf.keras.layers.SimpleRNN(1, return_sequences=True, return_state=False)


rnn1(x[..., np.newaxis])
rnn1_1(x[..., np.newaxis])
#   output [last_state]

rnn1_2(x[..., np.newaxis])  # return seq / state
#   output [seq_s, last_state]

rnn1_3(x[..., np.newaxis])  # return state
#   output [last_seq, last_state]

rnn1_4(x[..., np.newaxis])  # return seq
#   output [seq_s]


#####
rnn2 = tf.keras.layers.SimpleRNN(2)
rnn2_1 = tf.keras.layers.SimpleRNN(2, return_sequences=False, return_state=False)
rnn2_2 = tf.keras.layers.SimpleRNN(2, return_sequences=True, return_state=True)
rnn2_3 = tf.keras.layers.SimpleRNN(2, return_sequences=False, return_state=True)
rnn2_4 = tf.keras.layers.SimpleRNN(2, return_sequences=True, return_state=False)

rnn2(x[..., np.newaxis])
rnn2_1(x[..., np.newaxis])
#   output [last_state]

rnn2_2(x[..., np.newaxis])  # return seq / state
#   output [seq_s, last_state]

rnn2_3(x[..., np.newaxis])
#   output [last_seq, last_state]

rnn2_4(x[..., np.newaxis])  # return seq
#   output [seq_s]


# https://medium.com/@sanjivgautamofficial/lstm-in-keras-56a59264c0b2
# return_sequences: False: 아웃풋 시퀀스의 마지막 아웃풋을 반환  / True: 시퀀스 전체를 반환할지 여부.
# return_state: False : 아웃풋에 더해 마지막 hidden_state 및 cell_state 미반환 / True : 아웃풋에 더해 hidden_state 및 cell_state 반환 
# 일반적으로 output = final_hidden_state
lstm1 = tf.keras.layers.LSTM(1)
lstm1_1 = tf.keras.layers.LSTM(1, return_sequences=False, return_state=False)
lstm1_2 = tf.keras.layers.LSTM(1, return_sequences=True, return_state=True)
lstm1_3 = tf.keras.layers.LSTM(1, return_sequences=False, return_state=True)
lstm1_4 = tf.keras.layers.LSTM(1, return_sequences=True, return_state=False)

lstm2_1 = tf.keras.layers.LSTM(2, return_sequences=True, return_state=True)

lstm1(x[..., np.newaxis])
lstm1_1(x[..., np.newaxis])
#   output [final_memory_state]

lstm1_2(x[..., np.newaxis])  # return seq / state
#   output [whole_seq_output, final_memory_state, final_carry_state]

lstm1_3(x[..., np.newaxis])  # return state
#   output [last_seq_output, final_memory_state, final_carry_state]

lstm1_4(x[..., np.newaxis])  # return seq
#   output [whole_seq_output]

lstm2_1(x[..., np.newaxis])  # return seq
#   output [whole_seq_output, final_memory_state, final_carry_state]

############################################################################################################


emb1 = embed(x)
# lstm1 = lstm(emb1)
# lstm(emb1)

rnn1(x[..., np.newaxis])
rnn1_1(x[..., np.newaxis])
rnn1_2(x[..., np.newaxis])
rnn1_3(x[..., np.newaxis])

rnn2(x[..., np.newaxis])
rnn2_1(x[..., np.newaxis])
rnn2_2(x[..., np.newaxis])
rnn2_3(x[..., np.newaxis])

lstm1(x[..., np.newaxis])
lstm1_1(x[..., np.newaxis])
lstm1_2(x[..., np.newaxis])
lstm1_3(x[..., np.newaxis])

lstm2_1(x[..., np.newaxis])
# tf.keras.layers.UpSampling2D?
x[..., np.newaxis]

# --------------------------------------------------------------------------------------







### LSTM Tensorflow vs Torch ####################################################################
# params 확인
x = np.random.random((2,3,4)).astype('float32')
w1 = np.random.random((4,4))
w2 = np.random.random((1,4))
w3 = np.zeros(4)
for k,v in l1.state_dict().items():
    print(v.shape)



# tensorflow
l2 = tf.keras.layers.LSTM(1, return_state=True, return_sequences=True)

# set weights
l2.build(input_shape=(1,1,4))
l2.set_weights([tf.constant(w1),tf.constant(w2),tf.constant(w3)])
# l2.weights

# predict
r2 = l2(tf.constant(x))

r2      # return seq / (state / cell)
r2[0]
r2[1]
r2[2]




l2.weights[0].shape
l2.weights[1].shape
l2.weights[2].shape

# torch **
l1 = torch.nn.LSTM(4, 1, batch_first=True)

# set weights
set_params=  [torch.tensor(w1.T, requires_grad=True),
            torch.tensor(w2.T, requires_grad=True),
            torch.tensor(w3, requires_grad=True),
            torch.tensor(w3, requires_grad=True),
            ]
state_dict = l1.state_dict()
for k, vn in zip(state_dict, set_params):
    state_dict[k] = vn
l1.load_state_dict(state_dict)

# predict
r1 = l1(torch.Tensor(x))
r1      # return seq / (state / cell)
r1[0]
r1[1]

############################################################################################################




















































# Tensorflow.keras Sequential Time-Series RNN ------------------------------------------------------------------
# Rain_Tommorow
dict_data ={'Date': ['20081201', '20081202', '20081203', '20081204', '20081205',
            '20081206', '20081207', '20081208', '20081209', '20081210',
            '20081211', '20081212', '20081213', '20081214', '20081217', 
            '20081218', '20081219', '20081220', '20081221', '20081222'],
            'Humidity': [22, 25, 30, 16, 33, 23, 19, 19,  9, 27, 22, 91, 93, 43, 82, 65, 32, 26, 28, 28],
            'Rainfall': [ 0.6,  0. ,  0. ,  0. ,  1. ,  0.2,  0. ,  0. ,  0. ,  1.4,  0. ,
                        2.2, 15.6,  3.6,  0. , 16.8, 10.6,  0. ,  0. , 0.],
            'Pressure': [1007.1, 1007.8, 1008.7, 1012.8, 1006. , 1005.4, 1008.2, 1010.1, 1003.6, 1005.7,
                        1008.7, 1004.2, 993. , 1001.8, 1010.4, 1002.2,  1009.7, 1017.1, 1014.8, 1008.1],
            'Temp': [21.8, 24.3, 23.2, 26.5, 29.7, 28.9, 24.6, 25.5, 30.2, 28.2, 28.8,
                    17. , 15.8, 19.8, 18.1, 21.5, 21. , 23.2, 27.3, 31.6],
            'RainToday': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
            'RainTomorrow': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]}

df = pd.DataFrame(dict_data)
df

X_cols = ['Humidity', 'Rainfall', 'Pressure', 'Temp', 'RainToday']
y_cols = ['RainTomorrow']

X_np = df[X_cols].to_numpy().astype('float32')[np.newaxis, ...]
y_np = df[y_cols].to_numpy().astype('int32')[np.newaxis, ...]
# y_np = df[y_cols].to_numpy().astype('int32').flatten()[np.newaxis, ...]
X_np
y_np
print(X_np.shape, y_np.shape)


class RNN_Weather(tf.keras.Model):
    def __init__(self):
        super(RNN_Weather, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(units=10, return_sequences=True)
        # self.dense = tf.keras.layers.Dense(2, activation='softmax')
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.rnn(x)
        x = self.dense(x)
        # x = tf.reshape(tf.math.sigmoid(tf.reduce_sum(x, axis=-1)), (-1,1))
        return x

rnn_w = RNN_Weather()

rnn_w.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# rnn_w.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn_w.fit(X_np, y_np, epochs=50, verbose=2)
rnn_w.evaluate(X_np, y_np)

# rnn_w(X_np[0, 3][np.newaxis,np.newaxis,...]).numpy()
# rnn_w.predict(X_np[0, 3][np.newaxis,np.newaxis,...])    # predict



# --------------------------------------------------------------------------------------

























# Tensorflow.keras Sequential RNN Generator ------------------------------------------------------------------
# Dinosaurs Example ------------------------------------------------------------------------------
# training : many to many
# predict : one to may
# https://www.nhm.ac.uk/discover/dino-directory/name/name-az-all.html

dino_names = []
with open(absolute_path + '/dinosaurs.txt') as f:
    for line in f:
        dino_names.append(line.strip())
np.array(dino_names)
ord('A')        # ASCII Code

# to_ascii
names_ascii = []
for dino_name in dino_names:
    names_ascii.append(list(map(ord, dino_name)))

# to_numpy
inputs = np.array(names_ascii) - 64  # A: 65
inputs.shape

X = inputs[:,:-1]
y = inputs[:,1:]

print(X.shape, y.shape)
X[0], y[0]


# RNN Modeling
class DinoRNN(tf.keras.Model):
    def __init__(self):
        super(DinoRNN, self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim=27, output_dim=16)
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(27, activation='softmax')
    
    def call(self, x, training=False, print_shape=False):
        embed = self.embed(x)
        lstm, h1, h2 = self.lstm(embed)
        output = self.dense(lstm)

        if print_shape:
            print(f'input_shape: {x.shape}')
            print(f'embed_shape: {embed.shape}')
            print(f'lstm_shape: {lstm.shape}')
            print(f'output_shape: {output.shape}')
        return output

    def predict(self, x):
        begin = True
        
        predict_dino = []
        n = 0
        while n < 100:
            predict_dino.append(chr(x+64))
            x = np.array(x).reshape(1, 1)  # [[num]]

            x = self.embed(x)
            if begin:
                x, h1, h2 = self.lstm(x)
                begin = False
            else:
                x, h1, h2 = self.lstm(x, initial_state=(h1, h2))
            x = self.dense(x)
            # x = np.argmax(x, axis=-1)             # 확정적 이름
            x = np.random.choice(27, p=x.numpy().reshape(-1))    # 확률적 이름
            if x == 0:
                break
            n += 1
        return ''.join(predict_dino)

# chr(np.array([[64]]))
model = DinoRNN()
model(X[:10]).shape
forward = model(X[:10], print_shape=True)
# model.predict(1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, batch_size=128, epochs=500, verbose=2)

model.predict(0)
model.predict(1)
model.predict(2)
model.predict(3)
model.predict(4)
model.predict(5)
model.predict(6)

# input_shape: (10, 23)
# embed_shape: (10, 23, 16)
# rnn_shape: (10, 23, 32)
# output_shape: (10, 23, 27)










# (pseudocode)
# hidden_state_t = 0 # 초기 은닉 상태를 0(벡터)로 초기화
# for input_t in input_length: # 각 시점마다 입력을 받는다.
#     output_t = tanh(input_t, hidden_state_t) # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
#     hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉 상태가 된다.

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.

# --------------------------------------------------------------------------------------
