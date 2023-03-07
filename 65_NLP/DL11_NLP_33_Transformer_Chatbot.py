
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf

from konlpy.tag import Okt


####################################################################################################
database_path = r'C:\Users\Admin\Desktop\DataBase'

dataset_file = f'{database_path}/chatbot_data.csv' # acquired from 'http://www.aihub.or.kr' and modified
okt = Okt()

with open(dataset_file, 'r', encoding='utf-8-sig') as file:
    lines = file.readlines()
    seq = [' '.join(okt.morphs(line)) for line in lines]

questions = seq[::2]
answers = ['\t ' + lines for lines in seq[1::2]]

q_ = [q.replace('\n','').strip() for q in questions]
a_ = [a.replace('\n','').replace('\t','').strip() for a in answers]

print(len(q_), len(a_))

####################################################################################################
import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_NLP import NLP_Preprocessor

# Question processor ***
processor_q = NLP_Preprocessor(q_ + a_)
processor_q.replace().morphs_split(morphs=okt)
processor_q.fit_on_texts().texts_to_sequences(q_).add_sos_eos().pad_sequences()

processor_q.texts          # transformed data
processor_q.vocab_size     # vocab_size
processor_q.word_index     # word_index
processor_q.index_word     # index_word
processor_q.sequences_to_texts(processor_kr.texts, join=' ')   # inverse_transform

# processor_q.texts.shape

# Answer processor ***
processor_a = NLP_Preprocessor(q_ + a_)
processor_a.replace().morphs_split(morphs=okt)
processor_a.fit_on_texts().texts_to_sequences(a_).add_sos_eos().pad_sequences()

processor_a.texts          # transformed data
processor_a.vocab_size     # vocab_size
processor_a.word_index     # word_index
processor_a.index_word     # index_word
processor_a.sequences_to_texts(processor_kr.texts, join=' ')   # inverse_transform
# processor_a.texts.shape



print(f"Q_data_shape: {processor_q.texts.shape}, Q_vocab_size: {processor_q.vocab_size}" )
print(f"A_data_shape: {processor_a.texts.shape}, A_vocab_size: {processor_a.vocab_size}" )

X = processor_q.texts
y = processor_a.texts

vocab_size_X = processor_q.vocab_size
vocab_size_y = processor_a.vocab_size


####################################################################################################
from sklearn.model_selection import train_test_split

train_valid_idx, test_idx = train_test_split(range(len(processor_q.texts)), test_size=0.15, random_state=0)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.1, random_state=0)

print(len(train_idx), len(valid_idx), len(test_idx))

train_X, valid_X, test_X = X[train_idx,:], X[valid_idx,:], X[test_idx,:]
train_y, valid_y, test_y = y[train_idx,:], y[valid_idx,:], y[test_idx,:]


print(train_X.shape, valid_X.shape, test_X.shape)
print(train_y.shape, valid_y.shape, test_y.shape)


# torch dataset ----------
train_X_torch = torch.tensor(train_X)
valid_X_torch = torch.tensor(valid_X)
test_X_torch = torch.tensor(test_X)

train_y_torch = torch.tensor(train_y)
valid_y_torch = torch.tensor(valid_y)
test_y_torch = torch.tensor(test_y)

print(train_X_torch.shape, valid_X_torch.shape, test_X_torch.shape)
print(train_y_torch.shape, valid_y_torch.shape, test_y_torch.shape)


train_dataset = torch.utils.data.TensorDataset(train_X_torch, train_y_torch)
valid_dataset = torch.utils.data.TensorDataset(valid_X_torch, valid_y_torch)
test_dataset = torch.utils.data.TensorDataset(test_X_torch, test_y_torch)

batch_size=64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)




# Sample -----------------------------
# X_sample = train_X[:3]
# y_sample = test_y[:3]
# y_sample_oh = test_y_oh[:3]
X_sample = torch.tensor(train_X[:3])
y_sample = torch.tensor(test_y[:3])

print(X_sample.shape, y_sample.shape)

X_sample
y_sample
# ------------------------------------


##########모델 생성
# https://deep-learning-study.tistory.com/686
# https://americanoisice.tistory.com/53
import torch


# use_cuda = False
#  if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
# torch.cuda.empty_cache()

import numpy as np


# # customize library ***---------------------
import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=100)
# # ------------------------------------------
import time


# training prepare * -------------------------------------------------------------------------------------------------------
# X_sample.shape, y_sample.shape

model = Transformer(vocab_size_X, vocab_size_y, 0, 0).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0, n_layers=3, n_heads=8, dropout=0.5, pos_maxlen=150).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0, n_layers=2, n_heads=8, dropout=0.5, pos_maxlen=150).to(device)
# pred = model(X_sample.to(device), y_sample.to(device))




# model weights parameter initialize (가중치 초기화) ***
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)
model.apply(init_weights)

# trg_pad_idx = TRG.vocab.stoi[TRG.pad_token] ## pad에 해당하는 index는 무시합니다.

# loss_function = torch.nn.CrossEntropyLoss()     # ignore_index=trg_pad_idx
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())
epochs = 100

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()                   # wegiht initialize
        pred = model(batch_X.to(device), batch_y.to(device))                   # predict
        # pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict

        pred_eval = pred[:,:-1,:].reshape(-1, vocab_size_y)
        real_eval = batch_y[:,1:].reshape(-1).type(torch.int64).to(device)
        loss = loss_function(pred_eval, real_eval)     # loss
        loss.backward()                         # backward
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    # 기울기(gradient) clipping 진행
        # (gradient clipping) https://sanghyu.tistory.com/87
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )
    

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch_X, batch_y in valid_loader:
            pred = model(batch_X.to(device), batch_y.to(device))                   # predict
            # pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict

            pred_eval = pred[:,1:,:].reshape(-1, vocab_size_y)
            real_eval = batch_y[:,1:].reshape(-1).type(torch.int64).to(device)
            loss = loss_function(pred_eval, real_eval)     # loss
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )

    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')

        # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)
        if early_stop == 'break':
            break
        # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot

model.load_state_dict(es.optimum[2])    # optimum model (load weights)
# ------------------------------------------

pred_eval.shape
real_eval.shape