import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf

# Data Load ------------------------------------------------------------------------------------------
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'
# url_path = r'C:\Users\Admin\Desktop\DataBase'
df = pd.read_csv(f"{url_path}/NLP_EN_to_KR_0_Data.csv", encoding='utf-8-sig')
# path = r'C:\Users\Admin\Desktop\DataBase'
# df01 = pd.read_csv(f"{path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
# df02 = pd.read_csv(f"{path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')
# df = pd.concat([df02, df01],axis=0).reset_index(drop=True)


df.sample(6)
print(df.shape)

# Preprocessing --------------------------------------------------------------------------------------
from DS_NLP import NLP_Preprocessor

processor_en = NLP_Preprocessor(df['english'])
processor_en.replace().fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()

# processor_en.replace().word_prob()
# processor_en.word_prob_dict
# processor_en.word_cum_prob

processor_en.texts          # transformed data
processor_en.vocab_size     # vocab_size
processor_en.word_index     # word_index
processor_en.index_word     # index_word
processor_en.sequences_to_texts(processor_en.texts, join=' ')   # inverse_transform

print(f"y_data_shape: {processor_en.texts.shape}, y_vocab_size: {processor_en.vocab_size}" )
vocab_size_y = processor_en.vocab_size
index_word_y = processor_en.index_word
padseq_y = processor_en.texts



from konlpy.tag import Okt
okt = Okt()
processor_kr = NLP_Preprocessor(df['korean'])
processor_kr.replace().morphs_split(morphs=okt, stem=True)
processor_kr.fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()

processor_kr.texts          # transformed data
processor_kr.vocab_size     # vocab_size
processor_kr.word_index     # word_index
processor_kr.index_word     # index_word
processor_kr.sequences_to_texts(processor_kr.texts, join=' ')   # inverse_transform

print(f"X_data_shape: {processor_kr.texts.shape}, X_vocab_size: {processor_kr.vocab_size}" )
vocab_size_X = processor_kr.vocab_size
index_word_X = processor_kr.index_word
padseq_X = processor_kr.texts


# # Save Data ----------------------------------------------------------------------------------------------------------------------------
# data_base_url = r''

# index_word_y = pd.Series(processor_en.index_word).to_frame()
# index_word_y.index.name = 'index'
# index_word_y.columns = ['word']

# pd.DataFrame(processor_en.texts).to_csv(f"{data_base_url}/NLP_EN_to_KR_0_pad_seq_sentences(EN).csv", encoding='utf-8-sig', index=False)
# index_word_y.to_csv(f"{data_base_url}/NLP_EN_to_KR_0_index_word(EN).csv", encoding='utf-8-sig')


# index_word_X = pd.Series(processor_kr.index_word).to_frame()
# index_word_X.index.name = 'index'
# index_word_X.columns = ['word']

# pd.DataFrame(processor_kr.texts).to_csv(f"{data_base_url}/NLP_EN_to_KR_0_pad_seq_sentences(KR).csv", encoding='utf-8-sig', index=False)
# index_word_X.to_csv(f"{data_base_url}/NLP_EN_to_KR_0_index_word(KR).csv", encoding='utf-8-sig')
# ----------------------------------------------------------------------------------------------------------------------------------------


#####################################################################################################
# Load Preprocessed  Data from GIT ##################################################################
#####################################################################################################
# Read_from_csv *** ---------------------------------------------------------------------------------
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf

max_len = None
# max_len = 1000
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'


index_word_y = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_index_word(EN).csv', index_col='index', encoding='utf-8-sig')['word'])
index_word_X = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_index_word(KR).csv', index_col='index', encoding='utf-8-sig')['word'])
padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_pad_seq_sentences(EN).csv', encoding='utf-8-sig').to_numpy()
padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_pad_seq_sentences(KR).csv', encoding='utf-8-sig').to_numpy()


# index_word_y = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_1_index_word(EN).csv', index_col='index', encoding='utf-8-sig')['word'])
# index_word_X = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_1_index_word(KR).csv', index_col='index', encoding='utf-8-sig')['word'])
# padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR_1_pad_seq_sentences(EN).csv', encoding='utf-8-sig').to_numpy()
# padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR_1_pad_seq_sentences(KR).csv', encoding='utf-8-sig').to_numpy()

# index_word_y = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_2_index_word(EN).csv', index_col='index', encoding='utf-8-sig')['word'])
# index_word_X = dict(pd.read_csv(f'{url_path}/NLP_EN_to_KR_2_index_word(KR).csv', index_col='index', encoding='utf-8-sig')['word'])
# padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR_2_pad_seq_sentences(EN).csv', encoding='utf-8-sig').to_numpy()
# padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR_2_pad_seq_sentences(KR).csv', encoding='utf-8-sig').to_numpy()

vocab_size_y = len(index_word_y) + 1 #어휘수
vocab_size_X = len(index_word_X) + 1 #어휘수

y = padseq_y[:max_len]
X = padseq_X[:max_len]

print(f"vocab_size: {vocab_size_y}, {vocab_size_X}")
print(f"data_size: {y.shape}, {X.shape}")
# X_oh = tf.keras.utils.to_categorical(X, vocab_size_X)
# y_oh = tf.keras.utils.to_categorical(y, vocab_size_y)
################################################################################################




# (Train_Test_Split) -------------------------------------------
from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(range(len(X)), test_size=0.05, random_state=0)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.1, random_state=0)

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

# Seq2Seq -----------------------------------------------------------------------------------------------------------------------------
class Seq2Seq_Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_X, 256)
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True)
        self.rnn = torch.nn.LSTM(256, 512, batch_first=True)
    
    def forward(self, X):
        # X (batch_seq, X_word)
        self.enc_emb = self.embed(X)    # enc_emb (batch_seq, X_word, enc_emb)
        self.emb_dropout = self.dropout(self.enc_emb)
        self.enc_output, self.enc_hidden = self.rnn(self.emb_dropout)   # rnn_output (batch_seq, X_word, enc_rnn), rnn_hidden (1, batch_seq, enc_rnn)
        return self.enc_output, self.enc_hidden

class Seq2Seq_Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_y, 256)
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True)
        self.rnn = torch.nn.LSTM(256, 512, batch_first=True)
        self.fc = torch.nn.Linear(512, vocab_size_y)

    def forward(self, y_before, context_vector):
        # y_before (batch_seq, y_word)
        self.dec_emb = self.embed(y_before)    # dec_emb (batch_seq, y_word, dec_emb)
        self.emb_dropout = self.dropout(self.dec_emb)
        self.dec_output, self.dec_hidden = self.rnn(self.emb_dropout, context_vector)   # rnn_output (batch_seq, y_word, dec_rnn), rnn_hidden (1, batch_seq, dec_rnn)
        self.dec_fc = self.fc(self.dec_output)   # fc_output (batch_seq, y_word, dec_fc)
        return self.dec_fc, self.dec_hidden

class Seq2Seq(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y):
        super().__init__()
        self.vocab_size_X = vocab_size_X
        self.vocab_size_y = vocab_size_y

        self.encoder = Seq2Seq_Encoder(vocab_size_X)
        self.decoder = Seq2Seq_Decoder(vocab_size_y)

    def forward(self, X, y=None, teacher_forcing=0):
        # X (batch_seq, X_word)
        # y (batch_seq, y_word)
        if y is not None:
            with torch.no_grad():
                self.y_shape = y.shape
                self.init = np.array(y[0,0].to('cpu').detach()) # 학습시 초기값 저장

        # (encoding) --------------------------------------------------------------------------------------------
        self.enc_output, self.context_vector = self.encoder(X)
        #       enc_output (batch_seq, X_word, enc_rnn), context_vector (1, batch_seq, enc_rnn)

        # (decoding) --------------------------------------------------------------------------------------------
        if y is not None:
            y_before = y[:,0][:,None]     # y[:,0].unsqueeze(1)   # y_before (batch_seq, 1)
        else:
            y_before = torch.tensor(np.ones((X.shape[0],1))*self.init, dtype=torch.int64).to(X.device)  # 저장된 초기값을 예측시 활용

        # self.result, hidden_input = self.decoder(y_before, self.enc_output, self.context_vector)     # result (batch_seq, 1, dec_fc==vocab_size_y)
        self.result = torch.tensor(np.zeros((X.shape[0], 1, self.vocab_size_y)), dtype=torch.float64).to(X.device)  # 저장된 초기값을 예측시 활용
        hidden_input = self.context_vector

        for i in range(1, self.y_shape[1]):
            pred_output, dec_hidden = self.decoder(y_before, hidden_input) # (batch_seq, 1, dec_fc==vocab_size_y)
            hidden_input = dec_hidden

            self.result = torch.cat([self.result, pred_output],axis=1)  # (batch_seq, i->y_word, dec_fc==vocab_size_y)

            if teacher_forcing >= np.random.rand():     # teacher_forcing
                y_before = y[:,i][:,None] # y_before (batch_seq, 1)
            else:
                y_before = torch.argmax(pred_output, axis=2)  # y_before (batch_seq, 1)
        # ------------------------------------------------------------------------------------------------------
        return self.result      # (y_seq, y_word, dec_fc==vocab_size_y)

    def predict(self, X, return_word=True):
        with torch.no_grad():
            y_pred = self.forward(X, teacher_forcing=0)
        if return_word:
            return torch.argmax(y_pred, axis=2)
        else:
            return y_pred
# ------------------------------------------------------------------------------------------------------------------------------------


# # customize library ***---------------------
# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
# from DS_DeepLearning import EarlyStopping

# es = EarlyStopping()    
# # ------------------------------------------
import time


# training prepare * -------------------------------------------------------------------------------------------------------
# X_sample.shape, y_sample.shape

model = Seq2Seq(vocab_size_X, vocab_size_y).to(device)
# model(X_sample.to(device), y_sample.to(device))


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
epochs = 30


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
        pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict

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
        print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')

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

# loss graph * ------------------------------------------------------------------------------------------------------
plt.figure()
plt.title('Loss of Seq2Seq Model ')
plt.plot(train_losses, label='train_loss', marker='o')
plt.plot(valid_losses, label='valid_loss', marker='o')
plt.xlabel('epochs')
plt.ylabel('cross_entropy_loss')
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------------------


# predict * -------------------------------------------------------------------------------------------------------
idx = 100
sentence_input = train_X[[idx],:]
sentence_output = train_y[[idx],:]

with torch.no_grad():
    model.eval()
    pred_sentence = model.predict(torch.tensor(sentence_input).to(device))
    # pred_sentence = model(torch.tensor(sentence_input).to(device), torch.tensor(sentence_output).to(device))
pred_sentence

sentence_en = np.stack([[index_word_X[word] if word != 0 else '' for word in sentence] for sentence in sentence_input])[0]
sentence_kr_real = np.stack([[index_word_y[word] if word != 0 else '' for word in sentence] for sentence in sentence_output])[0]
sentence_kr_pred = np.stack([[index_word_y[word] if word != 0 else '' for word in sentence] for sentence in pred_sentence.to('cpu').detach().numpy()])[0]

print(sentence_kr_real)
print(sentence_kr_pred)
# ------------------------------------------------------------------------------------------------------------------



# Performance -------------------------------------------------------------------------------------------------------
# pip install torchmetrics
# pip install nltk
# pip install torchtext
import nltk.translate.bleu_score as bleu
# from torchtext.data.metrics import bleu_score
# bleu.sentence_bleu(references_corpus, candidate_corpus)

bleu_scores = []
for r, c in zip(batch_y.unsqueeze(1), torch.argmax(pred, axis=2).to('cpu').detach()):
    bleu_scores.append(bleu.sentence_bleu(r,c))
np.mean(bleu_scores)
# ------------------------------------------------------------------------------------------------------------------




