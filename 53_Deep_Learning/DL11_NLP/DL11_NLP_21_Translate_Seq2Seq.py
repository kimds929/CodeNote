# https://wikidocs.net/106259
# https://wikidocs.net/106254
# https://codetorial.net/tensorflow/natural_language_processing_in_tensorflow_01.html

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf

import datetime

# (Load Data) -------------------------------------------
# english_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.en', header=None)
# korean_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.ko', header=None)

# path = r'C:\Users\Admin\Desktop\DataBase'
# english_df.columns = ['english']
# english = english_df['english'].to_numpy()
# korean_df.columns = ['korean', 'etc']
# korean = korean_df['korean'].to_numpy()
# df = pd.concat([english_df, korean_df], axis=1).drop('etc',axis=1)

# df.to_csv(f"{path}/NLP_Translate_ENG_to_KOR.csv", index=False, encoding='utf-8-sig')


path = r'C:\Users\Admin\Desktop\DataBase'
df = pd.read_csv(f"{path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
# df = pd.read_csv(f"{path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')
df.head(6)


df1 = df.copy()
# df1 = df.iloc[:1000]


# (Preprocessing) -------------------------------------------
# english *
df1_en = df1['english']
df1_en = df1_en.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z ]","")
# df1_en = df1_en.str.replace('^ +', "")
df1_en = df1_en.apply(lambda x: np.nan if x =='' else x)
df1_en

# korean *
df1_kor = df1['korean']
df1_kor = df1_kor.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z]","")
# df1_kor = df1_kor.str.replace('^ +', "")
df1_kor = df1_kor.apply(lambda x: np.nan if x =='' else x)


# concat *
df2 = pd.concat([df1_en, df1_kor], axis=1).dropna()
print(df2.shape)

# (Tokenize) -----------------------------------------------------------------------
# pad_sequences *
# https://wikidocs.net/83544
# sample_seq= [[1,2,3], [1,2], [3]]
# tf.keras.preprocessing.sequence.pad_sequences(sample_seq, padding='post', maxlen=2, truncating='post')
#   . maxlen : (default) None                # 최대길이 지정
#   . dtype : (default) 'int32' / 'float32'  # data_type
#   . padding : (default) 'pre' / 'post'      # padding을 어디에 할 것인지?
#   . truncated : (default) 'pre' / 'post'      # maxlen때문에 데이터가 잘릴때 어느부분을 자를것인지?
# ----------------------------------------------------------------------------------

# english *
df2_en = df2['english']
tokenizer_en = tf.keras.preprocessing.text.Tokenizer() 
tokenizer_en.fit_on_texts(df2_en)
vocab_size_en = len(tokenizer_en.word_index) + 1 #어휘수
# tokenizer_en.word_index
# tokenizer_en.word_counts

# (text_to_sequence / pad_sequence) *
df2_en
seq_en = tokenizer_en.texts_to_sequences(df2_en)
padseq_en = tf.keras.preprocessing.sequence.pad_sequences(seq_en, padding='post')
print(padseq_en.shape)
# df_nlp2_e_inv = np.stack([['' if s ==0 else tokenizer_en.index_word[s] for s in sentence] for sentence in padseq_en])

# korean *
from konlpy.tag import Okt
okt = Okt()

df2_kor = df2['korean']
tokened_k = []
for sentence in df2_kor:
    sen_token = okt.morphs(sentence, stem=True)
    tokened_k.append(sen_token)

# okt.morphs(df2_kor[5], norm=False, stem=False)  
#   . norm: 문장을 정규화
#   . stem: 은 각 단어에서 어간을 추출하는 기능 (True: 동사의 원형을 찾아줌)
tokenizer_kor = tf.keras.preprocessing.text.Tokenizer()
tokenizer_kor.fit_on_texts(tokened_k)

len_k_tokens = len(tokenizer_kor.word_index)
tokenizer_kor.word_index['<SOS>'] =  len_k_tokens + 1
tokenizer_kor.word_index['<EOS>'] = len_k_tokens + 2
tokenizer_kor.index_word[tokenizer_kor.word_index['<SOS>']] = '<SOS>'
tokenizer_kor.index_word[tokenizer_kor.word_index['<EOS>']] = '<EOS>'
# tokenizer_kor.index_word = {v:k for k, v in tokenizer_kor.word_index.items()}
# list(tokenizer_kor.word_index.items())[-2:]
# list(tokenizer_kor.index_word.items())[-2:]
vocab_size_kor = len(tokenizer_kor.word_index) + 1 #어휘수


# (text_to_sequence / pad_sequence) *
seq_kor = tokenizer_kor.texts_to_sequences(tokened_k)

# SOS / EOS
# seq_kor_input = []
# seq_kor_output = []
seq_kor_inout = []
for sentence in seq_kor:
    # seq_kor_input.append([tokenizer_kor.word_index['<SOS>']] + sentence)
    # seq_kor_output.append(sentence + [tokenizer_kor.word_index['<EOS>']])
    seq_kor_inout.append([tokenizer_kor.word_index['<SOS>']] + sentence + [tokenizer_kor.word_index['<EOS>']])

# padseq_kor_input = tf.keras.preprocessing.sequence.pad_sequences(seq_kor_input, padding='post')
# padseq_kor_output = tf.keras.preprocessing.sequence.pad_sequences(seq_kor_output, padding='post')
padseq_kor_inout = tf.keras.preprocessing.sequence.pad_sequences(seq_kor_inout, padding='post')
# padseq_kor_inout_oh = tf.keras.utils.to_categorical(padseq_kor_inout, vocab_size_kor) #원핫 인코딩

# df_nlp2_k_inv_train = np.stack([['' if s ==0 else tokenizer_kor.index_word[s] for s in sentence] for sentence in padseq_kor_input])
# df_nlp2_k_inv_eval = np.stack([['' if s ==0 else tokenizer_kor.index_word[s] for s in sentence] for sentence in padseq_kor_output])
# df_nlp2_k_inv_train[0]
# df_nlp2_k_inv_eval[0]

# padseq_kor_input[0]
# padseq_kor_output[0]
print(padseq_kor_inout.shape)
# print(padseq_kor_input.shape, padseq_kor_output.shape, padseq_kor_inout.shape)



# class TokenizeTransformer():
#     def __init__(x, tokenizer, to_lang='kr'):
#         self.x = x
#         self.tokenizer = tokenizer
        
#     def tokenizer_en(x, tokenizer):
#         seq = tokenize.texts_to_sequences(x)
#         pad_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post')
    
#     def tokenizer_kr(x, tokenizer):
#         seq = tokenize.texts_to_sequences(x)
#         pad_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post')








################################################################################################
# Save_to_csv ***
word_index_X = pd.Series(tokenizer_en.word_index).reset_index()
word_index_y = pd.Series(tokenizer_kor.word_index).reset_index()
word_index_X.columns = ['word', 'index']
word_index_y.columns = ['word', 'index']

padseq_X = pd.DataFrame(padseq_en.copy())
padseq_y = pd.DataFrame(padseq_kor_inout.copy())

word_index_X.to_csv(f'{path}/NLP_EN_to_KR_word_index(EN).csv', index=False, encoding='utf-8-sig')
word_index_y.to_csv(f'{path}/NLP_EN_to_KR_word_index(KR).csv', index=False, encoding='utf-8-sig')
padseq_X.to_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
padseq_y.to_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

# word_index_X.to_csv(f'{path}/NLP_EN_to_KR2_word_index(EN).csv', index=False, encoding='utf-8-sig')
# word_index_y.to_csv(f'{path}/NLP_EN_to_KR2_word_index(KR).csv', index=False, encoding='utf-8-sig')
# padseq_X.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
# padseq_y.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

# Read_from_csv *** ---------------------------------------------------------------------------------
word_index_X = pd.read_csv(f'{path}/NLP_EN_to_KR_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
word_index_y = pd.read_csv(f'{path}/NLP_EN_to_KRword_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
padseq_X = pd.read_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
padseq_y = pd.read_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(KR).csv', encoding='utf-8-sig')

# word_index_X = pd.read_csv(f'{path}/NLP_EN_to_KR2_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
# word_index_y = pd.read_csv(f'{path}/NLP_EN_to_KR2_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
# padseq_X = pd.read_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
# padseq_y = pd.read_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', encoding='utf-8-sig')
# pd.Series(word_index_X.index, index=word_index_X)

vocab_size_X = len(word_index_X) + 1 #어휘수
vocab_size_y = len(word_index_y) + 1 #어휘수
X = padseq_X.to_numpy()
y = padseq_y.to_numpy()

X_oh = tf.keras.utils.to_categorical(X, vocab_size_X)
y_oh = tf.keras.utils.to_categorical(y, vocab_size_y)
################################################################################################



# (Train_Test_Split) -------------------------------------------
from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=0)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2, random_state=0)

train_X, valid_X, test_X = X[train_idx,:], X[valid_idx,:], X[test_idx,:]

train_y, valid_y, test_y = y[train_idx,:], y[valid_idx,:], y[test_idx,:]
train_y_oh, valid_y_oh, test_y_oh = y_oh[train_idx,:], y_oh[valid_idx,:], y_oh[test_idx,:]

print(train_X.shape, valid_X.shape, test_X.shape)
print(train_y.shape, valid_y.shape, test_y.shape)
print(train_y_oh.shape, valid_y_oh.shape, test_y_oh.shape)





# torch dataset ----------
train_X_torch = torch.tensor(train_X)
valid_X_torch = torch.tensor(valid_X)
test_X_torch = torch.tensor(test_X)

train_y_torch = torch.tensor(train_y)
valid_y_torch = torch.tensor(valid_y)
test_y_torch = torch.tensor(test_y)

train_y_oh_torch = torch.tensor(train_y_oh)
valid_y_oh_torch = torch.tensor(valid_y_oh)
test_y_oh_torch = torch.tensor(test_y_oh)

print(train_X_torch.shape, valid_X_torch.shape, test_X_torch.shape)
print(train_y_torch.shape, valid_y_torch.shape, test_y_torch.shape)
print(train_y_oh_torch.shape, valid_y_oh.shape, test_y_oh_torch.shape)


train_dataset = torch.utils.data.TensorDataset(train_X_torch, train_y_torch, train_y_oh_torch)
valid_dataset = torch.utils.data.TensorDataset(valid_X_torch, valid_y_torch, valid_y_oh_torch)
test_dataset = torch.utils.data.TensorDataset(test_X_torch, test_y_torch, test_y_oh_torch)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)




# Sample -----------------------------
# X_sample = train_X[:3]
# y_sample = test_y[:3]
# y_sample_oh = test_y_oh[:3]
X_sample = torch.tensor(train_X[:3])
y_sample = torch.tensor(test_y[:3])
y_oh_sample = torch.tensor(test_y_oh[:3])

print(X_sample.shape, y_sample.shape, y_oh_sample.shape)
X_sample
y_sample
y_oh_sample
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


# Seq2Seq ------- ---------------------------------------------------------------------------------------------------------------------
class Seq2Seq_Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_X, 256)
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True)
        self.rnn = torch.nn.LSTM(256, 512, batch_first=True)
    
    def forward(self, X):
        # X (seq, word)
        self.emb = self.embed(X)    # emb (seq, word, emb)
        self.emb_dropout = self.dropout(self.emb)
        self.rnn_output, self.rnn_hidden = self.rnn(self.emb_dropout)   # rnn_output (seq, word, rnn_layers), rnn_hidden (1, seq, rnn_layers)
        return self.rnn_output, self.rnn_hidden

class Seq2Seq_Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_y, 256)
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True)
        self.rnn = torch.nn.LSTM(256, 512, batch_first=True)
        self.fc = torch.nn.Linear(512, vocab_size_y)

    def forward(self, y, context_vector):
        # y (seq, word)
        self.emb = self.embed(y)    # emb (seq, word, emb)
        self.emb_dropout = self.dropout(self.emb)
        self.rnn_output, self.rnn_hidden = self.rnn(self.emb_dropout, context_vector)   # rnn_output (seq, word, rnn_layers), rnn_hidden (1, seq, rnn_layers)
        self.fc_output = self.fc(self.rnn_output)   # fc_output (seq, word, fc_layers)
        return self.fc_output

class Seq2Seq(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y):
        super().__init__()
        self.encoder = Seq2Seq_Encoder(vocab_size_X)
        self.decoder = Seq2Seq_Decoder(vocab_size_y)

    def forward(self, X, y=None, teacher_forcing=0):
        # X (seq, word)
        # y (seq, word)
        if y is not None:
            with torch.no_grad():
                self.y_shape = y.shape
                self.init = np.array(y[0,0].to('cpu').detach())

        # (encoding) --------------------------------------------------------------------------------------------
        self.enc_output, self.context_vector = self.encoder(X)
        #       enc_output (X_seq, X_word, X_rnn_layers), context_vector (1, X_seq, X_rnn_layers)

        # (decoding) --------------------------------------------------------------------------------------------
        if y is not None:
            pre_st = y[:,0][:,None]     # y[:,0].unsqueeze(1)   # pre_st (y_seq, 1)
        else:
            pre_st = torch.tensor(np.ones((X.shape[0],1))*self.init, dtype=torch.int64).to(X.device)
        self.result = self.decoder(pre_st, self.context_vector)     # result (y_seq, 1, fc_layers==vocab_size_y)

        for i in range(1, self.y_shape[1]):
            pred_output = self.decoder(pre_st, self.context_vector) # (y_seq, 1, fc_layers==vocab_size_y)
            
            self.result = torch.cat([self.result, pred_output],axis=1)  # (y_seq, i->y_word, fc_layers==vocab_size_y)

            if teacher_forcing >= np.random.rand():     # teacher_forcing
                pre_st = y[:,i][:,None] # pre_st (y_seq, 1)
            else:
                pre_st = torch.argmax(pred_output, axis=2)  # pre_st (y_seq, 1)
        # ------------------------------------------------------------------------------------------------------
        return self.result      # (y_seq, y_word, fc_layers==vocab_size_y)

    def predict(self, X, return_word=True):
        with torch.no_grad():
            y_pred = self.forward(X, teacher_forcing=0)
        if return_word:
            return torch.argmax(y_pred, axis=2)
        else:
            return y_pred
# ------------------------------------------------------------------------------------------------------------------------------------

# Seq2Seq + Attention
# https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_Attention_Tutorial.ipynb
class AttSeq2Seq_Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_X, 256)
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True, bidirectional=True)
        # self.rnn = torch.nn.LSTM(256, 512, batch_first=True, bidirectional=True)
        self.rnn = torch.nn.GRU(256, 512, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(512*2, 512)   # (enc_rnn_layers*2, fc_layers==dec_rnn_layers)
    
    def forward(self, X):
        # X (seq, X_word)
        self.emb = self.embed(X)    # emb (seq, X_word, emb)
        self.emb_dropout = self.dropout(self.emb)
        self.rnn_output, self.rnn_hidden = self.rnn(self.emb_dropout)   # rnn_output (seq, X_word, rnn_layers), rnn_hidden (1, seq, rnn_layers)
        self.rnn_concat = torch.cat([self.rnn_hidden[0,:,:], self.rnn_hidden[1,:,:]], axis=1)[None,...]  # rnn_concat (1, seq, rnn_layers * 2)
        # self.rnn_hidden[0,:,:] == self.rnn_output[:,-1,:512] : forward
        # self.rnn_hidden[1,:,:] == self.rnn_output[:,0,512:] : backward

        self.final_hidden = torch.tanh(self.fc(self.rnn_concat))       # final_hidden (1, seq, fc_layers==dec_rnn_layers)
        
        return self.rnn_output, self.final_hidden

class AttSeq2Seq_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_fc = torch.nn.Linear((512 * 2) + 512, 512)     # (enc_rnn_layer *2) + dec_hidden, dec_hidden
        self.att_fc = torch.nn.Linear(512, 1)     # dec_hidden, 1
    
    def forward(self, enc_output, dec_hidden):
        # enc_output (seq, X_word, enc_rnn_layers)
        # dec_hidden (1, seq, enc_final_hidden or dec_rnn_layers)
        with torch.no_grad():
            self.enc_output_shape = enc_output.shape
        
        self.dec_hidden_transform = dec_hidden.permute(1,0,2).repeat(1, self.enc_output_shape[1], 1)    # dec_hidden_transform (seq, X_word(repeat), dec_rnn_layers)
        self.att_concat = torch.cat([enc_output, self.dec_hidden_transform], axis=2)        # att_concat (seq, X_word, enc_rnn_layers + dec_rnn_layers)
        self.energy = torch.tanh(self.energy_fc(self.att_concat))   # energy (seq, X_word, dec_hidden)
        self.attention = self.att_fc(self.energy).squeeze(2)   # attention (seq, X_word, 1) → (seq, X_word)
        self.att_score = torch.nn.functional.softmax(self.attention, dim=1)  # att_score (seq, X_word)
        return self.att_score

enc = AttSeq2Seq_Encoder(vocab_size_X)
enc_result = enc(X_sample)
att = AttSeq2Seq_Attention()


# torch.cuda.empty_cache()  # 메모리 비우기

enc()[0].shape
enc(X_sample)[1].shape

enc(X_sample)[1].permute(1,0,2).repeat(1, enc(X_sample)[0].shape[1],1).shape


# training prepare * -------------------------------------------------------------------------------------------------------
model = Seq2Seq(vocab_size_en, vocab_size_kor).to(device)

# trg_pad_idx = TRG.vocab.stoi[TRG.pad_token] ## pad에 해당하는 index는 무시합니다.
loss_function = torch.nn.CrossEntropyLoss()     # ignore_index=trg_pad_idx
optimizer = torch.optim.Adam(model.parameters())
epochs = 50

# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch_X, batch_y, batch_y_oh in train_loader:
        optimizer.zero_grad()                   # wegiht initialize
        pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict
        loss = loss_function(pred, batch_y_oh.to(device))     # loss
        loss.backward()                         # backward
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )
    train_losses.append(np.mean(train_epoch_loss))

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch_X, batch_y, batch_y_oh in valid_loader:
            pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict
            loss = loss_function(pred, batch_y_oh.to(device))     # loss
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )
    valid_losses.append(np.mean(valid_epoch_loss))
    print(f"{e+1} epochs) train_loss: {train_losses[-1]},  valid_loss: {valid_losses[-1]}", end='\r')

# predict * -------------------------------------------------------------------------------------------------------
model.predict(X_sample.to(device))
np.stack([[word_index_y[word] for word in sentence] for sentence in model.predict(X_sample.to(device)).to('cpu').numpy()])


# loss graph * -------------------------------------------------------------------------------------------------------
plt.figure()
plt.title('Loss of Seq2Seq Model ')
plt.plot(train_losses, label='train_loss')
plt.plot(valid_losses, label='valid_loss')
plt.xlabel('epochs')
plt.ylabel('cross_entropy_loss')
plt.show()



# pip install torchmetrics
# pip install nltk
# pip install torchtext
# import nltk.translate.bleu_score as bleu
# from torchtext.data.metrics import bleu_score
# bleu.sentence_bleu(references_corpus, candidate_corpus)

# bleu_scores = []
# for r, c in zip(batch_y.unsqueeze(1), torch.argmax(pred, axis=2).to('cpu').detach()):
#     bleu_scores.append(bleu.sentence_bleu(r,c))
# np.mean(bleu_scores)






























































# tensorflow ----------------------------------------
class RNN_Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size_en, 32)
        self.rnn = tf.keras.layers.SimpleRNN(16, return_sequences=True, return_state=True)   # return seq / state
        
    def call(self, X):
        self.e1 = self.embed(X)
        self.e2_seq, self.e2_hidden = self.rnn(self.e1)
        return self.e2_seq, self.e2_hidden

class RNN_Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size_kor, 32)
        self.rnn = tf.keras.layers.SimpleRNN(16, return_sequences=True, return_state=True)   # return seq / state
        self.dense = tf.keras.layers.Dense(vocab_size_kor, activation='softmax')
        
    def call(self, X, hidden):
        self.d1 = self.embed(X)
        self.d2_seq, self.d2_hidden = self.rnn(self.d1, initial_state=[hidden] )
        self.result = self.dense(self.d2_seq)
        return self.result

class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = RNN_Encoder()
        self.decoder = RNN_Decoder()
    
    def call(self, Xy, training=True, teacher_forcing=1):
        X = Xy[0]
        y = Xy[1]
        # X = tf.constant(Xy[0], dtype=tf.int32)
        # y = tf.constant(Xy[1], dtype=tf.int32)

        self.seq_len = y.shape[1]
        # teacher_forcing = 0
        states, context_vector = self.encoder(X)
        y_word = y[:,0][..., tf.newaxis]
        self.result = self.decoder(y_word, context_vector)

        for i in range(1, self.seq_len):
            pred_output = self.decoder(y_word, context_vector)
            
            self.result = tf.concat([self.result, pred_output],axis=1)
            if teacher_forcing >= np.random.rand():
                y_word = y[:,i][..., tf.newaxis]
            else:
                y_word = tf.argmax(pred_output, axis=2)
        
        return self.result
    
    def predict(self, X, return_word=True):
        X_len = X.shape[0]
        states, context_vector = self.encoder(X)
        y_word = np.repeat(907, X_len).reshape(-1,1) 
        self.pred_result = self.decoder(y_word, context_vector)

        for i in range(1, self.seq_len):
            pred_output = self.decoder(y_word, context_vector)
            self.pred_result = tf.concat([self.pred_result, pred_output],axis=1)
            y_word = tf.argmax(pred_output, axis=2)
        self.pred_word = tf.argmax(self.pred_result, axis=2)

        return self.pred_word if return_word else self.pred_result

model = RNN_Seq2Seq()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.fit(x=[train_X, train_inout_y], y=train_inout_oh_y, batch_size=64, epochs=10)


# model([train_X, train_inout_y])
pred_result = model.predict(train_X)
[''.join(['' if w==0 else tokenizer_kor.index_word[w] for w in seq]) for seq in pred_result.numpy()]




target = ['I am a boy']
target_token = tokenizer_en.texts_to_sequences(target)
target_padseq = tf.keras.preprocessing.sequence.pad_sequences(target_token, maxlen=train_X.shape[1], padding='post')

model.predict(target_padseq)


# enco = RNN_Encoder()
# deco = RNN_Decoder()

# X_sample.shape
# y_inout_sample.shape
# y_inout_oh_sample.shape

# teacher_forcing = 0
# states, hidden = enco(tf.constant(X_sample, dtype=tf.int32))
# input_de = tf.constant(y_inout_sample[:,[0]], dtype=tf.int32)
# output = deco(input_de, hidden)
# for i in range(1, y_inout_sample.shape[1]):
#     pred_de = deco(input_de, hidden)
#     output = tf.concat([output,pred_de],axis=1)
#     if teacher_forcing >= np.random.rand():
#         input_de = tf.constant(y_inout_sample[:,[i]], dtype=tf.int32)
#     else:
#         input_de = tf.argmax(pred_de, axis=2)












####################################
rnn_model = RNN_Translate()
# rnn_model([X_sample, input_sample])
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(x=[train_X, train_input_y], y=train_output_y, 
          epochs=10,
          validation_data=([test_X, test_input_y], test_output_y))





# ##########################################################################################
# english_input = tf.keras.layers.Input(shape=(None,))
# encoder = tf.keras.layers.Embedding(input_dim=vocab_size_en, output_dim=32)(english_input)
# encoder_seq, encoder_hidden, encoder_cell = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True)(encoder)
# # model_encoder = tf.keras.models.Model(english_input)

# korean_input = tf.keras.layers.Input(shape=(None,))
# decoder = tf.keras.layers.Embedding(input_dim=vocab_size_kor, output_dim=32)(korean_input)
# decoder_seq, hidden_state, cell_state = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True)(decoder, initial_state=[encoder_hidden, encoder_cell])
# korean_output = tf.keras.layers.Dense(units=vocab_size_kor, activation='softmax')(decoder_seq)

# model = tf.keras.models.Model([english_input, korean_input], korean_output)
# # model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit([train_X, train_input_y], train_output_y, 
#           epochs=10,
#           validation_data=([test_X, test_input_y], test_output_y))
# ##########################################################################################












































# 인코더 네트워크
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()       
        self.input_dim = input_dim # 인코더 입력층
        self.embbed_dim = embbed_dim # 인코더 임베딩 계층
        self.hidden_dim = hidden_dim # 인코더 은닉층(이전 은닉층)
        self.num_layers = num_layers # GRU 계층 개수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim) # 임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        # 임베딩 차원, 은닉층 차원, gru 계층 개수를 이용하여 gru 계층 초기화
        
    def forward(self, src):      
        embedded = self.embedding(src).view(1,1,-1) # 임베딩
        outputs, hidden = self.gru(embedded) # 임베딩 결과를 GRU 모델에 적용
        return outputs, hidden

# 디코더 네트워크 
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim) # 임베딩 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers) # gru 초기화
        self.out = nn.Linear(self.hidden_dim, output_dim) # 선형 계층 초기화
        self.softmax = nn.LogSoftmax(dim=1)
      	
    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)       
        prediction = self.softmax(self.out(output[0]))      
        return prediction, hidden 

# Seq2seq 네트워크
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        # 인코더와 디코더 초기화
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
     
    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):

        input_length = input_lang.size(0) # 입력 문장 길이(문장 단어수)
        batch_size = output_lang.shape[1] 
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim      
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            # 문장의 모든 단어 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
            
        # 인코더 은닉층 -> 디코더 은닉층
        decoder_hidden = encoder_hidden.to(device)  
        # 예측 단어 앞에 SOS token 추가
        decoder_input = torch.tensor([SOS_token], device=device)  

        for t in range(target_length):   
            # 현재 단어에서 출력단어 예측
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            # teacher force 활성화하면 모표를 다음 입력으로 사용
            input = (output_lang[t] if teacher_force else topi)
            # teacher force 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token) :
                break
        return outputs
    
# teacher_force : seq2seq에서 많이 사용되는 기법. 번역(예측)하려는 목표 단어를 디코더의 다음 입력으로 넣어줌



# # Seq2Seq ###############################################################################################
# 하이퍼 파라미터 지정
input_dim = len(SRC.vocab)  # 7854
output_dim = len(TRG.vocab) # 5893
enc_emb_dim = 256 # 임베딩 차원
dec_emb_dim = 256
hid_dim = 512 # hidden state 차원
n_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

# # 모델 생성 ###############################################################################################
enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
# (input: X) src            (seq, batch)
# . emb : 7854, 256         (seq, batch, emb_dim)
# . lstm : 256, 512,        (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# (return) : hidden(lstm), cell(lstm)

dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)
# (input: y) trg[0,:]        (batch) : 1st word at each seqences
# . unsqueeze : 1, batch    (1, batch)
# . emb : 5893, 256         (1, batch, emb_dim)
# . lstm : 256, 512         (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# . linear : 512, 5893
# (return) : prediction(linear), hidden(lstm), cell(lstm)
############################################################################################################


batch_size = trg_s.shape[1]
trg_len = trg_s.shape[0]    # 타겟 토큰 길이 얻기
trg_vocab_size = output_dim     # context vector의 차원 : target_vocab
teacher_forcing_ratio = 0

# decoder의 output을 저장하기 위한 tensor
outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)   # (seq, batch, embeding)

# initial hidden state
hidden, cell = enc(src_s)   # (1,3,512), (1,3,512) : (seq, batch, rnn_nodes)
# embedded       # (31, 3, 256) : (seq, batch, emb_dim)
# rnn            # (1,3, 512), (1,3, 512), (1,3, 512)   : (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)

# 첫 번째 입력값 <sos> 토큰
input_de = trg_s[0,:]   # (3) : (batch)

# for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복
output, hidden, cell = dec(input_de, hidden, cell)
# print(output.shape, hidden.shape, cell.shape)
# (input)
# input_de.unsqueeze(0) # (1, 3) : (seq, batch)
# embedded       # (1,3, 256) : (seq, batch, emb_dim)
# rnn            # (1,3, 512), (1,3, 512), (1,3, 512)   : (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# linear         # (3, 5893) : (batch, y_vocab)

# prediction 저장
outputs[t] = output

# teacher forcing을 사용할지, 말지 결정
teacher_force = random.random() < teacher_forcing_ratio

# 가장 높은 확률을 갖은 값 얻기
top1 = output.argmax(1)

# teacher forcing의 경우에 다음 lstm에 target token 입력
input_de = trg[t] if teacher_force else top1
# return outputs
############################################################################################################################