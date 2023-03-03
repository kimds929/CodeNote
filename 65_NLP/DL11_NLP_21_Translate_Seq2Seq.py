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



# # Save Data ----------------------------------------------------------------------------------------------------------------------------
# data_base_url = r''

# word_index_y = pd.Series(processor_en.index_word).to_frame()
# word_index_y.index.name = 'index'
# word_index_y.columns = ['word']

# pd.DataFrame(processor_en.texts).to_csv(f"{data_base_url}/NLP_EN_to_KR_0_pad_seq_sentences(EN).csv", encoding='utf-8-sig', index=False)
# word_index_y.to_csv(f"{data_base_url}/NLP_EN_to_KR_0_index_word(EN).csv", encoding='utf-8-sig')


# word_index_X = pd.Series(processor_kr.index_word).to_frame()
# word_index_X.index.name = 'index'
# word_index_X.columns = ['word']

# pd.DataFrame(processor_kr.texts).to_csv(f"{data_base_url}/NLP_EN_to_KR_0_pad_seq_sentences(KR).csv", encoding='utf-8-sig', index=False)
# word_index_X.to_csv(f"{data_base_url}/NLP_EN_to_KR_0_index_word(KR).csv", encoding='utf-8-sig')
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

padseq_X.shape
word_index_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
word_index_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR_pad_seq_sentences(KR).csv', encoding='utf-8-sig')

# word_index_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
# word_index_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
# padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
# padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_pad_seq_sentences(KR).csv', encoding='utf-8-sig')

# word_index_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
# word_index_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
# padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
# padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', encoding='utf-8-sig')
# pd.Series(word_index_X.index, index=word_index_X)

# word_index_X[0] = ''
# word_index_y[0] = ''

vocab_size_X = len(word_index_X) + 1 #어휘수
vocab_size_y = len(word_index_y) + 1 #어휘수
X = padseq_X.to_numpy()[:max_len]
y = padseq_y.to_numpy()[:max_len]

print(f"vocab_size: {vocab_size_X}, {vocab_size_y}")
print(f"data_size: {X.shape}, {y.shape}")
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







################################################################################################
# max_len = None
max_len = 1000
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/53_Deep_Learning/DL11_NLP/'
word_index_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_word_index(DE_SRC).csv', index_col='index', encoding='utf-8-sig')['word']
word_index_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_word_index(EN_TRG).csv', index_col='index', encoding='utf-8-sig')['word']

train_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_train(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]
valid_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_valid(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]
test_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]

train_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_train(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]
valid_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_valid(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]
test_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]

vocab_size_X = len(word_index_X) + 1 #어휘수
vocab_size_y = len(word_index_y) + 1 #어휘수

train_y_oh = tf.keras.utils.to_categorical(train_y, vocab_size_y)
valid_y_oh = tf.keras.utils.to_categorical(valid_y, vocab_size_y)
test_y_oh = tf.keras.utils.to_categorical(test_y, vocab_size_y)

print(train_X.shape, valid_X.shape, test_X.shape)
print(train_y.shape, valid_y.shape, test_y.shape)
print(train_y_oh.shape, valid_y_oh.shape, test_y_oh.shape)

train_X
train_y


################################################################################################
# spaCy 라이브러리: 문장의 토큰화(tokenization), 태깅(tagging) 등의 전처리 기능을 위한 라이브러리
# 영어(Engilsh)와 독일어(Deutsch) 전처리 모듈 설치
import spacy

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# 영어(English) 및 독일어(Deutsch) 토큰화 함수 정의 -----------------------------------
# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


# 필드(field) 라이브러리를 이용해 데이터셋에 대한 구체적인 전처리 내용을 명시합니다. -----------------------------------
# 번역 목표
# 소스(SRC): 독일어
# 목표(TRG): 영어
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize=tokenize_de, init_token="", eos_token="", lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token="", eos_token="", lower=True, batch_first=True)

from torchtext.datasets import Multi30k
train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))


# 필드(field) 객체의 build_vocab 메서드를 이용해 영어와 독어의 단어 사전을 생성합니다.
# 최소 2번 이상 등장한 단어만을 선택합니다.
SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)


import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
# 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device)

index_dict_src
index_dict_src = {v: k for k, v in SRC.vocab.stoi.items()}
index_dict_trg = {v: k for k, v in TRG.vocab.stoi.items()}
for batch in train_iterator:
    break
np.stack([[index_dict_src[word] for word in seq] for seq in batch.src[:5,:].to('cpu').numpy()])
np.stack([[index_dict_src[word] for word in seq] for seq in batch.trg[:5,:].to('cpu').numpy()])
################################################################################################
















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

class Seq2Seq_Model(torch.nn.Module):
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

# Seq2Seq + Attention -----------------------------------------------------------------------------------------------------------------------------
# https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_Attention_Tutorial.ipynb

# torch mm, bmm, matmul 차이
# https://sunghee.kaist.ac.kr/entry/torch-mm-bmm-matmul-%EC%B0%A8%EC%9D%B4
class AttSeq2Seq_Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_X, 256)      # X_Sentece_Embedding
        self.dropout = torch.nn.Dropout(0.5)
        # self.rnn = torch.nn.RNN(256, 512, batch_first=True, bidirectional=True)
        # self.rnn = torch.nn.LSTM(256, 512, batch_first=True, bidirectional=True)
        self.rnn = torch.nn.GRU(256, 512, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(512*2, 512)   # (enc_rnn*2, enc_fc)
    
    def forward(self, X):
        # X (batch_seq, X_word)
        self.enc_emb = self.embed(X)    # enc_emb (batch_seq, X_word, enc_emb)
        self.emb_dropout = self.dropout(self.enc_emb)
        self.rnn_output, self.rnn_hidden = self.rnn(self.emb_dropout)   # rnn_output (batch_seq, X_word, rnn_enc), rnn_hidden (2, batch_seq, rnn_enc)
        self.rnn_concat = torch.cat([self.rnn_hidden[0,:,:], self.rnn_hidden[1,:,:]], axis=1)[None,...]  # rnn_concat (1, batch_seq, rnn_enc * 2) ←(cat)← (1, seq, rnn_enc_forward), (1, seq, rnn_enc_backward) 
        # self.rnn_hidden[0,:,:] == self.rnn_output[:,-1,:512] : forward
        # self.rnn_hidden[1,:,:] == self.rnn_output[:,0,512:] : backward

        self.enc_fc = torch.tanh(self.fc(self.rnn_concat))       # enc_fc (1, batch_seq, enc_fc)
        
        return self.rnn_output, self.enc_fc

class AttSeq2Seq_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_fc = torch.nn.Linear((512 * 2) + 512, 512)     # (enc_rnn_layer *2) + dec_rnn, energy_fc
        self.att_fc = torch.nn.Linear(512, 1)     # enc_fc, 1
    
    def forward(self, enc_output, dec_hidden):
        # enc_output (batch_seq, X_word, enc_rnn*2)
        # dec_hidden (1, batch_seq, dec_rnn)
        with torch.no_grad():
            self.enc_output_shape = enc_output.shape
        
        self.enc_hidden_expand = dec_hidden.permute(1,0,2).repeat(1, self.enc_output_shape[1], 1)    # enc_hidden_expand (batch_seq, X_word(repeat), dec_rnn)
        self.att_concat = torch.cat([enc_output, self.enc_hidden_expand], axis=2)        # att_concat (batch_seq, X_word, enc_rnn*2 + dec_rnn) ←(cat)← (seq, X_word, enc_rnn*2), (seq, X_word, dec_rnn)
        self.energy = torch.tanh(self.energy_fc(self.att_concat))       # energy (batch_seq, X_word, energy_fc)
        self.att = self.att_fc(self.energy).squeeze(2)        # att (batch_seq, X_word, 1) → (batch_seq, X_word)
        self.att_score = torch.nn.functional.softmax(self.att, dim=1)      # att_score (batch_seq, X_word)
        
        self.weighted = torch.bmm(self.att_score.unsqueeze(1), enc_output)      # weigted (batch_seq, 1, enc_rnn*2) ← (batch_seq, 1, X_word), (batch_seq, X_word, enc_rnn*2)  
        # self.weighted = (self.att_score.unsqueeze(1).permute(0,2,1) * enc_output).sum(1).unsqueeze(1)
        # (행렬 곱 함수) torch.bmm((i, n, m), (i, m, k)) → (i, n, k)
        
        return self.att_score, self.weighted    # att_score (batch_seq, X_word), weigted (batch_seq, 1, enc_rnn*2)

class AttSeq2Seq_Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y, attention_layer):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size_y, 256)      # y_Sentece_Embedding
        self.dropout = torch.nn.Dropout(0.5)
        self.attention = attention_layer
       
        # self.rnn = torch.nn.RNN(256 + (512 * 2), 512, batch_first=True)
        # self.rnn = torch.nn.LSTM(256 + (512 * 2), 512, batch_first=True)
        self.rnn = torch.nn.GRU(256 + (512 * 2), 512, batch_first=True)  # (enc_rnn_layer *2) + y_embed, enc_fc

        # self.fc = torch.nn.Linear(512, vocab_size_y)   # (dec_rnn, dec_fc=vocab_size_y)
        self.fc = torch.nn.Linear(256+512+(512 * 2), vocab_size_y)   # (dec_emb + dec_rnn + enc_rnn*2 , dec_fc=vocab_size_y)
    
    def forward(self, y_before, enc_output, dec_hidden):
        # y_before (batch_seq, y_word) ☞ (batch_seq, 1) * 1단어
        # enc_output (batch_seq, X_word, enc_rnn*2)
        # dec_hidden (1, batch_seq, dec_rnn)
        with torch.no_grad():
            self.y_shape = y_before.shape
            
        self.dec_embed = self.dropout(self.embed(y_before))     # dec_embed (batch_seq, y_word, dec_emb) ☞ (batch_seq, 1, dec_emb) * 1단어
        
        self.att_score, self.weighted = self.attention(enc_output, dec_hidden)  # att_score (batch_seq, X_word), weigted (batch_seq, 1, enc_rnn*2)
        self.weigthed_expand = self.weighted.repeat(1, self.y_shape[1], 1)     # weigthed_expand (batch_seq, y_word, enc_rnn*2) ☞ (batch_seq, 1, enc_rnn*2) * 1단어
        
        self.dec_rnn_input = torch.cat([self.dec_embed, self.weigthed_expand], dim=2)   # dec_rnn_input (batch_seq, y_word, dec_emb + enc_rnn*2) ←(cat)← (batch_seq, y_word, dec_emb), (seq, y_word, enc_rnn*2)
        
        self.dec_output, self.dec_hidden = self.rnn(self.dec_rnn_input, dec_hidden) # dec_output (batch_seq, y_word, dec_rnn), dec_hidden (1, batch_seq, dec_rnn)
        
        # self.dec_fc = torch.fc(self.dec_output)   # dec_fc (batch_seq, y_word, dec_fc)
        self.fc_input = torch.cat([self.dec_embed, self.dec_output, self.weigthed_expand], dim=2)  # fc_input (batch_seq, y_word, dec_emb + dec_rnn + enc_rnn*2) ←(cat)← (batch_seq, y_word, dec_emb), (seq, y_word, dec_rnn), (seq, y_word, enc_rnn*2)
        self.dec_fc = self.fc(self.fc_input)   # dec_fc (batch_seq, y_word, dec_fc)
        
        return self.dec_fc, self.dec_hidden     # dec_fc (batch_seq, y_word, dec_fc), dec_hidden (1, batch_seq, dec_rnn)

class AttSeq2Seq(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y):
        super().__init__()
        self.vocab_size_X = vocab_size_X
        self.vocab_size_y = vocab_size_y
        
        self.encoder = AttSeq2Seq_Encoder(vocab_size_X)
        self.attention = AttSeq2Seq_Attention()
        self.decoder = AttSeq2Seq_Decoder(vocab_size_y, self.attention)

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
        
        with torch.no_grad():
            # self.attention_scores = self.decoder.att_score
            self.attention_scores = torch.zeros(X.shape).to(X.device)

        for i in range(1, self.y_shape[1]):
            pred_output, dec_hidden = self.decoder(y_before, self.enc_output, hidden_input) # (batch_seq, 1, dec_fc==vocab_size_y)
            hidden_input = dec_hidden

            self.result = torch.cat([self.result, pred_output],axis=1)  # (batch_seq, i->y_word, dec_fc==vocab_size_y)

            if teacher_forcing >= np.random.rand():     # teacher_forcing
                y_before = y[:,i][:,None] # y_before (batch_seq, 1)
            else:
                y_before = torch.argmax(pred_output, axis=2)  # y_before (batch_seq, 1)
            
            with torch.no_grad():
                self.attention_scores = torch.cat([self.attention_scores, self.decoder.att_score],axis=0)
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


#######################################################################################################################################
# ★★★ Transformer
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y, X_pad_idx, y_pad_idx,
                 embed_dim=256, n_layers=1, dropout=0.5, n_heads=4, pos_maxlen=100, posff_dim=512):
        super().__init__()
        self.X_pad_idx = X_pad_idx
        self.y_pad_idx = y_pad_idx
        
        self.encoder = Transformer_Encoder(vocab_size_X, embed_dim, n_layers, n_heads, pos_maxlen, posff_dim, dropout)
        self.decoder = Transformer_Decoder(vocab_size_y, embed_dim, n_layers, n_heads, pos_maxlen, posff_dim, dropout)
    
    def make_X_mask(self, X):
        # X : (batch_seq, X_word)
        X_mask = (X != self.X_pad_idx).unsqueeze(1).unsqueeze(2)     # (batch_seq, 1, ,1, X_word)
        return X_mask   # (batch_seq, 1, ,1, X_word)
    
    def make_y_mask(self, y):
        # y : (batch_seq, y_word)
        pad_mask_y = (y != self.y_pad_idx).unsqueeze(1).unsqueeze(2)     # (batch_seq, 1, ,1, X_word)
        
        sub_mask_y = torch.tril(torch.ones((y.shape[1], y.shape[1])).to(y.device)).bool()  # (batch_seq, batch_seq)
        
        y_mask = pad_mask_y & sub_mask_y    # (batch_seq, 1, y_word, y_word)
        # (diagonal 이용하여) batch_seq에 따라 순차적  mask적용 
        
        return y_mask   # (batch_seq, 1, y_word, y_word)
    
    def forward(self, X, y):
        # X : (batch_seq, X_word)
        # y : (batch_seq, y_word)
        
        if y is not None:
            with torch.no_grad():
                self.y_shape = y.shape
                self.init = y[0,0].to('cpu').detach() # 학습시 초기값 저장

        # mask
        self.X_mask = self.make_X_mask(X)
        self.y_mask = self.make_y_mask(y)
        
        # encoder
        self.context_matrix, self.encoder_self_attention = self.encoder(X, self.X_mask)
        # decoder
        self.output, self.decoder_self_attention, self.encoder_attention = self.decoder(y, self.X_mask, self.y_mask, self.context_matrix)
        
        return self.output

    def predict(self, X, max_len=50, eos_word=None):
        # X : (batch_seq, X_word)
        with torch.no_grad():
            X_mask = self.make_X_mask(X)
            context_matrix, encoder_self_attention = self.encoder(X, X_mask)

            y_results = torch.LongTensor([self.init]).repeat(X.shape[0],1).to(X.device)

            for i in range(max_len):
                y_mask = self.make_y_mask(y_results)

                output, encoder_attention, decoder_self_attention = model.decoder(y_results, X_mask, y_mask, context_matrix)

                # 출력 문장에서 가장 마지막 단어만 사용
                pred_word = output.argmax(2)[:,[-1]]
                y_results = torch.cat([y_results, pred_word], axis=1)

        return y_results, encoder_attention


# ★★ Encoder
class Transformer_Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X, embed_dim=256, n_layers=1, n_heads=4, pos_maxlen=100, posff_dim=512, dropout=0.5):
        super().__init__()
        self.X_embed = torch.nn.Embedding(vocab_size_X, embed_dim)
        self.pos_X_embed = torch.nn.Embedding(pos_maxlen, embed_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.encoder_layers = torch.nn.ModuleList([Transformer_EncoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])

        self.scaled = torch.sqrt(torch.FloatTensor([embed_dim]))    # [1]   # sqrt([hidden_dim])
        # self.scaled = math.sqrt(embed_dim)
        # self.scaled = np.sqrt(embed_dim)

    def forward(self, X, X_mask=None):
        # X : (batch_Seq, X_word)
        
        # Scaled X
        self.X_emb_scaled = self.X_embed(X) * self.scaled.to(X.device)   # (batch_seq, X_word, emb)
        
        # positional vector (encoder)
        self.pos_X = torch.arange(0, X.shape[1]).unsqueeze(0).repeat(X.shape[0],1).to(X.device)     # [[0,1,2 ... W]...] (batch_seq, X_word)
        self.pos_emb_X = self.pos_X_embed(self.pos_X)     # (batch_seq, X_word, emb)

        # sum of X_emb_scaled and pos_emb_X
        self.X_input = self.dropout(self.X_emb_scaled + self.pos_emb_X)     # (batch_seq, X_word, emb)

        # # ---
        # self.encoder_output = self.X_input
        # self.encoder_self_attention = None

        # for enc_layer in self.encoder_layers:
        #     self.encoder_output, self.encoder_self_attention = enc_layer(self.encoder_output, X_mask)
        # return self.encoder_output, self.encoder_self_attention    # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)

        # ---
        self.encoder_layer_history = [(self.X_input, None)]

        for enc_layer in self.encoder_layers:
            enc_output, enc_self_att_score = enc_layer(self.encoder_layer_history[-1][0], X_mask)
            self.encoder_layer_history.append((enc_output, enc_self_att_score))
            
        self.encoder_output = self.encoder_layer_history[-1][0]
        return self.encoder_output, self.encoder_layer_history[-1][1]  # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)

# ★ Encoder_Layer
class Transformer_EncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, posff_dim, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.self_attention = Transformer_MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_attention_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.pos_feedforward = Transformer_PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.pos_feedforward_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, X_emb, X_mask):
        # X_emb : (batch_seq, X_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        
        # (Self Attention Layer) ------------------------------------------------------------------
        self.enc_self_att_output, self.enc_self_att_score = self.self_attention(key=X_emb, query=X_emb, value=X_emb, mask=X_mask)
        #  (batch_seq, X_word, fc_dim=emb), (batch_seq, n_heads, X_word, key_length=X_word)
        self.X_add_self_att_ouput = X_emb + self.dropout(self.enc_self_att_output)   # (batch_seq, X_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.layer_normed_self_att_X = self.self_attention_layer_norm(self.X_add_self_att_ouput)  # layer normalization
        
        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.posff_X = self.pos_feedforward(self.layer_normed_self_att_X)    # (batch_seq, X_word, emb)
        self.layer_normed_X_add_posff_output = self.layer_normed_self_att_X + self.dropout(self.posff_X)     # (batch_seq, X_word, emb)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.layer_normed_posff_output_X = self.pos_feedforward_layer_norm(self.layer_normed_X_add_posff_output)

        return self.layer_normed_posff_output_X, self.enc_self_att_score    # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)

# ★ Positionalwise_FeedForward_Layer
class Transformer_PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, embed_dim, pf_dim, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.fc1 = torch.nn.Linear(embed_dim, pf_dim)
        self.fc2 = torch.nn.Linear(pf_dim, embed_dim)
        
    def forward(self, X):
        # X : (batch_seq, X_word, emb)
        self.output_ff1 = self.dropout(torch.relu(self.fc1(X)))    # (batch_seq, X_word, pf_dim)
        self.output_ff2 = self.fc2(self.output_ff1)    # (batch_seq, X_word, emb)
        return self.output_ff2  # (batch_seq, X_word, emb)

# ★ Multihead_Attention_Layer
class Transformer_MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim은 n_head의 배수값 이어야만 합니다.'
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(embed_dim, embed_dim)
        
        self.scaled = torch.sqrt(torch.FloatTensor([embed_dim]))    # [1]   # sqrt([hidden_dim])
        # self.scaled = math.sqrt(embed_dim)
        # self.scaled = np.sqrt(embed_dim)

    def forward(self, query, key, value, mask=None):
        # query, key, value : (batch_seq, len, emb)
        batch_size = query.shape[0] 

        self.query = self.query_layer(query)    # (batch_seq, query_len, emb)
        self.key = self.query_layer(key)        # (batch_seq, key_len, emb)
        self.value = self.query_layer(value)    # (batch_seq, value_len, emb)

        self.query_multihead = self.query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, query_len, head_emb_dim)   ←permute←  (batch_seq, query_len, n_heads, head_emb_dim)
        self.key_multihead = self.key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)         # (batch_seq, n_heads, key_len, head_emb_dim)   ←permute←  (batch_seq, key_len, n_heads, head_emb_dim)
        self.value_multihead = self.value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, value_len, head_emb_dim)   ←permute←  (batch_seq, value_len, n_heads, head_emb_dim)

        self.energy = torch.matmul(self.query_multihead, self.key_multihead.permute(0,1,3,2)) / self.scaled.to(query.device)        # (B, H, QL, KL) ← (B, H, QL, HE), (B, H, HE, KL) 
        # self.energy = (self.query_multihead @ self.key_multihead.permute(0,1,3,2)) / scaled        # (B, H, QL, KL) ← (B, H, QL, HE), (B, H, HE, KL) 
        # np.matmul(A,B)[i,j,k] == np.sum(A[i,j,:] * B[i,:,k]) → i, j, k
        # * summation of muliply between embedding vectors : Query에 해당하는 각 Length(단어) embedding이 어떤 key의 Length(단어) embedding과 연관(내적)되는지?
        # (B, H, QL, KL) : (QL) query의 length(word),   (KL) queyr의 length(word) 대한 key의 length(word) 내적값

        if mask is not None:
            # masking 영역(==0)에 대해 -1e10으로 채우기 (softmax → 0)
            self.energy = self.energy.masked_fill(mask==0, -1e10)
        
        self.att_score = torch.softmax(self.energy, dim=-1)
        self.att_score_dropout = self.dropout(self.att_score)    

        self.weigted = torch.matmul(self.att_score_dropout, self.value_multihead)       # (B, H, QL, HE) ← (B, H, QL, KL), (B, H, VL, HE) 
        # self.weigted = self.att_score, @ self.value_multihead       # (B, H, QL, HE) ← (B, H, QL, KL), (B, H, VL, HE) 
        # * summation of muliply between softmax_score and embeding of value
        # (B, H, QL, HE) : (QL) query의 length(word)   (HE) attention의 softmax_socre에 대한 value embeding 의 내적값 (어떤 value의 embedding과 연관성이 있는지?)

        self.weighted_arange = self.weigted.permute(0,2,1,3).contiguous()        # (B, QL, H, HE) ← (B, H, QL, HE)
        self.weighted_flatten = self.weighted_arange.view(batch_size, -1, self.embed_dim)   # (B, QL, E) ← (B, H, E)

        self.multihead_output = self.fc(self.weighted_flatten)       # (B, QL, FC)
        return self.multihead_output, self.att_score        #  (batch_seq, query_length, fc_dim), (batch_seq, n_heads, query_length, key_length)


# ★★ Decoder 
class Transformer_Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y, embed_dim=256, n_layers=1, n_heads=4, pos_maxlen=100, posff_dim=512, dropout=0.5):
        super().__init__()
        
        self.y_embed = torch.nn.Embedding(vocab_size_y, embed_dim)
        self.pos_y_embed = torch.nn.Embedding(pos_maxlen, embed_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.decoder_layers = torch.nn.ModuleList([Transformer_DecoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])

        self.fc = torch.nn.Linear(embed_dim, vocab_size_y)
        
        self.scaled = torch.sqrt(torch.FloatTensor([embed_dim]))    # [1]   # sqrt([hidden_dim])
        # self.scaled = math.sqrt(embed_dim)
        # self.scaled = np.sqrt(embed_dim)
    
    def forward(self, y, X_mask, y_mask, context_matrix):
        # y : (batch_seq, y_word)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)
        
        # Scaled y
        self.y_emb_scaled = self.y_embed(y) * self.scaled.to(y.device)   # (batch_seq, y_word, emb)
        
        # positional vector (decoder)
        self.pos_y = torch.arange(0, y.shape[1]).unsqueeze(0).repeat(y.shape[0],1).to(y.device)     # [[0,1,2 ... W]...] (batch_seq, y_word)
        self.pos_emb_y = self.pos_y_embed(self.pos_y)     # (batch_seq, y_word, emb)  
        
        # sum of y_emb_scaled and pos_emb_y
        self.y_input = self.dropout(self.y_emb_scaled + self.pos_emb_y)     # (batch_seq, y_word, emb)
        
        # # ---
        # self.decoder_output = self.y_input
        # self.encoder_attention = None
        # self.decoder_self_attention = None

        # for dec_layer in self.decoder_layers:
        #     self.decoder_output, self.decoder_self_attention, self.encoder_attention  = dec_layer(self.decoder_output, X_mask, y_mask, context_matrix)
        
        # self.decoder_output = self.fc(self.decoder_output)
        # return self.decoder_output, self.encoder_attention, self.decoder_self_attention

        # ---
        self.decoder_layer_history = [(self.y_input, None, None)]

        for dec_layer in self.decoder_layers:
            dec_output, dec_self_att_score, enc_att_score = dec_layer(self.decoder_layer_history[-1][0], X_mask, y_mask, context_matrix)
            self.decoder_layer_history.append((dec_output, dec_self_att_score, enc_att_score))

        self.decoder_output = self.fc(self.decoder_layer_history[-1][0])
        return self.decoder_output, self.decoder_layer_history[-1][1], self.decoder_layer_history[-1][2]
    
# ★ Decoder_Layer
class Transformer_DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, posff_dim, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.self_attention = Transformer_MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_attention_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.encoder_attention = Transformer_MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.encoder_attention_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.pos_feedforward = Transformer_PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.pos_feedforward_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, y_emb, X_mask, y_mask, context_matrix):
        # y_emb : (batch_seq, y_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)     # encoder output
        
        # (Self Attention Layer) -------------------------------------------------------------------
        self.dec_self_att_output, self.dec_self_att_score = self.self_attention(query=y_emb, key=y_emb, value=y_emb, mask=y_mask)
        #  (batch_seq, y_word, fc_dim=emb), (batch_seq, n_heads, y_word, key_length=y_word)
        self.y_add_self_att_ouput = y_emb + self.dropout(self.dec_self_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.layer_normed_self_att_y = self.self_attention_layer_norm(self.y_add_self_att_ouput)  # layer normalization
        
        # (Encoder Attention Layer) ----------------------------------------------------------------
        self.y_enc_att_output, self.y_enc_att_score = self.encoder_attention(query=self.layer_normed_self_att_y, key=context_matrix, value=context_matrix, mask=X_mask)
        #  (batch_seq, y_word, fc_dim=emb), (batch_seq, n_heads, y_word, key_length=y_word)
        self.y_add_enc_att_ouput = self.layer_normed_self_att_y + self.dropout(self.y_enc_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 encoder_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.layer_normed_enc_att_y = self.encoder_attention_layer_norm(self.y_add_enc_att_ouput)  # layer normalization

        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.layer_normed_posff_output_y = self.pos_feedforward(self.layer_normed_enc_att_y)    # (batch_seq, y_word, emb)
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.layer_normed_posff_output_y = self.pos_feedforward_layer_norm(self.layer_normed_posff_output_y)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        return self.layer_normed_posff_output_y, self.dec_self_att_score, self.y_enc_att_score
        # (batch_seq, y_word, emb), (batch_seq, n_heads, y_word, y_word), (batch_seq, n_heads, y_word, X_word)
        
#######################################################################################################################################


import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping
import time


# training prepare * -------------------------------------------------------------------------------------------------------
X_sample.shape, y_sample.shape

# model = Seq2Seq_Model(vocab_size_X, vocab_size_y).to(device)
model = AttSeq2Seq(vocab_size_X, vocab_size_y).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0, n_layers=2, n_heads=8, dropout=0.5, pos_maxlen=150).to(device)
# model(X_sample.to(device), y_sample.to(device))


# model weights parameter initialize (가중치 초기화) ***
# def init_weights(model):
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             nn.init.normal_(param.data, mean=0, std=0.01)
#         else:
#             nn.init.constant_(param.data, 0)
# model.apply(init_weights)

# trg_pad_idx = TRG.vocab.stoi[TRG.pad_token] ## pad에 해당하는 index는 무시합니다.
loss_function = torch.nn.CrossEntropyLoss()     # ignore_index=trg_pad_idx
# loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())
epochs = 100

es = EarlyStopping()

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
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)

        if early_stop == 'break':
            break




# early_stopping plot
es.plot

# optimum model (load weights)
model.load_state_dict(es.optimum[2])


# model.predict(X_sample.to(device))

# predict * -------------------------------------------------------------------------------------------------------
# word_index_X = word_index_X.set_index('index')['word']
# word_index_y = word_index_y.set_index('index')['word']
# word_index_X = word_index_X['word']
# word_index_y = word_index_y['word']

idx = 600
sentence_input = train_X[[idx],:]
sentence_output = train_y[[idx],:]

with torch.no_grad():
    model.eval()
    # pred_sentence = model.predict(torch.tensor(sentence_input).to(device))
    pred_sentence, enc_attention = model.predict(torch.tensor(sentence_input).to(device))
    # pred_sentence = model(torch.tensor(sentence_input).to(device), torch.tensor(sentence_output).to(device))
pred_sentence

sentence_en = np.stack([[word_index_X[word] if word != 0 else '' for word in sentence] for sentence in sentence_input])[0]
sentence_kr_real = np.stack([[word_index_y[word] if word != 0 else '' for word in sentence] for sentence in sentence_output])[0]
sentence_kr_pred = np.stack([[word_index_y[word] if word != 0 else '' for word in sentence] for sentence in pred_sentence.to('cpu').detach().numpy()])[0]

sentence_kr_realD
Ssentence_kr_pred



# attention map
import seaborn as sns
sns.heatmap(model.attention_scores.to('cpu').numpy(), cmap='bone')
plt.xticks(np.arange(sentence_en.shape[0]), sentence_en, rotation=90)
plt.yticks(np.arange(sentence_kr_pred.shape[0]), sentence_kr_pred, rotation=0)

X_sample.shape
y_sample.shape

sns.heatmap(model.attention_scores.to('cpu').numpy(), cmap='bone')

word_index_X[419]

sentence_input[0][:12]
sentence_output[0][1:11]


model.encoder_self_attention.shape
model.decoder_self_attention.shape

f = plt.figure(figsize=(10,10))
for h in range(model.encoder_attention.shape[1]):
    plt.subplot(3,3,h+1)
    sns.heatmap(model.encoder_attention[0,h,:13,1:12].to('cpu').numpy(), cmap='bone')
    plt.xticks(np.arange(sentence_en.shape[0])[:13], sentence_en[:13], rotation=90)
    plt.yticks(np.arange(sentence_kr_pred.shape[0])[1:12], sentence_kr_pred[1:12], rotation=0)
plt.close()
f
sentence_en.shape[0][:]
sentence_en[:12]






# loss graph * -------------------------------------------------------------------------------------------------------
plt.figure()
plt.title('Loss of Seq2Seq Model ')
plt.plot(train_losses, label='train_loss')
plt.plot(valid_losses, label='valid_loss')
plt.xlabel('epochs')
plt.ylabel('cross_entropy_loss')
plt.legend()
plt.show()



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







# dot, matmul, bmm ---------------------------------------------------------------------------------------------------
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221356884894
# (*) numpy.dot(a, b, out=None) : Dot product of two arrays. (내적곱)
#   . 텐서(또는 고차원 배열)의 곱연산
#   . 5) 만약 a가 N차원 배열이고 b가 2이상의 M차원 배열이라면, 
#        dot(a,b)는 a의 마지막 축과 b의 뒤에서 두번째 축과의 내적으로 계산된다.
# np.dot(A,B)[i,j,k,m] == np.sum(A[i,j,:] * B[k,:,m]) → (i, j, k, m)

# (*) numpy.matmul(a, b, out=None) : Matrix product of two arrays. (행렬곱)
#   . 두번째 설명이 고차원 배열(N>2)에 대한 내용
#   . 2) 만약 배열이 2차원보다 클 경우, 
#        마지막 2개의 축으로 이루어진 행렬을 나머지 축에 따라 쌓아놓은 것이라고 생각한다.
# np.matmul(A,B)[i,j,k] == np.sum(A[i,j,:] * B[i,:,k]) → i, j, k


# (*) torch.bmm : matrix multiplicatoin
#  . bmm(A,B) [i,k] == np.sum(A[i,j], b[j,k]) → i, k



A0 = np.arange(2*3*4).reshape((2,3,4))

B1 = np.arange(2*3*4).reshape((2,3,4))
B2 = np.arange(2*3*4).reshape((2,4,3))
B3 = np.arange(2*3*4).reshape((3,2,4))
B4 = np.arange(2*3*4).reshape((3,4,2))
B5 = np.arange(2*3*4).reshape((4,2,3))
B6 = np.arange(2*3*4).reshape((4,3,2))

# (dot) operation
A0.shape    # (2,3,4)
np.dot(A0, B1) # (2,3,4) Error
np.dot(A0, B2) # (2,4,3) Ok -> (2,3,2,3)
np.dot(A0, B3) # (3,2,4) Error
np.dot(A0, B4) # (3,4,2) Ok -> (2,3,3,2)
np.dot(A0, B5) # (4,2,3) Error
np.dot(A0, B6) # (4,3,2) Error


# (matmul) opertaion
A0.shape    # (2,3,4)
np.matmul(A0, B1) # (2,3,4) Error
np.matmul(A0, B2) # (2,4,3) Ok -> (2,3,3)
np.matmul(A0, B3) # (3,2,4) Error
np.matmul(A0, B4) # (3,4,2) Error
np.matmul(A0, B5) # (4,2,3) Error
np.matmul(A0, B6) # (4,3,2) Error


# ---------------------------------------------------------------------------------------------------




































































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



####################################################################################################

torch.nn.Sequential()
# nn.Sequential은 input으로 준 module에 대해 순차적으로 forward() method를 호출해주는 역할
torch.nn.ModuleList()
# nn.ModuleList는 nn.Sequential과 마찬가지로 nn.Module의 list를 input으로 받는다.
# nn.Module을 저장하는 역할을 한다. index로 접근도 할 수 있다.
# nn.Sequential과 다르게 forward() method가 없다.
# 안에 담긴 module 간의 connection도 없다.
#  nn.ModuleList안에 Module들을 넣어 줌으로써 Module의 존재를 PyTorch에게 알려 주어야 한다.
# - 만약 nn.ModuleList에 넣어 주지 않고, Python list에만 Module들을 넣어 준다면, PyTorch는 이들의 존재를 알지 못한다.
# - 따라서 Module들을 Python list에 넣어 보관한다면, 꼭 마지막에 이들을 nn.ModuleList로 wrapping 해줘야 한다.

####################################################################################################
# https://sanghyu.tistory.com/3

# view() 
# . 원소의 수를 유지하면서 텐서의 shape를 변경하는 함수
# . contiguous tensor에서 사용할 수 있음
# . view 함수를 이용해서 반환된 값은 원본과 data(memory)를 공유하기 때문에 하나만 수정해도 반환 이전의 변수와 이후 변수 모두 수정된다.

# reshape()
# . contiguous하지 않는 함수에서도 작동한다.
# . reshape() == contiguous().view()

# transpose()
# [batch_size, hidden_dim, input_dim] -> [batch_size, input_dim, hidden_dim]
# 변환이후, contiguous한 성질을 잃어버리기 때문에 transpose().contiguous()와 같이 contiguous()함수와 같이 사용함

# permute()
# 모든 차원을 맞교환 할 수 있다. (transpose()의 일반화 버전이라고 생각한다.)
# 차원을 교환하면서 contiguous 한 성질이 사라진다. 
# view와 같은 contiguous한 성질이 보장될 때만 사용할 수 있는 함수를 사용해야 한다면, permute().contiguous()를 사용하자


# contiguous()
# 메모리상에 데이터를 contiguous 하게 배치한 값을 반환한다.

####################################################################################################


# (torch.tril)     ------------------------------------------------------------------------------------
# torch.tril(input, diagonal=0, *, out=None) → Tensor
# 행렬의 아래쪽 삼각형 부분 (2 차원 텐서) 또는 행렬의 배치 input 을 반환합니다 . 결과 텐서 out 의 다른 요소는 0으로 설정됩니다
# a = torch.rand(5,5)
# a
# torch.tril(a)
# torch.tril(a, diagonal=0)
# torch.tril(a, diagonal=1)
# torch.tril(a, diagonal=2)
# torch.tril(a, diagonal=3)
# torch.tril(a, diagonal=-1)
# torch.tril(a, diagonal=-2)
# torch.tril(a, diagonal=-3)

# a = torch.rand(3,1,1,5) > 0.5
# a_len = a.shape[-1]

# b = torch.tril(torch.ones((a_len, a_len))).bool()  # (batch_seq, batch_seq)  
# ---------------------------------------------------------------------------------------------------------



# (layer Normalization)   ------------------------------------------------------------------------
# https://wingnim.tistory.com/92
# https://velog.io/@tjdcjffff/Normalization-trend
# Batch-Normalization은 기존의 Batch들을 normalization했다면, Layer normalization은 Feature 차원에서 정규화를 진행한다.

# ○ 도입배경 *
# 기존 연구들은 training time을 줄이는 방법으로 batch normalization을 제안하였습니다.
# 그러나 batch normalization은 몇 가지 단점을 가지고 있습니다.
# batch normalization은 mini-batch size에 의존한다.
# recurrent neural network model인 경우, 어떻게 적용이 되는지 명백하게 설명하기 어렵다.
# 본 연구에서는 이러한 문제점을 해결하기 위해서 layer normalization을 제안하였습니다.
# layer normalization은 batch normalization과 달리 train/test time 일 때, 같은 computation을 수행한다는 점이 큰 특징입니다.

# layer normalization은 특정 layer가 가지고 있는 hidden unit에 대해 μ , σ를 공유함
# BN과 다르게 mini-batch-size에 제약이 없음
# RNN model에선 각각의 time-step마다 다른 BN이 학습됨
# LN은 layer의 output을 normalize함으로써 RNN계열 모델에서 좋은 성능을 보임

# ---------------------------------------------------------------------------------------------------------



# (contiguous 여부와 stride 의미)  ------------------------------------------------------------------------
# https://jimmy-ai.tistory.com/122
# https://f-future.tistory.com/entry/Pytorch-Contiguous
import torch

a = torch.randn(3, 4)
a.transpose_(0, 1)

b = torch.randn(4, 3)

# 두 tensor는 모두 (4, 3) shape
print(a)
print(b)

# a 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(a[i][j].data_ptr())

# b 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(b[i][j].data_ptr())


# 각 데이터의 타입인 torch.float32 자료형은 4바이트이므로, 메모리 1칸 당 주소 값이 4씩 증가함을 알 수 있습니다.
# 그런데 자세히 보시면 b는 한 줄에 4씩 값이 증가하고 있지만, a는 그렇지 않은 상황임을 알 수 있습니다.

# 즉, b는 axis = 0인 오른쪽 방향으로 자료가 순서대로 저장됨에 비해,
# a는 transpose 연산을 거치며 axis = 1인 아래 방향으로 자료가 저장되고 있었습니다.

# 여기서, b처럼 axis 순서대로 자료가 저장된 상태를 contiguous = True 상태라고 부르며,
# a같이 자료 저장 순서가 원래 방향과 어긋난 경우를 contiguous = False 상태라고 합니다.

# 각 텐서에 stride() 메소드를 호출하여 데이터의 저장 방향을 조회할 수 있습니다.
# 또한, is_contiguous() 메소드로 contiguous = True 여부도 쉽게 파악할 수 있습니다.

a.stride() # (1, 4)
b.stride() # (3, 1)
# 여기에서 a.stride() 결과가 (1, 4)라는 것은
# a[0][0] -> a[1][0]으로 증가할 때는 자료 1개 만큼의 메모리 주소가 이동되고,
# a[0][0] -> a[0][1]로 증가할 때는 자료 4개 만큼의 메모리 주소가 바뀐다는 의미입니다.


a.is_contiguous() # False
b.is_contiguous() # True

# 텐서의 shape을 조작하는 과정에서 메모리 저장 상태가 변경되는 경우가 있습니다.
# 주로 narrow(), view(), expand(), transpose() 등 메소드를 사용하는 경우에 이 상태가 깨지는 것으로 알려져 있습니다.
#   ㄴ 메모리를 따로 할당하지 않는 Tensor연산

# 해당 상태의 여부를 체크하지 않더라도 텐서를 다루는데 문제가 없는 경우가 많습니다.
# 다만, RuntimeError: input is not contiguous의 오류가 발생하는 경우에는
# input tensor를 contiguous = True인 상태로 변경해주어야 할 수 있습니다.

# 이럴 때에는 아래 예시 코드처럼 contiguous() 메소드를 텐서에 적용하여
# contiguous 여부가 True인 상태로 메모리 상 저장 구조를 바꿔줄 수 있습니다.

a.is_contiguous() # False

# 텐서를 contiguous = True 상태로 변경
a = a.contiguous()
a.is_contiguous() # True
# ---------------------------------------------------------------------------------------------------------









































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

