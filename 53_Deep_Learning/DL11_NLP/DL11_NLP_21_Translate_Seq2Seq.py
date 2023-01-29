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
vocab_size_kor = len(tokenizer_kor.word_index) + 1 #어휘수


# (text_to_sequence / pad_sequence) *
seq_kor = tokenizer_kor.texts_to_sequences(tokened_k)

# SOS / EOS
# seq_kor_input = []
# seq_kor_output = []
seq_kor_inout = []
for sentence in seq_kor:
    seq_kor_inout.append([tokenizer_kor.word_index['<SOS>']] + sentence + [tokenizer_kor.word_index['<EOS>']])

padseq_kor_inout = tf.keras.preprocessing.sequence.pad_sequences(seq_kor_inout, padding='post')
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
max_len = None
max_len = 5000
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


################################################################################################
# spaCy 라이브러리: 문장의 토큰화(tokenization), 태깅(tagging) 등의 전처리 기능을 위한 라이브러리
# 영어(Engilsh)와 독일어(Deutsch) 전처리 모듈 설치
import spacy
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

SRC = Field(tokenize=tokenize_de, init_token="", eos_token="", lower=True)
TRG = Field(tokenize=tokenize_en, init_token="", eos_token="", lower=True)

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

################################################################################################



path = r'D:\작업방\업무 - 자동차 ★★★\Dataset'
# Save_to_csv ***
word_index_X = pd.Series(tokenizer_en.word_index).reset_index()
word_index_y = pd.Series(tokenizer_kor.word_index).reset_index()
word_index_X.columns = ['word', 'index']
word_index_y.columns = ['word', 'index']

padseq_X = pd.DataFrame(padseq_en.copy())
padseq_y = pd.DataFrame(padseq_kor_inout.copy())

# word_index_X.to_csv(f'{path}/NLP_EN_to_KR1_word_index(EN).csv', index=False, encoding='utf-8-sig')
# word_index_y.to_csv(f'{path}/NLP_EN_to_KR1_word_index(KR).csv', index=False, encoding='utf-8-sig')
# padseq_X.to_csv(f'{path}/NLP_EN_to_KR1_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
# padseq_y.to_csv(f'{path}/NLP_EN_to_KR1_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

word_index_X.to_csv(f'{path}/NLP_EN_to_KR2_word_index(EN).csv', index=False, encoding='utf-8-sig')
word_index_y.to_csv(f'{path}/NLP_EN_to_KR2_word_index(KR).csv', index=False, encoding='utf-8-sig')
padseq_X.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
padseq_y.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

# Read_from_csv *** ---------------------------------------------------------------------------------
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/53_Deep_Learning/DL11_NLP/'
word_index_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
word_index_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR1_pad_seq_sentences(KR).csv', encoding='utf-8-sig')

# word_index_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_word_index(EN).csv', index_col='index', encoding='utf-8-sig')['word']
# word_index_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_word_index(KR).csv', index_col='index', encoding='utf-8-sig')['word']
# padseq_X = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', encoding='utf-8-sig')
# padseq_y = pd.read_csv(f'{url_path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', encoding='utf-8-sig')
# pd.Series(word_index_X.index, index=word_index_X)

word_index_X[0] = ''
word_index_y[0] = ''

vocab_size_X = len(word_index_X) + 1 #어휘수
vocab_size_y = len(word_index_y) + 1 #어휘수
X = padseq_X.to_numpy()[:1000]
y = padseq_y.to_numpy()[:1000]

# X_oh = tf.keras.utils.to_categorical(X, vocab_size_X)
# y_oh = tf.keras.utils.to_categorical(y, vocab_size_y)
################################################################################################

# (Train_Test_Split) -------------------------------------------
from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=0)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2, random_state=0)

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


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

class Seq2Seq_Model(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y):
        super().__init__()
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
        self.result, hidden_input = self.decoder(y_before, self.context_vector)     # result (batch_seq, 1, dec_fc==vocab_size_y)

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

        self.result, hidden_input = self.decoder(y_before, self.enc_output, self.context_vector)     # result (batch_seq, 1, dec_fc==vocab_size_y)
        with torch.no_grad():
            self.attention_scores = self.decoder.att_score

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


# enc = AttSeq2Seq_Encoder(vocab_size_X)
# att = AttSeq2Seq_Attention()
# dec = AttSeq2Seq_Decoder(vocab_size_y, att)

# output, hidden = enc(X_sample)
# y_result, dec_hidden = dec(y_sample[:,:1], output, hidden)
# y_result.shape, dec_hidden.shape

# y_result2, dec_hidden2 = dec(y_sample[:,:2], output, dec_hidden)
# y_result2.shape, dec_hidden2.shape


# training prepare * -------------------------------------------------------------------------------------------------------
# model = Seq2Seq_Model(vocab_size_X, vocab_size_y).to(device)
model = AttSeq2Seq(vocab_size_X, vocab_size_y).to(device)


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
optimizer = torch.optim.Adam(model.parameters())
epochs = 100

# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
# from DS_DeepLearning import EarlyStopping
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
        pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict

        pred_eval = pred[:,1:,:].reshape(-1, vocab_size_y)
        real_eval = batch_y[:,1:].reshape(-1).to(device)
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
            pred = model(batch_X.to(device), batch_y.to(device), teacher_forcing=1)                   # predict

            pred_eval = pred[:,1:,:].reshape(-1, vocab_size_y)
            real_eval = batch_y[:,1:].reshape(-1).to(device)
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



# predict * -------------------------------------------------------------------------------------------------------
idx = 15
sentence_input = train_X[[idx],:]
sentence_output = train_y[[idx],:]
model.predict(sentence.to(device))

sentence_en = np.stack([[word_index_X[word] for word in sentence] for sentence in sentence_input])[0]
sentence_kr_real = np.stack([[word_index_y[word] for word in sentence] for sentence in sentence_output])[0]
sentence_kr_pred = np.stack([[word_index_y[word] for word in sentence] for sentence in model.predict(torch.tensor(sentence_input).to(device)).to('cpu').numpy()])[0]

sentence_en
sentence_kr_real
sentence_kr_pred


# attention map
import seaborn as sns
sns.heatmap(model.attention_scores.to('cpu').numpy(), cmap='jet')
plt.xticks(np.arange(sentence_en.shape[0]), sentence_en, rotation=90)
plt.yticks(np.arange(sentence_kr_pred.shape[0]), sentence_kr_pred, rotation=0)





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