# https://wikidocs.net/44249

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

# (Error) JVMNotFoundException: No JVM shared library file (jvm.dll) found. Try setting up the JAVA_HOME environment variable properly.
# https://koreapy.tistory.com/902
# https://default-womyn.tistory.com/entry/konlpy-Okt-%EC%98%A4%EB%A5%98-No-JVM-shared-library-file-jvmdll-found
# conda install -c conda-forge jpype1
# pip install konlpy
from konlpy.tag import Okt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# import os
# os.getcwd()

# 1) 데이터 로드하기 --------------------------------
dataset_path = r"D:\작업방\업무 - 자동차 ★★★\Dataset"
train_data = pd.read_table(f'{dataset_path}/NLP_Raw_movie_ratings_train.txt')
test_data = pd.read_table(f'{dataset_path}/NLP_Raw_movie_ratings_test.txt')

print(train_data.shape, test_data.shape)
train_data[:5] # 상위 5개 출력


# 2) 데이터 정제하기 --------------------------------
train_data01 = train_data.copy()
test_data01 = test_data.copy()

# 한글과 공백을 제외하고 모두 제거 (정규표현식 활용)
#   (알파벳과 공백을 제외하고 모두 제거하는 전처리)
#   eng_text = 'do!!! you expect... people~ to~ read~ the FAQ, etc. and actually accept hard~! atheism?@@'
#   print(re.sub(r'[^a-zA-Z ]', '', eng_text))
#   'do you expect people to read the FAQ etc and actually accept hard atheism''do you expect people to read the FAQ etc and actually accept hard atheism'
train_data01['document'] = train_data01['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data01['document'] = train_data01['document'].str.replace('^ +', "")                  # white space 데이터를 empty value로 변경
train_data01['document'] = train_data01['document'].replace('', np.nan)

test_data01['document'] = test_data01['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data01['document'] = test_data01['document'].str.replace('^ +', "")                  # white space 데이터를 empty value로 변경
test_data01['document'] = test_data01['document'].replace('', np.nan)

train_data01[:5]
train_data01.isna().sum(0)

test_data01[:5]
test_data01.isna().sum(0)

# 중복제거, NA제거
train_data02 = train_data01.drop_duplicates(subset=['document']).dropna()
test_data02 = test_data01.drop_duplicates(subset=['document']).dropna()

# document 열과 label 열의 중복을 제외한 값의 개수
train_data02['document'].nunique(), train_data02['label'].nunique()
test_data02['document'].nunique(), test_data02['label'].nunique()
print(train_data02.shape, test_data02.shape)


# 3) 토큰화 --------------------------------

# 불용어를 제거 ***
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']     # 불용어 정의

from konlpy.tag import Okt

okt = Okt()
# okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)
#   ['오다', '이렇다', '것', '도', '영화', '라고', '차라리', '뮤직비디오', '를', '만들다', '게', '나다', '뻔']
# Okt는 위와 같이 KoNLPy에서 제공하는 형태소 분석기입니다. 
# 한국어을 토큰화할 때는 영어처럼 띄어쓰기 기준으로 토큰화를 하는 것이 아니라, 주로 형태소 분석기를 사용한다고 언급한 바 있습니다. 
# stem = True를 사용하면 일정 수준의 정규화를 수행해주는데, 
# 예를 들어 위의 예제의 결과를 보면 '이런'이 '이렇다'로 변환되었고 '만드는'이 '만들다'로 변환된 것을 알 수 있습니다.


X_train_all = []
for sentence in tqdm(train_data02['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train_all.append(stopwords_removed_sentence)


print(X_train_all[:3])
# [['아', '더빙', '진짜', '짜증나다', '목소리'], 
# ['흠', '포스터', '보고', '초딩', '영화', '줄', '오버', '연기', '조차', '가볍다', '않다'], 
# ['너', '무재', '밓었', '다그', '래서', '보다', '추천', '다']]


X_test_all = []
for sentence in tqdm(test_data02['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test_all.append(stopwords_removed_sentence)

################################################
dataset_path = r"D:\작업방\업무 - 자동차 ★★★\Dataset"

# train_token = pd.concat([train_data02[['id','label']].reset_index(drop=True), pd.Series(X_train_all, name='tokenized')], axis=1)
# test_token = pd.concat([test_data02[['id','label']].reset_index(drop=True), pd.Series(X_test_all, name='tokenized')], axis=1)

# train_token.to_csv(f"{dataset_path}/NLP_movie_review_train_tokenized.csv")
# test_token.to_csv(f"{dataset_path}/NLP_movie_review_test_tokenized.csv")
# train_token[:30000].to_csv(f"{dataset_path}/NLP_movie_review_simple_train_tokenized.csv")
# test_token[:3000].to_csv(f"{dataset_path}/NLP_movie_review_simple_test_tokenized.csv")

################################################
# train_token = pd.read_csv(f"{dataset_path}/NLP_movie_review_train_tokenized.csv", encoding='utf-8-sig')
# test_token = pd.read_csv(f"{dataset_path}/NLP_movie_review_test_tokenized.csv", encoding='utf-8-sig')
# train_X = list(map(lambda x: eval(x), train_token['tokenized']))
# test_X = list(map(lambda x: eval(x), test_token['tokenized']))

simple_train_token = pd.read_csv(f"{dataset_path}/NLP_movie_review_simple_train_tokenized.csv", encoding='utf-8-sig')
simple_test_token = pd.read_csv(f"{dataset_path}/NLP_movie_review_simple_test_tokenized.csv", encoding='utf-8-sig')
train_X = list(map(lambda x: eval(x), simple_train_token['tokenized']))
test_X = list(map(lambda x: eval(x), simple_test_token['tokenized']))

# train_y = train_token['label'].values
# test_y = test_token['label'].values
train_y = simple_train_token['label'].values
test_y = simple_test_token['label'].values
print(len(train_X), len(train_y), len(test_X), len(test_y))


# 4) 정수 인코딩 --------------------------------
#   기계가 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

word_numbering = pd.Series(tokenizer.word_index)
word_numbering
#   각 정수는 전체 훈련 데이터에서 등장 빈도수가 높은 순서대로 부여되었기 때문에, 높은 정수가 부여된 단어들은 등장 빈도수가 매우 낮다는 것을 의미
# {'영화': 1, '보다': 2, '을': 3, '없다': 4, '이다': 5, '있다': 6, '좋다': 7, ... 중략 ... '디케이드': 43751, '수간': 43752}

word_freq = pd.Series(tokenizer.word_counts)


# 희소단어 제거
rare_dict = {}
total_cnt = len(word_freq)      # 전체 단어수
for i in sorted(word_freq.reset_index(drop=True).unique()):
    prob =  len(word_freq[word_freq <= i]) / total_cnt     # 특정 빈도수 이하 단어
    rare_dict[i] = prob

rare_vec = pd.Series(rare_dict)
rare_vec[:20]

threshold_rareword = 0.7
rare_cnt = rare_vec[rare_vec <= threshold_rareword].index[-1]
# rare_cnt = 3
print(f"rare_cnt: {rare_cnt}")


plt.figure()
plt.title(f"Ratio of Rare_Word\n {rare_cnt}: {round(rare_vec[rare_cnt], 1)}% (total: {total_cnt})")
plt.plot(rare_vec.index, rare_vec, 'o-')
plt.xscale('log')
plt.axhline(threshold_rareword, color='red', alpha=0.1)
plt.text(rare_cnt, rare_vec[rare_cnt], f"  ← {rare_cnt}", color='red')
plt.scatter(rare_cnt, rare_vec[rare_cnt], color='red')
plt.show()


# 등장 빈도가 threshold 값인 3회 미만. 즉, 2회 이하인 단어들은 단어 집합에서 무려 절반 이상을 차지합니다.
#  하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 매우 적은 수치인 1.87%밖에 되지 않습니다.
#  아무래도 등장 빈도가 2회 이하인 단어들은 자연어 처리에서 별로 중요하지 않을 듯 합니다. 
# 그래서 이 단어들은 정수 인코딩 과정에서 배제시키겠습니다.

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
# vocab_size = total_cnt - rare_cnt + 1
vocab_size = len(word_freq[word_freq > rare_cnt]) + 1
# vocab_size = 6679

print(f'단어 집합의 크기 : {vocab_size} (total: {total_cnt})')
#   단어 집합의 크기 : 6679 (total: 21329)



# 단어 집합의 크기는 19,416개입니다. 
# 이를 케라스 토크나이저의 인자로 넘겨주고 텍스트 시퀀스를 정수 시퀀스로 변환합니다.

# texts_to_sequences ★★ -----------------------------------
# # ----------------------------------------
# # tokenizer
# AA = [['a','e','c','d'],
#  ['a','b','c','f'],
#  ['a','b','e','d'],
#  ['a','b','c','d']]
# t = Tokenizer()
# t.fit_on_texts(AA)
# dict(t2.word_counts)
# t2.word_index

# # tokenizer_filtering
# t2 = Tokenizer(num_words=6-1+1)
# t2.fit_on_texts(AA)
# np.array(dir(t2)[-50:])
# t2.texts_to_sequences(AA)
# # ----------------------------------------

tokenizer = Tokenizer(num_words=vocab_size)     # remove rare_word → re tokenize
tokenizer.fit_on_texts(train_X)

train_X_seq = tokenizer.texts_to_sequences(train_X)
test_X_seq = tokenizer.texts_to_sequences(test_X)

print(train_X_seq[:3])
print(test_X_seq[:3])
# [[50, 454, 16, 260, 659], [933, 457, 41, 602, 1, 214, 1449, 24, 961, 675, 19], [386, 2444, 2315, 5671, 2, 222, 9]]


# 5) 빈 샘플(empty samples) 제거 --------------------------------
train_seq_df = pd.concat([pd.Series(train_X_seq), pd.Series(train_y)], axis=1)
train_seq_df.columns = ['token_seq', 'label']

test_seq_df = pd.concat([pd.Series(test_X_seq), pd.Series(test_y)], axis=1)
test_seq_df.columns = ['token_seq', 'label']

train_seq_drop_df = train_seq_df[train_seq_df['token_seq'].apply(lambda x: len(x)) > 0]
test_seq_drop_df = test_seq_df[test_seq_df['token_seq'].apply(lambda x: len(x)) > 0]

print(train_seq_drop_df.shape, test_seq_drop_df.shape)


# 6) 패딩 : Max_Length ------------------------
train_len = train_seq_drop_df['token_seq'].apply(len).value_counts().sort_index()
train_cumprob = train_len.cumsum() / train_len.sum()

threshold_maxlen = 0.9
max_len = train_cumprob[train_cumprob >= threshold_maxlen].index[0]
# max_len = 22
print(max_len)


plt.figure()
plt.plot(train_cumprob.index, train_cumprob, 'o-')
plt.axhline(threshold_maxlen, color='red', alpha=0.1)
plt.text(max_len, train_cumprob[max_len], f"← {max_len}", color='red')
plt.show()


# pad_sequences ★★ --------------------------------------------
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_X_pad = pad_sequences(train_seq_drop_df['token_seq'].values, maxlen=max_len)
test_X_pad = pad_sequences(test_seq_drop_df['token_seq'].values, maxlen=max_len)

train_y_pad = train_seq_drop_df['label'].values
test_y_pad = test_seq_drop_df['label'].values

print(train_X_pad.shape, train_y_pad.shape)
print(test_X_pad.shape, test_y_pad.shape)

####################################################################################################################################
####################################################################################################################################


import sys
sys.path.append(r'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/DS_Module')
from DS_DeepLearning import EarlyStopping

# RNN으로 감성분석하기
#### tensorflow ########################################################################
embedding_dim = 100
hidden_units = 128
# pd.Series(dir()).to_clipboard()

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# device_lib.list_local_devices()[0].name

# vocab_size = 6679

class NLP_Movie_TF_RNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb_layer = tf.keras.layers.Embedding(vocab_size, 100)
        self.rnn_layer = tf.keras.layers.SimpleRNN(128, return_state=True, return_sequences=True)
        self.linear_layer = tf.keras.layers.Dense(1)
        self.sigmoid_layer = tf.keras.layers.Activation('sigmoid')

    def call(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 23(word), 100(embed))
        self.rnn_output, self.rnn_hidden = self.rnn_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        self.linear = self.linear_layer(self.rnn_hidden)             # (batch, 1)
        self.sigmoid = self.sigmoid_layer(self.linear)
        return self.sigmoid

class NLP_Movie_TF_LSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb_layer = tf.keras.layers.Embedding(vocab_size, 100)
        self.lstm_layer = tf.keras.layers.LSTM(128, return_state=True, return_sequences=True)
        self.linear_layer = tf.keras.layers.Dense(1)
        self.sigmoid_layer = tf.keras.layers.Activation('sigmoid')

    def call(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 23(word), 100(embed))
        self.lstm_output, self.lstm_hidden, self.lstm_cell = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        self.linear = self.linear_layer(self.lstm_hidden)             # (batch, 1)
        self.sigmoid = self.sigmoid_layer(self.linear)
        return self.sigmoid


# sample_pad_tf = tf.constant(train_X_pad[:3,:], dtype=tf.int32)
# sample_pad_tf.shape  # 3,23

train_ds = tf.data.Dataset.from_tensor_slices((train_X_pad, train_y_pad)).batch(32).shuffle(train_X_pad.shape[0])
test_ds = tf.data.Dataset.from_tensor_slices((test_X_pad, test_y_pad)).batch(32).shuffle(test_X_pad.shape[0])

tf_model = NLP_Movie_TF_RNN()
# tf_model(sample_pad_tf)

loss_function = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.RMSprop()
epochs = 10
min_epochs = 0

es_tf = EarlyStopping()

with tf.device(f"{device_lib.list_local_devices()[0].name}"):       #"GPU를 사용한 학습"
    losses = []
    for epoch in range(epochs):
        losses_train = []
        for batch_X, batch_y in train_ds:
            with tf.GradientTape() as tape:
                pred = tf_model(batch_X)
                loss = loss_function(batch_y, pred)
            gradients = tape.gradient(loss, tf_model.trainable_variables)
            optimize = optimizer.apply_gradients(zip(gradients, tf_model.trainable_variables))
            losses_train.append(loss.numpy())
        
        losses_test = []
        for t_batch_X, t_batch_y in test_ds:
            t_pred = tf_model(t_batch_X)
            t_loss = loss_function(t_batch_y, t_pred)
            losses_test.append(t_loss.numpy())
        
        es_result = es_tf.early_stop(score=np.mean(losses_test), reference_score=np.mean(losses_train),
                                     save=tf_model.weights, 
                                     sleep=0, verbose=2)
        if epoch+1 >= min_epochs:
            if es_result == 'break':
                break

# es_tf.plot
# es_tf.optimum
# es_tf.optimum[2]


# weights_save ***
# from six.moves import cPickle
# cPickle.dump(es_tf.optimum[2], open('NLP_tf_model_RNN_weights.pkl', 'wb'))
 
 
# weights_load ***
from six.moves import cPickle
weight_path = r'D:\작업방\업무 - 자동차 ★★★\Workspace_Python\Model\weights_tensorflow'

tf_model_weights = cPickle.load(open(f"{weight_path}/NLP_tf_model_LSTM_weights.pkl",'rb'))

# vocab_size = 6679
# tf_model = NLP_Movie_TF_RNN()
tf_model = NLP_Movie_TF_LSTM()
tf_model.build(input_shape=(1, max_len))
tf_model.set_weights(tf_model_weights)


# Model Evaluate ***
loss_function = tf.keras.losses.BinaryCrossentropy()
t_pred = tf_model(tf.constant(test_X_pad))
loss_function(tf.constant(test_y_pad), t_pred)
# ------------------------------------------------------------------------------------------------------------



    


#### torch ########################################################################
import torch
use_cuda = False
# if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)   

embedding_dim = 100
hidden_units = 128
# vocab_size = total_cnt - rare_cnt + 1
# vocab_size = train_X_pad.max() + 1
# vocab_size = 6679

class NLP_Movie_Torch_RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.rnn_layer = torch.nn.RNN(100, 128, batch_first=True)
        self.linear_layer = torch.nn.Linear(128, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 23(word), 100(embed))
        self.rnn_output, self.rnn_hidden = self.rnn_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        self.linear = self.linear_layer(self.rnn_hidden[0])             # (batch, 1)
        return self.linear
        # self.sigmoid = self.sigmoid_layer(self.linear)
        # return self.sigmoid

class NLP_Movie_Torch_LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.lstm_layer = torch.nn.LSTM(100, 128, batch_first=True)
        self.linear_layer = torch.nn.Linear(128, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 23(word), 100(embed))
        self.lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        self.linear = self.linear_layer(self.lstm_hidden[0])             # (batch, 1)
        # self.sigmoid = self.sigmoid_layer(self.linear)
        return self.linear       

# with attention (concat)
class NLP_Movie_Torch_LSTM_With_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.lstm_layer = torch.nn.LSTM(100, 128, batch_first=True)
        
        self.linear_layer = torch.nn.Linear(256, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 22(word), 100(embed))
        self.lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        # (B, 22(word), C), ((1, B, C), (1, B, C))
        
        # Attention_layer
        self.query = self.lstm_hidden.transpose(0,1)               # sT: (batch, 1, C)
        self.key = self.lstm_output                                 # hi:  (batch, 22(word), C)
        self.score = torch.sum(self.query * self.key, axis=2)               # et: (batch, 22(word)) : sum(query(batch, 1, C) * key(batch, 22(word), C), axis=2)
        self.score_softmax = torch.nn.functional.softmax(self.score, dim=1)             # et: (batch, 22(word))
        self.attention_value = torch.sum(torch.unsqueeze(self.score_softmax, 2) * self.lstm_output, axis=1)  # at: (batch, C) : sum((batch, 22(word), 1) * (batch, 1, C), axis=1)
        self.concat = torch.cat((self.attention_value, self.lstm_hidden[0]),axis=1)     # (batch, 2*C)
        
        self.linear = self.linear_layer(self.concat)             # (batch, 1)
        return self.linear

# with attention (product)
class NLP_Movie_Torch_LSTM_With_Attention2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.lstm_layer = torch.nn.LSTM(100, 128, batch_first=True)
        
        self.linear_layer = torch.nn.Linear(128, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 22(word), 100(embed))
        self.lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        # (B, 22(word), C), ((1, B, C), (1, B, C))
        
        # Attention_layer
        self.query = self.lstm_hidden.transpose(0,1)               # sT: (batch, 1, C)
        self.key = self.lstm_output                                 # hi:  (batch, 22(word), C)
        self.score = torch.sum(self.query * self.key, axis=2)               # et: (batch, 22(word)) : sum(query(batch, 1, C) * key(batch, 22(word), C), axis=2)
        self.score_softmax = torch.nn.functional.softmax(self.score, dim=1)             # et: (batch, 22(word))
        self.attention_value = torch.sum(torch.unsqueeze(self.score_softmax, 2) * self.lstm_output, axis=1)  # at: (batch, C) : sum((batch, 22(word), 1) * (batch, 1, C), axis=1)
        self.product = self.attention_value * self.lstm_hidden[0]     # (batch, C)
        
        self.linear = self.linear_layer(self.product)             # (batch, 1)
        return self.linear

# with attention concentrated_dot product
class NLP_Movie_Torch_LSTM_With_Attention3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.lstm_layer = torch.nn.LSTM(100, 128, batch_first=True)
        
        self.linear_layer = torch.nn.Linear(128, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 22(word), 100(embed))
        self.lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        # (B, 22(word), C), ((1, B, C), (1, B, C))
        
        # Attention_layer
        self.query = self.lstm_hidden.transpose(0,1)               # sT: (batch, 1, C)
        self.key = self.lstm_output                     # hi: (batch, 22(word), C)
        
        self.score = torch.sum(self.query * self.key, axis=2)      # et: (batch, 22(word))
        with torch.no_grad():
            self.score_softmax = torch.nn.functional.softmax(self.score, dim=1)                             # et: (batch, 22(word))
        self.attention_value = torch.unsqueeze(torch.sum(self.score, axis=1), dim=1)
        self.product = self.attention_value * self.lstm_hidden[0]
        
        self.linear = self.linear_layer(self.product)             # (batch, 1)
        return self.linear

# with attention ,weigts, concentrated_dot product
class NLP_Movie_Torch_LSTM_With_Attention4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_layer = torch.nn.Embedding(vocab_size, 100)
        self.lstm_layer = torch.nn.LSTM(100, 128, batch_first=True)
        
        self.att_wv = torch.rand(128,1).requires_grad_(True).to(device)
        # self.att_wv = torch.nn.init.xavier_normal_(torch.empty(128,1)).requires_grad_(True).to(device)
        
        self.linear_layer = torch.nn.Linear(128, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()
    
    def forward(self, X):
        self.emb = self.emb_layer(X)                                    # (batch, 22(word), 100(embed))
        self.lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        # (B, 22(word), C), ((1, B, C), (1, B, C))
        
        # Attention_layer
        self.query = self.lstm_hidden.transpose(0,1)               # sT: (batch, 1, C)
        self.key = self.lstm_output                     # hi: (batch, 22(word), C)
        
        self.score_hidden = self.query * self.key      # et: (batch, 22(word), C)
        self.score = torch.matmul(self.score_hidden, self.att_wv)[:,:,0]     # v_hi: (batch, 22(word), 1) : (batch, 22(word), C) @ (C, 1)
        with torch.no_grad():
            self.score_softmax = torch.nn.functional.softmax(self.score, dim=1)                             # et: (batch, 22(word))
        self.attention_value = torch.unsqueeze(torch.sum(self.score, axis=1), dim=1)
        self.product = self.attention_value * self.lstm_hidden[0]
        
        self.linear = self.linear_layer(self.product)             # (batch, 1)
        return self.linear


# Dataset
train_ds = torch.utils.data.TensorDataset(torch.tensor(train_X_pad, dtype=torch.int32)
                               ,torch.tensor(train_y_pad.reshape(-1,1), dtype=torch.float32) )
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

test_ds = torch.utils.data.TensorDataset(torch.tensor(test_X_pad, dtype=torch.int32)
                               ,torch.tensor(test_y_pad.reshape(-1,1), dtype=torch.float32) )
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True)

# sample_pad_torch = torch.tensor(train_X_pad[:3,:], dtype=torch.int32)
# sample_pad_torch.shape

# Prepare_Modeling
# torch_model = NLP_Movie_Torch_RNN().to(device)
# torch_model = NLP_Movie_Torch_LSTM().to(device)
torch_model = NLP_Movie_Torch_LSTM_With_Attention().to(device)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(torch_model.parameters())
epochs = 100
min_epochs = 0

es_torch = EarlyStopping()

# Modeling
losses = []
for epoch in range(epochs):
    losses_train = []
    
    for batch_X, batch_y in train_loader:
        torch_model.train()
        optimizer.zero_grad()
        pred = torch_model(batch_X.to(device))
        loss = loss_function(pred, batch_y.to(device))
        loss.backward()
        optimizer.step()

        losses_train.append(loss.cpu().detach().numpy())  # save loss
    
    losses_test = []
    with torch.no_grad():
        for t_batch_X, t_batch_y in test_loader:
            torch_model.eval()
            t_pred = torch_model(t_batch_X.to(device))
            t_loss = loss_function(t_pred, t_batch_y.to(device))
            losses_test.append(t_loss.cpu().detach().numpy())  # save loss
    
    es_result = es_torch.early_stop(score=np.mean(losses_test), reference_score=np.mean(losses_train), 
                                  save={k: v.cpu().detach() for k, v in torch_model.state_dict().items()}, 
                                  sleep=0, verbose=2)
    if epoch+1 >= min_epochs:
        if es_result == 'break':
            break

es_torch.plot
es_torch.optimum
es_torch.optimum[2]


# weights_save ***
# from six.moves import cPickle
# cPickle.dump(es_torch.optimum[2], open('NLP_torch_model_RNN_weights.pkl', 'wb'))


# weights_load ***
from six.moves import cPickle
weight_path = r'D:\작업방\업무 - 자동차 ★★★\Workspace_Python\Model\weights_torch'
# torch_model_weights = cPickle.load(open(f"{weight_path}/NLP_torch_model_LSTM_weights.pkl",'rb'))
torch_model_weights = cPickle.load(open(f"{weight_path}/NLP_torch_model_LSTM_Attention4_weights.pkl",'rb'))
# torch_model_weights

# vocab_size = 6679
# torch_model = NLP_Movie_RNN().to(device)
# torch_model = NLP_Movie_Torch_LSTM().to(device)
torch_model = NLP_Movie_Torch_LSTM_With_Attention4().to(device)
torch_model.load_state_dict(torch_model_weights)


# Model Evaluate ***
torch_model.eval()
with torch.no_grad():
    t_pred = torch_model(torch.tensor(test_X_pad, dtype=torch.int32))

# loss_function = torch.nn.BCELoss()
# loss_function(t_pred, torch.Tensor(test_y_pad.reshape(-1,1)))

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function(t_pred, torch.Tensor(test_y_pad.reshape(-1,1)))
# ------------------------------------------------------------------------------------------------------------



# predict ---------------------------------------------------------------
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']     # 불용어 정의
okt = Okt()
# tokenizer.texts_to_sequences()

# new_sentence = '이 영화 개꿀잼 ㅋㅋㅋ'
# new_sentence = '이 영화 핵노잼 ㅠㅠ'
# new_sentence = '이딴게 영화냐 ㅉㅉ'
# new_sentence = '와 개쩐다 정말 세계관 최강자들의 영화다'
# new_sentence = '최고'
# new_sentence = '이 돈내고 볼만한 영화는 아님'
# new_sentence = '인생작 인듯'

new_sentence_filter = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
new_sentence_okt = okt.morphs(new_sentence_filter, stem=True)
new_sentence_stopword = [word for word in new_sentence_okt if not word in stopwords] # 불용어 제거
new_sentence_encode = tokenizer.texts_to_sequences([new_sentence_stopword]) # 정수 인코딩
new_sentence_pad = pad_sequences(new_sentence_encode, maxlen=max_len) # 패딩



# # tensorflow prediction ***
# new_sentence_tfX = tf.constant(new_sentence_pad)

# score = tf_model.predict(new_sentence_pad)[0][0]
# print(f"긍정확률: {np.round(score*100,1)}%")


# torch prediction ***
new_sentence_torchX = torch.tensor(new_sentence_pad, dtype=torch.int32)

torch_model.eval()
with torch.no_grad():
    score = torch.nn.functional.sigmoid(torch_model(new_sentence_torchX).cpu().detach()).numpy()[0][0]
print(f"긍정확률: {np.round(score*100,1)}%")


plt.bar(range(torch_model.score_softmax.shape[1]), torch_model.score_softmax[0])





# class NLP_Movie_TF_LSTM(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.emb_layer = tf.keras.layers.Embedding(vocab_size, 100)
#         self.lstm_layer = tf.keras.layers.LSTM(128, return_state=True, return_sequences=True)
        
#         self.att_wk = self.add_weight(shape=(128,8), initializer='random_normal', trainable=True)
#         self.att_wq = self.add_weight(shape=(8,1), initializer='random_normal', trainable=True)
        
#         self.linear_layer = tf.keras.layers.Dense(1)
#         self.sigmoid_layer = tf.keras.layers.Activation('sigmoid')

#     def call(self, X):
#         self.emb = self.emb_layer(X)                                    # (batch, 23(word), 100(embed))
#         self.lstm_output, self.lstm_hidden, self.lstm_cell = self.lstm_layer(self.emb)     # total_seqs(batch, seq, features), last_seq (1, batch, features)
        
#         # Attention_Layer
#         self.value = self.lstm_output
#         self.key = tf.matmul(self.value, self.att_wk)
#         self.score = tf.matmul(self.key, self.att_wq)
#         self.score_prob = tf.nn.softmax(self.score, axis=1)
#         self.attention_value = tf.
        
#         self.linear = self.linear_layer(self.lstm_hidden)             # (batch, 1)
#         self.sigmoid = self.sigmoid_layer(self.linear)
#         return self.sigmoid






