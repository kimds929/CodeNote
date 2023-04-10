import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
# import tensorflow as tf



# Data Load ------------------------------------------------------------------------------------------
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'
# url_path = r'C:\Users\Admin\Desktop\DataBase'
df = pd.read_csv(f"{url_path}/NLP_EN_to_KR_0_Data.csv", encoding='utf-8-sig')
# path = r'C:\Users\Admin\Desktop\DataBase'
# df01 = pd.read_csv(f"{path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
# df02 = pd.read_csv(f"{path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')
# df = pd.concat([df02, df01],axis=0).reset_index(drop=True)
print(df.shape)

df.sample(6)


# Preprocessing --------------------------------------------------------------------------------------
# from DS_NLP import NLP_Preprocessor
import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
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


processor_kr.texts_to_sequence_transform('저는 사과를 좋아합니다')

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
# import tensorflow as tf

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

# [Torch Transformer] -------------------------------------------------
# https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/

# torch.nn.Transformer?
# torch.nn.TransformerDecoder?
# torch.nn.TransformerDecoderLayer?
# torch.nn.TransformerEncoder?
# torch.nn.TransformerEncoderLayer?

# self. encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=ff_dim, batch_first=True)

# # vocab_size=vocab_size,
# self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#         )


# x = torch.tensor(randint(3,15,high=6, sign='+'))
# y = torch.tensor(randint(3,10,high=2, sign='+'))

# tr = Transformer(7, 100, 0, 0)


####################################################################################################################################
# ★★★ Transformer
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y, X_pad_idx=0, y_pad_idx=0,
                 embed_dim=256, n_layers=1, dropout=0.1, n_heads=4, posff_dim=512, pos_encoding='sinusoid'):
        super().__init__()
        self.X_pad_idx = X_pad_idx
        self.y_pad_idx = y_pad_idx
        
        self.encoder = Encoder(vocab_size_X, embed_dim, n_layers, n_heads, posff_dim, dropout, pos_encoding=pos_encoding)
        self.decoder = Decoder(vocab_size_y, embed_dim, n_layers, n_heads, posff_dim, dropout, pos_encoding=pos_encoding)
        self.fc_layer = torch.nn.Linear(embed_dim, vocab_size_y)
    
    def forward(self, X, y):
        # X : (batch_seq, X_word)
        # y : (batch_seq, y_word)
        
        if y is not None:
            with torch.no_grad():
                self.y_shape = y.shape
                self.init = y[0,0].to('cpu').detach() # 학습시 초기값 저장

        # mask
        self.X_mask = make_mask(X, self.X_pad_idx).unsqueeze(1).unsqueeze(1) if self.X_pad_idx is not None else None
        self.y_mask = make_tril_mask(y, self.y_pad_idx).unsqueeze(1) if self.y_pad_idx is not None else None
        
        # encoder
        self.encoder_output = self.encoder(X, self.X_mask)
        # decoder
        self.decoder_output = self.decoder(y, self.encoder_output, self.X_mask, self.y_mask)
        
        # # fully connected layer 
        self.output = self.fc_layer(self.decoder_output)
        
        # attention_score
        with torch.no_grad():
            # self.attention_scores = [layer.attention_score for layer_name, layer in self.decoder.decoder_layers.named_children()]
            self.attention_score = self.decoder.decoder_layers[-1].attention_score
        
        return self.output

    def predict(self, X, max_len=50, eos_word=None):
        # X : (batch_seq, X_word)
        with torch.no_grad():
            X_mask = make_mask(X, self.X_pad_idx).unsqueeze(1).unsqueeze(1)
            encoder_output = self.encoder(X, X_mask)

            output = torch.LongTensor([self.init]).repeat(X.shape[0],1).to(X.device)

            for _ in range(max_len-1):
                y_mask = make_tril_mask(output, self.y_pad_idx).unsqueeze(1)

                decoder_output = self.decoder(output, encoder_output, X_mask, y_mask)
                predict_output = self.fc_layer(decoder_output)

                # 출력 문장에서 가장 마지막 단어만 사용
                pred_word = predict_output.argmax(2)[:,[-1]]
                output = torch.cat([output, pred_word], axis=1)

        return output



# ★ masking function ------------------------------------------------------------
# mask
def make_mask(x, pad_idx=0):
    # x : (batch_seq, x_word)
    mask = (x != pad_idx).to(x.device)    # (batch_seq, X_word)
    return mask   # (batch_seq, x_word)

# tril_mask
def make_tril_mask(x, pad_idx=0):
    # x : (batch_seq, x_word)
    pad_mask = (x != pad_idx).unsqueeze(1).to(x.device)     # (batch_seq, 1, x_word)
    
    tril_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).bool().to(x.device)  # (batch_seq, batch_seq)
    
    mask = (pad_mask & tril_mask)    # (batch_seq, x_word, x_word)
    # (diagonal 이용하여) batch_seq에 따라 순차적  mask적용 
    
    return mask   # (batch_seq, x_word, x_word)



# ★★ Encoder
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding=None):
        super().__init__()
        self.embed_layer = EmbeddingLayer(vocab_size_X, embed_dim)
        self.posembed_layer = PositionalEncodingLayer(encoding=pos_encoding)
        self.dropout = torch.nn.Dropout(dropout)

        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])

    def forward(self, X, X_mask=None):
        # X : (batch_Seq, X_word)
        
        # embedding layer
        self.X_embed = self.embed_layer(X)  # (batch_seq, X_word, emb)
        
        # positional encoding
        self.X_posembed = self.posembed_layer(self.X_embed).unsqueeze(0).repeat(X.shape[0], 1, 1)     # (batch_seq, X_word, emb)
        
        if X_mask is not None:
            mask = X_mask.squeeze().unsqueeze(-1).repeat(1, 1, self.X_posembed.shape[-1])
            self.X_posembed.masked_fill_(mask==0, 0)
        
        # sum of X_emb_scaled and pos_emb_X
        self.X_input = self.dropout(self.X_embed + self.X_posembed)     # (batch_seq, X_word, emb)

        # encoder layer
        next_input = self.X_input
        
        for enc_layer in self.encoder_layers:
            next_input = enc_layer(next_input, X_mask)
        self.encoder_output = next_input

        return self.encoder_output  # (batch_seq, X_word, emb)


# ★ EmbeddingLayer
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embed_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.scaled = embed_dim ** (1/2)
        
        # attribute
        self.weight = self.embed_layer.weight
    
    def forward(self, X):
        self.emb_scaled = self.embed_layer(X) * self.scaled   # (batch_seq, X_word, emb)

        return self.emb_scaled



# ★ PositionalEncodingLayer
# https://velog.io/@sjinu/Transformer-in-Pytorch#3-positional-encoding
# functional : postional_encoding
def positional_encoding(x, encoding=None):
    """
     encoding : None, 'sinusoid'
    """
    if x.ndim == 2:
        batch_size, seq_len = x.shape
        
        if encoding is None:
            pos_encode = torch.arange(seq_len).requires_grad_(False).to(x.device)
        elif 'sin' in encoding:
            pos_encode = torch.sin(torch.arange(seq_len)).requires_grad_(False).to(x.device)
            
    elif x.ndim == 3:
        batch_size, seq_len, embed_dim = x.shape
        
        if encoding is None:
            pos_encode = torch.arange(seq_len).unsqueeze(-1).repeat(1, embed_dim).requires_grad_(False).to(x.device)
        elif 'sin' in encoding:
            pos_encode = torch.zeros(seq_len, embed_dim).requires_grad_(False).to(x.device)         # (seq_len, emb)
            pos_inform = torch.arange(0, seq_len).unsqueeze(1) # (seq_len, 1)
            index_2i = torch.arange(0, embed_dim, step=2)       # (emb)
            pos_encode[:, ::2] = torch.sin(pos_inform/(10000**(index_2i/embed_dim)))       # (seq_len, emb)

            if embed_dim % 2 == 0:
                pos_encode[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i/embed_dim)))
            else:
                pos_encode[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i[:-1]/embed_dim)))
    return pos_encode

# Class : PositionalEncodingLayer
class PositionalEncodingLayer(torch.nn.Module):
    def __init__(self, encoding=None):
        super().__init__()
        self.encoding = encoding
        
    def forward(self, x):
        return positional_encoding(x, self.encoding)


# class LinearPositionalEncodingLayer(torch.nn.Module):
#     def __init__(self, max_len=300, embed_dim=256):
#         super().__init__()
#         pos_embed = torch.arange(0, max_len).unsqueeze(1).repeat(1,embed_dim).requires_grad_(False)

#         # self.pos_embed = pos_embed    # (max_len, emb)
#         self.register_buffer('pos_embed', pos_embed)      # 학습되지 않는 변수로 등록

#     def forward(self, x):
#         # x : (batch_Seq, x_word, emb)
#         self.pos_embed = self.pos_embed[:x.shape[1]]
#         self.pos_embed_output = torch.autograd.Variable(self.pos_embed, requires_grad=False).to(x.device)
#         return self.pos_embed_output       # (x_word, emb)

# class PositionalEncodingLayer(torch.nn.Module):
#     def __init__(self, max_len=300, embed_dim=256):
#         super().__init__()

#         pos_embed = torch.zeros(max_len, embed_dim).requires_grad_(False)         # (max_len, emb)
#         pos_inform = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
#         index_2i = torch.arange(0, embed_dim, step=2)       # (emb)
#         pos_embed[:, ::2] = torch.sin(pos_inform/(10000**(index_2i/embed_dim)))       # (max_len, emb)

#         if embed_dim % 2 == 0:
#             pos_embed[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i/embed_dim)))
#         else:
#             pos_embed[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i[:-1]/embed_dim)))

#         # self.pos_embed = pos_embed    # (max_len, emb)
#         self.register_buffer('pos_embed', pos_embed)      # 학습되지 않는 변수로 등록

#     def forward(self, x):
#         # x : (batch_Seq, x_word, emb)
#         self.pos_embed = self.pos_embed[:x.shape[1]]
#         self.pos_embed_output = torch.autograd.Variable(self.pos_embed, requires_grad=False).to(x.device)
#         return self.pos_embed_output       # (x_word, emb)







# ★ Encoder_Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, X_emb, X_mask=None):
        # X_emb : (batch_seq, X_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        
        # (Self Attention Layer) ------------------------------------------------------------------
        self.X_self_att_output  = self.self_att_layer((X_emb, X_emb, X_emb), mask=X_mask)
        self.self_attention_score = self.self_att_layer.attention_score
        
        #  (batch_seq, X_word, fc_dim=emb), (batch_seq, n_heads, X_word, key_length=X_word)
        self.X_skipconnect_1 = X_emb + self.dropout(self.X_self_att_output)   # (batch_seq, X_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_1 = self.self_att_layer_norm(self.X_skipconnect_1)  # layer normalization
        
        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.X_posff = self.posff_layer(self.X_layer_normed_1)    # (batch_seq, X_word, emb)
        self.X_skipconnect_2 = self.X_layer_normed_1 + self.dropout(self.X_posff)     # (batch_seq, X_word, emb)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_2 = self.posff_layer_norm(self.X_skipconnect_2)

        return self.X_layer_normed_2   # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)


# ☆ MultiHeadAttentionLayer
class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, dropout=0):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim은 n_head의 배수값 이어야만 합니다.'

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

        self.att_layer = ScaledDotProductAttention(embed_dim ** (1/2), dropout)
        self.fc_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        query, key, value = x
        # query, key, value : (batch_seq, len, emb)
        with torch.no_grad():
            batch_size = query.shape[0]

        self.query = self.query_layer(query)    # (batch_seq, query_len, emb)
        self.key   = self.key_layer(key)        # (batch_seq, key_len, emb)
        self.value = self.value_layer(value)    # (batch_seq, value_len, emb)

        self.query_multihead = self.query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, query_len, head_emb_dim)   ←permute←  (batch_seq, query_len, n_heads, head_emb_dim)
        self.key_multihead   =   self.key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, key_len, head_emb_dim)   ←permute←  (batch_seq, key_len, n_heads, head_emb_dim)
        self.value_multihead = self.value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, value_len, head_emb_dim)   ←permute←  (batch_seq, value_len, n_heads, head_emb_dim)

        self.weighted, self.attention_score = self.att_layer((self.query_multihead, self.key_multihead, self.value_multihead), mask=mask)
        # self.weightd          # (B, H, QL, HE)
        # self.attention_score  # (B, H, QL, QL)     ★

        self.weighted_arange = self.weighted.permute(0,2,1,3).contiguous()        # (B, QL, H, HE) ← (B, H, QL, HE)
        self.weighted_flatten = self.weighted_arange.view(batch_size, -1, self.embed_dim)   # (B, QL, E) ← (B, H, E)

        self.multihead_output = self.fc_layer(self.weighted_flatten)       # (B, QL, FC)
        return self.multihead_output       #  (batch_seq, query_length, fc_dim)


# * ScaledDotProductAttention
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, scaled=1, dropout=0):
        super().__init__()
        self.scaled = scaled
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, epsilon=1e-10):
        query, key, value = x

        self.energy = torch.matmul(query, key.transpose(-1,-2)) / self.scaled    # (B, ..., S, S) ← (B, ..., S, W), (B, ..., W, S)
        # * summation of muliply between embedding vectors : Query에 해당하는 각 Length(단어) embedding이 어떤 key의 Length(단어) embedding과 연관(내적)되는지?

        if mask is not None:
            # masking 영역(==0)에 대해 -epsilon 으로 채우기 (softmax → 0)
            self.energy = self.energy.masked_fill(mask==0, -epsilon)

        self.attention_score = torch.softmax(self.energy, dim=-1)

        self.weighted = torch.matmul(self.dropout_layer(self.attention_score), value)    # (B, ..., S, W) ← (B, ..., S, S), (B, ..., S, W)
        # * summation of muliply between softmax_score and embeding of value

        return self.weighted, self.attention_score


# ☆ Positionalwise_FeedForward_Layer
class PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, posff_dim=512, dropout=0, activation='ReLU'):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.fc_layer_1 = torch.nn.Linear(embed_dim, posff_dim)
        self.activation =  eval(f"torch.nn.{activation}()") if type(activation) == str else (activation() if isinstance(activation, type) else None)
        self.fc_layer_2 = torch.nn.Linear(posff_dim, embed_dim)
        
    def forward(self, X):
        # X : (batch_seq, X_word, emb)
        self.ff_output_1 = self.dropout(self.activation(self.fc_layer_1(X)))    # (batch_seq, X_word, posff_dim)
        self.ff_output_2 = self.fc_layer_2(self.ff_output_1)    # (batch_seq, X_word, emb)
        return self.ff_output_2  # (batch_seq, X_word, emb)


# ★★ Decoder 
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding=None):
        super().__init__()
        
        self.embed_layer = EmbeddingLayer(vocab_size_y, embed_dim)
        self.posembed_layer = PositionalEncodingLayer(encoding=pos_encoding)
        self.dropout = torch.nn.Dropout(dropout)

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])
        
    def forward(self, y, context_matrix, X_mask=None, y_mask=None):
        # y : (batch_seq, y_word)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)
        

        # embedding layer
        self.y_embed = self.embed_layer(y)  # (batch_seq, y_word, emb)
        
        # positional encoding
        self.y_posembed = self.posembed_layer(self.y_embed).unsqueeze(0).repeat(y.shape[0], 1, 1)     # (batch_seq, y_word, emb)
        
        # if y_mask is not None:
            # mask = y_mask.squeeze().unsqueeze(-1).repeat(1, 1, self.y_posembed.shape[-1])
            # self.y_posembed.masked_fill_(mask==0, 0)
            
        # sum of X_emb_scaled and pos_emb_X
        self.y_input = self.dropout(self.y_embed + self.y_posembed)     # (batch_seq, y_word, emb)
        
        # decoder layer
        next_input = self.y_input
        
        for dec_layer in self.decoder_layers:
            next_input = dec_layer(next_input, context_matrix, X_mask, y_mask)
        self.decoder_output = next_input
        
        return self.decoder_output

# ★ Decoder_Layer
class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.enc_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.enc_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, y_emb, context_matrix, X_mask=None, y_mask=None):
        # y_emb : (batch_seq, y_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)     # encoder output
        
        # (Self Attention Layer) -------------------------------------------------------------------
        self.y_self_att_output = self.self_att_layer((y_emb, y_emb, y_emb), mask=y_mask)
        #  (batch_seq, y_word, fc_dim=emb)        
        self.self_attention_score = self.self_att_layer.attention_score
        # (batch_seq, n_heads, y_word, key_length=y_word)
        
        self.y_skipconnect_1 = y_emb + self.dropout(self.y_self_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_1 = self.self_att_layer_norm(self.y_skipconnect_1)  # layer normalization
        
        # (Encoder Attention Layer) ----------------------------------------------------------------
        self.y_enc_att_output = self.enc_att_layer((self.y_layer_normed_1, context_matrix, context_matrix), mask=X_mask)
        #  (batch_seq, y_word, fc_dim=emb)
        self.attention_score = self.enc_att_layer.attention_score
        # (batch_seq, n_heads, y_word, key_length=y_word)
        
        self.y_skipconnect_2 = self.y_layer_normed_1 + self.dropout(self.y_enc_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 encoder_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_2 = self.enc_att_layer_norm(self.y_skipconnect_2)  # layer normalization

        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.y_posff = self.posff_layer(self.y_layer_normed_2)    # (batch_seq, y_word, emb)
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_3 = self.posff_layer_norm(self.y_posff)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        return self.y_layer_normed_3    # (batch_seq, y_word, emb)
#######################################################################################################################################


# x = torch.tensor(randint(3,15,high=6, sign='+'))
# y = torch.tensor(randint(3,10,high=2, sign='+'))

# x = (torch.rand(3,15)*6).type(torch.long)
# y = (torch.rand(3,10)*2).type(torch.long)

# tr = Transformer(7, 5, 0, 0, n_layers=3, pos_encoding=None)
# tr = Transformer(7, 5, 0, 0, pos_encoding='sin')
# tr(x,y).shape
# tr.predict(x).shape
# tr.attention_score.shape




# # customize library ***---------------------
# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
# from DS_DeepLearning import EarlyStopping

import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=100)
# # ------------------------------------------
import time


# training prepare * -------------------------------------------------------------------------------------------------------
# X_sample.shape, y_sample.shape

model = Transformer(vocab_size_X, vocab_size_y, 0, 0).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0, n_layers=3, n_heads=8, dropout=0.5, pos_maxlen=150).to(device)
# model = Transformer(vocab_size_X, vocab_size_y, 0, 0, n_layers=2, n_heads=8, dropout=0.5, pos_maxlen=150).to(device)
# model(X_sample.to(device), y_sample.to(device)).shape


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
    pred_sentence, attention_scores = model.predict(torch.tensor(sentence_input).to(device))
    # pred_sentence_ = model(torch.tensor(sentence_input).to(device), torch.tensor(sentence_output).to(device))
    # pred_sentence = torch.argmax(pred_sentence_, dim=2)
print(pred_sentence)



sentence_en = np.stack([[index_word_X[word] if word != 0 else '' for word in sentence] for sentence in sentence_input])[0]
sentence_kr_real = np.stack([[index_word_y[word] if word != 0 else '' for word in sentence] for sentence in sentence_output])[0]
sentence_kr_pred = np.stack([[index_word_y[word] if word != 0 else '' for word in sentence] for sentence in pred_sentence.to('cpu').detach().numpy()])[0]

print(sentence_kr_real)
print(sentence_kr_pred)



# attention map ---------------------------------------------------------------------------------------
import seaborn as sns
pred_slice_end = np.where(sentence_kr_pred=='<EOS>')[0][0]
trg_slice_end = np.where(sentence_en=='<EOS>')[0][0]

f = plt.figure(figsize=(10,10))
for h in range(model.encoder_attention.shape[1]):
    plt.subplot(3,3,h+1)
    sns.heatmap(model.encoder_attention[0,h,:13,1:12].to('cpu').numpy(), cmap='bone')
    plt.xticks(np.arange(sentence_en.shape[0])[:13], sentence_en[:13], rotation=90)
    plt.yticks(np.arange(sentence_kr_pred.shape[0])[1:12], sentence_kr_pred[1:12], rotation=0)
plt.close()







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






