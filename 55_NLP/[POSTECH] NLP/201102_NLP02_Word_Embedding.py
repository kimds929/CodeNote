import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import collections

import torch
from torch import nn
import torch.nn.functional as F

# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 8) Natural_Language_Processing\Dataset'
path = r'/home/pirl/data/8_NLP/'
origin_path = os.getcwd()
os.chdir(path)


# Word Embedding : 컴퓨터가 이해할 수 있도록 단어를 vector로 표현



# 학습을 위한 데이터 로딩
class TextIterator(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):     # Generator
		for line in open(self.fname, "r", encoding="utf-8"):
			yield line.split()      # 띄어쓰기 기준으로 하나씩 들어감

filename = 'newskor.txt'
sentences = TextIterator(filename)

# Dictonary (사전)
#   . word2idx : 이 word는 몇번인지?
#   . index2word : index → word
class Dictionary(object):
    def __init__(self, data, size):
        self.word2idx = {'<pad>':0, '<unk>': 1, '<sos>': 2, '<eos>': 3}     # init vocab
        self.build_dict(data, size - 4) # 초기에 4개의 token이 포함되어 있으므로 이를 제외한 나머지 단어로 사전 구성

    def __call__(self, word):
        return self.word2idx.get(word, 1) # 사전에 없는 단어가 call을 받으면 unknown token으로 반환

    def add_word(self, word):
        if word not in self.word2idx:
            length = len(self.word2idx)
            self.word2idx[word] = length
        return self.word2idx[word]

    def build_dict(self, data, dict_size):
        total_words = (word for sent in data for word in sent) # 데이터에 있는 모든 단어 추출
        word_freq = collections.Counter(total_words) # 단어의 갯수를 세서 (단어, 갯수) 형태의 tuple로 이루어진 list 생성
        vocab = sorted(word_freq.items(), reverse=True, key=lambda x: x[1]) # 갯수에 따른 내림차순으로 정렬
        vocab = vocab[:dict_size] # 원하는 갯수만큼 절삭
        for (word, count) in vocab:
            self.add_word(word) # 사전에 해당 단어를 추가

    def __len__(self):
        return len(self.word2idx)
    
    def idx2word(self):
        return list(self.word2idx.keys())



# 사전 생성 후 확인
vocab = Dictionary(sentences, 10000) # 주어진 데이터에서 크기가 10000인 사전 생성

print(len(vocab)) # 사전의 크기 확인
print(vocab.word2idx) # 사전에 포함된 단어와 index 확인



# 단어의 index 확인

WORD = '포스코'
print("{}(이)라는 단어는 {} 번 단어 입니다.".format(WORD, vocab(WORD)))

WORD = '인공지능대학원'
print("{}(이)라는 단어는 {} 번 단어 입니다.".format(WORD, vocab(WORD)))



# 원하는 단어 추가
vocab.add_word('인공지능대학원')

WORD = '인공지능대학원'
print("{}(이)라는 단어는 {} 번 단어 입니다.".format(WORD, vocab(WORD)))



# indexing으로 단어를 찾을 수 있는 list 생성
i2w = vocab.idx2word()
print(i2w[4159])



# Step 2. nn.Embedding() 모듈을 활용하여 embedding layer 생성
# torch.nn.Embedding?
# torch.nn.Embedding(
#     num_embeddings: int,
#     embedding_dim: int,
#     padding_idx: Union[int, NoneType] = None,
#     max_norm: Union[float, NoneType] = None,
#     norm_type: float = 2.0,
#     scale_grad_by_freq: bool = False,
#     sparse: bool = False,
#     _weight: Union[torch.Tensor, NoneType] = None,
# )

# num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
# max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool

# num_embeddings (int) – embedding시킬 사전의 크기
# embedding_dim (int) – embedding vector의 차원 수
# padding_idx (int, optional) – 값을 준다면 해당 값의 index가 들어왔을 때 zero vector로 반환


emb = nn.Embedding(len(vocab), 300, padding_idx=0)
emb         # Embedding(10001, 300, padding_idx=0) # 10,001개의 단어를 300개로 만들겠다


# Step 3. Weight 확인 및 word embedding 결과 vector 확인
# weight 확인
print(emb.weight)
print(emb.weight.size())



# 한국어 문장 tokenizing
from konlpy.tag import Kkma

kkma=Kkma()
sample = "이 시간이 언제 끝날까?"
tokens = kkma.morphs(sample)
print(tokens)


# tokenizing이 된 문장 indexing

def indexing(tokens): # token의 sequence를 사전의 정보를 바탕으로 indexing하는 함수
        idxes = [vocab(word) for word in tokens]
        return torch.tensor(idxes, dtype=torch.int64)

indexed = indexing(tokens)
print(indexed)


# embedding layer를 활용하여 embedding vector look-up
emb(indexed)













# ==================================================================================
import gensim


# Step 1. 주어진 data로 gensim을 활용하여 word2vec 모델 학습

# 학습을 위한 데이터 로딩
class TextIterator(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		for line in open(self.fname):
			yield line.split()

filename = 'newskor.txt'
sentences = TextIterator(filename)

# gensim.model.Word2Vec 을 이용하여 모델 생성
SIZE = 300
WINDOW = 5
SG = 1
MIN_COUNT = 50
WORKERS = 20

# Model
model = gensim.models.Word2Vec(size=SIZE, window=WINDOW, sg=SG, min_count=MIN_COUNT, workers=WORKERS)

# _.build_vocab() 메소드를 이용해서 vocab 구축
model.build_vocab(sentences)

print(list(model.wv.vocab.keys())[:100]) # 구축된 vocabulary 확인

WORD = '사람'

model.wv[WORD] # 주어진 단어의 embedding vector 확인

# _.train()과 _.save() 메소드를 이용하여 모델 학습 및 저장
model.train(sentences, total_words=len(model.wv.vocab), epochs=model.epochs)
model.save('newskor.model')



# Step 2. 모델 활용
# 단어 검증 함수
vocab = model.wv.vocab


def word_checker(word, vocab): # 특정 단어를 입력받아 앞서 구축한 사전에 해당 단어가 있는지 확인하는 함수
    while True:
        if word not in vocab:
            alt_word = input("{} 단어는 사전에 없습니다. 다시 입력해 주세요. ".format(word))
            word = alt_word
        else:
            return word
            break;

# 단어간 유사도를 통해 검증
word1 = input('단어를 입력해주세요: ')
word1 = word_checker(word1, vocab)

word2 = input('비교할 단어를 입력해주세요: ')
word2 = word_checker(word2, vocab)


similarity = model.wv.similarity(word1, word2)
print ('{} 단어와 {} 단어의 유사도는 {:.5f} 입니다. '.format(word1, word2, similarity))






# 주어진 단어들 중 의미가 가장 다른 단어 찾기
text = input('단어를 입력해주세요. (예시: 소프트웨어 네트워크 프로그램 데이터): ')
words = text.split()

for i, word in enumerate(words): # 제시한 단어들 중 사전에 없는 단어를 찾고 변환
    words[i] = word_checker(word, vocab)

mismatched_word = model.wv.doesnt_match(words)
print ('{} 단어들 중 가장 관련성이 낮은 단어는 {} 입니다.'.format(text, mismatched_word))




# 주어진 단어에 가장 유사한 top N 단어 찾기
text = input("단어를 입력해주세요. (예시: 소프트웨어 네트워크 프로그램 데이터): ")
words = text.split() if ' ' in text else [text]
topn = int(input("표시하고 싶은 단어의 갯수를 입력해주세요. : "))

for i, word in enumerate(words): # 제시한 단어들 중 사전에 없는 단어를 찾고 변환
    words[i] = word_checker(word, vocab)
    
positive_words = model.wv.most_similar(positive=word, topn=topn)
#negative_words = model.wv.most_similar(negative=word, topn=topn)

print ('가장 연관성이 높은 단어는 {} 입니다.'.format([word[0] for word in positive_words]))
#print (negative_words)





# 주어진 단어들을 조합했을 때 가장 연관성 높은 단어 찾기
positive_text = input("유사한 단어를 입력해주세요. (예시: 소프트웨어 네트워크 프로그램 데이터): ")
negative_text = input("거리가 먼 단어를 입력해주세요. (예시: 소프트웨어 네트워크 프로그램 데이터): ")
positive_words = positive_text.split() if ' ' in positive_text else [positive_text]
negative_words = negative_text.split() if ' ' in negative_text else [negative_text]
topn = int(input("표시하고 싶은 단어의 갯수를 입력해주세요. : "))

for i, word in enumerate(positive_words): # 제시한 단어들 중 사전에 없는 단어를 찾고 변환
    positive_words[i] = word_checker(word, vocab)
    
for i, word in enumerate(negative_words): # 제시한 단어들 중 사전에 없는 단어를 찾고 변환
    negative_words[i] = word_checker(word, vocab)


mostsimilar = model.wv.most_similar(positive=positive_words, negative=negative_words, topn=3)
print('가장 관련 있는 단어는 {} 입니다.'.format([word[0] for word in mostsimilar]))





# Step 3. 시각화
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# get_ipython().run_line_magic('matplotlib', 'notebook')

from sklearn.decomposition import PCA

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None: # 입력하는 단어 리스트가 없을 때 활용
        if sample > 0: # 샘플의 수를 입력하면 랜덤 초이스
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else: # 샘플의 수를 0 이하의 수로 넣으면 전체 vocab 활용
            words = [ word for word in model.wv.vocab ]
    else:
        for i, word in enumerate(words):
            words[i] = word_checker(word, vocab)
        
    word_vectors = np.array([model.wv[w] for w in words]) # sklearn에 활용할 수 있도록 array type의 embedding vector를 numpy type으로 변환

    twodim = PCA().fit_transform(word_vectors)[:,:2] # 2-dim의 vector로 변형
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
        
display_pca_scatterplot(model, sample=300)





# 번외
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/home/pirl/NLP/2.WE/glove.6B.300d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.save('glove.model')



print(list(model.vocab.keys())[:100])
print(model['of'])



similarity = model.similarity('apple', 'korea')
print(similarity)


















import torch
import torch.nn as nn
import gensim
import collections



# 학습을 위한 데이터 로딩
class TextIterator(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		for line in open(self.fname):
			yield line.split()

filename = 'newskor.txt'
sentences = TextIterator(filename)



class Dictionary(object):
    def __init__(self, data, size):
        self.word2idx = {'<pad>':0, '<unk>': 1, '<sos>': 2, '<eos>': 3} # init vocab
        self.build_dict(data, size - 4) # 초기에 4개의 token이 포함되어 있으므로 이를 제외한 나머지 단어로 사전 구성

    def __call__(self, word):
        return self.word2idx.get(word, 1) # 사전에 없는 단어가 call을 받으면 unknown token으로 반환

    def add_word(self, word):
        if word not in self.word2idx:
            length = len(self.word2idx)
            self.word2idx[word] = length
        return self.word2idx[word]

    def build_dict(self, data, dict_size):
        total_words = (word for sent in data for word in sent) # 데이터에 있는 모든 단어 추출
        word_freq = collections.Counter(total_words) # 단어의 갯수를 세서 (단어, 갯수) 형태의 tuple로 이루어진 list 생성
        vocab = sorted(word_freq.items(), reverse=True, key=lambda x: x[1]) # 갯수에 따른 내림차순으로 정렬
        vocab = vocab[:dict_size] # 원하는 갯수만큼 절삭
        for (word, count) in vocab:
            self.add_word(word) # 사전에 해당 단어를 추가

    def __len__(self):
        return len(self.word2idx)
    
    def idx2word(self):
        return list(self.word2idx.keys())
    
vocab = Dictionary(sentences, 10000)


model = gensim.models.Word2Vec.load('newskor.model')


w2v_tokens = list(model.wv.vocab.keys()) # word2vec model에 있는 단어들의 list
vocab_tokens = vocab.idx2word() # 생성한 사전에 있는 단어들의 list


dim = len(model.wv['.']) # word vector의 dimension의 수
print(dim)

pretrained_emb = torch.Tensor(len(vocab_tokens), dim) # (사전 x 차원의 수)의 크기를 가지는 tensor 선언

for i, token in enumerate(vocab_tokens):
    # 기존 word2vec model에 있는 단어는 해당 emb에 저장하고 없는 경우 random으로 값 부여
    pretrained_emb[i] = torch.FloatTensor(model.wv[token]) if token in w2v_tokens else torch.randn(dim)
pretrained_emb[0] = torch.zeros(dim) # padding token은 zero vector로 저장

print(pretrained_emb)



emb = nn.Embedding.from_pretrained(pretrained_emb, padding_idx=0)
emb

emb.weight
emb.weight.shape

emb(torch.LongTensor([10]))
