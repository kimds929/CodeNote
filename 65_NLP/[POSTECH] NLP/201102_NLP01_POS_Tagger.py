import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch
from torch import nn
import torch.nn.functional as F

path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 8) Natural_Language_Processing\Dataset'
# path = r'/home/pirl/data/Lecture12/'
origin_path = os.getcwd()
os.chdir(path)

# Vector Space Model
# Hidden Markov Model (HMM)
# - 모델이 가볍고 쉽게 구현 가능



from collections import defaultdict
import math

def sent_processing(lines):
    if isinstance(lines, list):
        lines = [line.strip().split(" ") for line in lines]

        corpus = []
        for line in lines:
            sent = []
            for word in line:
                word = tuple(word.rsplit("/", 1))
                sent.append(word)
            corpus.append(sent)

        return corpus

    elif isinstance(lines, str):
        line = []
        for word in lines.strip().split(" "):
            word = tuple(word.rsplit("/", 1))
            line.append(word)
        return line

    else:
        print("wrong type of input sentence")
        exit(1)

    
with open("corpus.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()

corpus = sent_processing(lines)
# len(corpus)
# corpus[0]


    
# Train
def train(corpus):
    def bigram_count(sent):
        poslist = [pos for _, pos in sent] # [NN, VBD, DT, NN]
        return [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])]

    # 자료형
    pos2words_freq = defaultdict(lambda: defaultdict(int)) # number of (word, tag)
    trans_freq = defaultdict(int) # bigram count --> (tag-1, tag)
    bos_freq = defaultdict(int) # count for bos bigram --> number of (BOS, tag)

    # sent format: [(word, tag), (word, tag), ....)]
    for sent in corpus: # counting
        for word, pos in sent:
            pos2words_freq[pos][word] +=1

        for bigram in bigram_count(sent):
            trans_freq[bigram] +=1

        bos_freq[sent[0][1]] +=1 # number of (BOS, tag) bigram
        trans_freq[(sent[-1][1], 'EOS')] +=1 # number of (tag, EOS) bigram

    # obervation p(x|y) - emission prob.
    base = {pos: sum(words.values()) for pos, words in pos2words_freq.items()}# P(y) for every y (count for each tag) 
    
    pos2words_prob = defaultdict(lambda: defaultdict(int))  # log(p(x, y)/p(y)) for every (x, y)
    
    # practice 1: emission prob tables: log(p(x, y)/p(y)) ***
    #for ....:
    #    for ....:
    #        pos2words_prob[pos][word] = # calcuate log_prob for p(x|y) here
    for pos, words in pos2words_freq.items():
        for word, count in words.items():
            pos2words_prob[pos][word] = math.log(count / base[pos])

    # practice 2 : transition prob tables: log(p(y_k|p_(k-1))) ***
    base = defaultdict(int)

    #for ...:
    #    complete base[] for trans_prob
    # trans_prob = # calculate log prob for p(y_k, y_(k-1))/p(y_(k-1)) here
    for (pos0, pos1), count in trans_freq.items():
        base[pos0] += count
    trans_prob = {(pos0, pos1): math.log(count/base[pos0]) for (pos0, pos1), count in trans_freq.items()}

    # BOS
    base = sum(bos_freq.values()) # p(BOS) --> 100 + 200 + 55 + .... + 
    bos_prob = {pos: math.log(count/base) for pos, count in bos_freq.items()} # log P(tag|BOS) 

    return pos2words_prob, trans_prob, bos_prob


pos2words, trans, bos = train(corpus)

print('명사 라면의 확률:', pos2words['CMC']['라면']) # 명사 '라면'의 확률 (신라면, 진라면 등.)
print('연결어미 라면의 확률:', pos2words['fmoc']['라면']) # 연결어미 '라면'의 확률 (~ 이라면)


# Hidden Markov Model Calculation
class HMM_tagger(object):
    def __init__(self, pos2words, trans, bos):
        self.pos2words = pos2words
        self.trans = trans
        self.bos = bos
        self.unk = -15      # unknown word (학습된 Table, Dictionary 등에 존재하지 않는 정보에 대한 처리)
        self.eos ='EOS'

    # sent format: [(word, tag), (word, tag), ....)]
    def sent_log_prob(self, sent):
        # emission prob.
        # get emission prob. for each (w, t), otherwise unk value

        # log_prob = 0
        # for word, tag in sent:
        #     log_prob += self.pos2words.get(tag, {}).get(word, self.unk)
        log_prob = sum(
            (self.pos2words.get(tag, {}).get(word, self.unk)) for word, tag in sent
        )

        # bos
        log_prob += self.bos.get(sent[0][1], self.unk) # get BOS prob for the first (w, t)
        # dictionary.get('key', 없는키일경우의 return 값)
        # a = {'a':1, 'b':2}
        # a.get('c', 10)

        # transition prob.
        # sent: [(w_1, t_1), (w_2, t_2), ....)]
        # sent[1:]: [(w_2, t_2), (w_3, t_3), ....)]
        bigrams = [(t0, t1) for (_, t0), (_, t1) in zip(sent, sent[1:])] # every bigram in sentence
        log_prob += sum( (self.trans.get(bigram, self.unk)) for bigram in bigrams)

        # eos
        log_prob += self.trans.get(
            (sent[-1][1], self.eos), self.unk)

        # length norm.  # Normalizing
        log_prob /= len(sent)

        return log_prob


tagger = HMM_tagger(pos2words, trans, bos)
test_sent1= "감기/CMC 는/fjb 줄이/YBD 다/fmof ./g"
test_sent2= "감기/fmotg 는/fjb 줄/CMC 이다/fjj ./g"
print("%s: %f" % (test_sent1, tagger.sent_log_prob(sent_processing(test_sent1))))
print("%s: %f" % (test_sent2, tagger.sent_log_prob(sent_processing(test_sent2))))





# 품사태거 (KoNLPy)
# https://konlpy.org/ko/latest/