import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np

import os
import random
import collections
import time

from konlpy.tag import Kkma

# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 8) Natural_Language_Processing\Dataset'
path = r'/home/pirl/data/8_NLP/'
origin_path = os.getcwd()
os.chdir(path)



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # gpu가 사용가능 할 때 cuda로 입력시키기 위한 변수
tokenizer = Kkma().morphs # 추후에 사용할 한국어 문장 tokenizer

# 학습을 위한 데이터 로딩

class TextIterator(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		for line in open(self.fname):
			yield line.split()

filename = 'newskor.txt'
sentences = TextIterator(filename)




# vocab을 만들기 위한 class 생성
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


# 데이터를 Train과 Valid set으로 분류하고 indexing을 하는 클래스 생성
class Corpus(object):
    def __init__(self, sentences, device, vocab_size, train_ratio):
        self.device = device
        
        dataset = self.data_shuffle(sentences)
        # dataset = dataset[:int(len(dataset) / 10)]  # 
        train_size = int(len(dataset) * train_ratio)
        
        self.train = dataset[:train_size]
        self.valid = dataset[train_size:]
        self.dictionary = Dictionary(self.train, vocab_size)

    def data_shuffle(self, sentences): # Data를 전체적으로 한 번 섞는 함수
        data = []
        for sentence in sentences:
            data.append(sentence)
        random.shuffle(data)
        
        return data
    
    def iteration(self, batch_size, isTrain=True):  # 배치단위로 뱉어낼때 쓰는 함수
        data = self.train if isTrain else self.valid
        if isTrain: # Train set의 경우 shuffle을 진행해서 학습 때 마다 batch 안에 다른 데이터가 들어갈 수 있게 만듬
            random.shuffle(data)

        for i in range((len(data) // batch_size) + 1): # Batch의 크기만큼 데이터를 잘라서 indexing을 시킴
            batch = data[i * batch_size: (i+1) * batch_size]
            src, tgt = self.indexing(batch)
            yield (src, tgt)    
            
    def indexing(self, batch):
        src_idxes = []
        tgt_idxes = []
        
        for sent in batch:
            src_idx = [self.dictionary('<sos>')] + [self.dictionary(word) for word in sent]
            tgt_idx = [self.dictionary(word) for word in sent] + [self.dictionary('<eos>')]
            src_idxes.append(torch.tensor(src_idx).type(torch.int64))
            tgt_idxes.append(torch.tensor(tgt_idx).type(torch.int64))
        # self.dictionary(w1)
        # batch 안에 문장을 하나씩 불러서 src와 tgt 문장으로 indexing
        # src는 앞에 start token, tgt는 뒤에 end token을 붙임
        # tensor로 변환해서 idxes에 저장
        
        # Padding이 들어간 tensor로 변환
        src_idxes = rnn.pad_sequence(src_idxes, batch_first=True).to(self.device) # size= [B, L]
        tgt_idxes = rnn.pad_sequence(tgt_idxes, batch_first=True).to(self.device).view(-1) # flatten = [B * L]

        return src_idxes, tgt_idxes




# Corpus를 구축할 때 입력할 hyperparameters
vocab_size = 15000
train_ratio = 0.9

# Corpus 구축
corpus = Corpus(sentences, device, vocab_size, train_ratio)

# 구축한 corpus가 제대로 작동하는지 확인

sent1 = tokenizer("나는 교육생입니다.")
sent2 = tokenizer("오늘 빨리 마쳤으면 좋겠다.")

batch = [sent1, sent2]
idx_src, idx_tgt = corpus.indexing(batch)

print("배치에 들어간 문장은 {}입니다.".format(batch))
print("Src 문장은 {}입니다.".format(idx_src))
print("Tgt 문장은 {}입니다.".format(idx_tgt))





# 모델 정의
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers, dropout=0.1):
        super(LSTMLanguageModel, self).__init__()
        # 모델 layer 생성
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder = nn.LSTM(hidden_size, hidden_size, nlayers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # 모델 안에서 사용할 hyperparameter 선언
        self.ntoken = vocab_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        # layer 내에 parameter 초기화
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        emb = self.embeddings(x) # emb = (B, L, D)
        x, hidden = self.encoder(emb, hidden) # x = (B, L, D)
        x = self.drop(x)
        out = self.decoder(x) # out = (B, L, V)
        out = out.view(-1, self.ntoken) # out = (B * L, V)

        return self.softmax(out), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.hidden_size),
                weight.new_zeros(self.nlayers, batch_size, self.hidden_size))
    



# Hyperparameters 정의
batch_size = 100
hidden_size = 200
dropout = 0.3
nlayer = 2
max_epoch = 40
lr = 0.003
model_name = "practice"

# 모델 building
model = LSTMLanguageModel(vocab_size, hidden_size, nlayer, dropout).to(device)


# loss func과 optimizer 정의
criterion = nn.NLLLoss(ignore_index=0)
optim = torch.optim.Adam(model.parameters(), lr=lr)

model




# Train data가 들어왔을 때 parameter를 업데이트 하며 학습하는 함수
def train(batch_size, model, corpus, criterion, optim):
    model.train() # dropout을 수행하도록 .train() 메서드 사용.

    for i, (src, tgt) in enumerate(corpus.iteration(batch_size)):
        start_time = time.time() # 시간을 기록하기 위해 start time 기록
        hidden = model.init_hidden(len(src)) # h_0를 0으로 설정
        optim.zero_grad()
        output, hidden = model(src, hidden)

        loss = criterion(output, tgt)
        loss.backward()
        optim.step()
        
        if i % 100 == 99:
            total_time = time.time() - start_time
            print("\r Batch {:3d} | times {:3.3f} | loss: {:3.3f} ".format(i + 1, total_time, loss.item()), end='')





# Valid data가 들어왔을 때 모델을 검증
def evaluate(batch_size, model, corpus, criterion):
    mean_loss = []
    model.eval()

    for (src, tgt) in corpus.iteration(batch_size, isTrain=False):
        with torch.no_grad():
            hidden = model.init_hidden(len(src))
            output, hidden = model(src, hidden)
            loss = criterion(output, tgt)
            mean_loss.append(loss.item())
    # _.eval()을 활용하여 dropout을 적용하지 않음
    # 전체 loss를 mean_loss로 계산
    # torch.no_grad()를 활용하여 parameter update를 수행하지 않음
    mean_loss = torch.mean(torch.FloatTensor(mean_loss), 0)

    return mean_loss





# Trainer 함수를 통해 epoch 마다 loss를 반환하고 일정 epoch이 반복되면 valid의 loss를 반환
def trainer(max_epoch, batch_size, model, corpus, criterion, optim, model_name, valid_every=1):
    start_time = time.time()

    print("Train")
    print('=' * 60)
    best_loss = float('inf') # model selection을 위한 loss 초기화
    for epoch in range(1, max_epoch+1):
        print('epoch {:3d}'.format(epoch))
        print('=' * 60)
        train(batch_size, model, corpus, criterion, optim) # 모델 학습
        print()
        
        if epoch % valid_every == 0: # n epoch 마다 검증 수행
            loss = evaluate(batch_size, model, corpus, criterion)
            if loss < best_loss: # valid loss가 줄어들었을 때 model을 바꾸는 조건문
                best_loss = loss
                torch.save(model, './' + model_name  + '.pt')
            print('=' * 60)
            print('Evaluation | loss: {:3.3f} '.format(loss))
            print('=' * 60)
            
    total_time = time.time() - start_time
    print("학습 끝! 총 학습시간은 {:7.1} 입니다.".format(total_time))


# 학습 시작
trainer(max_epoch, batch_size, model, corpus, criterion, optim, model_name, valid_every=1)

model = torch.load(model_name + '.pt')




# 문장의 점수를 예측하는 함수
def pred_sent_prob(sent, model=model):
    model.eval()

    with torch.no_grad():
        src, tgt = corpus.indexing([sent])
        hidden = model.init_hidden(batch_size=1)
        output, hidden = model(src, hidden)
    
    word_prob = output[np.arange(len(tgt)), tgt].tolist()
    sent_prob = np.mean(word_prob)
    # sent_prob = word_prob[-1]

    # _.eval() 로 dropout 해제
    # torch.no_grad()로 update 안함
    # 입력받는 문장 indexing
    # hidden 초기화
    # model을 통해 vocab 크기의 log softmax 결과값 출력
    # 찾고자 하는 단어의 log 확률값 추출
    # 마지막 값을 반환
    
    return sent_prob


sent1 = tokenizer("오늘 저녁은 맛있을 것 같습니다.")
sent2 = tokenizer("오늘 저녁은 잘 들릴 것 같습니다.")

print("문장 {}의 점수는 {}입니다.".format(sent1, pred_sent_prob(sent1)))
print("문장 {}의 점수는 {}입니다.".format(sent2, pred_sent_prob(sent2)))


# 다음 단어를 예측하는 함수
vocab = corpus.dictionary.idx2word()[3:]    # padding, ukn, start (tocken은 제외)
len(vocab)

def pred_next_token(sent, model=model, vocab=vocab):
    # 현재 문장을 정답 문장으로 선언
    # 현재 문장에 vocab에 있는 단어들을 붙이면서 점수를 계산
    # 점수가 더 높을 경우 정답 문장 update
    answer = sent
    
    for word in vocab:
        new_sent = sent + [word]
        if pred_sent_prob(new_sent) > pred_sent_prob(answer):
            answer = new_sent
    return answer


sent = tokenizer("마치는 시간")
print("{}의 다음 단어는 {}입니다.".format(sent, pred_next_token(sent)[-1]))





# ===== assignment ===========================================================
# 임의의 문장을 입력 받았을 때 문장 완성하는 함수
# 문장의 일부 혹은 단어가 주어졌을때 문장 완성
#   무한 loop에 빠지면 감점 (max_length 지정)
#   문장이 끝나야 하는 조건이 없으면 감점
def writer(sent, max_length=100):
    new_sentence = sent
    print(f'원래문장: {new_sentence}')

    while (len(new_sentence) < max_length):
        recommand_next_word = pred_next_token(sent)[-1]
        if recommand_next_word == '<eos>':
            break
        new_sentence += [recommand_next_word]
        print(f'\r 만든문장: {new_sentence}', end='')
    print()
    return new_sentence

print("완성된 문장은 {} 입니다.".format(writer(sent)))
