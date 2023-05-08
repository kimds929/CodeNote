from torchtext.data import Field, BucketIterator, interleave_keys
from torchtext.datasets import TranslationDataset
from torchtext.data import Example
from mosestokenizer import *
import torch

from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math
import time

import os
# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 8) Natural_Language_Processing\Dataset'
path = r'/home/pirl/data/8_NLP/'
origin_path = os.getcwd()
os.chdir(path)


# Author: WonKee Lee (POSTECH)
# "Neural Machine Translation by Jointly Learning to Align and Translate" 논문의 model 재현 (Toy code)
#  (https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html 를 참고하여 수정함.)




### torchtext #####
BOS = '<s>'    # Start symbol
EOS = '</s>'   # End symbol
PAD = '<pad>'  # padding symbol

# ex) 'I am a boy.' -> ['I', 'am', 'a', 'boy']
tok_en = MosesTokenizer('en')
tok_fr = MosesTokenizer('fr')

# Field: Tensor로 표현할 데이터의 타입, 처리 프로세스 등을 정의하는 객체
src = Field(sequential=True,
            use_vocab=True,
            pad_token=PAD,
            tokenize=tok_en,
            lower=True,
            batch_first=True) # if=True shape:[Batch, length] else shape=[length, Batch]

tgt = Field(sequential=True,
            use_vocab=True,
            pad_token=PAD,
            tokenize=tok_fr,
            lower=True,
            init_token=BOS,
            eos_token=EOS,
            batch_first=True)



prefix_f = 'data'

# parallel data 각각 (en, de) 을 src Field 와 tgt Field에 정의된 형태로 처리.
parallel_dataset = TranslationDataset(path=prefix_f, exts=('.en', '.fr'), 
                                      fields=[('src', src), ('tgt', tgt)])

print(parallel_dataset) 

print(parallel_dataset.examples[22222].__dict__.items()) # src 및 tgt 에 대한 samples 를 포함.



print(parallel_dataset.examples[22222].src) # src 출력 방법

print(parallel_dataset.examples[22222].tgt) # tgt 출력 방법


##### 사전 구축 ########
# src, tgt 필드에 사전 구축
src.build_vocab(parallel_dataset, max_size=15000)
tgt.build_vocab(parallel_dataset, max_size=15000)


# 사전 내용 
print(src.vocab.__dict__.keys())
print('')
# stoi : string to index 의 약자
for i, (k, v) in enumerate(src.vocab.stoi.items()):
    print ('{:>10s} | {:>3d}'.format(k, v))
    if i == 15 : break

train, valid = parallel_dataset.split(split_ratio=0.95) # 0.95 = train / 0.05 = valid 데이터로 분할


# Batch iterator 생성.
# iterator 를 반복하며 batch (src, tgt) 가 생성 됨.
BATCH_SIZE = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = BucketIterator.splits((train, valid), batch_size=BATCH_SIZE,
                                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.tgt)),
                                                    device=device)


# iterator 는 Batch 객체 (Tensor) 를 출력해주며, 
# Batch.src / Batch.tgt 로 parallel data각각에 대해 접근가능.

# 예시.
Batch = next(iter(train_iterator)) 


# src 에 저장된 데이터 출력
# Field에 정의된 형식으로 데이터 전처리 (indexing 포함.)
# 가장 긴 문장을 기준으로, 그 보다 짧은 문장은 Padding idx(=1) 을 부여.
Batch.src 


# Field에 정의된 형식으로 데이터 전처리 (indexing + bos + eos + pad 토큰 처리 됨.)
Batch.tgt 





## Network 정의


### Encoder 정의.

class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, src_ntoken: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.src_ntoken = src_ntoken

        self.embedding = nn.Embedding(src_ntoken, hidden_dim, 
                                      padding_idx=src.vocab.stoi['<pad>'])
        
        self.rnn = nn.GRU(hidden_dim, hidden_dim, bidirectional = True, 
                          batch_first=True) # batch_first = [B, L, dim]
        
        
        # bidirectional hidden을 하나의 hidden size로 mapping해주기 위한 Linear
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = (Batch, Length) Tensor
        embedded = self.dropout(self.embedding(src)) # shape = (Batch, Length, hidden_dim)

        # outputs: [B, L, D*2], hidden: [2, B, D] -> [1, B, D] + [1, B, D]
        # Note: Bidirection=False 인 경우:  outputs: [B, L, D], hidden: [1, B, D]
        outputs, hidden = self.rnn(embedded)

        last_hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) # bidirection Dim(x2)을 projection --> [B, D]
        hidden = torch.tanh(last_hidden).unsqueeze(0) # last bidirectional hidden (=Decoder init hidden) --> [1, B, D]

        return outputs, hidden


### Attention 모듈 정의 ###

class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        attn_in = (enc_hid_dim * 2) + dec_hid_dim # bidirectional hidden + dec_hidden
        self.linear = nn.Linear(attn_in, attn_dim)
        self.merge = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hiden = (Batch, 1, Dim) 길이가 1씩 들어오기 때문.
        src_len = encoder_outputs.shape[1] 
        repeated_decoder_hidden = decoder_hidden.repeat(1, src_len, 1) # [B, src_len, D] -> 각각의 src단어와 연산해주기 위해 늘려준 결과.

        # enc의 각 step의 hidden + decoder의 hidden 의 결과값 # [B, src_len, D*2] --> [B, src_len, D]
        # tanh(W*h_dec  + U*h_enc) 수식 부분.
        energy = torch.tanh(self.linear(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2))) 

        score = self.merge(energy).squeeze(-1) # [B, src_len] 각 src 단어에 대한 점수 -> V^T tanh(W*h_dec  + U*h_enc) 부분
        normalized_score = F.softmax(score, dim=1)  # softmax를 통해 확률분포값으로 변환
        return  normalized_score



### Decoder 모듈 정의 ####

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, dec_ntoken: int, dropout: int):
        super().__init__()

        self.hidden_dim = hidden_dim # Decoder RNN의 previous hidden
        self.dropout = dropout
        self.attention = Attention(enc_hid_dim=hidden_dim, 
                                   dec_hid_dim=hidden_dim, 
                                   attn_dim=hidden_dim) # attention layer
        
        self.dec_ntoken = dec_ntoken # tgt vocab_size

        self.embedding = nn.Embedding(dec_ntoken, hidden_dim, 
                                      padding_idx=tgt.vocab.stoi['<pad>'])
        
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True) # bidirectinal=False 임.
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_dim*3, dec_ntoken) # Vocab 크기로 linear projection
        self.sm = nn.LogSoftmax(dim=-1) # 확률 분포 값.

    def _context_rep(self, dec_out, enc_outs):
        scores = self.attention(dec_out, enc_outs) # score = [B, src_len]
        scores = scores.unsqueeze(1) # [B, 1, src_len] -> weight value (softmax)

        # scores: (batch, 1, src_len),  ecn_outs: (Batch, src_len, dim)
        context_vector = torch.bmm(scores, enc_outs) # weighted average -> (batch, 1, dec_dim): encoder의 각 hidden의 weighted sum
        return context_vector

    def forward(self, input, decoder_hidden, encoder_outputs):
        dec_outs = []
        embedded = self.dropout(self.embedding(input)) # (Batch, length, Dim)
        
        # (Batch, 1, dim)  (batch, 1, dim) , ....,
        for emb_t in embedded.split(1, dim=1): # Batch 별 각 단어 (=각 time step) 에 대한 embedding 출력 
            rnn_out, decoder_hidden = self.rnn(emb_t, decoder_hidden) # feed input with previous decoder hidden at each step

            context = self._context_rep(rnn_out, encoder_outputs) # C_t vector
            rnn_context = self.dropout(torch.cat([rnn_out, context], dim=2)) 
            dec_out = self.linear(rnn_context) # W(H + C) 
            dec_outs += [self.sm(dec_out)]
        
        if len(dec_outs) > 1:
            dec_outs = dec_outs[:-1] # trg = trg[:-1] # <E> 는 Decoder 입력으로 고려하지 않음.
            dec_outs = torch.cat(dec_outs, dim=1) # convert list into tensor : [B, L, vocab]

        else: # step-wise 로 decoding 하는 경우,
            dec_outs = dec_outs[0] # [B=1, L=1, vocab]

        return dec_outs, decoder_hidden



### Seq-to-Seq 모델 정의 ###

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        encoder_outputs, hidden = self.encoder(src) # encoder_outputs = (Batch, length, Dim * 2) , hidden = (Batch, Dim)
        dec_out, _ = self.decoder(trg, hidden, encoder_outputs)
        return dec_out



INPUT_DIM = len(src.vocab)  # src 사전 크기
OUTPUT_DIM = len(tgt.vocab) # tgt 사전 크기
HID_DIM = 128 # rnn, embedding, 등. 모든 hidden 크기를 해당 값으로 통일함. (실습의 용이성을 위함.)
D_OUT = 0.1 # Dropout  확률
BATCH_SIZE = 26

train_iterator, valid_iterator = BucketIterator.splits((train, valid), batch_size=BATCH_SIZE,
                                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.tgt)),
                                                    device=device)



# 인코더 및 디코더 생성
# Seq2Seq 모델 생성
encoder = Encoder(HID_DIM, INPUT_DIM, D_OUT)
decoder = Decoder(HID_DIM, OUTPUT_DIM, D_OUT)
model = Seq2Seq(encoder, decoder, device).to(device)



def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights) # 모델 파라미터 초기화
optimizer = optim.Adam(model.parameters(), lr=0.0005) # Optimizer 설정
criterion = nn.NLLLoss(ignore_index=tgt.vocab.stoi['<pad>'], reduction='mean') # LOSS 설정



# 모델 정보 및 파라미터 수 출력
def count_parameters(model: nn.Module):
    print(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')



## 모델 학습 함수 ###
def train(model, iterator, optimize, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.tgt

        optimizer.zero_grad()

        output = model(src, tgt) # [batch, length, vocab_size]
        output = output.view(-1, output.size(-1)) # flatten --> (batch * length, vocab_size)

        tgt = tgt.unsqueeze(-1)[:,1:,:].squeeze(-1).contiguous() # 정답에는 <S>가 포함되지 않음으로, 이를 삭제
        tgt = tgt.view(-1) # flatten = (batch * length)

        loss = criterion(output, tgt) # tgt 이 내부적으로 one_hot으로 변환됨 --> (batch * length, vocab_size)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        if(((i+1) % int(len(iterator)*0.2)) == 0):
            num_complete = batch.batch_size * (i+1)
            total_size = batch.batch_size * int(len(iterator))
            ratio = num_complete/total_size * 100
            print('| Current Epoch:  {:>4d} / {:<5d} ({:2d}%) | Train Loss: {:3.3f}'.
                  format(num_complete, batch.batch_size * int(len(iterator)), round(ratio), loss.item())
                  )

    return epoch_loss / len(iterator)



### 모델 평가 함수 ###
def evaluate(model: nn.Module, iterator: BucketIterator,
             criterion: nn.Module):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt

            output = model(src, tgt)
            output = output.view(-1, output.size(-1)) # flatten (batch * length, vocab_size)

            tgt = tgt.unsqueeze(-1)[:,1:,:].squeeze(-1).contiguous() # remove <S> placed at first from targets
            tgt = tgt.view(-1) # flatten target with shape = (batch * length)
            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



# 학습 시간 카운트를 위한 함수 #
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



N_EPOCHS = 15 # 최대 epoch 크기
CLIP = 0.2 # weight cliping 
isTrain = True # True 인 경우 아래 학습 코드 실행, False인 경우 저장된 model 로드만 수행.

if isTrain:
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('='*65)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('='*65)

    with open('NMT.pt', 'wb') as f:
        print("model saving..")
        torch.save(model, f)

else:
    with open('NMT.pt', 'rb') as f:
        model = torch.load(f).to(device)





## Greedy decoding 
def greedy_decoding(model, input, fields, maxLen=20):
    src_field = [('src', fields[0])]
    tgt_field = fields[1]

    ex = Example.fromlist([input], src_field) # field에 정의된 내용으로 전처리 (tokenizing) 수행
    src_tensor = src.numericalize([ex.src], device) # torch.Tensor로 치환, indexing, bos, eos 등의 처리과정도 함께 적용됨.
    tgt_tensor = torch.tensor([[tgt_field.vocab.stoi['<s>']]], device=device) # Decoder 초기 입력 
    model.eval()

    dec_result = []
    with torch.no_grad():
        enc_out, hidden = model.encoder(src_tensor)
        for i in range(maxLen):
            dec_step, hideen = model.decoder(tgt_tensor, hidden, enc_out)
            val, idx = torch.topk(dec_step, 1)

            if tgt_field.vocab.itos[idx] == '</s>':
                break
            else:
                dec_result.append(idx.item())
                tgt_tensor = idx.view(1,1)
            # Step1: tgt_tensor (입력) 과 인코더의 출력을 이용하여 디코더 결과 출력
            # Do someting about Step1 here..
            # --> dec_step, hidden = model.decoder(....)
            
            # Step2: 디코더의 출력결과 (확룰분포) 에서 Top1 에 해당하는 word Index 추출
            # Do someting about Step2 here..
            # use torch.topk(..) 
            
            # Step3: 
            # if: 출력된 word Index == EOS 인 경우 디코딩 중지 (break).
            # else: 출력된 word Index를 저장하고, 다음 step의 디코더 입력 (tgt_tensor)으로 전달
           
    dec_result = [tgt_field.vocab.itos[w] for w in dec_result] # Word index를 단어로 치환
    return dec_result




# Greedy decoding 수행
input_sent = input('Enter a english sentence:  ')
output = greedy_decoding(model, input_sent, fields=(src, tgt))
output = MosesDetokenizer('fr')(output)
print('> ', input_sent)
print('< ', output)
print()

