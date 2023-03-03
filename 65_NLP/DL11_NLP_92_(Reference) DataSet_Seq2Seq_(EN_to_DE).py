# https://deep-learning-study.tistory.com/686
# https://coffeedjimmy.github.io/transformer/2021/01/02/transformer_implementation_01/
# https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_Attention_Tutorial.ipynb

# pip install -U pip setuptools wheel
# pip install -U spacy
# pip install torchtext==0.6.0
# python -m spacy download en_core_web_sm --user



# %%capture
# !python -m spacy download en
# !python -m spacy download de

# 데이터 전처리(Preprocessing) #######################################################################################
# spaCy 라이브러리: 문장의 토큰화(tokenization), 태깅(tagging) 등의 전처리 기능을 위한 라이브러리
# 영어(Engilsh)와 독일어(Deutsch) 전처리 모듈 설치
import spacy

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')
# spacy_en = spacy.load('en') # 영어 토큰화(tokenization)
# spacy_de = spacy.load('de') # 독일어 토큰화(tokenization)

# 간단히 토큰화(tokenization) 기능 써보기
tokenized = spacy_en.tokenizer("I am a graduate student.")

for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")


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

print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")


# 학습 데이터 중 하나를 선택해 출력
print(vars(train_dataset.examples[30])['src'])
print(vars(train_dataset.examples[30])['trg'])


# 필드(field) 객체의 build_vocab 메서드를 이용해 영어와 독어의 단어 사전을 생성합니다.
# 최소 2번 이상 등장한 단어만을 선택합니다.
SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")


print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
print(TRG.vocab.stoi[""]) # : 2
print(TRG.vocab.stoi[""]) # : 3
print(TRG.vocab.stoi["hello"])
print(TRG.vocab.stoi["world"])
TRG.vocab.stoi



# 한 문장에 포함된 단어가 연속적으로 RNN에 입력되어야 합니다.
# 따라서 하나의 배치에 포함된 문장들이 가지는 단어의 개수가 유사하도록 만들면 좋습니다.
# 이를 위해 BucketIterator를 사용합니다.
# 배치 크기(batch size): 128


import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

BATCH_SIZE = 128

# 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device)

TRG.vocab.stoi[TRG.pad_token]

# import numpy as np
# import pandas as pd
# word_index_src = pd.Series(SRC.vocab.stoi).reset_index()
# word_index_trg = pd.Series(TRG.vocab.stoi).reset_index()
# word_index_src.columns = ['word', 'index']
# word_index_trg.columns = ['word', 'index']

# word_index_src['index'] = word_index_src.apply(lambda x: -1 if (x['word'] != '<unk>') & (x['index']==0) else x['index'], axis=1)
# word_index_trg['index'] = word_index_trg.apply(lambda x: -1 if (x['word'] != '<unk>') & (x['index']==0) else x['index'], axis=1)
# path = r'C:\Users\Admin\Desktop\DataScience\★★ Python_정리자료(Git)\53_Deep_Learning\DL11_NLP'
# word_index_src.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index(DE_SRC).csv', index=False, encoding='utf-8-sig')
# word_index_trg.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index(EN_TRG).csv', index=False, encoding='utf-8-sig')

# for e, batch in enumerate(test_iterator):
#     print(e)
#     src_data.append(batch.src.numpy())
#     trg_data.append(batch.trg.numpy())

# src_frame = pd.DataFrame(batch.src.numpy().T)
# trg_frame = pd.DataFrame(batch.trg.numpy().T)
# print(src_frame.shape, trg_frame.shape)

# src_frame.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(DE_SRC).csv', index=False, encoding='utf-8-sig')
# trg_frame.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(EN_TRG).csv', index=False, encoding='utf-8-sig')



import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# 인코더(Encoder) 아키텍처 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # 양방향(bidirectional) GRU 레이어
        self.rnn = nn.GRU(embed_dim, enc_hidden_dim, bidirectional=True)

        # FC 레이어
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환        
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보
        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, hidden = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 인코더 히든 차원]: 현재까지의 모든 단어의 정보

        # hidden은 [forward_1, backward_1, forward_2, backward_2, ...] 형태로 구성
        # 따라서 hidden[-2, :, :]은 forwards의 마지막 값
        # 따라서 hidden[-1, :, :]은 backwards의 마지막 값
        # 디코더(decoder)의 첫 번째 hidden (context) vector는 인코더의 마지막 hidden을 이용
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        # outputs은 Attention 목적으로, hidden은 context vector 목적으로 사용
        return outputs, hidden

        import torch.nn.functional as F

# 어텐션(Attention) 아키텍처 정의
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        # hidden: [배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        batch_size = enc_outputs.shape[1]
        src_len = enc_outputs.shape[0]

        # 현재 디코더의 히든 상태(hidden state)를 src_len만큼 반복
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # hidden: [배치 크기, 단어 개수, 디코더 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        # energy: [배치 크기, 단어 개수, 디코더 히든 차원]

        attention = self.v(energy).squeeze(2)
        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        return F.softmax(attention, dim=1)


# 디코더(Decoder) 아키텍처 정의
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # GRU 레이어
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embed_dim, dec_hidden_dim)

        # FC 레이어
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embed_dim, output_dim)
        
        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환     
    def forward(self, input, hidden, enc_outputs):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [배치 크기, 히든 차원]
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, 배치 크기]

        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수 = 1, 배치 크기, 임베딩 차원]

        attention = self.attention(hidden, enc_outputs)
        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들
        attention = attention.unsqueeze(1)
        # attention: [배치 크기, 1, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        weighted = torch.bmm(attention, enc_outputs) # 행렬 곱 함수
        # weighted: [배치 크기, 1, 인코더 히든 차원 * 방향의 수]

        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, 배치 크기, 인코더 히든 차원 * 방향의 수]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, 배치 크기, 인코더 히든 차원 * 방향의 수 + embed_dim]: 어텐션이 적용된 현재 단어 입력 정보
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [단어 개수, 배치 크기, 디코더 히든 차원 * 방향의 수]
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 디코더 히든 차원]: 현재까지의 모든 단어의 정보

        # 현재 예제에서는 단어 개수, 레이어 개수, 방향의 수 모두 1의 값을 가짐
        # 따라서 output: [1, 배치 크기, 디코더 히든 차원], hidden: [1, 배치 크기, 디코더 히든 차원]
        # 다시 말해 output과 hidden의 값 또한 동일
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [배치 크기, 출력 차원]
        
        # (현재 출력 단어, 현재까지의 모든 단어의 정보)
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 전체 출력과 문맥 벡터(context vector)를 추출
        enc_outputs, hidden = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0] # 단어 개수
        batch_size = trg.shape[1] # 배치 크기
        trg_vocab_size = self.decoder.output_dim # 출력 차원
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력은 항상  토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_outputs)

            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기
        return outputs







# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0
    
    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()

        output = model(src, trg)
        # output: [출력 단어 개수, 배치 크기, 출력 차원]
        output_dim = output.shape[-1]
        
        # 출력 단어의 인덱스 0은 사용하지 않음
        output = output[1:].view(-1, output_dim)
        # output = [(출력 단어의 개수 - 1) * batch size, output dim]
        trg = trg[1:].view(-1)
        # trg = [(타겟 단어의 개수 - 1) * batch size]
        
        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward() # 기울기(gradient) 계산
        
        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 파라미터 업데이트
        optimizer.step()
        
        # 전체 손실 값 계산
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0
    
    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 평가할 때 teacher forcing는 사용하지 않음
            output = model(src, trg, 0)
            # output: [출력 단어 개수, 배치 크기, 출력 차원]
            output_dim = output.shape[-1]
            
            # 출력 단어의 인덱스 0은 사용하지 않음
            output = output[1:].view(-1, output_dim)
            # output = [(출력 단어의 개수 - 1) * batch size, output dim]
            trg = trg[1:].view(-1)
            # trg = [(타겟 단어의 개수 - 1) * batch size]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)





def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs








INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENCODER_EMBED_DIM = 256
DECODER_EMBED_DIM = 256
ENCODER_HIDDEN_DIM = 512
DECODER_HIDDEN_DIM = 512
ENC_DROPOUT_RATIO = 0.5
DEC_DROPOUT_RATIO = 0.5


# 어텐션(attention) 객체 선언
attn = Attention(ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, ENCODER_EMBED_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM, ENC_DROPOUT_RATIO)
dec = Decoder(OUTPUT_DIM, DECODER_EMBED_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM, DEC_DROPOUT_RATIO, attn)

# Seq2Seq 객체 선언
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

import torch.optim as optim

# Adam optimizer로 학습 최적화
optimizer = optim.Adam(model.parameters())

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


import time
import math
import random

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')


for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'seq2seq_with_attention.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')





# 학습된 모델 저장
from google.colab import files

# files.download('seq2seq_with_attention.pt')

model.load_state_dict(torch.load('seq2seq_with_attention.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')






example_idx = 10

src = vars(test_dataset.examples[example_idx])['src']
trg = vars(test_dataset.examples[example_idx])['trg']

print(f'소스 문장: {src}')
print(f'타겟 문장: {trg}')
print("모델 출력 결과:", " ".join(translate_sentence(src, SRC, TRG, model, device)))



src = tokenize_de("Guten Abend.")

print(f'소스 문장: {src}')
print("모델 출력 결과:", " ".join(translate_sentence(src, SRC, TRG, model, device)))



















































for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg

    print(f"첫 번째 배치 크기: {src.shape}")

    # 현재 배치에 있는 하나의 문장에 포함된 정보 출력
    for i in range(src.shape[0]):
        print(f"인덱스 {i}: {src[i][0].item()}")

    # 첫 번째 배치만 확인
    break

src
trg



# model = Seq2Seq(enc, dec, device).to(device)
# model = Seq2Seq_Model(INPUT_DIM, OUTPUT_DIM).to(device)
model = AttSeq2Seq(INPUT_DIM, OUTPUT_DIM).to(device)
# model(src.permute(1,0), trg.permute(1,0)).shape


import numpy as np
import torch.optim as optim

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)
     

model.train() # 학습 모드
epoch_loss = 0

# Adam optimizer로 학습 최적화
optimizer = optim.Adam(model.parameters())

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
clip = 1

# 전체 학습 데이터를 확인하며
for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg
    
    optimizer.zero_grad()

    # # ----------------------------------------------------------------
    # output = model(src, trg)    # 29, 128, 5892
    # # output: [출력 단어 개수, 배치 크기, 출력 차원]
    # output_dim = output.shape[-1]   # 5892
    
    # # 출력 단어의 인덱스 0은 사용하지 않음
    # output_eval = output[1:].view(-1, output_dim)
    # # output = [(출력 단어의 개수 - 1) * batch size, output dim]
    # trg_eval = trg[1:].view(-1)
    # # trg = [(타겟 단어의 개수 - 1) * batch size]
    # # ----------------------------------------------------------------

    # # ----------------------------------------------------------------
    output = model(src.permute(1,0), trg.permute(1,0))    # 128, 29, 5892
    output_dim = output.shape[-1]   # 5892
    output_eval = output[:,1:,:].reshape(-1, output_dim)
    # output[:,1:,:].is_contiguous()
    trg_eval = trg.permute(1,0)[:,1:].reshape(-1)
    # ----------------------------------------------------------------

    
    # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
    loss = criterion(output_eval, trg_eval)
    loss.backward() # 기울기(gradient) 계산
    
    # 기울기(gradient) clipping 진행
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    # 파라미터 업데이트
    optimizer.step()
    
    # 전체 손실 값 계산
    epoch_loss += loss.item()
    break
epoch_loss

