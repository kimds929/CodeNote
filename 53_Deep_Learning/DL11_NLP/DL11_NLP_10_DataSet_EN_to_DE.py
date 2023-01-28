# https://deep-learning-study.tistory.com/686
# https://coffeedjimmy.github.io/transformer/2021/01/02/transformer_implementation_01/
# https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_Attention_Tutorial.ipynb

# pip install -U pip setuptools wheel
# pip install -U spacy
# pip install torchtext==0.6.0
# python -m spacy download en_core_web_sm --user

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

SRC = Field(tokenize=tokenize_de, init_token="", eos_token="", lower=True)
TRG = Field(tokenize=tokenize_en, init_token="", eos_token="", lower=True)


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




# 한 문장에 포함된 단어가 연속적으로 RNN에 입력되어야 합니다.
# 따라서 하나의 배치에 포함된 문장들이 가지는 단어의 개수가 유사하도록 만들면 좋습니다.
# 이를 위해 BucketIterator를 사용합니다.
# 배치 크기(batch size): 128


import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

BATCH_SIZE = np.inf

# 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device)



# import numpy as np
# import pandas as pd
word_index_src = pd.Series(SRC.vocab.stoi).reset_index()
word_index_trg = pd.Series(TRG.vocab.stoi).reset_index()
word_index_src.columns = ['word', 'index']
word_index_trg.columns = ['word', 'index']

word_index_src_filtered = word_index_src[~word_index_src.apply(lambda x:(x['word'] != '<unk>') & (x['index']==0), axis=1)]
word_index_trg_filtered = word_index_trg[~word_index_trg.apply(lambda x:(x['word'] != '<unk>') & (x['index']==0), axis=1)]

path = r'C:\Users\Admin\Desktop\DataScience\★★ Python_정리자료(Git)\53_Deep_Learning\DL11_NLP'
word_index_src.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index_full(DE_SRC).csv', index=False, encoding='utf-8-sig')
word_index_trg.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index_full(EN_TRG).csv', index=False, encoding='utf-8-sig')
word_index_src_filtered.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index(DE_SRC).csv', index=False, encoding='utf-8-sig')
word_index_trg_filtered.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_word_index(EN_TRG).csv', index=False, encoding='utf-8-sig')

# for e, batch in enumerate(test_iterator):
#     print(e)
#     src_data.append(batch.src.numpy())
#     trg_data.append(batch.trg.numpy())

# src_frame = pd.DataFrame(batch.src.numpy().T)
# trg_frame = pd.DataFrame(batch.trg.numpy().T)
# print(src_frame.shape, trg_frame.shape)

# src_frame.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(DE_SRC).csv', index=False, encoding='utf-8-sig')
# trg_frame.to_csv(f'{path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(EN_TRG).csv', index=False, encoding='utf-8-sig')


