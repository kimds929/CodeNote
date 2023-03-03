

################################################################################################
# max_len = None
max_len = 1000
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/53_Deep_Learning/DL11_NLP/'
index_word_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_word_index(DE_SRC).csv', index_col='index', encoding='utf-8-sig')['word']
index_word_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_word_index(EN_TRG).csv', index_col='index', encoding='utf-8-sig')['word']

train_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_train(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]
valid_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_valid(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]
test_X = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(DE_SRC).csv', encoding='utf-8-sig').to_numpy()[:max_len]

train_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_train(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]
valid_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_valid(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]
test_y = pd.read_csv(f'{url_path}/NLP_Multi30k_EN_to_DE_pad_seq_sentences_test(EN_TRG).csv', encoding='utf-8-sig').to_numpy()[:max_len]

vocab_size_X = len(index_word_X) + 1 #어휘수
vocab_size_y = len(index_word_y) + 1 #어휘수

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

