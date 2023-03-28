# https://wikidocs.net/106259
# https://wikidocs.net/106254
# https://codetorial.net/tensorflow/natural_language_processing_in_tensorflow_01.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf



# (Load Data) -------------------------------------------
# english_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.en', header=None)
# korean_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.ko', header=None)


# english_df.columns = ['english']
# english = english_df['english'].to_numpy()
# korean_df.columns = ['korean', 'etc']
# korean = korean_df['korean'].to_numpy()
# df = pd.concat([english_df, korean_df], axis=1).drop('etc',axis=1)


url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'


df = pd.read_csv(f"{url_path}/NLP_EN_to_KR_Data.csv", encoding='utf-8-sig')     # (4573, 2)
# df = pd.read_csv(f"{url_path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')    # (720, 2)
# df = pd.read_csv(f"{url_path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')    # (3853, 2)

# df01 = pd.read_csv(f"{url_path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
# df02 = pd.read_csv(f"{url_path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')
# df = pd.concat([df02, df01],axis=0).reset_index(drop=True)


df.sample(6)
print(df.shape)




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

# add_tokens = pd.DataFrame([['<sos>', '<sos>'], ['<eos>', '<eos>']], columns=df2.columns)
# df2_add_tokens = pd.concat([add_tokens, df2], axis=0)


# (english) *
df2_en = df2['english']
# tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') 
# tokenizer_en.fit_on_texts(df2_add_tokens['english'])
tokenizer_en = tf.keras.preprocessing.text.Tokenizer() 
tokenizer_en.fit_on_texts(df2_en)



# tokenizer_en.word_counts
vocab_size_en = len(tokenizer_en.word_index) + 1 #어휘수
print(f"vocab_size_en : {vocab_size_en}")


# (text_to_sequence / pad_sequence) *
seq_en = tokenizer_en.texts_to_sequences(df2_en)


# SOS / EOS
seq_en_inout = []
for sentence in seq_en:
    seq_en_inout.append([tokenizer_en.word_index['<sos>']] + sentence + [tokenizer_en.word_index['<eos>']])
padseq_en = tf.keras.preprocessing.sequence.pad_sequences(seq_en_inout, padding='post')
# padseq_en = tf.keras.preprocessing.sequence.pad_sequences(seq_en, padding='post')

padseq_en[:3,:]
print(padseq_en.shape)



# (korean) *
# from konlpy.tag import Mecab
# mecab = Mecab()
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh
from konlpy.tag import Okt
okt = Okt()

df2_kor = df2['korean']

tokened_kr = []
for sentence in df2_kor:
    sen_token = okt.morphs(sentence, stem=False)
    # sen_token = okt.morphs(sentence, stem=True)
    tokened_kr.append(sen_token)

# okt.morphs(df2_kor[5], norm=False, stem=False)  
#   . norm: 문장을 정규화
#   . stem: 은 각 단어에서 어간을 추출하는 기능 (True: 동사의 원형을 찾아줌)
tokenizer_kr = tf.keras.preprocessing.text.Tokenizer()
tokenizer_kr.fit_on_texts([['<sos>'],['<eos>']] + tokened_kr)

tokenizer_kr.word_index
# tokenizer_kor.index_word = {v:k for k, v in tokenizer_kor.word_index.items()}
vocab_size_kr = len(tokenizer_kor.word_index) + 1 #어휘수
print(f"vocab_size_kr : {vocab_size_kr}")


# (text_to_sequence / pad_sequence) *
seq_kr = tokenizer_kor.texts_to_sequences(tokened_kr)

# SOS / EOS
seq_kr_inout = []
for sentence in seq_kr:
    seq_kr_inout.append([tokenizer_kor.word_index['<sos>']] + sentence + [tokenizer_kor.word_index['<eos>']])
padseq_kr = tf.keras.preprocessing.sequence.pad_sequences(seq_kr_inout, padding='post')

padseq_kr[:3,:]
print(padseq_kr.shape)


print(padseq_en.shape, padseq_kr.shape)



#################################################################################################################################
# df01 = pd.read_csv(f"{path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
# df02 = pd.read_csv(f"{path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')
# df = pd.concat([df02, df01],axis=0).reset_index(drop=True)

df = pd.read_csv(f"{path}/NLP_EN_to_KR1_Data.csv", encoding='utf-8-sig')
df = pd.read_csv(f"{path}/NLP_EN_to_KR2_Data.csv", encoding='utf-8-sig')

processor_en = NLP_PreProcessor(df['english'])
processor_en.replace().fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()
processor_en.texts.shape
vocab_size_y = processor_en.vocab_size
# processor_en.sequences_to_texts(processor_en.texts, join=' ')

tokenizer_en = processor_en.tokenizer
padseq_en = processor_en.texts


processor_kr = NLP_PreProcessor(df['korean'])
processor_kr.replace().morphs_split(morphs=okt, stem=True).fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()
processor_kr.texts.shape
vocab_size_X = processor_en.vocab_size

# processor_kr.sequences_to_texts(processor_kr.texts, join=' ')
tokenizer_kr = processor_kr.tokenizer
padseq_kr = processor_kr.texts



path = r'C:\Users\Admin\Desktop\DataBase'
# Save_to_csv ***
word_index_X = pd.Series(tokenizer_en.word_index).reset_index()
word_index_y = pd.Series(tokenizer_kr.word_index).reset_index()
word_index_X.columns = ['word', 'index']
word_index_y.columns = ['word', 'index']

padseq_X = pd.DataFrame(padseq_en.copy())
padseq_y = pd.DataFrame(padseq_kr.copy())


# word_index_X.to_csv(f'{path}/NLP_EN_to_KR_word_index(EN).csv', index=False, encoding='utf-8-sig')
# word_index_y.to_csv(f'{path}/NLP_EN_to_KR_word_index(KR).csv', index=False, encoding='utf-8-sig')
# padseq_X.to_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
# padseq_y.to_csv(f'{path}/NLP_EN_to_KR_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

# word_index_X.to_csv(f'{path}/NLP_EN_to_KR1_word_index(EN).csv', index=False, encoding='utf-8-sig')
# word_index_y.to_csv(f'{path}/NLP_EN_to_KR1_word_index(KR).csv', index=False, encoding='utf-8-sig')
# padseq_X.to_csv(f'{path}/NLP_EN_to_KR1_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
# padseq_y.to_csv(f'{path}/NLP_EN_to_KR1_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')

# word_index_X.to_csv(f'{path}/NLP_EN_to_KR2_word_index(EN).csv', index=False, encoding='utf-8-sig')
# word_index_y.to_csv(f'{path}/NLP_EN_to_KR2_word_index(KR).csv', index=False, encoding='utf-8-sig')
# padseq_X.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(EN).csv', index=False, encoding='utf-8-sig')
# padseq_y.to_csv(f'{path}/NLP_EN_to_KR2_pad_seq_sentences(KR).csv', index=False, encoding='utf-8-sig')