import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import tensorflow as tf



#################################################################################################################################
class NLP_Preprocessor():
    """
    【Required Library】 import collections, import numpy as np, import matplotlib.pyplot as plt, import tensorflow as tf
    """
    def __init__(self, texts=None, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs):
        self.texts = texts
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, filters=filters, lower=lower,
                             split=split, char_level=char_level, oov_token=oov_token, document_count=document_count, **kwargs)
        self.fit_token = False
        self.use_morphs = False
    
    def input_texts(self, texts):
        self.texts = texts
    
    def replace(self, texts=None, re="[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z ]", replace="", inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts

        texts_result = []
        for sentence in texts:
            texts_result.append(sentence.replace(re, replace))

        self.texts_replace = texts_result.copy()
        if verbose > 0:
            print("→ self.texts_replace")
        if inplace:
            self.texts = texts_result
        return self

    def remove_stopword(self, texts=None, stopword=[], inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts

        if stopword:
            texts_result = []
            for sentence in texts:
                sentence = [word for word in sentence if word != stopword]
                texts_result.append(sentence)
        
        self.texts_remove_stopword = texts_result
        if verbose > 0:
            print("→ self.texts_replace")
        if inplace:
            self.texts = texts_result
        return self

    def morphs_split(self, texts=None, morphs=None, inplace=True, verbose=1, **kwargs):
        texts = texts if texts is not None else self.texts

        texts_result = []
        for sentence in texts:
            if 'okt' in str(morphs):
                texts_result.append(morphs.morphs(sentence, **kwargs))
        
        self.texts_morphs = texts_result
        self.morphs = morphs
        self.use_morphs = True
        if verbose > 0:
            print("→ self.texts_morphs")
        if inplace:
            self.texts = texts_result
        return self  

    def word_prob(self, texts=None, plot=True):
        if self.fit_token is False:
            texts = texts if texts is not None else self.texts
            self.fit_on_texts(texts)

        # 누적 희소단어 비율
        self.word_counts = self.tokenizer.word_counts

        total_cnt = len(self.word_counts)
        rare_dict = {}
        for i in sorted(np.unique(list(self.word_counts.values()))):
            filtered_count = len( dict(filter(lambda e: e[1]<=i, self.word_counts.items())) )
            prob =  filtered_count / total_cnt     # 특정 빈도수 이하 단어 비율
            rare_dict[i] = prob

        # 단어 수별 점유비
        rare_cum_prob = []
        for p in [0.7, 0.8, 0.9, 0.95, 0.99]:
            target_cum_prob = list(tuple(filter(lambda e: e[1] < p, sorted(rare_dict.items())))[-1])
            rare_cum_prob.append([*target_cum_prob, p] )
        rare_cum_prob   # word_count, prob, target_prob

        # 점유비 Plot
        fig = plt.figure()
        plt.title(f"Ratio of Rare_Word (total: {total_cnt})")
        plt.plot(rare_dict.keys(), rare_dict.values(), 'o-')
        plt.xscale('log')
        for cp, p, tp in rare_cum_prob:
            plt.axhline(p, color='red', alpha=0.1)
            plt.text(cp, p, f"    {cp} (nw: {total_cnt +1 - cp})\n ←   ({round(p*100,1)}%, aim:{int(tp*100)}%)", color='red')
            plt.scatter(cp, p, color='red')
        plt.xlabel('Word_Count (log_scale)')
        plt.ylabel('Word_Prob')
        if plot is True:
            plt.show()
        else:
            plt.close()

        self.word_prob_dict = rare_dict
        self.word_cum_prob = rare_cum_prob
        self.word_prob_plot = fig

        self.tokenizer = None

    def tokenizer(self, filter_words=None, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs):
        if filter_word is not None:
            num_words = self.word_counts +1 - filter_words
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, filters=filters, lower=lower,
                        split=split, char_level=char_level, oov_token=oov_token, document_count=document_count, **kwargs)

    def fit_on_texts(self, texts=None, filter_words=None, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs):
        texts = texts if texts is not None else self.texts
        if self.tokenizer is None:
            if filter_word is not None:
                num_words = self.word_counts +1 - filter_words
            self.tokenizer(num_words=num_words, filters=filters, lower=lower,
                        split=split, char_level=char_level, oov_token=oov_token, document_count=document_count, **kwargs)
        
        self.tokenizer.fit_on_texts(texts)
        self.fit_token = True
        
        self.word_counts = self.tokenizer.word_counts

        self.tokenizer.word_index[''] = 0
        self.tokenizer.index_word[0] = ''

        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        self.vocab_size = len(self.word_index) + 1
        return self

    def texts_to_sequences(self, texts=None, inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts

        texts_result = self.tokenizer.texts_to_sequences(texts)

        self.texts_texts_to_seq = texts_result
        if verbose > 0:
            print("→ self.texts_texts_to_seq")
        if inplace:
            self.texts = texts_result
            return self    
        else:
            return texts_result

    def add_sos_eos(self, texts=None, sos='<SOS>', eos='<EOS>', inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts

        self.sos = sos
        self.eos = eos

        if bool(sos):
            self.tokenizer.word_index[sos] =  self.vocab_size - 1
            self.tokenizer.index_word[self.tokenizer.word_index[sos]] = sos
            self.vocab_size += 1
        
        if bool(eos):
            self.tokenizer.word_index[eos] =  self.vocab_size - 1
            self.tokenizer.index_word[self.tokenizer.word_index[eos]] = eos
            self.vocab_size += 1

        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        # return self
        if bool(sos) + bool(eos) > 0:
            texts_result = []
            for sentence in texts:
                if bool(sos) is True and bool(eos) is True:
                    texts_result.append([self.word_index[sos]] + sentence + [self.word_index[eos]])
                elif bool(sos) is True:
                    texts_result.append([self.word_index[sos]] + sentence)
                elif bool(eos) is True:
                    texts_result.append(sentence + [self.word_index[eos]])
        else:
            texts_result = list(texts)
        
        self.texts_add_sos_eos = texts_result
        if verbose > 0:
            print("→ self.texts_add_sos_eos")
        if inplace:
            self.texts = texts_result
        return self     

    def seq_length_prob(self, texts=None, plot=True):
        texts = texts if texts is not None else self.texts
        
        seq_lens = np.array([len(c) for c in texts])

        seq_len_counts = {}
        cumsum_cw = 0
        for length in sorted(np.unique(seq_lens)): 
            cw = np.count_nonzero(seq_lens==length)
            cumsum_cw += cw
            seq_len_counts[length] = (cw, cumsum_cw, cumsum_cw/len(seq_lens))   # seq_count, seq_cumsum, cumsum_prob
        # seq_len_counts    # 

        # 단어 수별 점유비
        seq_len_prob = []
        for p in [0.7, 0.8, 0.9, 0.95, 0.99]:
            target_cum_prob = list(tuple(filter(lambda e: e[1][2] < p, sorted(seq_len_counts.items())))[-1])
            seq_len_prob.append([target_cum_prob[0], *target_cum_prob[1], p] )
        # seq_cum_prob   # seq_length, seq_count, sqe_cumsum, prob, target_prob


        # 점유비 Plot
        fig = plt.figure()
        plt.title(f"Ratio of Seq_Length (max_len: {seq_lens})")
        plt.plot(seq_len_counts.keys(), np.array(list(seq_len_counts.values()))[:,2], 'o-')
        for seq_len, seq_count, seq_cumsum, seq_prob, target_prob in seq_len_prob:
            plt.axhline(seq_prob, color='red', alpha=0.1)
            plt.text(seq_len, seq_prob, f" ← {seq_len} ({round(seq_prob*100,1)}%, aim:{int(target_prob*100)}%)", color='red')
            plt.scatter(seq_len, seq_prob, color='red')
        plt.xlabel('Seq_Length (log_scale)')
        plt.ylabel('Cum_Sum_Prob')

        if plot is True:
            plt.show()
        else:
            plt.close()

        self.seq_len_counts = seq_len_counts
        self.seq_len_prob = seq_len_prob
        self.seq_len_prob_plot = fig

    def pad_sequences(self, texts=None, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.0, inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts

        texts_result = tf.keras.preprocessing.sequence.pad_sequences(sequences=texts, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating, value=value)
        
        self.texts_pad_seq = texts_result
        if verbose > 0:
            print("→ self.texts_pad_seq")
        if inplace:
                self.texts = texts_result
        return self

    def to_categorical(self, texts=None, num_classes=None, dtype='float32', inplace=True, verbose=1):
        texts = texts if texts is not None else self.texts
        num_classes = num_classes if num_classes is not None else self.vocab_size

        texts_result = tf.keras.utils.to_categorical(y=texts, num_classes=num_classes, dtype=dtype)
        
        self.texts_categorical = texts_categorical
        if verbose > 0:
            print("self.texts_categorical")
        if inplace:
            self.texts = texts_result
        return self
    
    def sequences_to_texts(self, texts, index_word=None, join=None, 
                                sos_text=False, eos_text=False, padding_text=False, sos=None, eos=None):
        index_word = index_word if index_word is not None else self.index_word
        sos = sos if sos is not None else self.sos
        eos = eos if eos is not None else self.eos
        padding = ''
        indexes = index_word.keys()
        
        texts_result = []
        for sentence in texts:
            sentence_result = []
            for word in sentence:
                if word in indexes:
                    word_text = index_word[word]
                    if sos_text is False and word_text == sos:
                        pass
                    elif eos_text is False and word_text == eos:
                        pass
                    elif padding_text is False and word_text == padding:
                        pass
                    else:
                        sentence_result.append(word_text)
                else:
                    sentence_result.append('')
            if join is not None:
                sentence_result = join.join(sentence_result)
            texts_result.append(sentence_result)
        
        return texts_result

    def texts_to_sequence_transform(self, texts, with_morphs=True):
        if with_morphs is True and self.use_morphs is True:
            target_texts = self.morphs.morphs(texts)
        else:
            target_texts = texts
        
        return self.texts_to_sequences(target_texts, inplace=False, verbose=0)


# (Load Data) -------------------------------------------
# english_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.en', header=None)
# korean_df = pd.read_fwf('https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/master/korean-english-jhe/jhe-koen-dev.ko', header=None)

# path = r'C:\Users\Admin\Desktop\DataBase'
# english_df.columns = ['english']
# english = english_df['english'].to_numpy()
# korean_df.columns = ['korean', 'etc']
# korean = korean_df['korean'].to_numpy()
# df = pd.concat([english_df, korean_df], axis=1).drop('etc',axis=1)

# processor_en = NLP_PreProcessor(df['english'])
# processor_en.replace().fit_on_texts().texts_to_sequences().add_sos_eos()

# # processor_en = NLP_PreProcessor(df['english'])
# # processor_en.replace()
# # processor_en.replace().word_prob()
# # processor_en.word_prob_dict
# # processor_en.word_cum_prob

# processor_en = NLP_PreProcessor(df['english'])
# processor_en.replace().fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()
# processor_en.texts.shape
# vocab_size_y = processor_en.vocab_size
# processor_en.sequences_to_texts(processor_en.texts, join=' ')


# from konlpy.tag import Okt
# okt = Okt()
# processor_kr = NLP_PreProcessor(df['korean'])
# processor_kr.replace().morphs_split(morphs=okt, stem=True)
# processor_kr.fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()
# processor_kr.texts.shape
# vocab_size_X = processor_en.vocab_size

# processor_kr.sequences_to_texts(processor_kr.texts, join=' ')

# processor_kr.texts_to_sequence_transform('저는 사과를 좋아합니다')