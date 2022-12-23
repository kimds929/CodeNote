import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/mobile2014.csv')

# sklearn Dataset Load : iris, wine, breast_cancer
def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df
dir(datasets)

    # wine Dataset
df.info()
df.head()
df.describe().T

df.groupby('Target').count()


# 【 텍스트 마이닝(Text Mining) 】 ------------------------------------------------------------------------------------------------------------------
# 단어처리방안, 빈도처리 방안

# 텍스트 데이터의 처리
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
# nltk.download('punkt')

df['Texts'][0]
sent_tokenize(df['Texts'][0])       # 문장단위로 분할
word_tokenize(df['Texts'][0])       # 단어단위로 분할

# http://www.nextree.co.kr/p4327/       # 정규표현식 소개
# http://soooprmx.com/archives/7718     # 정규표현식의 개념과 패턴 총정리


# 문장기호 제거
from nltk.tokenize import RegexpTokenizer
    # 정규표현식 활용 -  [\w]+ : 알파벳이나 숫자로 되어있는 내용만 추출
retokenize = RegexpTokenizer("[\w]+")
retokenize.tokenize(df['Texts'][0])

nltk.word_tokenize(str(df["Texts"]))

df['tokenized_sents'] = df.apply( lambda row: nltk.word_tokenize(str(row["Texts"])), axis = 1 ) # 단어 분할 Column 생성
df.head()
df['sents_length'] = df.apply( lambda row: len(row['tokenized_sents']), axis = 1)       # 단어갯수 Column 생성
df.head().to_clipboard(index=False)

# 어간추출(Steming) : fly, flies, flying, flew, flown →(어간) fly
# Steming: 변화하지 않는 어간부분을 추출
from nltk.stem import PorterStemmer, LancasterStemmer

st1 = PorterStemmer()
st2 = LancasterStemmer()

words = ['fly', 'flies', 'flying', 'flew', 'flown']
print(f'Porter Stemer: {[st1.stem(w) for w in words]}')
print(f'Lancaster Stemer: {[st2.stem(w) for w in words]}')
    # 전처리 과정을 거치지 않으면 다른단어로 인식 (사전에 없는 단어가 발생하여 재처리 과정에서 문제가 되기도 함)


# 표제어 추출방법 : 사전에 등재되어 있는 기본 형태로 단어 변경
# lemmatizing
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')        # wordnet : 영어사전역할

lm = WordNetLemmatizer()
[lm.lemmatize(w, pos='v') for w in words]       # 표제어는 사전에 표현되어있는 기본형으로 찾아감

# 품사 추출 가능 : 문장에서 단어들만 추출하고 품사를 붙일 수 있음.
nltk.download('tagsets')
    # 품사추출 : part-of-speech (POS)
# NNP: 단수 고유명사
# VB: 동사
# VVP: 동사 현재형
# TO: to전치사
# NN: 명사
# DT: 관형사

nltk.help.upenn_tagset('VB')

from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')
tagged_list = pos_tag( word_tokenize(df['Texts'][0]) )  # 단어별 품사 추출
tagged_list

nouns_list = [t[0] for t in tagged_list if t[1]== 'NN']
nouns_list


from nltk import Text
text = Text( retokenize.tokenize(str(df['Texts'])) )
text
retokenize.tokenize(str(df['Texts']))


import matplotlib.pyplot as plt
text.plot(20)       # frequency plot
plt.show()
# 관심있는 품사 부분 추출 후 그래프를 그리면 효과적임

text.dispersion_plot(['phone', 'call', 'price'])        # 단어 빈도 탐색 Plot
text.concordance('phone')

# 지워야하는 단어들을 제외하고 빈도구하기
from nltk import FreqDist
stopwords = ['I', 'i', 'this', 'the', 'a']
df_tokens = pos_tag( retokenize.tokenize(str(df['Texts'])) )  # 단어별 품사 추출
df_tokens

name_list = [t[0] for t in df_tokens if t[1] == 'NN' and t[0] not in stopwords ]        # 명사이면서 stopwords에 없는 단어들
name_list

fd_names = FreqDist(name_list)
fd_names

fd_names.N()                # 명사의 갯수
fd_names['phone']           # 'phone' 명사의 갯수
fd_names.freq('phone')      # 'phone' 명사의 빈도비  = fd_names['phone'] / fd_names.N() 

fd_names.most_common(5)     # 빈도가 많은 n개의 단어 추출       # 특별한 옵션이 없으면 대소문자를 구별


# 워드 클라우드 그리기 : 단어의 빈도를 고려하여 작성
from wordcloud import WordCloud
wc = WordCloud(width = 1000, height = 600, background_color='white', random_state=0)

plt.imshow(wc.generate_from_frequencies(fd_names))
plt.axis('off')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------





# 【 감성 분석 】 ------------------------------------------------------------------------------------------------------------------
    # 방법 : ① 감성사전(단어마다 감성점수가 부여된 사전) 만들기 → 감성 스코어를 계산하여 긍/부정 판단
    #       ② 기계학습 알고리즘을 이용한 감성 분류 → 문장이나 문서의 감성이 미리 분류된 학습 데이터 필요 (학습데이터를 활용하여 긍/부정 판단)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_score(sentence):
    score = analyzer.polarity_scores(sentence)
    return score


sentiment_analyzer_score('I am happy.')     # neg : 부정, neu : 중립, pos : 긍정  / compound : neg + neu + pos
sentiment_analyzer_score('I am very happy.')
sentiment_analyzer_score('I am not happy.')

# Test Data(Review)의 긍정, 부정 스코어 계산
df.iloc[0,4]
sentiment_analyzer_score(df.iloc[0,4])

# 기계학습 알고리즘 활용
    # 학습데이터 분류 → 머신러닝 알고리즘 학습 → 긍정/부정 분별하는지 여부 판단
df['Sentiment'].value_counts()

import matplotlib.pyplot as plt

df.info()
df['Sentiment'] = pd.Categorical(df['Sentiment'])       # int64 형식의 데이터타입을 category타입으로 변경
Sentiment_count = df.groupby('Sentiment').count()

df[df['Texts'].isna()]
na_index = df[df['Texts'].isna()].index
df['Texts'] = df['Texts'].fillna('')
df.iloc[na_index]

# 시각화
plt.bar(Sentiment_count.index.values, Sentiment_count['Texts'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# 기계학습을 사용하여 리뷰의 긍정, 부정 스코어 분류 모형 만들기
    # 리뷰에서 알파벳과 숫자로 시작하는 단어만 남기고 다른 단어들은 제거한 후 에 남은 단어들과 이들의 문서별 출현빈도를 만들기
    # 단어들을 모두 소문자로 변경하고, 불용어는 제거하며 한 단어가 분석의 대상이 된다datetime A combination of a date and a time. Attributes: ()
    # 이렇게 만들어진행렬을 단어-문서 행렬이라고 부르면 단어가 해당 문서에 몇 번 등장하였는지가 행렬의 원소값이 된다.
      
      # [ 빈도활용방법 ]
from sklearn.feature_extraction.text import CountVectorizer     # 단어가 문서내 몇번 들어가있는지 세는 library
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase = True, stop_words = 'english', ngram_range = (1,1), tokenizer = token.tokenize )
    # lowercase : 소문자로 변경, stop_words : 영어 불용어 제거, ngram_range : 추출 단어단위, tokenizer : 정규표현식
text_counts = cv.fit_transform(df['Texts'])
text_counts

# 학습데이터, 테스트 데이터 분류
from sklearn.model_selection import train_test_split
c_train_x, c_test_x, c_train_y, c_test_y = train_test_split(text_counts, df['Sentiment'], test_size = 0.3, random_state=1)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf_count = MultinomialNB()
clf_count.fit(c_train_x, c_train_y)
clf_count

predict_count = clf_count.predict(c_test_x)
predict_count
print(f'(Count) MultinomialNB Accuracy: {metrics.accuracy_score(c_test_y, predict_count)}')

# tf-idf 방법 : 특정 단어가 문서에 존재하는 것이 다른 문서와 구분 짓는데 중요한 역할을 하는 경우
# tf(Term Frequency) 
# idf(Inverse Documnet Frequency) = log(N/nt) : 해당단어가 몇개의 문서에 포함되어 있는지? → 작은 문서에 포함되어 있을 수록 값이 커짐

      # [ tfidf ]
# TfidfVectorizer 활용하여 모형, 행렬 만들기
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
text_tf = tf.fit_transform(df['Texts'])
text_tf

t_train_x, t_test_x, t_train_y, t_test_y = train_test_split(text_tf, df['Sentiment'], test_size = 0.3, random_state=123)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf_tfidf = MultinomialNB()
clf_tfidf.fit(t_train_x, t_train_y)
clf_tfidf

predict_tfidf = clf_tfidf.predict(t_test_x)
predict_tfidf
print(f'(Tfidf) MultinomialNB Accuracy: {metrics.accuracy_score(t_test_y, predict_tfidf)}')


# ---------------------------------------------------------------------------------------------------------------------------------------------------





# 【 한글 데이터 분석 】 ------------------------------------------------------------------------------------------------------------------------------
    # 영어 : 공백을 기준으로 단어 추출
    # 한글 : 한글 라이브러리에서 자체 처리 및 추출
    # https://ellun.tistory.com/46          # Konlpy 관련 오류 해결
    # https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/       # Konlpy 공식 홈페이지
        # pip install --upgrade pip
        # pip install JPype1-0.5.7-cp27-none-win_amd64.whl

import konlpy
from konlpy.corpus import kolaw
    # konlpy : 한글 형태소 분석을 위한 도구들이 포함되어 있음
    # Hannanum(한나눔) : http://semanticweb.kaist.ac.kr/hannanum/
    # Kkma(꼬꼬마) : http://kkma.snu.ac.kr/
    # Komoran(코모란) : https://github.com/shin285/KOMORAN
    # Open Korea Text : http://github.com/open-korean-text/open-korean-text
        # 형태소 분석기에 따라 한글문장의 형태소 분석결과가 상이하며, 문장의 종류에 따라서 적합한 결과를 내놓은 분석기가 다르므로
        # 분석된 결과를 보고 적합한 도구를  선택하는 것이 좋음

kolaw.fileids()

c = kolaw.open('constitution.txt').read()   # 데이터 로드
print(c[:40])

from konlpy.tag import *
hannanum = Hannanum()
okt = Okt()                 # 문장 띄어쓰기 등 잘 안되어있는 경우에 좋음  *ex) 소셜미디어
hannanum.nouns(c[:40])
okt.nouns(c[:40])

hannanum.pos(c[:40])        # 단어와 형태
okt.pos(c[:40])        # 단어와 형태 : 영어기반의 문법을 한글에 적용시킨 형태


import matplotlib.pyplot as plt
from nltk import Text
kolaw = Text(okt.nouns(c), name='kolaw')
kolaw.plot(30)
plt.show()

from wordcloud import WordCloud
wc = WordCloud(width=1000, height=600, background_color='white', font_path=path)
plt.imshow(wc.generate_from_frequencies(kolaw.vocab()))
plt.axis('off')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
