
## Word2Vec 
# https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb

corpus = ['king is a strong man', 
          'queen is a wise woman', 
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong', 
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']


# Remove stop words ----------------------------------------------------
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    
    return results

corpus = remove_stop_words(corpus)
corpus


words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)




# data generation ----------------------------------------------------

word2int = {}
for i,word in enumerate(words):
    word2int[word] = i
word2int

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
sentences


WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])
data




import pandas as pd
for text in corpus:
    print(text)

df = pd.DataFrame(data, columns = ['input', 'label'])
df


# Define Tensorflow Graph ----------------------------------------------------
import tensorflow as tf
import numpy as np

ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # input word
Y = [] # target word
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)


# Train ----------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(12, activation='softmax')
])

model.compile(optimizer='adam', loss='mse', metrics='mse')
model.fit(X_train, Y_train, epochs=10000, shuffle=True, batch_size=16,verbose=0)
model.summary()


# word vector in table ----------------------------------------------------
sample1 = np.array([[1,0,0,0,0,0,0,0,0,0,0,0]])

word2int
word2int_inv = {v: k for k,v in word2int.items()}
sample = np.eye(12,12)

model.predict(sample1)
latent_np = model.layers[0](sample).numpy()
model.layers[0](sample).numpy().T


# word vector in 2d chart ----------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(*latent_np.T)
for e, (xp, yp) in enumerate(latent_np):
    plt.text(xp,yp, word2int_inv[e])

