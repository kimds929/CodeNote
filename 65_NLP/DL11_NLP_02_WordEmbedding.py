
########################################################################################
# Embedding Layer #####################################################################
# 단어 집합(vocabulary) 생성 
vocab_size = 10
word2idx = {"hello": 0, "world": 1, "this": 2, "is": 3, "a": 4, 
    "test": 5, "example": 6, "for": 7, "numpy": 8, "embedding": 9} 


# numpy embedding ########################
import numpy as np 
# 임베딩 벡터 생성
embed_dim = 5
rng = np.random.RandomState(1)

embedding_matrix = rng.uniform(-1, 1, (vocab_size, embed_dim))    # -1~1, shape: (vocab_size, embed_dim)
embedding_matrix.shape  # (10,5) → (word, embeding)
embedding_matrix

# 입력 시퀀스 생성
input_seq_shape = (3,7)
input_seq = rng.randint(vocab_size, size=(3,7))
# array([[5, 9, 3, 6, 8, 0, 2],
#        [7, 7, 9, 7, 3, 0, 8],
#        [7, 7, 1, 1, 3, 0, 8]])


# Embedding layer 적용
emb_output = embedding_matrix[input_seq]  # 3, 7, 5

# 결과 출력
print(emb_output.shape)
print(emb_output)



# torch embedding ########################
import torch
emb = torch.nn.Embedding(vocab_size, embed_dim)
# emb.state_dict()['weight']    # embedding layer weight 확인

# weight update
from collections import OrderedDict
emb_weight = OrderedDict()
emb_weight['weight'] = torch.tensor(embedding_matrix)   
emb.load_state_dict(emb_weight)

# embedding layer forward
torh_emb_output = emb(torch.tensor(input_seq))
print(torh_emb_output)

print('numpy 결과와 torch결과의 비교')
print(np.allclose(emb_output, torh_emb_output.detach().numpy()))



########################################################################################
# Simple Embedding Classification Example ##############################################
import torch

# Data ---------------------
data = [("good movie",1),
("bad movie",0),
("awesome film",1),
("terrible film",0),
("great movie",1),
("awful film",0),
("excellent movie",1),
("poor film",0),
("fantastic movie",1),
("horrible movie",0),
("brilliant film",1),
("lousy film",0),
("superb film",1),
("disappointing film",0),
("amazing movie",1),
("dreadful film",0),
("outstanding film",1),
("atrocious movie",0),
("marvelous movie",1),
("terrible movie",0)]




# Preprocessing ---------------------
data_text = [i[0] for i in data]
data_label = [i[1] for i in data]

# vocab = {}
# for text in data_text:
#     words = text.split()
#     for word in words:
#         if word not in vocab:
#             vocab[word] = len(vocab)


vocab = {'good': 0, 'movie': 1, 'bad': 2, 'awesome': 3, 'film': 4,
 'terrible': 5, 'great': 6, 'awful': 7, 'excellent': 8, 'poor': 9,
 'fantastic': 10, 'horrible': 11, 'brilliant': 12, 'lousy': 13, 'superb': 14,
 'disappointing': 15, 'amazing': 16, 'dreadful': 17, 'outstanding': 18, 'atrocious': 19,
 'marvelous': 20}
vocab_size = len(vocab)

# text_to_Sequence ----------------------------
sequences = []
for text in data_text:
    words = text.split()
    sequence = [vocab[word] for word in words]
    sequences.append(sequence)


# pad_sequence ---------------------------------
max_length = max(len(sequence) for sequence in sequences)
padded_sequences = []
for sequence in sequences:
    padding = [0] * (max_length - len(sequence))
    padded_sequence = sequence + padding
    padded_sequences.append(padded_sequence)

tensor_sequences = torch.LongTensor(padded_sequences)
# tensor_sequences

# label -----------------------------------------
tensor_labels = torch.tensor(data_label)



# Define the classifier ------------------------------
class EmbeddingClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        embedded = self.embedding_layer(x)
        avg_embedded = embedded.mean(dim=1)
        # Average the embeddings over the sentence length and pass them through the classifier
        output = torch.softmax(avg_embedded, -1)
        return output

# vocab_size
embed_dim = 2
classifier = EmbeddingClassifier(vocab_size, embed_dim)
# classifier(tensor_sequences)

# Word to Vector Visualization (initial) ------------------------------
import matplotlib.pyplot as plt
vocab_weights = classifier.embedding_layer.state_dict()['weight'].detach().numpy()

init_fig = plt.figure()
plt.scatter(vocab_weights[:,0], vocab_weights[:,1])
for text, idx in vocab.items():
    plt.text(*vocab_weights[idx], text)    
plt.show()


# Define the optimizer and loss function ------------------------------
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()


# Training ------------------------------
epochs = 1000
for _ in range(epochs):
    pred = classifier(tensor_sequences)
    
    # Compute the loss and perform backpropagation
    loss = criterion(pred, tensor_labels)
    loss.backward()
    optimizer.step()

# Print the output and loss
print(pred)
print(loss.item())




# Word to Vector Visualization ------------------------------
import matplotlib.pyplot as plt
len(vocab)
vocab_weights = classifier.embedding_layer.state_dict()['weight'].detach().numpy()

trained_fig = plt.figure()
plt.scatter(vocab_weights[:,0], vocab_weights[:,1])
for text, idx in vocab.items():
    plt.text(*vocab_weights[idx], text)    
plt.show()

# compare word fig
init_fig
trained_fig
########################################################################################
