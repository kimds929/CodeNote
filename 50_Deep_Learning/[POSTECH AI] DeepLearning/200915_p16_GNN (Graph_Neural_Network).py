import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_hub as tf_hub

import cv2
import node2vec
import networkx as nx

import tqdm

# !wget https://raw.githubusercontent.com/aditya-grover/node2vec/master/graph/karate.edgelist
# !rm karate.edgelist


G = nx.Graph()
with open('karate.edgelist') as f:
    for line in f:
        a, b = line.split()
        G.add_edge(int(a), int(b))

dir(G)
G.size()
G.number_of_nodes()
G.number_of_edges()

G.nodes
G.edges

# draw_graph
nx.draw(G, with_labels=True)


# Node2vec
karate_n2v = node2vec.Node2Vec(G, dimensions=20, walk_length=16, 
                            num_walks=100, p=0.5, q=1)
# dimension: latent_vector길이,  walk_length: walk길이,  num_walks : random_walk 갯수
# 1/p : 되돌아갈 확률 (small p : BFS)
# 1/q : 멀어질 확률 (small q: DFS)
nx.draw(karate_n2v.graph, with_labels=True)

karate_model = karate_n2v.fit(window=10, min_count=1)       # latent_node_vector구하기
# window : 주변에 몇개의 node를 볼 것인지?
# min_count : 

for node, _ in karate_model.most_similar('34',topn=3): # 34번노드와 가장 가까운 순서대로 추출(상위 10개만 추출)
    print(node)

karate_model.wv.get_vector('34')    # latent vector

def cos_similarity(a,b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

vect01 = karate_model.wv.get_vector('1')
vect33 = karate_model.wv.get_vector('33')
vect34 = karate_model.wv.get_vector('34')
cos_similarity(vect33, vect34)      # similarity
cos_similarity(vect01, vect34)      # similarity






# Face_Book Dataset --------------------------------------------------------------------

# !wget https://snap.stanford.edu/data/facebook_combined.txt.gz
# !gzip -d facebook_combined.txt.gz     # 압출풀기
# !ls       # File List

facebook_graph = nx.Graph()
with open('facebook_combined.txt') as f:
    for line in f:
        a, b = line.split()
        facebook_graph.add_edge(int(a), int(b))

# nx.draw(facebook_graph, with_labels=True)     # Very Large

n2v_facebook = node2vec.Node2Vec(facebook_graph, dimensions=300, 
                walk_length=30, num_walks=6, p=0.5, q=0.1)
facebook_model = n2v_facebook.fit(window=5, min_count=1)       # latent_node_vector구하기

facebook_model.wv.get_vector('2950').shape    # latent vector

# node_vector를 꺼내기
node_vectors_facebook = []
for i in tqdm.tqdm_notebook(range(facebook_graph.number_of_nodes())):
    node_vectors_facebook.append( facebook_model.wv.get_vector(str(i)) )

nodevec_array_facebook = np.array(node_vectors_facebook)
nodevec_array_facebook.shape

# Gender를 예측하는 모델 ------------------------------------------------
# !wget https://snap.stanford.edu/data/facebook.tar.gz
# !tar -zxvf facebook.tar.gz facebook


# with open('facebook/0.feat') as f:
#     for line in tqdm.tqdm_notebook(f):
#         split = line.split()
#         node_number = int(split[0])
#         train_x.append(nodevec_array_facebook[node_number])
#         train_y.append(int(split[78]))      # 77번이 gender

# import dataset function
def load_data(file_column_obj):
    X_list = []
    y_list = []
    for file, gender_idx in tqdm.tqdm_notebook(file_column_obj.items()):
        with open('facebook/'+file+'.feat') as f:
            for line in f:
                split = line.split()
                node_number = int(split[0])
                X_list.append(nodevec_array_facebook[node_number])
                y_list.append(int(split[gender_idx]))      # 77번이 gender
    return np.array(X_list), np.array(y_list)

# import dataset
train_file_gender = {'0':78, '107':265, '348':87, '414':64, '686':42, '1684':148, '1912':260}
train_x, train_y = load_data(train_file_gender)

test_file_gender = {'3437':118, '3980':20}
test_x, test_y = load_data(test_file_gender)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(600, input_shape=(300,), activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model

gender_model = make_model()
gender_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
gender_model.fit(x=train_x, y=train_y, batch_size=300, epochs=300, verbose=0)

gender_model.evaluate(train_x, train_y) # trainset_result
gender_model.evaluate(test_x, test_y)   # testset_result



# 다른 Attribute 정보를 포함한 Latent를 추출하여 학습하는 GCN 모델 학습을 시켜서 예측성능을 높일 수 있음


