# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch



# url_path ='/home/kimds929/DataSet/'
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'
# url_path = r'D:\작업방\업무 - 자동차 ★★★\Dataset\99_DataSet'
train_df = pd.read_csv(f'{url_path}/NLP_movie_review_train_tokenized.csv', encoding='utf-8-sig')
# train_df = pd.read_csv(f'{url_path}/NLP_movie_review_simple_train_tokenized.csv', encoding='utf-8-sig')
print(train_df.shape)

train_X = train_df['tokenized']


# from DS_NLP import NLP_Preprocessor
import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_NLP import NLP_Preprocessor


processor = NLP_Preprocessor(texts=train_X)
processor.word_prob()
# processor.fit_on_texts().texts_to_sequences()

num_words = 4380
processor.fit_on_texts(num_words=num_words).texts_to_sequences().add_sos_eos().pad_sequences()
processor.vocab_size
processor.word_index
train_y = train_df['label'].to_numpy()[processor.texts_index]
print(processor.texts.shape, train_y.shape)


# processor.sequences_to_texts()
vocab_size = processor.vocab_size
print(vocab_size)

n_cls = 30000

pretrain_X = processor.texts[:-n_cls]
pretrain_y = train_y[:-n_cls]
cls_X = processor.texts[-n_cls:]
cls_y = train_y[-n_cls:]

print(pretrain_X.shape, pretrain_y.shape, cls_X.shape, cls_y.shape)


# ## Pre-Train Dataset
# url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'
# pretrain_seq = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_pad_seq_sentences(KR).csv', encoding='utf-8-sig').to_numpy()

# index_word_df = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_index_word(KR).csv', encoding='utf-8-sig')
# index_word = index_word_df.set_index('index')['word'].to_dict()
# word_index = index_word_df.set_index('word')['index'].to_dict()


# class TextPreprocessor():
#     def __init__(self, index_word=None, word_index=None):
#         self.index_word = index_word
#         self.word_index = word_index
#         self.index_word_np = None
        
#         self.sort = False
#         self.vocab_size = None
        
#         self.make_dict()

#     def make_dict(self):
#         if self.index_word is not None and self.word_index is None:
#             self.word_index = {v: k for k, v in self.index_word.items()}
#         if self.word_index is not None and self.index_word is None :
#             self.index_word = {k: v for v, k in word_index.items()}

#         if self.sort is False:
#             if ~(0 in index_word.keys()):
#                 self.index_word[0] = ''
#                 self.word_index[''] = 0
#             self.index_word = dict(sorted(self.index_word.items(), key=lambda item: item[0]))
#             self.word_index = dict(sorted(self.word_index.items(), key=lambda item: item[1]))
#             self.sort = True

#         if self.word_index is not None and self.index_word_np is None:
#             self.index_word_np = np.array(list(self.index_word.values()))     
        
#         self.vocab_size = len(index_word)+1
        
#     def seq_to_text(self, x, text=False):
#         self.make_dict()
        
#         result = self.index_word_np[x]
#         if text is True:
#             result = [''.join(list(l)) for l in result]
#         return result

# tp = TextPreprocessor(index_word)
# tp.seq_to_text(pretrain_seq, text=True)
# vocab_size = tp.vocab_size
# # pretrain_seq.shape    # 4573, 132


# from DS_Torch import TorchDataLoader
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader

# X = np.random.rand(100,3)
# y = np.random.rand(100)
# loader =  TorchDataLoader(X, y, split_size=(0.7, 0.1, 0.2), random_state=1)
# loader.make_dataloader()
# loader.dataloader
# loader.sample
# train_loader, valid_loader, test_loader = loader.dataloader



# pretrain_dataset
pretrain_loader = TorchDataLoader(pretrain_X, pretrain_y, split_size=(0.8,0.1,0.1), random_state=1)
pretrain_loader.make_dataloader(batch_size=64, shuffle=True)
pretrain_loader.dataloader
pretrain_train_loader, pretrain_valid_loader, pretrain_test_loader = pretrain_loader.dataloader
pretrain_sample_X, pretrain_sample_y = pretrain_loader.sample

# classification_dataset
cls_loader = TorchDataLoader(cls_X, cls_y, split_size=(0.8,0.1,0.1), random_state=1)
cls_loader.make_dataloader(batch_size=64, shuffle=True)
cls_loader.dataloader
cls_train_loader, cls_valid_loader, cls_test_loader = cls_loader.dataloader
cls_sample_X, cls_sample_y = cls_loader.sample

# # sample data
# sample_df = pd.read_csv(f'{dataset_path}/Sample_Sequence_Data_1000.csv', encoding='utf-8-sig')
# sample_X = torch.tensor(sample_df.to_numpy())[:3, :5]    # (8,5)
# vocab_size = 7
# sample_X.shape  # 3,5, 7


# torch -----------------------------------------------------------------------
# self.register_buffer('pos_embed', pos_embed)      # 학습되지 않는 변수로 등록
# self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
# torch.autograd.Variable(self.pos_embed[:X.shape[1]], requires_grad=False) # Variable
# .masked_fill((sample_X == 0), 0)      # apply
# .masked_fill_((sample_X == 0), 0)     # inplace
# -----------------------------------------------------------------------------
# from DS_TorchModule import EmbeddingLayer, PositionalEncodingLayer, PositionwiseFeedForwardLayer, MultiHeadAttentionLayer, make_tril_mask
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_TorchModule import EmbeddingLayer, PositionalEncodingLayer, PositionwiseFeedForwardLayer, MultiHeadAttentionLayer, make_tril_mask

#######################################################################################################################################
# https://paul-hyun.github.io/gpt-01/
# https://paul-hyun.github.io/transformer-02/


# # Pretrain : NULL
# gpt_pretain = GPT1(vocab_size)
# gpt_pretain.pretrain()
# gpt_pretain(cls_sample_X).shape
# gpt_pretain.pretrain_output.shape   # pretrain
# gpt_pretain.downstream_output.shape # downstream

# # DownStream : Classification
# gpt_cls = GPT1(vocab_size, downstream_layer=GPT1_ClassifierLayer())
# gpt_cls.downstream()
# gpt_cls(cls_sample_X).shape
# gpt_cls.pretrain_output.shape        # pretrain
# gpt_pretain.downstream_output.shape  # downstream

# ★★★ GPT1 (Pretrain, DownStream, Fine-Tunnning)
class GPT1(torch.nn.Module):
    def __init__(self, vocab_size, downstream_layer=None, embed_dim=256
                 ,n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding='sinusoid'
                 ,pretrainable=True, downstreamable=True
                 ):
        super().__init__()
        
        # pretrain
        self.gpt_decoder = GPT1_Decoder(vocab_size, embed_dim, n_layers, n_heads, 
                                   posff_dim, dropout, pos_encoding)
        self.projection_layer = torch.nn.Linear(embed_dim, vocab_size, bias=False)
        self.projection_layer.weight = self.gpt_decoder.embed_layer.weight     # We.T
        
        
        # downstream
        if downstream_layer is None:
            self.downstream_layer = NullLayer()
        else:
            self.downstream_layer = downstream_layer
            
        self.pretrainable = True
        self.downstreamable = True
    
    def forward(self, x):
        # GPT1_Decoder
        mask_tril = make_tril_mask(x).unsqueeze(1)
        self.decoder_output = self.gpt_decoder(x, mask_tril)
        with torch.no_grad():
            # self.attention_scores = [layer.attention_score for layer_name, layer in self.decoder.decoder_layers.named_children()]
            self.self_attention_score = self.gpt_decoder.decoder_layers[-1].self_attention_score

        # pretrain 
        if self.pretrainable is True:
            self.pretrain_output = self.projection_layer(self.decoder_output)
            
        # downstream
        if self.downstreamable is True:
            self.downstream_output = self.downstream_layer(self.decoder_output)
        
        # return
        if (self.pretrainable is True) and (self.downstreamable is True):        
            return self.pretrain_output, self.downstream_output
        elif self.pretrainable is True:
            return self.pretrain_output
        elif self.downstreamable is True:
            return self.downstream_output
            
    
    def pretrain(self):
        self.pretrainable = True
        self.downstreamable = False
    
    def downstream(self):
        self.pretrainable = False
        self.downstreamable = True
    
    def fine_tunning(self):
        self.pretrainable = True
        self.downstreamable = True



# GPT1_ClassificationLayer
class GPT1_ClassifierLayer(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.classifier_lm = torch.nn.Linear(embed_dim, 2, bias=False)
    
    def forward(self, x):
        self.lm_output = self.classifier_lm(x[:,-1, :].contiguous())
        self.output = torch.softmax(self.lm_output, dim=1)
        return self.output

# NullLayer
class NullLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


# ★★ GPT1_Decoder 
#   gpt = GPT1_Decoder(vocab_size, embed_dim=2, n_heads=1)
#   gpt(sample_X)
#   mask_tril = make_tril_mask(sample_X,0).unsqueeze(1)
#   gpt(sample_X, mask_tril)
class GPT1_Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding='sinusoid'):
        super().__init__()
        
        self.embed_layer = EmbeddingLayer(vocab_size, embed_dim)
        self.posembed_layer = PositionalEncodingLayer(pos_encoding)
        self.dropout = torch.nn.Dropout(dropout)

        self.decoder_layers = torch.nn.ModuleList([GPT1_DecoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])
        
    def forward(self, x, mask=None):
        # x : (batch_seq, x_word)
        # mask : (batch_seq, 1, y_word, y_word)

        # embedding layer
        self.x_embed = self.embed_layer(x)  # (batch_seq, x_word, emb)
        
        # positional encoding
        self.x_posembed = self.posembed_layer(self.x_embed)     # (batch_seq, x_word, emb)
        
        # sum of X_emb_scaled and pos_emb_X
        self.x_input = self.dropout(self.x_embed + self.x_posembed)     # (batch_seq, x_word, emb)
        
        # decoder layer
        next_input = self.x_input
        
        for dec_layer in self.decoder_layers:
            next_input = dec_layer(next_input, mask)
        self.decoder_layer_output = next_input

        return self.decoder_layer_output



# ★ GPT1_DecoderLayer
class GPT1_DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, y_emb, y_mask=None):
        # y_emb : (batch_seq, y_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        
        # (Self Attention Layer) -------------------------------------------------------------------
        self.y_self_att_output = self.self_att_layer((y_emb, y_emb, y_emb), mask=y_mask)
        #  (batch_seq, y_word, fc_dim=emb)        
        self.self_attention_score = self.self_att_layer.attention_score
        # (batch_seq, n_heads, y_word, key_length=y_word)
        
        self.y_skipconnect_1 = y_emb + self.dropout(self.y_self_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_1 = self.self_att_layer_norm(self.y_skipconnect_1)  # layer normalization
        
        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.y_posff = self.posff_layer(self.y_layer_normed_1)    # (batch_seq, y_word, emb)
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_2 = self.posff_layer_norm(self.y_posff)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        return self.y_layer_normed_2   # (batch_seq, y_word, emb)
#######################################################################################################################################


# pretrain_train_loader, pretrain_valid_loader, pretrain_test_loader = pretrain_loader.dataloader
# pretrain_sample_X, pretrain_sample_y = pretrain_loader.sample

# cls_train_loader, cls_valid_loader, cls_test_loader = cls_loader.dataloader
# cls_sample_X, cls_sample_y = cls_loader.sample
# device = 'cpu'

# Pretrain : NULL
gpt_pretain = GPT1(vocab_size)
gpt_pretain.pretrain()
gpt_pretain(cls_sample_X).shape
gpt_pretain.pretrain_output.shape   # pretrain
# gpt_pretain.downstream_output.shape # downstream

# DownStream : Classification
gpt_cls = GPT1(vocab_size, downstream_layer=GPT1_ClassifierLayer())
gpt_cls.downstream()
gpt_cls(cls_sample_X).shape
# gpt_cls.pretrain_output.shape        # pretrain
gpt_cls.downstream_output.shape  # downstream




# Training Setting ########################################################################
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

# # customize library ***---------------------
# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
# from DS_DeepLearning import EarlyStopping
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping
# # ------------------------------------------
import time
import copy

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def torch_accuracy_score(dataloader, estimator, device='cpu', verbose=0):
    estimator.eval()
    with torch.no_grad():
        acc_list = []

        for batch in dataloader:
            batch_X = batch[0].type(torch.long).to(device)
            batch_y = batch[1].to(device)
            
            pred = estimator(batch_X)
            pred_cls = torch.argmax(pred, axis=1).to('cpu').detach().numpy()
            real_cls = batch_y.to('cpu').detach().numpy()
            acc_list.append((pred_cls == real_cls))
        acc_array = np.hstack(acc_list)
        test_accuracy = acc_array.sum()/len(acc_array)
        if verbose > 0:
            print(test_accuracy)
    return test_accuracy

torch_accuracy_score(cls_valid_loader, gpt_cls, 'cpu')








# Classification without pre-training ########################################################################
print('* Classification without pre-training ----------------------------------------------------------------')
gpt_cls = GPT1(vocab_size, GPT1_ClassifierLayer(), n_layers=3).to(device)
gpt_cls.downstream()
# gpt_cls(cls_sample_X)     # (64, 2)
# gpt_cls(cls_sample_X).shape

model = copy.deepcopy(gpt_cls)


# # model weights parameter initialize (가중치 초기화) ***
# def init_weights(model):
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             torch.nn.init.normal_(param.data, mean=0, std=0.01)
#         else:
#             torch.nn.init.constant_(param.data, 0)
# model.apply(init_weights)


es = EarlyStopping(patience=100)
loss_function = torch.nn.CrossEntropyLoss()

# learning_rate = 0.001
learning_rate = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100


# training * -------------------------------------------------------------------------------------------------------
train_loader = cls_train_loader
valid_loader = cls_valid_loader
test_loader = cls_test_loader

train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch  in train_loader:
        batch_X = batch[0].type(torch.long).to(device)
        batch_y = batch[1].to(device)
        optimizer.zero_grad()                   # wegiht initialize
        model(batch_X)                          # forward
        pred = model.downstream_output          # predict
        loss = loss_function(pred, batch_y)     # loss
        loss.backward()                         # backward
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    # 기울기(gradient) clipping 진행
        # (gradient clipping) https://sanghyu.tistory.com/87
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )
    

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch in valid_loader:
            batch_X = batch[0].to(device)
            batch_y = batch[1].to(device)
            model(batch_X)                          # forward
            pred = model.downstream_output          # predict
            loss = loss_function(pred, batch_y)     # loss
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )

    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')

        # # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)
        if early_stop == 'break':
            break
        # # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot
# ------------------------------------------

gpt_cls.load_state_dict(es.optimum[2])    # optimum model (load weights)



# Performance Evaluate (Accuracy_score)
torch_accuracy_score(cls_valid_loader, gpt_cls, device)

























# Classification with pre-training ########################################################################




# Pre-training ★ #########
print('* Classification with pre-training : pre-training ----------------------------------------------------------------')
# sample_X.shape    # (64, 69)
gpt_pretain = GPT1(vocab_size, n_layers=3).to(device)
gpt_pretain.pretrain()
# gpt_pretain(cls_sample_X)  # (64, 69, 4383)
# gpt_pretain(cls_sample_X).shape

model = copy.deepcopy(gpt_pretain)

# # model weights parameter initialize (가중치 초기화) ***
# def init_weights(model):
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             torch.nn.init.normal_(param.data, mean=0, std=0.01)
#         else:
#             torch.nn.init.constant_(param.data, 0)
# model.apply(init_weights)

es = EarlyStopping(patience=100)

# loss_function = torch.nn.CrossEntropyLoss()     # ignore_index=trg_pad_idx
learning_rate = 5e-5
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100


# training * -------------------------------------------------------------------------------------------------------
train_loader = pretrain_train_loader
valid_loader = pretrain_valid_loader
test_loader = pretrain_test_loader


train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch  in train_loader:
        batch_X = batch[0].type(torch.long).to(device)
        optimizer.zero_grad()                   # wegiht initialize
        model(batch_X)                          # forward
        pred_eval = model.pretrain_output[:,:-1,:].reshape(-1, vocab_size)  # self-supervised predict
        real_eval = batch_X[:,1:].reshape(-1)
         
        loss = loss_function(pred_eval, real_eval)     # loss
        loss.backward()                         # backward
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    # 기울기(gradient) clipping 진행
        # (gradient clipping) https://sanghyu.tistory.com/87
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )
    

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch in valid_loader:
            batch_X = batch[0].type(torch.long).to(device)
            model(batch_X)                   # forward
            pred_eval = model.pretrain_output[:,:-1,:].reshape(-1, vocab_size)  # self-supervised predict
            real_eval = batch_X[:,1:].reshape(-1)

            loss = loss_function(pred_eval, real_eval)     # loss
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )

    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')

        # # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)
        if early_stop == 'break':
            break
        # # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot
# ------------------------------------------

gpt_pretain.load_state_dict(es.optimum[2])    # optimum model (load weights)















# Transfer Learning & Fine-Tunning for Classification ★ #########################################################
print('* Classification with pre-training : fine-tunning (classification downstream) ----------------------------')

gpt_cls_with_pretrain = GPT1(vocab_size, GPT1_ClassifierLayer(), n_layers=3).to(device)
gpt_cls_with_pretrain.fine_tunning()
# gpt_cls_with_pretrain(cls_sample_X)       # (64, 25, 2160), (64, 2)

# parameter transfer ★★★
gpt_cls_with_pretrain.gpt_decoder.load_state_dict(gpt_pretain.gpt_decoder.state_dict())


model = copy.deepcopy(gpt_cls_with_pretrain)

# # model weights parameter initialize (가중치 초기화) ***
# def init_weights(model):
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             torch.nn.init.normal_(param.data, mean=0, std=0.01)
#         else:
#             torch.nn.init.constant_(param.data, 0)
# model.apply(init_weights)


es = EarlyStopping(patience=100)
loss_function = torch.nn.CrossEntropyLoss()

# learning_rate = 0.001
learning_rate = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 100


lamb = 0.5

# training * -------------------------------------------------------------------------------------------------------
train_loader = cls_train_loader
valid_loader = cls_valid_loader
test_loader = cls_test_loader

train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for ei, batch  in enumerate(train_loader):
        batch_X = batch[0].type(torch.long).to(device)
        batch_y = batch[1].to(device)
        
        optimizer.zero_grad()                   # wegiht initialize
        pred_pretrain, pred_cls = model(batch_X)                   # forward
        # loss_function for GPT fine-tunning ----------------------------------------
        # Loss_L1
        pretrain_pred_eval = pred_pretrain[:,:-1,:].reshape(-1, vocab_size)
        pretrain_real_eval = batch_X[:,1:].reshape(-1)
        loss_L1 = loss_function(pretrain_pred_eval, pretrain_real_eval)     # loss

        # Loss_L2
        loss_L2 = loss_function(pred_cls, batch_y)     # loss

        # Loss = L2 + λ·L1
        loss = loss_L2 + lamb * loss_L1
        loss.backward()                         # backward
        # ------------------------------------------------------------------------------
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    # 기울기(gradient) clipping 진행
        # (gradient clipping) https://sanghyu.tistory.com/87
        optimizer.step()                        # update_weight

        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )

    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch in valid_loader:
            batch_X = batch[0].type(torch.long).to(device)
            batch_y = batch[1].to(device)
            
            pred_pretrain, pred_cls = model(batch_X)                   # forward
            # loss_function for GPT fine-tunning ----------------------------------------
            # Loss_L1
            pretrain_pred_eval = pred_pretrain[:,:-1,:].reshape(-1, vocab_size)
            pretrain_real_eval = batch_X[:,1:].reshape(-1)
            loss_L1 = loss_function(pretrain_pred_eval, pretrain_real_eval)     # loss

            # Loss_L2
            loss_L2 = loss_function(pred_cls, batch_y)     # loss

            # Loss = L2 + λ·L1
            loss = loss_L2 + lamb * loss_L1
            # ------------------------------------------------------------------------------
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )

    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')

        # # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)
        if early_stop == 'break':
            break
        # # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot
# ------------------------------------------

gpt_cls_with_pretrain.load_state_dict(es.optimum[2])    # optimum model (load weights)

# Performance Evaluate (Accuracy_score)
gpt_cls_with_pretrain.downstream()
torch_accuracy_score(cls_test_loader, gpt_cls_with_pretrain, device)




# Compare Performance ###########################################################
torch_accuracy_score(cls_valid_loader, gpt_cls, device, verbose=1)
torch_accuracy_score(cls_test_loader, gpt_cls_with_pretrain, device, verbose=1)
print()








