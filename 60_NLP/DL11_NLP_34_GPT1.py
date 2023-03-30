import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch



# url_path ='/home/kimds929/DataSet/'

# train_df = pd.read_csv(f'{url_path}/NLP_movie_review_simple_train_tokenized.csv', encoding='utf-8-sig')

# from DS_NLP import NLP_Preprocessor

# processor = NLP_Preprocessor(texts=train_df['tokenized'])
# processor.fit_on_texts().texts_to_sequences().add_sos_eos().pad_sequences()
# processor.sequences_to_texts()
# vocab_size = processor.vocab_size

# train_X = processor.texts
# train_y = train_df['label'].to_numpy()
# print(train_X.shape)


## Pre-Train Dataset
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/'
pretrain_seq = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_pad_seq_sentences(KR).csv', encoding='utf-8-sig').to_numpy()

index_word_df = pd.read_csv(f'{url_path}/NLP_EN_to_KR_0_index_word(KR).csv', encoding='utf-8-sig')
index_word = index_word_df.set_index('index')['word'].to_dict()
word_index = index_word_df.set_index('word')['index'].to_dict()


class TextPreprocessor():
    def __init__(self, index_word=None, word_index=None):
        self.index_word = index_word
        self.word_index = word_index
        self.index_word_np = None
        
        self.sort = False
        self.vocab_size = None
        
        self.make_dict()

    def make_dict(self):
        if self.index_word is not None and self.word_index is None:
            self.word_index = {v: k for k, v in self.index_word.items()}
        if self.word_index is not None and self.index_word is None :
            self.index_word = {k: v for v, k in word_index.items()}

        if self.sort is False:
            if ~(0 in index_word.keys()):
                self.index_word[0] = ''
                self.word_index[''] = 0
            self.index_word = dict(sorted(self.index_word.items(), key=lambda item: item[0]))
            self.word_index = dict(sorted(self.word_index.items(), key=lambda item: item[1]))
            self.sort = True

        if self.word_index is not None and self.index_word_np is None:
            self.index_word_np = np.array(list(self.index_word.values()))     
        
        self.vocab_size = len(index_word)+1
        
    def seq_to_text(self, x, text=False):
        self.make_dict()
        
        result = self.index_word_np[x]
        if text is True:
            result = [''.join(list(l)) for l in result]
        return result

tp = TextPreprocessor(index_word)
tp.seq_to_text(pretrain_seq, text=True)
vocab_size = tp.vocab_size



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
from DS_TorchModule import EmbeddingLayer, PositionalEncodingLayer, PositionwiseFeedForwardLayer, MultiHeadAttentionLayer, make_tril_mask

#######################################################################################################################################
# https://paul-hyun.github.io/gpt-01/
# https://paul-hyun.github.io/transformer-02/

# ★★ GPT1_Pretrain
#   gpt_pt = GPT1_Pretrain(vocab_size, 2, n_heads=1)
#   op = gpt_pt(sample_X)
#   op.shape
class GPT1_Pretrain(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding='sinusoid'):
        super().__init__()
        self.gpt_decoder = GPT1_Decoder(vocab_size, embed_dim, n_layers, n_heads, 
                                   posff_dim, dropout, pos_encoding)
        
        self.projection_layer = torch.nn.Linear(embed_dim, vocab_size, bias=False)
        self.projection_layer.weight = self.gpt_decoder.embed_layer.weight     # We.T
    
    def forward(self, x):
        mask_tril = make_tril_mask(x).unsqueeze(1)
        self.decoder_output = self.gpt_decoder(x, mask_tril)
        with torch.no_grad():
            # self.attention_scores = [layer.attention_score for layer_name, layer in self.decoder.decoder_layers.named_children()]
            self.self_attention_score = self.gpt_decoder.decoder_layers[-1].self_attention_score

        self.projection_output = self.projection_layer(self.decoder_output)
        self.output = self.projection_output[:,:-1,:]
        # 입력에 다한 다음 단어를 예측하는 것이므로 결과의 마지막을 제외한 나머지를 리턴 합니다. 
        return self.output
    
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




# DataSet ########################################################################
# import numpy as np
# import torch
from sklearn.model_selection import train_test_split
class TorchDataLoader():
    def __init__(self, *args, split_size=(0.7, 0.1, 0.2), random_state=None, **kwargs):
        self.args = args
        assert (np.array(list(map(len, self.args)))/len(self.args[0])).all() == True, 'Arguments must have same length'
        self.idx = np.arange(len(self.args[0]))
        
        self.split_size = [s/np.sum(split_size) for s in split_size]
        
        self.train_test_split_size = None
        self.train_valid_split_size = None
        
        if len(self.split_size) == 2:
            self.train_test_split_size = self.split_size
        elif len(self.split_size) == 3:
            self.train_test_split_size = [self.split_size[0]+self.split_size[1], self.split_size[2]]
            self.train_valid_split_size = [s/self.train_test_split_size[0] for s in self.split_size[:2]]
        
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.torch_data = None
        self.dataset = None
        self.dataloader = None
        
    def split(self, dtypes=None, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        self.train_idx, self.test_idx = train_test_split(self.idx, test_size=self.train_test_split_size[-1], random_state=random_state)
        self.index = (self.train_idx, self.test_idx)
        if self.train_valid_split_size is not None:
            self.train_idx, self.valid_idx = train_test_split(self.train_idx, test_size=self.train_valid_split_size[-1], random_state=random_state)
            self.index = (self.train_idx, self.valid_idx, self.test_idx)
        
        [print(len(index), end=', ') for index in self.index]
        if dtypes is None:
            self.torch_data = tuple([tuple([torch.tensor(arg[idx]) for idx in self.index]) for arg in self.args])
        else:
            self.torch_data = tuple([tuple([torch.tensor(arg[idx]).type(dtype) for idx in self.index]) for arg, dtype in zip(self.args, dtypes)])
    
    def make_dataset(self, dtypes=None, random_state=None):
        if self.torch_data is None:
            self.split(dtypes, random_state)
            
        self.dataset = tuple([torch.utils.data.TensorDataset(*data) for data in zip(*self.torch_data)])

    def make_dataloader(self, dtypes=None, random_state=None, **kwargs):
        if self.dataset is None:
            self.make_dataset(dtypes, random_state)
        if len(kwargs) > 0:
            self.kwargs = kwargs
            
        self.dataloader = tuple([torch.utils.data.DataLoader(dataset, **self.kwargs) for dataset in self.dataset])
        
        for sample in self.dataloader[0]:
            break
        self.sample = sample

# X = np.random.rand(100,3)
# y = np.random.rand(100)
# loader =  TorchDataLoader(X, y, split_size=(0.7, 0.1, 0.2), random_state=1)
# loader.make_dataloader()
# loader.dataloader
# loader.sample
# train_loader, valid_loader, test_loader = loader.dataloader

loader = TorchDataLoader(pretrain_seq, split_size=(0.8,0.1,0.1), random_state=1)
loader.make_dataloader(batch_size=64, shuffle=True)
loader.dataloader
train_loader, valid_loader, test_loader = loader.dataloader
sample_X = loader.sample[0]





# Training ########################################################################
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

# # customize library ***---------------------
import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=100)
# # ------------------------------------------
import time
import copy


# sample_X.shape    # (64, 132)
gpt_pretain = GPT1_Pretrain(vocab_size).to(device)
# gpt_pretain(sample_X)  # (64, 131, 3986)


model = copy.deepcopy(gpt_pretain)

# model weights parameter initialize (가중치 초기화) ***
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)
model.apply(init_weights)

# trg_pad_idx = TRG.vocab.stoi[TRG.pad_token] ## pad에 해당하는 index는 무시합니다.

# loss_function = torch.nn.CrossEntropyLoss()     # ignore_index=trg_pad_idx
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())
epochs = 20

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch  in train_loader:
        batch_data = batch[0]
        optimizer.zero_grad()                   # wegiht initialize
        pred = model(batch_data.to(device))                   # predict
        pred_eval = pred.reshape(-1, vocab_size)
        real_eval = batch_data[:,1:].reshape(-1).to(device)
         
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
            batch_data = batch[0]
            pred = model(batch_data.to(device))                   # predict
            pred_eval = pred.reshape(-1, vocab_size)
            real_eval = batch_data[:,1:].reshape(-1).to(device)              # predict

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

gpt_pretain.load_state_dict(es.optimum[2])    # optimum model (load weights)
# ------------------------------------------



















