


# tensorflow ----------------------------------------
class RNN_Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size_en, 32)
        self.rnn = tf.keras.layers.SimpleRNN(16, return_sequences=True, return_state=True)   # return seq / state
        
    def call(self, X):
        self.e1 = self.embed(X)
        self.e2_seq, self.e2_hidden = self.rnn(self.e1)
        return self.e2_seq, self.e2_hidden

class RNN_Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size_kor, 32)
        self.rnn = tf.keras.layers.SimpleRNN(16, return_sequences=True, return_state=True)   # return seq / state
        self.dense = tf.keras.layers.Dense(vocab_size_kor, activation='softmax')
        
    def call(self, X, hidden):
        self.d1 = self.embed(X)
        self.d2_seq, self.d2_hidden = self.rnn(self.d1, initial_state=[hidden] )
        self.result = self.dense(self.d2_seq)
        return self.result

class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = RNN_Encoder()
        self.decoder = RNN_Decoder()
    
    def call(self, Xy, training=True, teacher_forcing=1):
        X = Xy[0]
        y = Xy[1]
        # X = tf.constant(Xy[0], dtype=tf.int32)
        # y = tf.constant(Xy[1], dtype=tf.int32)

        self.seq_len = y.shape[1]
        # teacher_forcing = 0
        states, context_vector = self.encoder(X)
        y_word = y[:,0][..., tf.newaxis]
        self.result = self.decoder(y_word, context_vector)

        for i in range(1, self.seq_len):
            pred_output = self.decoder(y_word, context_vector)
            
            self.result = tf.concat([self.result, pred_output],axis=1)
            if teacher_forcing >= np.random.rand():
                y_word = y[:,i][..., tf.newaxis]
            else:
                y_word = tf.argmax(pred_output, axis=2)
        
        return self.result
    
    def predict(self, X, return_word=True):
        X_len = X.shape[0]
        states, context_vector = self.encoder(X)
        y_word = np.repeat(907, X_len).reshape(-1,1) 
        self.pred_result = self.decoder(y_word, context_vector)

        for i in range(1, self.seq_len):
            pred_output = self.decoder(y_word, context_vector)
            self.pred_result = tf.concat([self.pred_result, pred_output],axis=1)
            y_word = tf.argmax(pred_output, axis=2)
        self.pred_word = tf.argmax(self.pred_result, axis=2)

        return self.pred_word if return_word else self.pred_result

model = RNN_Seq2Seq()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.fit(x=[train_X, train_inout_y], y=train_inout_oh_y, batch_size=64, epochs=10)


# model([train_X, train_inout_y])
pred_result = model.predict(train_X)
[''.join(['' if w==0 else tokenizer_kor.index_word[w] for w in seq]) for seq in pred_result.numpy()]




target = ['I am a boy']
target_token = tokenizer_en.texts_to_sequences(target)
target_padseq = tf.keras.preprocessing.sequence.pad_sequences(target_token, maxlen=train_X.shape[1], padding='post')

model.predict(target_padseq)


# enco = RNN_Encoder()
# deco = RNN_Decoder()

# X_sample.shape
# y_inout_sample.shape
# y_inout_oh_sample.shape

# teacher_forcing = 0
# states, hidden = enco(tf.constant(X_sample, dtype=tf.int32))
# input_de = tf.constant(y_inout_sample[:,[0]], dtype=tf.int32)
# output = deco(input_de, hidden)
# for i in range(1, y_inout_sample.shape[1]):
#     pred_de = deco(input_de, hidden)
#     output = tf.concat([output,pred_de],axis=1)
#     if teacher_forcing >= np.random.rand():
#         input_de = tf.constant(y_inout_sample[:,[i]], dtype=tf.int32)
#     else:
#         input_de = tf.argmax(pred_de, axis=2)












####################################
rnn_model = RNN_Translate()
# rnn_model([X_sample, input_sample])
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(x=[train_X, train_input_y], y=train_output_y, 
          epochs=10,
          validation_data=([test_X, test_input_y], test_output_y))





# ##########################################################################################
# english_input = tf.keras.layers.Input(shape=(None,))
# encoder = tf.keras.layers.Embedding(input_dim=vocab_size_en, output_dim=32)(english_input)
# encoder_seq, encoder_hidden, encoder_cell = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True)(encoder)
# # model_encoder = tf.keras.models.Model(english_input)

# korean_input = tf.keras.layers.Input(shape=(None,))
# decoder = tf.keras.layers.Embedding(input_dim=vocab_size_kor, output_dim=32)(korean_input)
# decoder_seq, hidden_state, cell_state = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True)(decoder, initial_state=[encoder_hidden, encoder_cell])
# korean_output = tf.keras.layers.Dense(units=vocab_size_kor, activation='softmax')(decoder_seq)

# model = tf.keras.models.Model([english_input, korean_input], korean_output)
# # model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit([train_X, train_input_y], train_output_y, 
#           epochs=10,
#           validation_data=([test_X, test_input_y], test_output_y))
# ##########################################################################################



####################################################################################################

torch.nn.Sequential()
# nn.Sequential은 input으로 준 module에 대해 순차적으로 forward() method를 호출해주는 역할
torch.nn.ModuleList()
# nn.ModuleList는 nn.Sequential과 마찬가지로 nn.Module의 list를 input으로 받는다.
# nn.Module을 저장하는 역할을 한다. index로 접근도 할 수 있다.
# nn.Sequential과 다르게 forward() method가 없다.
# 안에 담긴 module 간의 connection도 없다.
#  nn.ModuleList안에 Module들을 넣어 줌으로써 Module의 존재를 PyTorch에게 알려 주어야 한다.
# - 만약 nn.ModuleList에 넣어 주지 않고, Python list에만 Module들을 넣어 준다면, PyTorch는 이들의 존재를 알지 못한다.
# - 따라서 Module들을 Python list에 넣어 보관한다면, 꼭 마지막에 이들을 nn.ModuleList로 wrapping 해줘야 한다.

####################################################################################################
# https://sanghyu.tistory.com/3

# view() 
# . 원소의 수를 유지하면서 텐서의 shape를 변경하는 함수
# . contiguous tensor에서 사용할 수 있음
# . view 함수를 이용해서 반환된 값은 원본과 data(memory)를 공유하기 때문에 하나만 수정해도 반환 이전의 변수와 이후 변수 모두 수정된다.

# reshape()
# . contiguous하지 않는 함수에서도 작동한다.
# . reshape() == contiguous().view()

# transpose()
# [batch_size, hidden_dim, input_dim] -> [batch_size, input_dim, hidden_dim]
# 변환이후, contiguous한 성질을 잃어버리기 때문에 transpose().contiguous()와 같이 contiguous()함수와 같이 사용함

# permute()
# 모든 차원을 맞교환 할 수 있다. (transpose()의 일반화 버전이라고 생각한다.)
# 차원을 교환하면서 contiguous 한 성질이 사라진다. 
# view와 같은 contiguous한 성질이 보장될 때만 사용할 수 있는 함수를 사용해야 한다면, permute().contiguous()를 사용하자


# contiguous()
# 메모리상에 데이터를 contiguous 하게 배치한 값을 반환한다.

####################################################################################################


# (torch.tril)     ------------------------------------------------------------------------------------
# torch.tril(input, diagonal=0, *, out=None) → Tensor
# 행렬의 아래쪽 삼각형 부분 (2 차원 텐서) 또는 행렬의 배치 input 을 반환합니다 . 결과 텐서 out 의 다른 요소는 0으로 설정됩니다
# a = torch.rand(5,5)
# a
# torch.tril(a)
# torch.tril(a, diagonal=0)
# torch.tril(a, diagonal=1)
# torch.tril(a, diagonal=2)
# torch.tril(a, diagonal=3)
# torch.tril(a, diagonal=-1)
# torch.tril(a, diagonal=-2)
# torch.tril(a, diagonal=-3)

# a = torch.rand(3,1,1,5) > 0.5
# a_len = a.shape[-1]

# b = torch.tril(torch.ones((a_len, a_len))).bool()  # (batch_seq, batch_seq)  
# ---------------------------------------------------------------------------------------------------------



# (layer Normalization)   ------------------------------------------------------------------------
# https://wingnim.tistory.com/92
# https://velog.io/@tjdcjffff/Normalization-trend
# Batch-Normalization은 기존의 Batch들을 normalization했다면, Layer normalization은 Feature 차원에서 정규화를 진행한다.

# ○ 도입배경 *
# 기존 연구들은 training time을 줄이는 방법으로 batch normalization을 제안하였습니다.
# 그러나 batch normalization은 몇 가지 단점을 가지고 있습니다.
# batch normalization은 mini-batch size에 의존한다.
# recurrent neural network model인 경우, 어떻게 적용이 되는지 명백하게 설명하기 어렵다.
# 본 연구에서는 이러한 문제점을 해결하기 위해서 layer normalization을 제안하였습니다.
# layer normalization은 batch normalization과 달리 train/test time 일 때, 같은 computation을 수행한다는 점이 큰 특징입니다.

# layer normalization은 특정 layer가 가지고 있는 hidden unit에 대해 μ , σ를 공유함
# BN과 다르게 mini-batch-size에 제약이 없음
# RNN model에선 각각의 time-step마다 다른 BN이 학습됨
# LN은 layer의 output을 normalize함으로써 RNN계열 모델에서 좋은 성능을 보임

# ---------------------------------------------------------------------------------------------------------



# (contiguous 여부와 stride 의미)  ------------------------------------------------------------------------
# https://jimmy-ai.tistory.com/122
# https://f-future.tistory.com/entry/Pytorch-Contiguous
import torch

a = torch.randn(3, 4)
a.transpose_(0, 1)

b = torch.randn(4, 3)

# 두 tensor는 모두 (4, 3) shape
print(a)
print(b)

# a 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(a[i][j].data_ptr())

# b 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(b[i][j].data_ptr())


# 각 데이터의 타입인 torch.float32 자료형은 4바이트이므로, 메모리 1칸 당 주소 값이 4씩 증가함을 알 수 있습니다.
# 그런데 자세히 보시면 b는 한 줄에 4씩 값이 증가하고 있지만, a는 그렇지 않은 상황임을 알 수 있습니다.

# 즉, b는 axis = 0인 오른쪽 방향으로 자료가 순서대로 저장됨에 비해,
# a는 transpose 연산을 거치며 axis = 1인 아래 방향으로 자료가 저장되고 있었습니다.

# 여기서, b처럼 axis 순서대로 자료가 저장된 상태를 contiguous = True 상태라고 부르며,
# a같이 자료 저장 순서가 원래 방향과 어긋난 경우를 contiguous = False 상태라고 합니다.

# 각 텐서에 stride() 메소드를 호출하여 데이터의 저장 방향을 조회할 수 있습니다.
# 또한, is_contiguous() 메소드로 contiguous = True 여부도 쉽게 파악할 수 있습니다.

a.stride() # (1, 4)
b.stride() # (3, 1)
# 여기에서 a.stride() 결과가 (1, 4)라는 것은
# a[0][0] -> a[1][0]으로 증가할 때는 자료 1개 만큼의 메모리 주소가 이동되고,
# a[0][0] -> a[0][1]로 증가할 때는 자료 4개 만큼의 메모리 주소가 바뀐다는 의미입니다.


a.is_contiguous() # False
b.is_contiguous() # True

# 텐서의 shape을 조작하는 과정에서 메모리 저장 상태가 변경되는 경우가 있습니다.
# 주로 narrow(), view(), expand(), transpose() 등 메소드를 사용하는 경우에 이 상태가 깨지는 것으로 알려져 있습니다.
#   ㄴ 메모리를 따로 할당하지 않는 Tensor연산

# 해당 상태의 여부를 체크하지 않더라도 텐서를 다루는데 문제가 없는 경우가 많습니다.
# 다만, RuntimeError: input is not contiguous의 오류가 발생하는 경우에는
# input tensor를 contiguous = True인 상태로 변경해주어야 할 수 있습니다.

# 이럴 때에는 아래 예시 코드처럼 contiguous() 메소드를 텐서에 적용하여
# contiguous 여부가 True인 상태로 메모리 상 저장 구조를 바꿔줄 수 있습니다.

a.is_contiguous() # False

# 텐서를 contiguous = True 상태로 변경
a = a.contiguous()
a.is_contiguous() # True
# ---------------------------------------------------------------------------------------------------------









































# 인코더 네트워크
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()       
        self.input_dim = input_dim # 인코더 입력층
        self.embbed_dim = embbed_dim # 인코더 임베딩 계층
        self.hidden_dim = hidden_dim # 인코더 은닉층(이전 은닉층)
        self.num_layers = num_layers # GRU 계층 개수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim) # 임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        # 임베딩 차원, 은닉층 차원, gru 계층 개수를 이용하여 gru 계층 초기화
        
    def forward(self, src):      
        embedded = self.embedding(src).view(1,1,-1) # 임베딩
        outputs, hidden = self.gru(embedded) # 임베딩 결과를 GRU 모델에 적용
        return outputs, hidden

# 디코더 네트워크 
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim) # 임베딩 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers) # gru 초기화
        self.out = nn.Linear(self.hidden_dim, output_dim) # 선형 계층 초기화
        self.softmax = nn.LogSoftmax(dim=1)
      	
    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)       
        prediction = self.softmax(self.out(output[0]))      
        return prediction, hidden 

# Seq2seq 네트워크
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        # 인코더와 디코더 초기화
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
     
    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):

        input_length = input_lang.size(0) # 입력 문장 길이(문장 단어수)
        batch_size = output_lang.shape[1] 
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim      
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            # 문장의 모든 단어 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
            
        # 인코더 은닉층 -> 디코더 은닉층
        decoder_hidden = encoder_hidden.to(device)  
        # 예측 단어 앞에 SOS token 추가
        decoder_input = torch.tensor([SOS_token], device=device)  

        for t in range(target_length):   
            # 현재 단어에서 출력단어 예측
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            # teacher force 활성화하면 모표를 다음 입력으로 사용
            input = (output_lang[t] if teacher_force else topi)
            # teacher force 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token) :
                break
        return outputs
    
# teacher_force : seq2seq에서 많이 사용되는 기법. 번역(예측)하려는 목표 단어를 디코더의 다음 입력으로 넣어줌



# # Seq2Seq ###############################################################################################
# 하이퍼 파라미터 지정
input_dim = len(SRC.vocab)  # 7854
output_dim = len(TRG.vocab) # 5893
enc_emb_dim = 256 # 임베딩 차원
dec_emb_dim = 256
hid_dim = 512 # hidden state 차원
n_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

# # 모델 생성 ###############################################################################################
enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
# (input: X) src            (seq, batch)
# . emb : 7854, 256         (seq, batch, emb_dim)
# . lstm : 256, 512,        (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# (return) : hidden(lstm), cell(lstm)

dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)
# (input: y) trg[0,:]        (batch) : 1st word at each seqences
# . unsqueeze : 1, batch    (1, batch)
# . emb : 5893, 256         (1, batch, emb_dim)
# . lstm : 256, 512         (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# . linear : 512, 5893
# (return) : prediction(linear), hidden(lstm), cell(lstm)
############################################################################################################


batch_size = trg_s.shape[1]
trg_len = trg_s.shape[0]    # 타겟 토큰 길이 얻기
trg_vocab_size = output_dim     # context vector의 차원 : target_vocab
teacher_forcing_ratio = 0

# decoder의 output을 저장하기 위한 tensor
outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)   # (seq, batch, embeding)

# initial hidden state
hidden, cell = enc(src_s)   # (1,3,512), (1,3,512) : (seq, batch, rnn_nodes)
# embedded       # (31, 3, 256) : (seq, batch, emb_dim)
# rnn            # (1,3, 512), (1,3, 512), (1,3, 512)   : (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)

# 첫 번째 입력값 <sos> 토큰
input_de = trg_s[0,:]   # (3) : (batch)

# for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복
output, hidden, cell = dec(input_de, hidden, cell)
# print(output.shape, hidden.shape, cell.shape)
# (input)
# input_de.unsqueeze(0) # (1, 3) : (seq, batch)
# embedded       # (1,3, 256) : (seq, batch, emb_dim)
# rnn            # (1,3, 512), (1,3, 512), (1,3, 512)   : (seq, batch, h_dim*n_dir) (n_lay*n_dir, batch, h_dim), (n_lay*n_dir, batch, h_dim)
# linear         # (3, 5893) : (batch, y_vocab)

# prediction 저장
outputs[t] = output

# teacher forcing을 사용할지, 말지 결정
teacher_force = random.random() < teacher_forcing_ratio

# 가장 높은 확률을 갖은 값 얻기
top1 = output.argmax(1)

# teacher forcing의 경우에 다음 lstm에 target token 입력
input_de = trg[t] if teacher_force else top1
# return outputs
############################################################################################################################

