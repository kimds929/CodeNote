# (Python) Tabular Torch TransferLearning 230316

# df = df01.copy()


# y_cols = ['초_YP', '초_TS', '초_EL', '초_U_EL', '초_HER_평균']
y_cols = ['초_YP', '초_TS', '초_EL']
# X_cols_str = ['인장_방향', '인장_호수', '열연공장']
X_cols_str = [ '열연공장']
X_cols_num = ['주문두께', '소둔_폭', 
    'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적', 'SolAl_실적', 'Nb_실적', 'Cr_실적', 'Ti_실적', 'N_실적', 
    'SRT', 'FDT', 'CT', '냉간압하율', 
    'LS', 'HS_POS', 'SS_POS', 'SCS_POS', 'RCS_POS', 'RHS_POS', 'SPM_EL', 'SPM_RollForce']
X_cols = X_cols_str + X_cols_num
cols = y_cols + ['품종명','소둔작업완료일시'] + X_cols

df0 = df[~(df[y_cols + X_cols].isna()).any(1)]


df0[cols].to_clipboard()

df0.groupby(['품종명','소둔공장']).size().unstack('소둔공장')


df1 = df0.copy()
df1 = df0.query("(소둔공장=='5CGL') & (품종명 == 'GA')")

for c in df1[X_cols].columns:
    if c in X_cols_num:
        df1[c] = df1[c].astype('float')
    elif c in X_cols_str:
        df1[c] = df1[c].astype('str')
df1[y_cols] = df1[y_cols].astype('float')

from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(df1.index, test_size=0.2, random_state=0)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.15, random_state=0)

from DS_MachineLearning import ScalerEncoder

se_y = ScalerEncoder()
se_y.fit(df1.loc[train_idx][y_cols])

se_X = ScalerEncoder()
se_X.fit(df1.loc[train_idx][X_cols])
se_X.transform(df1.loc[train_idx][X_cols])


X_cols_str_t = np.stack([seX.match_columns[c] for c in X_cols_str if se.match_columns[c]]).ravel().tolist()
X_cols_num_t = np.stack([seX.match_columns[c] for c in X_cols_num if se.match_columns[c]]).ravel().tolist()
X_cols_t = X_cols_str_t + X_cols_num_t



train_X = se_X.transform(df1.loc[train_idx][X_cols])
valid_X = se_X.transform(df1.loc[valid_idx][X_cols])
test_X = se_X.transform(df1.loc[test_idx][X_cols])

train_y = se_y.transform(df1.loc[train_idx][y_cols])
valid_y = se_y.transform(df1.loc[valid_idx][y_cols])
test_y = se_y.transform(df1.loc[test_idx][y_cols])

print(train_X.shape, valid_X.shape, test_X.shape)
print(train_y.shape, valid_y.shape, test_y.shape)



############################################################################
y_cols_pred = ['초_YP', '초_TS', '초_EL']
# y_cols_pred = ['초_YP']

train_dataset = torch.utils.data.TensorDataset(
         torch.tensor(train_X[X_cols_str_t].to_numpy())
        ,torch.tensor(train_X[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(train_y[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

valid_dataset = torch.utils.data.TensorDataset(
         torch.tensor(valid_X[X_cols_str_t].to_numpy())
        ,torch.tensor(valid_X[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(valid_y[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

test_dataset = torch.utils.data.TensorDataset(
         torch.tensor(test_X[X_cols_str_t].to_numpy())
        ,torch.tensor(test_X[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(test_y[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



##########################################################################################
import torch
import copy


# --- Model ---------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
# torch.cuda.empty_cache()

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embed_layer = torch.nn.Embedding(vocab_size, emb_dim)
        
    def forward(self, X):
        self.embed_output = self.embed_layer(X)
        return self.embed_output
    
xs = torch.tensor(train_X[X_cols_str_t].to_numpy())
xn = torch.tensor(train_X[X_cols_num_t].to_numpy()).type(torch.float32)


# 3 layer relu (128, 256, 128)
# batch norm
class DNN_Regressor(torch.nn.Module):
    def __init__(self, str_cols, num_cols, emb_dim=2, output_dim=1, hidden_dims=[128,128,128], dropout=0.1):
        super().__init__()
        self.embed_layer = Embedding(str_cols+1, emb_dim)
        self.flatten_layer = torch.nn.Flatten()
        
        fc_modules = []
        for node_in, node_out in zip([(str_cols)*emb_dim + num_cols, *hidden_dims], hidden_dims+[output_dim]):
            fc_modules.append( torch.nn.Linear(node_in, node_out) )
            fc_modules.append( torch.nn.BatchNorm1d(node_out) )
            fc_modules.append( torch.nn.ReLU() )
            
        self.fc_modules = torch.nn.ModuleList(fc_modules[:-2])
        self.dropout_layer = torch.nn.Dropout(dropout)
        
    def forward(self, X):
        # X : [X(str), X(num)]
        self.embed_output = self.embed_layer(X[0])
        self.embed_flatten = self.flatten_layer(self.embed_output)
        
        self.concat = torch.cat([self.embed_flatten, X[1]], dim=-1)
        output = self.concat
        
        for e, fc_layer in enumerate(self.fc_modules):
            # print(e, fc_layer)
            if e % 3 == 2:
                output_ = fc_layer(output)
                output = self.dropout_layer(output + output_)   # residual connection
            else:
                output = fc_layer(output)
        
        return output


for batch in train_loader:
    batch
    break

dr = DNN_Regressor(batch[0].shape[-1], batch[1].shape[-1], output_dim=batch[2].shape[-1])
# dr((batch[0], batch[1]))

model = copy.deepcopy(dr)


# model weights parameter initialize (가중치 초기화) ***
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)
model.apply(init_weights)


loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# # customize library ***---------------------
import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=50)
# # ------------------------------------------
import time


epochs = 50

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

torch.autograd.set_detect_anomaly(False)
# with torch.autograd.detect_anomaly():
#     input = torch.rand(5, 10, requires_grad=True)
#     output = function_A(input)
#     output.backward()


# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch in train_loader:
        batch = [batch_data.to(device) for batch_data in batch]
        
        optimizer.zero_grad()                   # wegiht initialize
        pred = model((batch[0], batch[1]))                   # predict
        loss = loss_function(pred, batch[2])     # loss
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
            batch = [batch_data.to(device) for batch_data in batch]
            
            pred = model((batch[0], batch[1]))                   # predict
            loss = loss_function(pred, batch[2])     # loss
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

        # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(),
                                   verbose=2)
        if early_stop == 'break':
            break
        # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot


dr_5cgGA = DNN_Regressor(batch[0].shape[-1], batch[1].shape[-1], output_dim=batch[2].shape[-1])
dr_5cgGA.load_state_dict(es.optimum[2])





# dr.load_state_dict(es.optimum[2])
class DL_Estimator():
    def __init__(self, model, scaler_X=None, scaler_y=None):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def predict(self, x):
        transformed_x = self.input_transform(x)

        with torch.no_grad():
            self.model.eval()
            pred = self.model.forward(transformed_x)
        output = self.inverse_transform(pred, self.scaler_y)
        
        return output
        
    def input_transform(self, x, scaler=None):
        scaler = self.scaler_X if scaler is None else scaler
        
        if scaler is None:
            return torch.tensor(x)
        else:
            x_transformed = scaler.transform(x)
            return torch.tensor(np.array(x_transformed)).type(torch.float32)

                
    def output_transform(self, x, scaler=None):
        scaler = self.scaler_y if scaler is None else scaler

        if scaler is None:
            return torch.tensor(x)
        else:
            x_transformed = scaler.transform(x)
            return torch.tensor(np.array(x_transformed)).type(torch.float32)
        
    def inverse_transform(self, x, scaler=None):
        x_np = x.to('cpu').detach().numpy()
        if scaler is None:
            return x_np
        else:
            return scaler.inverse_transform(x_np)


# dr.load_state_dict(es.optimum[2])
class DL_Estimator2(PredictDL):
    def __init__(self, model, scaler_X=None, scaler_y=None, pred_position=None):
        super().__init__(model, scaler_X, scaler_y)
        self.pred_position = pred_position

    def input_transform(self, x, scaler=None):
        scaler = self.scaler_X if scaler is None else scaler
        self.scaler_X = scaler
        
        if scaler is None:
            return torch.tensor(x)
        else:
            x_transformed_x = scaler.transform(x)
            x_torch = torch.tensor(x_transformed_x.to_numpy()).type(torch.float32)
            return (x_torch[:,:1].type(torch.int32), x_torch[:,1:])

    def predict(self, x):
        transformed_x = self.input_transform(x)

        with torch.no_grad():
            self.model.eval()
            pred = self.model.forward(transformed_x)
        output = self.inverse_transform(pred, self.scaler_y)
        
        if self.pred_position is None:
            return output
        else:
            return output[:,self.pred_position]


# from sklearn.ensemble import RandomForestRegressor
# RF_YP = RandomForestRegressor()
# RF_YP.fit(train_X,  df1.loc[train_idx]['초_YP'])

# RF_TS = RandomForestRegressor()
# RF_TS.fit(train_X,  df1.loc[train_idx]['초_TS'])

# RF_EL = RandomForestRegressor()
# RF_EL.fit(train_X,  df1.loc[train_idx]['초_EL'])


# ModelEvaluate(test_X, df1.loc[test_idx]['초_YP'], model=RF_YP)
# ModelEvaluate(test_X, df1.loc[test_idx]['초_TS'], model=RF_TS)
# ModelEvaluate(test_X, df1.loc[test_idx]['초_EL'], model=RF_EL)



est_5cgGA_yp = DL_Estimator2(dr_5cgGA, se_X, se_y, pred_position=0)
est_5cgGA_ts = DL_Estimator2(dr_5cgGA, se_X, se_y, pred_position=1)
est_5cgGA_el = DL_Estimator2(dr_5cgGA, se_X, se_y, pred_position=2)

ModelEvaluate(df1.loc[test_idx][X_cols], df1.loc[test_idx]['초_YP'], model=est_5cgGA_yp, model_type='regressor')  # 18.35
ModelEvaluate(df1.loc[test_idx][X_cols], df1.loc[test_idx]['초_TS'], model=est_5cgGA_ts, model_type='regressor')  # 15.27
ModelEvaluate(df1.loc[test_idx][X_cols], df1.loc[test_idx]['초_EL'], model=est_5cgGA_el, model_type='regressor')  # 0.69




























# coils = ['CQNB176', 'CQNB177', 'CQNB178', 'CQP2146', 'CQP8459', 'CQP8460', 'CQN1019', 'CQH1369', 'CQM3897', 'CQM3898', 'CQM3899', 'CQN1020', 'CQN1021']
df_special = pd.read_clipboard(sep='\t')
df_3m = pd.read_clipboard(sep='\t')

cols = ['소둔작업완료일시','소둔공장','품종명'] + y_cols+X_cols


df_dict = {}
for ri, rv in df_special.iterrows():
    name = rv['냉연코일번호'] + '_' + rv['루틴특별구분'] + '_'+ rv['위치'] 
    
    temp = df_3m[df_3m['냉연코일번호'].apply(lambda x: x[:7] == rv['냉연코일번호'])].iloc[-1]
    row_dict = {c: None for c in cols}
    for c in cols:
        if c in rv.index:
            row_dict[c] = rv[c]
        else:
            row_dict[c] = temp[c]

    df_dict[name] = row_dict


pd.DataFrame(df_dict).T.to_clipboard()


df3 = df0.query("(소둔공장=='7CGL') & (품종명=='GI')")
df3[['냉연코일번호','소둔작업완료일시','소둔공장','품종명']+y_cols + X_cols].to_clipboard()






df980_all.loc[df2.index][['냉연코일번호','소둔작업완료일시','소둔공장','품종명']].to_clipboard()
##### Transfer Learning ########################################################
# df2 = df0.query("(소둔공장=='5CGL') & (품종명=='GI')")

df_clip = pd.read_clipboard(sep='\t')
    

# df_transfer = df_clip.query("(소둔공장=='5CGL') & (품종명=='GI')")
df_transfer = df_clip.query("(소둔공장=='7CGL') & (품종명=='GI')")
# df_transfer[y_cols + X_cols].to_clipboard()

for c in df_transfer[X_cols].columns:
    if c in X_cols_num:
        df_transfer[c] = df_transfer[c].astype('float')
    elif c in X_cols_str:
        df_transfer[c] = df_transfer[c].astype('str')
df_transfer[y_cols] = df_transfer[y_cols].astype('float')

print(len(df_transfer))

from sklearn.model_selection import train_test_split
train_valid_idx_tr, test_idx_tr = train_test_split(df_transfer.index, test_size=0.2, random_state=0)
train_idx_tr, valid_idx_tr = train_test_split(train_valid_idx_tr, test_size=0.2, random_state=0)

train_X_tr = se_X.transform(df_transfer.loc[train_idx_tr][X_cols])
valid_X_tr = se_X.transform(df_transfer.loc[valid_idx_tr][X_cols])
test_X_tr = se_X.transform(df_transfer.loc[test_idx_tr][X_cols])

train_y_tr = se_y.transform(df_transfer.loc[train_idx_tr][y_cols])
valid_y_tr = se_y.transform(df_transfer.loc[valid_idx_tr][y_cols])
test_y_tr = se_y.transform(df_transfer.loc[test_idx_tr][y_cols])

print(train_X_tr.shape, valid_X_tr.shape, test_X_tr.shape)
print(train_y_tr.shape, valid_y_tr.shape, test_y_tr.shape)

############################################################################
y_cols_pred = ['초_YP', '초_TS', '초_EL']
# y_cols_pred = ['초_YP']

train_dataset_tr = torch.utils.data.TensorDataset(
         torch.tensor(train_X_tr[X_cols_str_t].to_numpy())
        ,torch.tensor(train_X_tr[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(train_y_tr[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

valid_dataset_tr = torch.utils.data.TensorDataset(
         torch.tensor(valid_X_tr[X_cols_str_t].to_numpy())
        ,torch.tensor(valid_X_tr[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(valid_y_tr[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

test_dataset_tr = torch.utils.data.TensorDataset(
         torch.tensor(test_X_tr[X_cols_str_t].to_numpy())
        ,torch.tensor(test_X_tr[X_cols_num_t].to_numpy(), dtype=torch.float32)
        ,torch.tensor(test_y_tr[y_cols_pred].to_numpy(), dtype=torch.float32)
    )

batch_size = 64
train_loader_tr = torch.utils.data.DataLoader(train_dataset_tr, batch_size=batch_size, shuffle=True)
valid_loader_tr = torch.utils.data.DataLoader(valid_dataset_tr, batch_size=batch_size, shuffle=True)
test_loader_tr = torch.utils.data.DataLoader(test_dataset_tr, batch_size=batch_size, shuffle=True)



##########################################################################################

# 기존 5CGL GA모델 사용시 -----------
# 17.81  # 23.97  # 1.57
# ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_YP'], model=est_5cgGA_yp, model_type='regressor')  # 17.81
# ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_TS'], model=est_5cgGA_ts, model_type='regressor')  # 23.91
# ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_EL'], model=est_5cgGA_el, model_type='regressor')  # 1.57



# dr_5cgGI_init = DNN_Regressor(batch[0].shape[-1], batch[1].shape[-1], output_dim=batch[2].shape[-1])
dr_7cgGI_init = DNN_Regressor(batch[0].shape[-1], batch[1].shape[-1], output_dim=batch[2].shape[-1])
# dr((batch[0], batch[1]))

# model = copy.deepcopy(dr_5cgGI_init)
model = copy.deepcopy(dr_7cgGI_init)


# model weights parameter initialize (가중치 초기화) ***
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)
model.apply(init_weights)


loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# # customize library ***---------------------
import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=50)
# # ------------------------------------------
import time


epochs = 50

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

torch.autograd.set_detect_anomaly(False)
# with torch.autograd.detect_anomaly():
#     input = torch.rand(5, 10, requires_grad=True)
#     output = function_A(input)
#     output.backward()


# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch in train_loader_tr:
        batch = [batch_data.to(device) for batch_data in batch]
        
        optimizer.zero_grad()                   # wegiht initialize
        pred = model((batch[0], batch[1]))                   # predict
        loss = loss_function(pred, batch[2])     # loss
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
        for batch in valid_loader_tr:
            batch = [batch_data.to(device) for batch_data in batch]
            
            pred = model((batch[0], batch[1]))                   # predict
            loss = loss_function(pred, batch[2])     # loss
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

        # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(),
                                   verbose=2)
        if early_stop == 'break':
            break
        # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


# customize library ***---------------------
es.plot     # early_stopping plot 
es.optimum[1]   


# (5CG GI init) 0.445 ----------------------------------------------------------------
# dr_5cgGI_init.load_state_dict(es.optimum[2])

# estimate
est_5cgGI_yp0 = DL_Estimator2(dr_5cgGI_init, se_X, se_y, pred_position=0)
est_5cgGI_ts0 = DL_Estimator2(dr_5cgGI_init, se_X, se_y, pred_position=1)
est_5cgGI_el0 = DL_Estimator2(dr_5cgGI_init, se_X, se_y, pred_position=2)


# 12.8  # 14.06  # 1.07
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_YP'], model=est_5cgGI_yp0, model_type='regressor')  # 17.15
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_TS'], model=est_5cgGI_ts0, model_type='regressor')  # 16.32
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_EL'], model=est_5cgGI_el0, model_type='regressor')  # 1.40

# (7CG GI init) 0.685 ----------------------------------------------------------------
# dr_7cgGI_init.load_state_dict(es.optimum[2])

# estimate
est_7cgGI_yp0 = DL_Estimator2(dr_7cgGI_init, se_X, se_y, pred_position=0)
est_7cgGI_ts0 = DL_Estimator2(dr_7cgGI_init, se_X, se_y, pred_position=1)
est_7cgGI_el0 = DL_Estimator2(dr_7cgGI_init, se_X, se_y, pred_position=2)


# 26.34  # 16.58  # 1.097
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_YP'], model=est_7cgGI_yp0, model_type='regressor')  # 17.15
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_TS'], model=est_7cgGI_ts0, model_type='regressor')  # 16.32
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_EL'], model=est_7cgGI_el0, model_type='regressor')  # 1.40



# pd.Series(X_cols).to_clipboard(index=False)
# model 0 (init)
fi0_yp = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_yp0, n_points=50)
fi0_ts = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_ts0, n_points=50)
fi0_el = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_el0, n_points=50)

# fi0_yp.influence_summary()
# fi0_ts.influence_summary()
# fi0_el.influence_summary()

fi0_yp.summary_table.to_clipboard(index=False)
fi0_ts.summary_table.to_clipboard(index=False)
fi0_el.summary_table.to_clipboard(index=False)

img_to_clipboard(fi0_yp.summary_plot)
img_to_clipboard(fi0_ts.summary_plot)
img_to_clipboard(fi0_el.summary_plot)


# model 1
fi1_yp = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_yp1, n_points=50)
fi1_ts = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_ts1, n_points=50)
fi1_el = FeatureInfluence(df_transfer.loc[train_idx_tr][X_cols], est_7cgGI_el1, n_points=50)

fi1_yp.influence_summary()
fi1_ts.influence_summary()
fi1_el.influence_summary()

fi1_yp.summary_table.to_clipboard(index=False)
fi1_ts.summary_table.to_clipboard(index=False)
fi1_el.summary_table.to_clipboard(index=False)

img_to_clipboard(fi1_yp.summary_plot)
img_to_clipboard(fi1_ts.summary_plot)
img_to_clipboard(fi1_el.summary_plot)









## Transfer Learning ############################################################
dr_7cgGI = DNN_Regressor(batch[0].shape[-1], batch[1].shape[-1], output_dim=batch[2].shape[-1])
dr_7cgGI.load_state_dict(dr_5cgGI_init.state_dict())


model = copy.deepcopy(dr_7cgGI)

# # [print(name) for name, parameter in model.named_parameters()]
# (parameter freezing) ---------------------------------
for name, parameter in model.named_parameters():
    # if name not in ['fc_modules.0.weight', 'fc_modules.0.bias']:
    # if name in ['fc_modules.7.weight', 'fc_modules.7.bias', 'fc_modules.9.weight', 'fc_modules.9.bias']:
    if name in ['fc_modules.9.weight', 'fc_modules.9.bias']:
        # print(name)
        parameter.requires_grad = True
    else:
        parameter.requires_grad = False

for name, parameter in model.named_parameters():
    print(name, parameter.requires_grad)



# # customize library ***---------------------
import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=30)
# # ------------------------------------------
import time


epochs = 30

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

torch.autograd.set_detect_anomaly(False)
# with torch.autograd.detect_anomaly():
#     input = torch.rand(5, 10, requires_grad=True)
#     output = function_A(input)
#     output.backward()


# training * -------------------------------------------------------------------------------------------------------
train_losses = []
valid_losses = []
for e in range(epochs):
    start_time = time.time() # 시작 시간 기록
    # train_set learning*
    model.train()
    train_epoch_loss = []
    for batch in train_loader_tr:
        batch = [batch_data.to(device) for batch_data in batch]
        
        optimizer.zero_grad()                   # wegiht initialize
        pred = model((batch[0], batch[1]))                   # predict
        loss = loss_function(pred, batch[2])     # loss
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
        for batch in valid_loader_tr:
            batch = [batch_data.to(device) for batch_data in batch]
            
            pred = model((batch[0], batch[1]))                   # predict
            loss = loss_function(pred, batch[2])     # loss
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

        # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(),
                                   verbose=2)
        if early_stop == 'break':
            break
        # ------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# customize library ***---------------------
es.plot     # early_stopping plot 
es.optimum[1]   


# 
dr_7cgGI.load_state_dict(es.optimum[2])


# estimate
est_7cgGI_yp1 = DL_Estimator2(dr_7cgGI, se_X, se_y, pred_position=0)
est_7cgGI_ts1 = DL_Estimator2(dr_7cgGI, se_X, se_y, pred_position=1)
est_7cgGI_el1 = DL_Estimator2(dr_7cgGI, se_X, se_y, pred_position=2)


# 44.17  # 35.38  # 1.57
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_YP'], model=est_7cgGI_yp1, model_type='regressor')
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_TS'], model=est_7cgGI_ts1, model_type='regressor')
ModelEvaluate(df_transfer.loc[test_idx_tr][X_cols], df_transfer.loc[test_idx_tr]['초_EL'], model=est_7cgGI_el1, model_type='regressor')



