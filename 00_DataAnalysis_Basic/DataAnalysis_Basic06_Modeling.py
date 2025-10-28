

###########################################################################################
###########################################################################################
# 【 예측모델 preprocessing 】 ###################################################
import sys
sys.path.append(r'D:\WorkForPython\00_DataAnalysis_Basic')
sys.path.append(r'D:\WorkForPython\DS_Library')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DS_Basic_Module import search, DF_Summary


    


###########################################################################################
columns_order = ['강종_소구분', '품종명', '소둔작업완료일시', '규격약호', '출강목표N', '재질시험_대표구분']
columns_size =  ['주문두께', '소둔_폭']
columns_target = ['YP','TS', 'EL', 'BMB']
columns_criteria = ['YP_보증범위', 'TS_보증범위', 'EL_보증범위']
columns_criteria_cond = ['인장_방향', '인장_폭방향', '인장_호수', 'BMB_방향', 'BMB_폭방향']
columns_craim = ['소둔_HS목표온도', '소둔_SS목표온도', '소둔_SCS목표온도', '소둔_RCS목표온도', 
           '소둔_RHS목표온도', '소둔_OAS목표온도', '소둔_SPM_EL목표']
columns_cr = ['LS_POS', 'HS_POS', 'SS_POS', 'SCS_POS', 'RCS_POS', 'RHS_POS','OAS_POS', 'FCS_POS', 'SPM_RollForce_ST1', 'SPM_RollForce_ST2']
columns_hr = ['SRT', 'FDT', 'CT']
columns_alloy = ['C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적', 'SolAl_실적', 'TotAl_실적', 'Cu_실적', 'Nb_실적',
                'B_실적', 'Ni_실적', 'Cr_실적', 'Mo_실적', 'Ti_실적', 'V_실적', 'Sn_실적', 'Ca_실적', 'Sb_실적', 'N_실적']
columns_fac = ['PCM공장', '소둔공장', '열연공장', '제강공장']



###############################################################################################################
# 【 YP Regression 】##########################################################################################
# 파일 불러오기
file_path = r'D:\WorkForPython\DB\Data_Education'
df04 = pd.read_csv(f"{file_path}/SampleData_980DP_YS_Modeling.csv", encoding='utf-8-sig')

# Column 선택
modeling_columns = columns_size + columns_target + columns_cr + columns_hr + columns_alloy
modeling_columns_final = [c for c in df04.columns if c in modeling_columns]
df05 = df04[modeling_columns_final]
df05.head(3).T

df_summary = DF_Summary(df05)
df_summary.summary_plot(on=['SPM_RollForce_ST1', 'SPM_RollForce_ST2'])


# X, y column
X = df05.drop(columns_target + ['SPM_RollForce_ST1'], axis=1)
y = df05[['YP']]



# train-test split
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(df05.index, test_size=0.2)
train_X = X.loc[train_idx]
train_y = y.loc[train_idx]
test_X = X.loc[test_idx]
test_y = y.loc[test_idx]


# normalize
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_X_norm = ss.fit_transform(train_X)
test_X_norm = ss.transform(test_X)





###############################################################################################################
# 【 모델링 (ML) 】 ###################################################
from sklearn.ensemble import RandomForestRegressor

RF_reg = RandomForestRegressor()
RF_reg.fit(train_X_norm, train_y)


from sklearn.metrics import mean_squared_error
# train_set evaluate
pred_y_train = RF_reg.predict(train_X_norm)
mean_squared_error(train_y, pred_y_train)    # MSE
np.sqrt(mean_squared_error(train_y, pred_y_train))             # RMSE

# test_set evaluate
pred_y_test = RF_reg.predict(test_X_norm)
mean_squared_error(test_y, pred_y_test)  # MSE
np.sqrt(mean_squared_error(test_y, pred_y_test))             # RMSE

# 변수중요도
plt.figure(figsize=(5,8))
plt.barh(X.columns[::-1], RF_reg.feature_importances_[::-1])
plt.show()


# 【 설명가능 인공지능 PartialDependence 】 ###################################################
from DS_Basic_Module import FeatureInfluence

fi = FeatureInfluence(train_X=train_X, estimator=RF_reg, encoderX=ss, y_name='YP')
fi.influence_summary(n_points=20)
fi.feature_importances_plot
fi.summary_table
fi.summary_plot


fi.plot_element(x='SS_POS',n_points=20)
fi.plot_element(x='소둔_폭',n_points=20)
fi.plot_element(x='SS_POS', x2='소둔_폭', n_points=20)





# 【 설명가능 인공지능 SHAP - ML 】 ###################################################
import shap 
shap.initjs()

explainer = shap.TreeExplainer(RF_reg)

# data_select -----------------------------------------------------
# train_data
shap_data = train_X_norm

# sampled train_data
sample_idx = sorted(np.random.choice(np.arange(len(train_X_norm)), size=100, replace=False))
shap_data = train_X_norm[sample_idx]

# test_data
pd.DataFrame(ss.inverse_transform(test_X_norm), columns=test_X.columns)
shap_data = test_X_norm[[1]]

# -----------------------------------------------------------------
# shap value
shap_values = explainer.shap_values(shap_data)
shap_values

# Summary Plot
shap.summary_plot(shap_values, shap_data, feature_names=train_X.columns)
shap.summary_plot(shap_values, shap_data, plot_type='bar', feature_names=train_X.columns)

# plt.figure(figsize=(3,6))
# plt.barh(train_X.columns[::-1], np.abs(shap_values).mean(axis=0)[::-1])

# Force Plot : 개별 예측에 대한 "기여도"를 직관적으로 보여줌
shap.force_plot(explainer.expected_value, shap_values, ss.inverse_transform(shap_data), feature_names=train_X.columns)



# Force Plot each data
shap.force_plot(explainer.expected_value, 
                shap_values[1], 
                ss.inverse_transform(shap_data)[1], 
                feature_names=train_X.columns)



# Dependence Plot
col_idx = np.where(train_X.columns=='SS_POS')[0].item()
shap.dependence_plot(ind=col_idx,
                    shap_values=shap_values,
                    features=ss.inverse_transform(shap_data),
                    feature_names=list(train_X.columns),
                    interaction_index=None,
                    )



# Decision Plot : 모델이 예측값을 만들어내는 과정을 각 feature의 기여(=SHAP 값) 관점에서 누적적으로 시각화해주는 그래프
#               예측값이 feature가 하나씩 추가될 때 어떻게 누적적으로 변화하는지 보여줌
fig= plt.figure(figsize=(5,12))
shap.decision_plot(
    explainer.expected_value,   # base value (모델의 평균 예측값)
    shap_values[1],                # SHAP 값 (n_samples, n_features)
    ss.inverse_transform(shap_data)[1],                          # feature 값 (DataFrame 또는 array)
    feature_names=list(train_X.columns),    # (선택) feature 이름
    # highlight=0                 # (선택) 0번째 샘플을 강조
    feature_order= np.arange(len(train_X.columns))[::-1],
    feature_display_range = range(len(train_X.columns))[::-1],       # 전체 feature보기
    auto_size_plot=False,
    # show=False
)


# SHAP value 직접 활용
# pd.Series(shap_values[1], index=train_X.columns).plot.barh()


# Partial Dependence Plot
col_idx = np.where(train_X.columns=='SS_POS')[0].item()

shap.partial_dependence_plot(
    ind=col_idx,                # feature 인덱스 또는 이름
    model=RF_reg.predict,
    data=shap_data,                    # 입력 데이터 (numpy array 또는 DataFrame)
    feature_names=train_X.columns,   # feature 이름 리스트 (옵션)
    model_expected_value=True,      # 모델의 base value 표시 여부
    feature_expected_value=True,     # feature의 평균값 표시 여부
    # shap_values=shap_values
    show=False
)
# 현재 x축 tick 값 가져오기
ax = plt.gca()
xticks = ax.get_xticks()

# scaling된 tick을 원래 값으로 역변환
xticks_original = xticks * np.sqrt(ss.var_[col_idx]) + ss.mean_[col_idx]

# x축 tick label을 원래 값으로 변경
ax.set_xticklabels([f"{x:.2f}" for x in xticks_original])
plt.xlabel(f"{train_X.columns[col_idx]} (원래 값)")
plt.show()









# 【 DeepLearning 】 ###################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from tqdm.auto import tqdm
from tqdm.notebook import tqdm
# from tqdm import trange

# Dataset & DataLoader
train_x_norm_torch = torch.FloatTensor(np.array(train_X_norm))
test_x_norm_torch = torch.FloatTensor(np.array(test_X_norm))

train_y_torch = torch.FloatTensor(np.array(train_y))
test_y_torch = torch.FloatTensor(np.array(test_y))

train_dataset = TensorDataset(train_x_norm_torch, train_y_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
class DL_Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        return x3


mdl_reg = DL_Regressor(input_dim=32).to(device)
# mdl_reg(torch.rand(10,32))

optimizer = optim.Adam(mdl_reg.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

mdl_reg.train()
N_EPOCHS = 300

loss_history = []
epoch_iter = tqdm(range(N_EPOCHS), desc="Epochs", total=N_EPOCHS)
for epoch in epoch_iter:
    batch_loss = 0.0
    for batch in train_loader:
        batch_X, batch_y = map(lambda x: x.to(device), batch)
        
        pred_y = mdl_reg(batch_X)
        loss = loss_function(pred_y, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
    epoch_loss = batch_loss / len(train_loader)
    loss_history.append(epoch_loss)
    epoch_iter.set_postfix(mse_loss=epoch_loss, rmse_loss=np.sqrt(epoch_loss))


# loss histogram
plt.plot(loss_history, 'o-', alpha=0.5)
plt.yscale('symlog')
plt.show()

with torch.no_grad():
    mdl_reg.eval()
    pred_y_test_torch = mdl_reg(test_x_norm_torch)
    test_mse = loss_function(pred_y_test_torch, test_y_torch).to('cpu').detach().numpy()
    test_rmse = np.sqrt(test_mse)
    print(f"test_mse : {test_mse:.3f}, test_rmse : {test_rmse:.3f}")






# 【 설명가능 인공지능 PartialDependence - torch 】 ###################################################
from DS_Basic_Module import FeatureInfluence
# from DS_DeepLearning import PredictDL

mdl_torch = PredictDL(model=mdl_reg)
# mdl_torch.predict(train_X_norm)

fi = FeatureInfluence(train_X=train_X, estimator=mdl_torch, encoderX=ss, y_name='YP')
fi.influence_summary(n_points=20)
fi.feature_importances_plot
fi.summary_table
fi.summary_plot


fi.plot_element(x='SS_POS',n_points=20)
fi.plot_element(x='소둔_폭',n_points=20)
fi.plot_element(x='SS_POS', x2='소둔_폭', n_points=20)



# 【 설명가능 인공지능 SHAP - Torch 】 ###################################################
import shap 
shap.initjs()

# 1. 모델을 평가 모드로 전환
mdl_reg.eval()

# 2. 샘플 데이터 준비 (설명에 사용할 데이터)
# sample_X, sample_y = map(lambda x: x.to(device), next(iter(train_loader)))
sample_idx = sorted(np.random.choice(range(len(train_x_norm_torch)), size=100, replace=False) )
sample_X = train_x_norm_torch[sample_idx].to(device)

# 3. SHAP explainer 생성
explainer = shap.DeepExplainer(mdl_reg, sample_X)

# 4. 설명할 데이터 준비 
shap_data = sample_X

# shap value
shap_values = explainer.shap_values(shap_data, check_additivity=False)
shap_values = shap_values.squeeze(-1)

# Summary Plot
shap.summary_plot(shap_values, shap_data, feature_names=train_X.columns)
shap.summary_plot(shap_values, shap_data, plot_type='bar', feature_names=train_X.columns)


# Force Plot
shap.force_plot(explainer.expected_value, shap_values, ss.inverse_transform(shap_data), feature_names=train_X.columns)

# Force Plot each data
shap.force_plot(explainer.expected_value, 
                shap_values[0], 
                ss.inverse_transform(shap_data)[0], 
                feature_names=train_X.columns)


# Dependence Plot
col_idx = np.where(train_X.columns=='SS_POS')[0].item()
shap.dependence_plot(ind=col_idx,
                    shap_values=shap_values,
                    features=ss.inverse_transform(shap_data),
                    feature_names=list(train_X.columns),
                    interaction_index=None,
                    )



# # Decision Plot : 모델이 예측값을 만들어내는 과정을 각 feature의 기여(=SHAP 값) 관점에서 누적적으로 시각화해주는 그래프
# #               예측값이 feature가 하나씩 추가될 때 어떻게 누적적으로 변화하는지 보여줌
# fig= plt.figure(figsize=(5,12))
# shap.decision_plot(
#     explainer.expected_value,   # base value (모델의 평균 예측값)
#     shap_values[1],                # SHAP 값 (n_samples, n_features)
#     ss.inverse_transform(shap_data)[1],                          # feature 값 (DataFrame 또는 array)
#     feature_names=list(train_X.columns),    # (선택) feature 이름
#     # highlight=0                 # (선택) 0번째 샘플을 강조
#     feature_order= np.arange(len(train_X.columns))[::-1],
#     feature_display_range = slice(None, None, -1),       # 전체 feature보기
#     auto_size_plot=False,
#     # show=False
# )



# # Partial Dependence Plot
# col_idx = np.where(train_X.columns=='SS_POS')[0].item()

# shap.partial_dependence_plot(
#     ind=col_idx,                # feature 인덱스 또는 이름
#     model=RF_reg.predict,
#     data=shap_data,                    # 입력 데이터 (numpy array 또는 DataFrame)
#     feature_names=train_X.columns,   # feature 이름 리스트 (옵션)
#     model_expected_value=True,      # 모델의 base value 표시 여부
#     feature_expected_value=True,     # feature의 평균값 표시 여부
#     # shap_values=shap_values
#     show=False
# )
# # 현재 x축 tick 값 가져오기
# ax = plt.gca()
# xticks = ax.get_xticks()

# # scaling된 tick을 원래 값으로 역변환
# xticks_original = xticks * np.sqrt(ss.var_[col_idx]) + ss.mean_[col_idx]

# # x축 tick label을 원래 값으로 변경
# ax.set_xticklabels([f"{x:.2f}" for x in xticks_original])
# plt.xlabel(f"{train_X.columns[col_idx]} (원래 값)")
# plt.show()
































###############################################################################################################
# 【 BMB Classification 】##########################################################################################
# 파일 불러오기
file_path = r'D:\WorkForPython\DB\Data_Education'
df04 = pd.read_csv(f"{file_path}/SampleData_980DP_EL_Modeling.csv", encoding='utf-8-sig')

df04['BMB'] = df04['BMB'].apply(lambda x: 0 if x == 1 else 1).astype(str)
# df04.groupby(['BMB_방향', 'BMB_폭방향']).size()

# Column 선택
modeling_columns = columns_size + columns_target + columns_cr + columns_hr + columns_alloy
modeling_columns_final = [c for c in df04.columns if c in modeling_columns]
df05 = df04[modeling_columns_final]
df05.head(3).T

df_summary = DF_Summary(df05)
# df_summary.summary_plot()
df_summary.summary_plot(on=['BMB'])


df05.to_clipboard(index=False)

# X, y column
X = df05.drop(columns_target, axis=1)
y = df05[['BMB']].astype(np.int64)



# train-test split
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(df05.index, test_size=0.2, stratify=df05['BMB'])

train_X = X.loc[train_idx]
train_y = y.loc[train_idx]
test_X = X.loc[test_idx]
test_y = y.loc[test_idx]

# train_y.value_counts()
# test_y.value_counts()

# train_y['BMB'].value_counts()
# test_y['BMB'].value_counts()

# normalize
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_X_norm = ss.fit_transform(train_X)
test_X_norm = ss.transform(test_X)



###############################################################################################################
# 【 모델링 (ML) 】 ###################################################
from sklearn.ensemble import RandomForestClassifier

RF_cls = RandomForestClassifier()
RF_cls.fit(train_X_norm, train_y)


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# train_set evaluate
pred_y_train = RF_cls.predict(train_X_norm)

accuracy_score(train_y, pred_y_train)    # accuracy
f1_score(train_y, pred_y_train)    # f1_score
confusion_matrix(pred_y_train, train_y) # confusion_matrix

# test_set evaluate
pred_y_test = RF_cls.predict(test_X_norm)
accuracy_score(test_y, pred_y_test)    # accuracy
f1_score(test_y, pred_y_test)    # f1_score
confusion_matrix(pred_y_test, test_y) # confusion_matrix

# 변수중요도
plt.figure(figsize=(5,8))
plt.barh(X.columns[::-1], RF_cls.feature_importances_[::-1])
plt.show()



# 【 설명가능 인공지능 SHAP - ML 】 ###################################################
import shap 
shap.initjs()

explainer = shap.TreeExplainer(RF_cls)

# data_select -----------------------------------------------------
# train_data
shap_data = train_X_norm

# sampled train_data
sample_idx = sorted(np.random.choice(np.arange(len(train_X_norm)), size=100, replace=False))
shap_data = train_X_norm[sample_idx]

# # test_data
# pd.DataFrame(ss.inverse_transform(test_X_norm), columns=test_X.columns)
# shap_data = test_X_norm[[1]]

# -----------------------------------------------------------------
# shap value
shap_values = explainer.shap_values(shap_data)[:,:,1]   # 불량원인분석


# Summary Plot
shap.summary_plot(shap_values, shap_data, feature_names=train_X.columns)
shap.summary_plot(shap_values, shap_data, plot_type='bar', feature_names=train_X.columns)


# Force Plot
shap.force_plot(explainer.expected_value[1], shap_values, ss.inverse_transform(shap_data), feature_names=train_X.columns)

# Force Plot each data
shap.force_plot(explainer.expected_value[1], 
                shap_values[0], 
                ss.inverse_transform(shap_data)[0], 
                feature_names=train_X.columns)



####################################################################################
df04.groupby(['주문두께','BMB']).size().unstack('BMB')
df04.groupby(['주문두께'])['LS_POS'].describe()

from DS_Basic_Module import group_plots
group_plots(data=df04, x='LS_POS', group='주문두께')
















####################################################################################
# (1.8t detail modeling)

# 파일 불러오기
file_path = r'D:\WorkForPython\DB\Data_Education'
df04 = pd.read_csv(f"{file_path}/SampleData_980DP_EL_Modeling.csv", encoding='utf-8-sig')

df04['BMB'] = df04['BMB'].apply(lambda x: 0 if x == 1 else 1).astype(str)
# df04.groupby(['BMB_방향', 'BMB_폭방향']).size()


# Column 선택
modeling_columns = columns_size + columns_target + columns_cr + columns_hr + columns_alloy
modeling_columns_final = [c for c in df04.columns if c in modeling_columns]
df05 = df04[modeling_columns_final]
df05.head(3).T

# -------------------------------------------------------------------------
# 1.8t filtering
df06 = df05[df05['주문두께'] == 1.8]

df_summary = DF_Summary(df06)
# df_summary.summary_plot()
df_summary.summary_plot(on=['BMB'])



# X, y column
X = df06.drop(columns_target +['주문두께'], axis=1)
y = df06[['BMB']].astype(np.int64)


# train-test split
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(df06.index, test_size=0.2, stratify=df06['BMB'])

train_X = X.loc[train_idx]
train_y = y.loc[train_idx]
test_X = X.loc[test_idx]
test_y = y.loc[test_idx]

# train_y['BMB'].value_counts()
# test_y['BMB'].value_counts()

# normalize
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_X_norm = ss.fit_transform(train_X)
test_X_norm = ss.transform(test_X)



###############################################################################################################
# 【 모델링 (ML) 】 ###################################################
from sklearn.ensemble import RandomForestClassifier

RF_cls = RandomForestClassifier()
RF_cls.fit(train_X_norm, train_y)


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# train_set evaluate
pred_y_train = RF_cls.predict(train_X_norm)

accuracy_score(train_y, pred_y_train)    # accuracy
f1_score(train_y, pred_y_train)    # f1_score
confusion_matrix(pred_y_train, train_y) # confusion_matrix

# test_set evaluate
pred_y_test = RF_cls.predict(test_X_norm)
accuracy_score(test_y, pred_y_test)    # accuracy
f1_score(test_y, pred_y_test)    # f1_score
confusion_matrix(pred_y_test, test_y) # confusion_matrix

# 변수중요도
plt.figure(figsize=(5,8))
plt.barh(X.columns[::-1], RF_cls.feature_importances_[::-1])
plt.show()



# 【 설명가능 인공지능 SHAP - ML 】 ###################################################
import shap 
shap.initjs()

explainer = shap.TreeExplainer(RF_cls)

# data_select -----------------------------------------------------
# train_data
shap_data = train_X_norm

# sampled train_data
sample_idx = sorted(np.random.choice(np.arange(len(train_X_norm)), size=100, replace=False))
shap_data = train_X_norm[sample_idx]

# # test_data
# pd.DataFrame(ss.inverse_transform(test_X_norm), columns=test_X.columns)
# shap_data = test_X_norm[[1]]

# -----------------------------------------------------------------
# shap value
shap_values = explainer.shap_values(shap_data)[:,:,1]   # 불량원인분석


# Summary Plot
shap.summary_plot(shap_values, shap_data, feature_names=train_X.columns)
shap.summary_plot(shap_values, shap_data, plot_type='bar', feature_names=train_X.columns)

# Force Plot
shap.force_plot(explainer.expected_value[1], shap_values, ss.inverse_transform(shap_data), feature_names=train_X.columns)

# Force Plot each data
shap.force_plot(explainer.expected_value[1], 
                shap_values[0], 
                ss.inverse_transform(shap_data)[0], 
                feature_names=train_X.columns)













# 【 DeepLearning 】 ###################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from tqdm.auto import tqdm
from tqdm.notebook import tqdm
# from tqdm import trange

# Dataset & DataLoader
train_x_norm_torch = torch.FloatTensor(np.array(train_X_norm))
test_x_norm_torch = torch.FloatTensor(np.array(test_X_norm))

train_y_torch = torch.LongTensor(np.array(train_y))
test_y_torch = torch.LongTensor(np.array(test_y))

train_dataset = TensorDataset(train_x_norm_torch, train_y_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
class DL_Clssifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.layer3 = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        return x3


mdl_cls = DL_Clssifier(input_dim=30).to(device)
# mdl_cls(torch.rand(10,30))

optimizer = optim.Adam(mdl_cls.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

mdl_cls.train()
N_EPOCHS = 300

loss_history = []
epoch_iter = tqdm(range(N_EPOCHS), desc="Epochs", total=N_EPOCHS)
for epoch in epoch_iter:
    batch_loss = 0.0
    batch_accuracy = 0.0
    batch_f1_score = 0.0
    for batch in train_loader:
        batch_X, batch_y = map(lambda x: x.to(device), batch)

        pred_y = mdl_cls(batch_X)
        loss = loss_function(pred_y, batch_y.ravel())
        pred_y.shape
        batch_y.shape
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        with torch.no_grad():
            batch_y_ravel = batch_y.to('cpu').detach().numpy().ravel()
            pred_y_prob_ravel = torch.argmax(torch.softmax(pred_y, dim=-1),dim=-1).to('cpu').detach().numpy()
            batch_accuracy += accuracy_score(batch_y_ravel, pred_y_prob_ravel)    # accuracy
            batch_f1_score += f1_score(batch_y_ravel, pred_y_prob_ravel)    # f1_score

    epoch_loss = batch_loss / len(train_loader)
    epoch_accuracy = batch_accuracy / len(train_loader)
    epoch_f1_score = batch_f1_score / len(train_loader)
    loss_history.append(epoch_loss)
    epoch_iter.set_postfix(cross_entropy_loss=epoch_loss, accuracy=epoch_accuracy, f1_score=epoch_f1_score)


# loss histogram
plt.plot(loss_history, 'o-', alpha=0.5)
plt.yscale('log')
plt.show()

with torch.no_grad():
    mdl_cls.eval()
    pred_y_test_torch = mdl_cls(test_x_norm_torch)

    test_y_ravel = test_y.to_numpy().ravel()
    test_pred_y_prob_ravel = torch.argmax(torch.softmax(pred_y_test_torch, dim=-1),dim=-1).to('cpu').detach().numpy()
    
    test_accuracy = accuracy_score(test_y_ravel, test_pred_y_prob_ravel)    # accuracy
    test_f1_score = f1_score(test_y_ravel, test_pred_y_prob_ravel)    # f1_score
    test_confusion_matrix = confusion_matrix(test_pred_y_prob_ravel, test_y_ravel)    # confusion_matrix
    
    print(f"test_accuracy : {test_accuracy:.3f}, test_f1_scroe : {test_f1_score:.3f}")
    print(test_confusion_matrix)





# 【 설명가능 인공지능 SHAP - Torch 】 ###################################################
import shap 
shap.initjs()

# 1. 모델을 평가 모드로 전환
mdl_cls.eval()

# 2. 샘플 데이터 준비 (설명에 사용할 데이터)
# sample_X, sample_y = map(lambda x: x.to(device), next(iter(train_loader)))
sample_idx = sorted(np.random.choice(range(len(train_x_norm_torch)), size=100, replace=False) )
sample_X = train_x_norm_torch[sample_idx].to(device)

# 3. SHAP explainer 생성
explainer = shap.DeepExplainer(mdl_cls, sample_X)

# 4. 설명할 데이터 준비 
shap_data = sample_X

# shap value
shap_values = explainer.shap_values(shap_data, check_additivity=False)[:,:,1]

# Summary Plot
shap.summary_plot(shap_values, shap_data, feature_names=train_X.columns)
shap.summary_plot(shap_values, shap_data, plot_type='bar', feature_names=train_X.columns)

# Force Plot
shap.force_plot(explainer.expected_value[1], shap_values, ss.inverse_transform(shap_data), feature_names=train_X.columns)

# Force Plot each data
shap.force_plot(explainer.expected_value[1], 
                shap_values[0], 
                ss.inverse_transform(shap_data)[0], 
                feature_names=train_X.columns)




