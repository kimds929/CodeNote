import torch

################################################################
# (Normalization) 
ten = torch.arange(24).reshape(2,4,3).type(torch.float32)

bn = torch.nn.BatchNorm1d(4)    # 모든 Batch에 걸쳐, sequence-feature 단위로 norm
ln = torch.nn.LayerNorm(3)      # 각 Batch/Sequence내 feature 단위로 norm
In = torch.nn.InstanceNorm1d(2) # Batch내 모든 원소를 normaliza

bn(ten)
ln(ten)
In(ten)

bn(ten).mean(dim=[0,2])
ln(ten).mean(dim=[2])     # 각 Batch의 sequence별로
In(ten).mean(dim=[1,2])



################################################################
# (DropOut)
ten = torch.rand(10,3) 
dropout_layer = torch.nn.Dropout(0.5)
dropout_layer(ten)




################################################################


