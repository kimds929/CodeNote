## 【 Bayesian Deep Learning Regression 】##############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)



import httpimport
remote_library_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    from DS_DataFrame import dtypes_split, ScalerEncoder, DF_Summary, SummaryPlot
    
# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_MachineLearning import DataSet

################################################################################
import copy
class DataSet():
    """
    【required (Library)】copy, functools, numpy(np), pandas(pd), torch, tensorflow(tf)
    
    < Attribute >
      self.inputdata_info
      self.dataloader_info
      self.dataloader
      
    < Method >
      . self.Dataset
      . self.Split
      . self.Encoding (self.Decoding)
      . self.Batch
      . self.Reset_dataloader
      
    < Funtion >
      . self.make_data_info
      . self.dataloader_to_info
      . self.info_to_dataloader
      . self.data_transform_from_numpy
      . self.split_size
      . self.data_slicing
      . self.data_split
      . self.data_transform
      . self.make_batch
      
    
    """
    def __init__(self, X=None, y=None,  X_columns=None, y_columns=None, kwargs_columns={},
                type = ['pandas', 'numpy', 'tensorflow', 'torch'], set_type={},
                X_encoder=None, y_encoder=None, encoder={},
                shuffle=False, random_state=None, **kwargs):
        
        # all input value save
        local_values = locals().copy()
        
        dataloader_info = {}
        # dataloader = {}
        self.data_columns_couple = {None: None, 'X':X_columns, 'y': y_columns}
        
        for arg_name, arg_value in local_values.items():
            if arg_name not in ['self', 'random_state', 'kwargs']:
                exec(f"self.{arg_name} = arg_value")
                
                if arg_name in ['X', 'y']:
                    if arg_value is not None and arg_value.shape[0] > 0:
                        dataloader_info[arg_name] = {}
                        dataloader_info[arg_name]['data'] = self.make_data_info(arg_value, prefix=arg_name, columns=self.data_columns_couple[arg_name], return_data=True) if arg_value is not None else None
                    
        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_value is not None and kwarg_value.shape[0] > 0:
                kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
                dataloader_info[kwarg_name] = {}
                dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
        
        # self.local_values = local_values
        self.set_type = set_type
        # self.inputdata_info = dataloader_info
        
        initialize_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
        self.dataloader = initialize_object['dataloader']
        self.dataloader_info = initialize_object['data_info']
        self.inputdata_info = copy.deepcopy(self.dataloader_info)
                
        self.random_state = random_state
        self.define_random_generate(random_state)
        self.indices = {}
        self.length = {}
        
        self.dataloader_process = ''

    # [funcation] ---------------------------
    def define_random_generate(self, random_state=None):
        self.random_generate = np.random.RandomState(self.random_state) if random_state is None else np.random.RandomState(random_state)
        # self.random_generate = np.random.default_rng(random_state)

    def make_data_info(self, data, prefix=None, columns=None, iloc_index=None, loc_index=None, dtype=None, return_data=False):
        prefix = 'v' if prefix is None else prefix
        type_of_data = str(type(data))
        
        array_data = np.array(data)
        shape = array_data.shape
        n_dim = len(shape)
        
        # type
        result_of_type = None
        if 'pandas' in type_of_data:
            result_of_type = 'pandas'
        elif 'numpy' in type_of_data:
            result_of_type = 'numpy'
        elif 'tensorflow' in type_of_data:
            result_of_type = 'tensorflow'
            # result_of_type =  'tensorflow - variable' if 'variable' in type_of_data.lower() else 'tensorflow - constant'      
        elif 'torch' in type_of_data:
            result_of_type = 'torch'
        else:
            result_of_type = 'else'
        
        # dtype
        if dtype is None:
            if  result_of_type == 'pandas':
                dtype = data.dtype if n_dim == 1 else data.dtypes.to_dict()
            else:
                dtype = data.dtype
            
            
        # columns
        if columns is not None:
            set_columns = columns
        elif result_of_type == 'pandas':
            set_columns = str(data.name) if n_dim == 1 else np.array(data.columns)
        else:
            if n_dim == 1:
                set_columns = prefix
            elif n_dim > 1:
                col_base = np.array(range(shape[-1])).astype('str')
                set_columns = np.tile(col_base, np.array(shape[:-2]).prod()).reshape(*list(shape[:-2]), shape[-1]).astype('str')
                set_columns = np.char.add(prefix, set_columns)
                
        # index
        if iloc_index is None and loc_index is None:
            if result_of_type == 'pandas':
                loc_index = np.array(data.index)
                iloc_index = np.array(range(shape[0]))
            else:
                loc_index = np.array(range(shape[0]))
                iloc_index = np.array(range(shape[0]))
        else:
            if result_of_type == 'pandas':
                loc_index = np.array(data.index) if loc_index is None else np.array(loc_index)
                iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
            else:
                loc_index = (np.array(range(shape[0])) if iloc_index is None else iloc_index) if loc_index is None else np.array(loc_index)
                iloc_index = (np.array(range(shape[0])) if loc_index is None else loc_index) if iloc_index is None else np.array(iloc_index)
        
        if return_data:
            return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index, 'data': array_data}
        else:
            return {'type': result_of_type, 'ndim': n_dim, 'dtype': dtype, 'columns': set_columns, 'loc_index' :loc_index, 'iloc_index':iloc_index}

    def dataloader_to_info(self, dataloader, data_info=None, update_loader=False):
        return_data_info = {}
        for name, dataset in dataloader.items():
            return_data_info[name] = {}
            for dataset_name, data in dataset.items():
                if data_info is None:
                    return_data_info[name][dataset_name] = self.make_data_info(data=data)
                else:
                    info = data_info[name][dataset_name]
                    return_data_info[name][dataset_name] = self.make_data_info(data=data, columns=info['columns'], iloc_index=info['iloc_index'], loc_index=info['loc_index'], return_data=True)
        if update_loader is True:
            return_dataloader = self.info_to_dataloader(return_data_info, update_info=False)
            return {'dataloader': return_dataloader['dataloader'], 'data_info':return_data_info}
        else:
            return {'dataloader': dataloader, 'data_info':return_data_info}

    def info_to_dataloader(self, data_info, set_type={}, update_info=False):
        return_dataloader = {}
        self.set_type.update(set_type)
        
        for name, dataset in data_info.items():
            return_dataloader[name] = {}
            for dataset_name, info in dataset.items():
                if self.set_type is None:
                    apply_type = info['type']
                elif type(self.set_type) == str:
                    apply_type = self.set_type
                elif type(self.set_type) == dict:
                    apply_type = self.set_type[name] if name in self.set_type.keys() else info['type']
                return_dataloader[name][dataset_name] = self.data_transform_from_numpy(numpy_data=info['data'], set_type=apply_type,
                                                        index=info['loc_index'], columns=info['columns'], dtype=info['dtype'])
        if len(self.set_type) > 0 or update_info is True:
            return_data_info = self.dataloader_to_info(return_dataloader, data_info=data_info, update_loader=False)
            return {'dataloader': return_dataloader, 'data_info':return_data_info['data_info']}
        else:
            return {'dataloader':return_dataloader, 'data_info':data_info}

    def data_transform_from_numpy(self, numpy_data, set_type, index=None, columns=None, dtype=None):
        shape = numpy_data.shape
        ndim = len(shape)
        
        # dtype
        if dtype is not None:
            if type(dtype) == dict:     # pandas
                if set_type == 'pandas':
                    apply_dtype = dtype
                else:
                    apply_dtype = np.dtype('float32')
            elif 'torch' in str(dtype):     # torch
                apply_dtype = np.dtype(str(dtype).split('.')[-1])
            elif '<dtype:' in str(dtype):   # tensorflow
                apply_dtype = np.dtype(dtype.as_numpy_dtype)
            else:
                apply_dtype = dtype
        else:
            apply_dtype = dtype
                
        # set_type transform
        if set_type == 'numpy':
            return numpy_data
        elif set_type == 'pandas':
            if ndim == 1:
                return pd.Series(numpy_data, index=index, name=columns, dtype=apply_dtype)
            elif ndim == 2:
                if apply_dtype is None:
                    return pd.DataFrame(numpy_data, index=index, columns=columns)
                else:
                    if type(apply_dtype) != dict:
                        if columns is None:
                            apply_dtype = {c: apply_dtype for c in range(numpy_data.shape[1])}
                        else:
                            apply_dtype = {c: apply_dtype for c in columns}
                    return pd.DataFrame(numpy_data, index=index, columns=columns).astype(apply_dtype)
        elif set_type == 'tensorflow':
            return tf.constant(numpy_data, dtype=apply_dtype)
        elif set_type == 'torch':
            return torch.FloatTensor(numpy_data)
            # return torch.tensor(numpy_data, dtype=eval(f'torch.{str(apply_dtype)}'))

    def split_size(self, valid_size, test_size):
        self.valid_size = valid_size
        self.test_size = test_size
        
        self.train_size = 1 - test_size
        self.train_train_size = self.train_size - valid_size
        self.train_valid_size = valid_size

    def generate_index(self, data, data_info=None, valid_size=None, test_size=None, shuffle=True,
                      random_state=None):
        try:
            data_length = len(data)
        except:
            data_length = data.shape[0]
        if data_info is None:
            data_info = self.make_data_info(data=data)
            
        index = data_info['iloc_index']
        indices = {}

        # shuffle
        if shuffle is True:
            random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
            apply_index = random_generate.permutation(index)
        else:
            apply_index = index
        
        # split_size
        self.split_size(valid_size = 0.0 if valid_size is None else valid_size,
                        test_size = 0.3 if test_size is None else test_size)
        
        # train_valid_test split
        train_len = int(data_length * (1-self.test_size))
        train_train_len = int(train_len * (1-self.train_valid_size))
        train_valid_len = train_len - train_train_len
        test_len = data_length - train_len
        
        for k, v in zip(['data','train', 'train_train', 'train_valid', 'test'], [train_len, train_train_len, train_valid_len, test_len]):
            if v > 0:
                self.length[k] = v
        
        # save
        indices['all_index'] = index
        indices['apply_index'] = apply_index
        indices['train_index'] = apply_index[:train_len]
        indices['train_train_index'] = indices['train_index'][:train_train_len]
        if train_valid_len > 0:
            indices['train_valid_index'] = indices['train_index'][train_train_len:]
        indices['test_index'] = apply_index[train_len:]
        
        return indices

    def data_slicing(self, data=None, apply_index=None, data_info=None, index_type='iloc'):
        # data_info
        if data_info is None:
            data_info = self.make_data_info(data=data, return_data=True)

        numpy_data = data_info['data']
        
        if index_type == 'iloc':
            index_series = pd.Series(data_info['loc_index'], index=range(len(numpy_data)) )
            iloc_index = np.array(index_series[apply_index].index) 
            loc_index = np.array(index_series[apply_index].values)
        elif index_type == 'loc':
            index_series = pd.Series(range(len(numpy_data)), index=data_info['loc_index'])
            iloc_index = np.array(index_series[apply_index].values) 
            loc_index = np.array(index_series[apply_index].index)
        
        # slicing
        sliced_data = np.take(numpy_data, iloc_index, axis=0)
        # sliced_index = np.take(data_index, apply_index, axis=0)
        return self.data_transform_from_numpy(numpy_data=sliced_data, index=loc_index, 
                                            set_type=data_info['type'], columns=data_info['columns'], dtype=data_info['dtype'])       

    def data_split(self, data, data_info=None, indices=None, valid_size=0, test_size=0, shuffle=True,
                   index_type='iloc', random_state=None, verbose=0):
        # data_info
        if data_info is None:
            data_info = self.make_data_info(data=data, return_data=True)
            
        if indices is None:
            if len(self.indices) == 0:
                indices = self.generate_index(data=data, valid_size=valid_size, test_size=test_size, 
                                              shuffle=shuffle, random_state=random_state)
            else:
                indices = self.indices
        
        split_data = {}
        for key, index in indices.items():
            split_data[key] = self.data_slicing(data=data, apply_index=index, data_info=data_info, index_type=index_type)
            if verbose > 0:
                print(f"{key}: {split_data[key].shape}")
            
        return {'split_data': split_data, 'indices':indices}

    def data_transform(self, data, data_info=None, encoder=None, type='encoding'):
        if encoder is None:
            return data
        else:
            try:
                if 'enco' in type:
                    return encoder.transform(data)
                elif 'deco' in type:
                    return encoder.inverse_transform(data)
            except:
                # data_info
                if data_info is None:
                    data_info = self.make_data_info(data=data, return_data=True)
                if 'enco' in type:
                    transformed_np_data = encoder.transform(data_info['data'])
                elif 'deco' in type:
                    transformed_np_data = encoder.inverse_transform(data_info['data'])
                transformed_data = self.data_transform_from_numpy(numpy_data=transformed_np_data, 
                                               set_type=data_info['type'], columns=data_info['columns'], dtype=None)
                return transformed_data

    def make_batch(self, data, batch_size=None):
        try:
            data_length = len(data)
        except:
            data_length = data.shape[0]
        
        batch_size = data_length if batch_size is None or batch_size <= 0 or batch_size > data_length else batch_size
        batch_index = 0
        batch = []
        while True:
            if batch_index + batch_size >= data_length:
                batch.append(data[batch_index:])
                batch_index = data_length
                # batch = np.array(batch)
                break
            else:
                batch.append(data[batch_index:batch_index + batch_size])
                batch_index = batch_index + batch_size
        return batch

    # [method] ---------------------------
    def Reset_dataloader(self):
        self.dataloader_info = copy.deepcopy(self.inputdata_info)
        self.dataloader = self.info_to_dataloader(self.dataloader_info)['dataloader']
        self.dataloader_process = ''

    def Dataset(self, kwargs_columns={}, set_type={}, **kwargs):
        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_value is not None and kwarg_value.shape[0] > 0:
                kwargs_column = kwargs_columns[kwarg_name] if kwarg_name in kwargs_columns.keys() else None
                self.dataloader_info[kwarg_name] = {}
                self.dataloader_info[kwarg_name]['data'] = self.make_data_info(kwarg_value, prefix=kwarg_name, columns=kwargs_column, return_data=True) if kwarg_value is not None else None
                self.inputdata_info[kwarg_name] = {}
                self.inputdata_info[kwarg_name] = self.dataloader_info[kwarg_name]['data']
        # set dataset / data_info
        update_object = self.info_to_dataloader(dataloader_info, set_type=set_type)
        self.dataloader = self.info_to_dataloader(self.dataloader_info, set_type=set_type)['dataloader']
        self.dataloader_info = update_object['data_info']
        self.inputdata_info = copy.deepcopy(self.dataloader_info)
        return self

    def Encoding(self, X_encoder=None, y_encoder=None, encoder={}):
        # encoder dictionary
        self.encoder.update(encoder)
        X_encoder = self.X_encoder if X_encoder is None else X_encoder
        y_encoder = self.y_encoder if y_encoder is None else y_encoder
        if X_encoder is not None:
            encoder['X'] = X_encoder
        if y_encoder is not None:
            encoder['y'] = y_encoder
        
        # encoding
        encoder_keys = encoder.keys()
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                if name in encoder_keys and encoder[name] is not None:
                    data_info = self.dataloader_info[name][dataset_name]
                    encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='encoding')
                    self.dataloader[name][dataset_name] = encodered_data
                    self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
                                                                                    iloc_index= data_info['iloc_index'],
                                                                                    return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Encoding'

    def Decoding(self, X_encoder=None, y_encoder=None, encoder={}):
        # encoder dictionary
        self.encoder.update(encoder)
        X_encoder = self.X_encoder if X_encoder is None else X_encoder
        y_encoder = self.y_encoder if y_encoder is None else y_encoder
        if X_encoder is not None:
            encoder['X'] = X_encoder
        if y_encoder is not None:
            encoder['y'] = y_encoder
        
        # decoding
        encoder_keys = encoder.keys()
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                if name in encoder_keys and encoder[name] is not None:
                    data_info = self.dataloader_info[name][dataset_name]
                    encodered_data = self.data_transform(data=data, data_info=data_info, encoder=encoder[name], type='decoding')
                    self.dataloader[name][dataset_name] = encodered_data
                    self.dataloader_info[name][dataset_name] = self.make_data_info(data=encodered_data, columns=data_info['columns'], 
                                                                                    iloc_index= data_info['iloc_index'],
                                                                                    dtype=self.inputdata_info[name]['data']['dtype'],
                                                                                    return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Decoding'

    def Split(self, valid_size=0, test_size=0.3, shuffle=True, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        
        # train_valid_test_split
        for name, data_dict in self.dataloader.items():
            data_info = self.dataloader_info[name]['data']
            splited_dict = self.data_split(data=data_dict['data'], data_info=data_info,
                            valid_size=valid_size, test_size=test_size, shuffle=shuffle, 
                            index_type='iloc', random_state=self.random_state)
            del splited_dict['split_data']['all_index']
            del splited_dict['split_data']['apply_index']
            del splited_dict['indices']['all_index']
            del splited_dict['indices']['apply_index']
            
            for dataset_name, index in splited_dict['indices'].items():
                data = splited_dict['split_data'][dataset_name]
                set_name = dataset_name.replace('index','set')
                self.dataloader[name][set_name] = data
                self.dataloader_info[name][set_name] = self.make_data_info(data=data,
                                                                   columns=data_info['columns'],
                                                                   iloc_index=index,
                                                                   loc_index=np.take(data_info['loc_index'], index),
                                                                   dtype=data_info['dtype'],
                                                                   return_data=True)
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + 'Split'

    def Batch(self, batch_size=None, shuffle=True, random_state=None):
        if shuffle is True:
            random_generate = self.random_generate if random_state is None else np.random.RandomState(random_state)
        
        # transform to batch
        for name, data_dict in self.dataloader.items():
            for dataset_name, data in data_dict.items():
                batch_data = []
                data_info = self.dataloader_info[name][dataset_name]
                dataset_index = data_info['loc_index']
                if shuffle is True:
                    dataset_index = random_generate.permutation(dataset_index)
                batch_indices = self.make_batch(dataset_index, batch_size=batch_size)
                
                for batch_index in batch_indices:
                    batch_sliced_data = self.data_slicing(data, data_info=data_info, apply_index=batch_index, index_type='loc')
                    batch_data.append( batch_sliced_data )
                
                # update dataloader
                self.dataloader[name][dataset_name] = batch_data
                
                # update dataloader_info
                self.dataloader_info[name][dataset_name]['batch_iloc_index'] = batch_indices
        # __repr__
        self.dataloader_process = self.dataloader_process + ('\n . ' if self.dataloader_process == '' else ' > ') + f'Batch({batch_size})'

    def __repr__(self):
        data_set_names = ', '.join(list(self.dataloader.keys()))
        return f'<class: DataLoader Object ({data_set_names})>{self.dataloader_process}'    
#################################################################################
# from sklearn.datasets import load_boston
# load_skdata = load_boston()

# data_boston = pd.DataFrame(load_skdata['data'], columns=load_skdata.feature_names)
# data_boston['Target']=load_skdata['target']

data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/Data_Tabular/'
# data_boston = pd.read_csv(data_url + "boston_house.csv")
data_titanic = pd.read_csv(data_url + "titanic.csv")


DF_Summary(data_titanic)
data_titanic
# data = data_boston.drop('CHAS', axis=1)

data = data_titanic.copy()

# Preprocessing ########################################################################
data =  data.drop('age', axis=1).dropna()
data['pclass'] = data['pclass'].astype('str')
data['survived'] = data['survived'].astype('str')

df_summary = DF_Summary(data)
df_summary.summary
df_summary.summary_plot()


# #####################################################################################
# train test split
from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(range(len(data)), test_size=0.3)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2)

y_columns = ['survived']
X_columns = [c for c in data.columns if c not in y_columns]

train_X = data.iloc[train_idx, :][X_columns]
train_y = data.iloc[train_idx, :][y_columns]
valid_X = data.iloc[valid_idx, :][X_columns]
valid_y = data.iloc[valid_idx, :][y_columns]
test_X = data.iloc[test_idx, :][X_columns]
test_y = data.iloc[test_idx, :][y_columns]


# #####################################################################################
# train test split
data_dtypes = dtypes_split(data, return_type='columns_list')
X_numeric_columns = [c for c in data_dtypes['numeric'] if c not in y_columns]
X_object_columns = [c for c in data_dtypes['object'] if c not in y_columns]
X_object_categories = {k: int(v.split(' ')[0]) for k,v in df_summary.summary['info'][X_object_columns].items()}
# X_object_categories




# Normalize
se_X = ScalerEncoder(encoder= {'#numeric':'StandardScaler', '#object':'LabelEncoder', '#time': 'StandardScaler'})
train_X_normalized = se_X.fit_transform(train_X)
valid_X_normalized = se_X.transform(valid_X)
test_X_normalized = se_X.transform(test_X)

se_y = ScalerEncoder(encoder= {'#numeric':'StandardScaler', '#object':'LabelEncoder', '#time': 'StandardScaler'})
train_y_normalized = se_y.fit_transform(train_y)
valid_y_normalized = se_y.transform(valid_y)
test_y_normalized = se_y.transform(test_y)


# #####################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# train test split
train_X_tensor_numeric = torch.FloatTensor(train_X_normalized[X_numeric_columns].to_numpy())
train_X_tensor_object = torch.IntTensor(train_X_normalized[X_object_columns].to_numpy())

torch.rand(4).unsqueeze(0).expand(5,3,4,-1)




########################################################################################
train_X_tensor_numeric
# Feature 자체의 고유한 특성을 학습하기 위한 embedding 생성 (Feature값에 independent)
class EmbeddingFeature_Layer(nn.Module):
    def __init__(self, embedding_dim: int, max_embedding: int = 100):
        super().__init__()
        self.feature_embedding_layer = nn.Embedding(num_embeddings=max_embedding, embedding_dim=embedding_dim)

    def forward(self, x, x_index=None):
        # x.shape (B, T)
        # B: batch, T: seq_len,  D:embedding_dim
        *batch_shape, T = x.shape

        # feature index: (T,)
        if x_index is None:
            x_index = torch.arange(T, device=x.device)
        else:
            x_index = x_index.to(x.device)

        # feature embedding: (T, D)
        feature_embed = self.feature_embedding_layer(x_index)

        # reshape for broadcast: (1,...,1,T,D)
        expand_shape = [1] * len(batch_shape) + [T, feature_embed.shape[-1]]
        feature_embed = feature_embed.view(*expand_shape)  # (1,...,1,T,D)

        # expand to match batch: (..., T, D)
        output_embedding_feature = feature_embed.expand(*batch_shape, T, feature_embed.shape[-1])

        return output_embedding_feature  # shape: (..., T, D)

# efl = EmbeddingFeature_Layer(5)
# efl(torch.rand(10,5,3,2)).shape

# Numeric Feature마다 독립적인 Linear 표현력(embedding)을 부여 (Feature값에 dependent)
class EmbeddingFC_Layer(nn.Module):
    def __init__(self, embedding_dim: int, max_embedding: int = 100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight_embedding_layer = nn.Embedding(max_embedding, embedding_dim)
        self.bias_embedding_layer = nn.Embedding(max_embedding, embedding_dim)
        nn.init.xavier_uniform_(self.weight_embedding_layer.weight)
        nn.init.zeros_(self.bias_embedding_layer.weight)

    def forward(self, x, x_index=None):
        # x.shape (B, T)
        # B: batch, T: seq_len,  D:embedding_dim
        *batch_shape, T = x.shape

        # feature index: (T,)
        if x_index is None:
            x_index = torch.arange(T, device=x.device)
        else:
            x_index = x_index.to(x.device)

        # (T, D)
        weight = self.weight_embedding_layer(x_index)
        bias = self.bias_embedding_layer(x_index)

        # reshape for broadcasting: (1,...,1,T,D)
        expand_shape = [1] * len(batch_shape) + [T, self.embedding_dim]
        weight = weight.view(*expand_shape)  # (1,...,1,T,D)
        bias = bias.view(*expand_shape)      # (1,...,1,T,D)
        
        # embedding fc features
        output_embedding_fc = x.unsqueeze(-1) * weight + bias  # (..., T, D)

        return output_embedding_fc  # (..., T, D)

# efl = EmbeddingFC_Layer(4)
# efl(torch.rand(10,5,3,2)).shape

class NumericCombineEmbedding(nn.Module):
    def __init__(self, embedding_dim=6, max_embedding: int = 100, permute_groups=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.permute_groups = permute_groups
        self.feature_embedding = EmbeddingFeature_Layer(max_embedding=max_embedding, embedding_dim=embedding_dim)
        self.fc_embedding = EmbeddingFC_Layer(max_embedding=max_embedding, embedding_dim=embedding_dim)
        self.sine_embedding = EmbeddingFC_Layer(max_embedding=max_embedding, embedding_dim=embedding_dim)
    
    def forward(self, x):
        x_shape = x.shape
        x1 = self.feature_embedding(x)      # (B,T,D)
        x2 = self.fc_embedding(x)       # (B,T,D)
        x3 = torch.sin(self.sine_embedding(x))  # (B,T,D)
        x_embed = torch.cat([x1, x2, x3], dim=-1)     # (B,T,3D)
        
        if self.permute_groups is None:
            return x_embed
        else:
            # 각 embedding dimesion을 골고루 분배
            assert x_embed.shape[-1] % self.permute_groups == 0, "D must be divisible by num_groups"
            n_elements = x_embed.shape[-1] // self.permute_groups     # 각 group내 element수
            x_embed_out = x_embed.view(*x_shape, n_elements, self.permute_groups).transpose(-1,-2).contiguous().view(*x_shape, -1)
            return x_embed_out

# ce = NumericCombineEmbedding(permute_groups=6)
# ce(torch.rand(5,3))
# ce(train_X_tensor_numeric)



########################################################################################
train_X_tensor_object

# Categorical class마다 embedding부여
class CategoricalEmbedding(nn.Module):
    def __init__(self, embedding_dim=4, max_embeddings=[100, 100, 100]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_embedding, embedding_dim)
            for max_embedding in max_embeddings
        ])
        
    def forward(self, x):
        # x.shape: (B, T)  where T = num of categorical features
        cat_embedding_outputs = [embed(x[:, i]).unsqueeze(-2) for i, embed in enumerate(self.embeddings)]     # [(B,T,1),...]
        print(len(cat_embedding_outputs))
        cat_embedding_output = torch.cat(cat_embedding_outputs, dim=-2)  # (B,T,D)
        return cat_embedding_output
        

ce = CategoricalEmbedding(max_embeddings=list(X_object_categories.values()))
ce(train_X_tensor_object).shape   # (B)




class CategoricalCombineEmbedding(nn.Module):
    def __init__(self, cat_embed_dim=4, feature_embed_dim=2, max_cat_embeddings=[100, 100, 100], max_feature_embedding=100):
        super().__init__()
        self.cat_embedding_layer = CategoricalEmbedding(embedding_dim=cat_embed_dim, max_embeddings=max_cat_embeddings)
        self.cat_feature_embedding_layer = EmbeddingFeature_Layer(embedding_dim=feature_embed_dim, max_embedding=max_feature_embedding)
        

    def forward(self, x):
        cat_embedding_output = self.cat_embedding_layer(x)
        embedding_feature_output = self.cat_feature_embedding_layer(x)
        categorical_embedding_output = torch.cat([cat_embedding_output, embedding_feature_output], dim=-1)
        return categorical_embedding_output

cce = CategoricalCombineEmbedding(max_cat_embeddings=list(X_object_categories.values()))
cce(train_X_tensor_object).shape




########################################################################################
class UnifiedInputEmbedding(nn.Module):
    def __init__(
        self,
        numeric_embed_dim=2,
        cat_embed_dim=4,
        feature_embed_dim=2,
        max_feature_embedding=100,
        max_cat_embeddings=[100, 100, 100],
        permute_groups=None
    ):
        super().__init__()
        self.numeric_embed = NumericCombineEmbedding(
            embedding_dim=numeric_embed_dim,
            max_embedding=max_feature_embedding,
            permute_groups=permute_groups
        )
        self.categorical_embed = CategoricalCombineEmbedding(
            cat_embed_dim=cat_embed_dim,
            feature_embed_dim=feature_embed_dim,
            max_cat_embeddings=max_cat_embeddings,
            max_feature_embedding=max_feature_embedding
        )
    
    def forward(self, x_numeric, x_categorical):
        # x_numeric.shape: (B, T_num), x_categorical.shape: (B, T_cat)
        n_out = self.numeric_embed(x_numeric)        # (B, T_num, 3 * numeric_embed_dim)
        c_out = self.categorical_embed(x_categorical)  # (B, T_cat, cat_embed_dim + feature_embed_dim)
        print(n_out.shape, c_out.shape)
        return torch.cat([n_out, c_out], dim=-1)  # concat along sequence (T) axis → (B, T_total, D)

# uie = UnifiedInputEmbedding()
# uie(train_X_tensor_numeric, train_X_tensor_object).shape
# train_X_tensor_numeric.shape
# train_X_tensor_object.shape


# ---------------------------------------------------------------------------
# UnifiedInputEmbedding() -> fc_mixture(linear) -> LayerNorm -> Multi-Head Attention
# self.mix = nn.Sequential(
#     nn.Linear(D_total, D_total),
#     nn.GELU(),
#     nn.Dropout(0.1),
#     nn.LayerNorm(D_total)
# )


# class MixtureAndMHA_Layer(nn.Module):
#     def __init__(self, input_dim, proj_dim=64, dropout=0.1):
#         super().__init__()
#         self.mix = nn.Sequential(
#             nn.Linear(input_dim, proj_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LayerNorm(proj_dim)
#         )
#         self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, batch_first=True)
    
#     def forward(self, x):  # x: (B, T, D_input)
#         x_proj = self.mix(x)  # (B, T, D_proj)
#         x_attn, _ = self.attn(x_proj, x_proj, x_proj)  # (B, T, D_proj)
#         return x_attn





















####################################################################################################
# (참고 : 어떤 embedding에 집중할지?) #################################################################
# Squeeze-and-Excitation Networks (CVPR 2018)
# Squeeze-and-Excitation (SE)  : 채널 어텐션 메커니즘

# class SEAttention(nn.Module):
#     def __init__(self, dim, reduction=4):
#         super().__init__()
#         self.se_weight_layer = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),    # Squeeze : (B,D,T) → (B,D,1)
#             nn.Flatten(),               # (B,D,1) → (B,D)
#             nn.Linear(dim, dim//reduction),     # Excitation : (B,D) → (B,dim//reduction)
#             nn.ReLU(),  
#             nn.Linear(dim//reduction, dim),     # Channel-Recalibration : (B,dim//reduction) → (B,D)
#             nn.Sigmoid()                # Probability
#         )
#     def forward(self, x):
#         # x: (B, T, D)
#         weights = self.se_weight_layer(x.transpose(-1,-2))  # (B,D)
#         weights = weights.unsqueeze(-2)  # (B,D) → (B,1,D)
#         return x * weights  # 채널별 강화/억제
