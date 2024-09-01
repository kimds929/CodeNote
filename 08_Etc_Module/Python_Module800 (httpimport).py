
import importlib
importlib.reload(httpimport)


import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'


with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    # import DF_DataFrame
    from DS_DataFrame import DF_Summary

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_Plot import ttest_each, violin_box_plot, distbox

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_MachineLearning import ScalerEncoder, DataSet

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_NLP import NLP_Preprocessor

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Temporal_12_TemporalEmbedding import PeriodicEmbedding

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Spatial_11_SpatialEmbedding import SpatialEmbedding


with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML



# import requests
# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/60_Graph_Neural_Network/GNN01_GenerateGraph.py")
# exec(response_files.text)
# response_files.text

# response_DS_DataFrame = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_DataFrame.py")
# response_DS_Plot = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Plot.py")
# response_DS_MachineLearning = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_MachineLearning.py")
# response_DS_DeepLearning = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_DeepLearning.py")
# response_DS_Torch = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Torch.py")

# exec(response_DS_DataFrame.text)
# exec(response_DS_Plot.text)
# exec(response_DS_MachineLearning.text)
# exec(response_DS_DeepLearning.text)
# exec(response_DS_Torch.text)