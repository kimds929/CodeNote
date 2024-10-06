import os
base_path = os.getcwd()
import json


json_dict = {'a': 1, 'b':[1,2,3], 'c': 'kkk'}
json_list = [{'a': 1, 'b':[1,2,3], 'c': 'kkk'}, {'ab':'abc', 'cb':9}]

save_data = json_dict.copy()
# save_data = json_list.copy()

# save info to json
with open(f"{base_path}/save_json.json", 'w', encoding='utf-8') as file:
    json.dump(save_data, file, indent=4, ensure_ascii=False) 
print('save complete.') 

# load json
with open(f"{base_path}/save_json.json", "r") as file:
    load_json = json.load(file)