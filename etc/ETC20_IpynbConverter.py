
import os
from functools import reduce
import datetime
# from IPython.display import clear_output
# pip install nbconvert
# conda install nbconvert
# pip install ipynb-py-convert
# jupyter nbconvert --to markdown JUPYTER_NOTEBOOK.ipynb

# path =  'D:/Workspace_Python/기타'
# path1 = 'D:/Workspace_Python/기타/test03.py'
# path2 = 'D:/Workspace_Python/기타/test01.ipynb'
# path3 = 'D:/Workspace_Python/기타/test02.ipynb'
# path4 = [path2, path3]

# path.split('.')
# path1.split('.')
# os.system(f'jupyter nbconvert --to script "{path2}" "{path1}"')
# os.system(f'jupyter nbconvert --to notebook "{path1}" "{path3}"')
# os.system(f'jupyter nbconvert --config "{path1}"')

# os.system(f'ipynb-py-convert "{path2}" "{path1}"')
# os.system(f'ipynb-py-convert "{path1}" "{path3}"')



# ic = IpynbConverter()
# # ic.listdir(path)
# ic.convert(path, input='ipynb', output='py')
# ic.convert(path, input='py', output='html')
# ic.convert(path, input='ipynb', output='py')
# ic.convert(path1, output='ipynb')





# from functools import reduce
class IpynbConverter:
    """
    【Requried Library】import os, import nbconvert, import ipynb-py-convert, from functools import reduce
    """
    def __init__(self, path=None):
        self.listdir(path)
        self.convert_dict = {'markdown': 'markdown', 'md': 'markdown', 'html': 'html', 'pdf':'pdf'}
        self.splitext_dict = {'py':'py', 'python': 'py',
                              'notebook':'ipynb', 'ipynb':'ipynb', 'ipython': 'ipynb',
                              'markdown': 'md', 'md': 'md', 'html': 'html', 'pdf':'pdf'}

    # filtering py, ipynb file     
    def _filter_sep_files(self, path):
        path = path.replace('\\', '/')
        
        if os.path.splitext(path)[1] == '': # folder
            folder_path = path + '/'
            path_files = os.listdir(path)
        else:
            folder_path = ''
            path_files = [path]
        path_dict = {}
        path_dict['py'] = [folder_path + f for f in path_files if os.path.splitext(f)[1] == '.py' in f]
        path_dict['ipynb'] = [folder_path + f for f in path_files if os.path.splitext(f)[1] == '.ipynb' in f]
        return path_dict
    
    def _convert_list_type_path(self, path, splitext):
        if path is None:
            path = self.path_dict[splitext]
        elif type(path) == list:
            path = [p.replace('\\', '/') for p in path]
        elif type(path) == dict:
            path = path[splitext]
        else:
            path = self.listdir(path, verbose=0)[splitext]
        return path
    
    # list py, ipynb file
    def listdir(self, path=None, verbose=1):
        if path is not None:
            self.path_dict = self._filter_sep_files(path)
            if verbose > 0:
                print("self.path_dict")
            return self.path_dict
    
    # debug_recording
    def _debug_record(self, syscode, input_path, debug=False):
        if debug is False:
            try:
                os.system(syscode)
                self.convert_success.append(input_path)
            except:
                self.convert_failure.append(input_path)
        else:
            os.system(syscode)
    
    # ipynb-py-convert
    def _ipynb_py_convert_command(self, input_path, output_path, debug=False):
        syscode = f'ipynb-py-convert "{input_path}" "{output_path}"'
        self._debug_record(syscode=syscode, input_path=input_path, debug=debug)
    
    # nbcovert
    def _nbcovert_command(self, input_path, output, execute=True, debug=False):
        execute_code = '--execute ' if execute is True else ''
        syscode = f'jupyter nbconvert {execute_code}--to {self.convert_dict[output]} "{input_path}"'
        self._debug_record(syscode=syscode, input_path=input_path, debug=debug)
       
    # Convert ipynb to py
    def convert(self, path=None, input=None, output='py', execute=True, verbose=1, debug=False):
        if (input is None) and (type(path) == str):
            input = os.path.splitext(path)[1][1:]
        elif (input is None) and (type(path) == list):
            splitext_list = [os.path.splitext(p)[1][1:] for p in path]
            splitext = reduce(lambda x,y: x if x == y else 0, splitext_list)
            if splitext != 0:
                input = splitext
        path_list = self._convert_list_type_path(path=path, splitext=input)
        len_input_splitext = len(input)
        
        # return path
        self.convert_success = []
        self.convert_failure = []
        e = 1
        
        output_splitext = self.splitext_dict[output]
        
        # return path_list
        for p in path_list:
            if verbose > 0:
                print(f'({round((e)/len(path_list)*100,1)}%) "{p.split("/")[-1]}" Converting...', end='\r')

            if (input == 'ipynb' and output_splitext == 'py') or (input=='py' and output_splitext == 'ipynb'):
                # ipynb → py / py → ipynb
                self._ipynb_py_convert_command(input_path=p, 
                                               output_path=f'{p[:-len_input_splitext]}{output_splitext}', 
                                               debug=debug)
            elif input == 'ipynb':
                # ipynb → html, md, pdf
                self._nbcovert_command(input_path=p, output=output_splitext, execute=execute, debug=debug)
                
            elif input == 'py':
                # py → ipynb → html, md, pdf
                time_now = '_' + str(datetime.datetime.now()).replace('-','').replace(' ','').replace(':','').replace('.','')
                temp_name = f'{p[:-len_input_splitext-1]}{time_now}.ipynb'
                
                self._ipynb_py_convert_command(input_path=p, output_path=temp_name, debug=debug)
                self._nbcovert_command(input_path=temp_name, output=output_splitext, execute=execute, debug=debug)
                os.remove(temp_name)
                os.rename(f"{temp_name[:-6]}.{output_splitext}", f"{temp_name[:-(len(b)+6)]}.{output_splitext}")
                
            e += 1
    
        if verbose > 0 and len(self.convert_failure) >0:
            print('convert_failure file list')
            print(self.convert_failure)



