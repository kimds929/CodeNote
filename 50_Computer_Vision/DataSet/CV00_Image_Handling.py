import os 
from glob import glob
from PIL import Image   # PIL는 이미지를 load 할 때 이용
from IPython.display import clear_output
from IPython.core.display import display, HTML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# [ Functions ]

# get image label 
def get_label_from_path(path, label_dict=False):
    if '\\' in path:
        label_return = path.split('\\')[-2]
    elif '/' in path:
        label_return = path.split('/')[-2]
    try:
        if label_dict:
            return label_dict[label_return]
        else:
            return int(label_return)
    except:
        if label_dict:
            return label_dict[label_return]
        else:
            return label_return

# image load
def load_image(path, label_dict=False, label_from_folder=True, dtype='float32'):
    images_data = []
    labels = []
    for p in path:
        images_data.append(np.array(Image.open(p)))
        if label_from_folder:
            labels.append(get_label_from_path(path=p, label_dict=label_dict))
        
    images_data = np.array(images_data).astype(dtype)
    labels = np.array(labels).reshape(len(labels),)
    if 'int' in str(labels.dtype) or 'float' in str(labels.dtype):
        labels = labels.astype(dtype)

    if label_from_folder:
        return images_data, labels
    else:
        return images_data


# show image to plot
def show_image(image_data, n_samples=48, image_index=[], label=[], label_dict={}, sparse=False, title_size=12):
    len_data = len(image_data)
    if list(image_index):
        data_index = image_index
    else:
        data_index = np.arange(len_data)

    if len_data >= n_samples:
        summary_index = list(map(int, np.linspace(0, len(image_data)-1, n_samples)))
        immage_array = np.array(image_data[summary_index]).astype('uint8')
        len_image = len(immage_array)
        data_index = summary_index

        if list(label):
            label = label[summary_index]
    else:
        immage_array = np.array(image_data).astype('uint8')
        len_image = len_data

    if len_image <= 8:
        n_subplot = len_image
        width = 2 * len_image
        height = 3
    else:
        n_subplot = 8
        width = 16
        height = 2 * ((len_image-1) // 8 + 1)
    
    if label_dict:
        label_name = {v:k for k, v in label_dict.items()}
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.7)   # 위아래, 상하좌우 간격

    for seq, (index, image) in enumerate(zip(data_index, immage_array)):
        plt.subplot((len_image-1) // 8 + 1, n_subplot, seq + 1)
        plt.imshow(image)
        if list(label):
            if 'int' in str(label.dtype) or 'float' in str(label.dtype):
                title_list = list(map(int,label))
            else:
                title_list = list(label)

            if label_dict:
                plt.title(label_name[title_list[seq]] + ') ' + str(index), fontsize=title_size)
            else:
                plt.title(title_list[seq] + ') ' + str(index), fontsize=title_size)
    plt.show()
    if sparse:
        return fig



# [ Class ] 
# Image_load Class
class Image_load():
    def __init__(self):
        pass

    def load_data(self, path, dtype='float32', label_dict=False, label_from_folder=True, sparse=False, 
                image_show=False, display_samples=48):
        self.data, self.labels = load_image(path=path, dtype=dtype, label_dict=label_dict, label_from_folder=label_from_folder)
        self.feature_labels_dict = label_dict

        if sparse:
            return self.data, self.labels

        if image_show:
            self.draw_images(sparse=sparse, n_samples=display_samples)
    
    def draw_images(self, n_samples=48, title_size=12, sparse=False):
        self.images = show_image(image_data=self.data.astype('uint8'), label=self.labels, 
                                label_dict=self.feature_labels_dict, n_samples=n_samples, sparse=sparse, title_size=title_size)
        return self.images


# All Images load from folder with auto labeling
class LoadImage_from_folder(Image_load):
    def __init__(self):
        pass
    
    def load_folder(self, path, level='folder', folders=None, random_choice=None):
        """
         . level : 'folder', 'file
        """
        
        # feature 정보
        if level == 'folder':
            self.feature_path = path
            feature_list = folders if folders is not None else os.listdir(self.feature_path)  # file list
        elif level == 'file':
            path_split = reduce(lambda x,y: x+y, map(lambda x:x.split('\\'), path.split('/')))
            self.feature_path = '/'.join(path_split[:-1])
            feature_list = [path_split[-1]]

        # feature label dictionary
        self.feature_labels_dict = {}
        self.feature_labels_reverse_dict = {}
        for vi, vn in enumerate(feature_list):
            self.feature_labels_dict[vn] = vi
            self.feature_labels_reverse_dict[vi] = vn

        # image_path
        self.image_path_dict ={}
        self.image_index_dict ={}
        for vn in feature_list:
            self.image_path_dict[vn] = np.array(list(map(lambda x: x.replace('\\','/'), glob(self.feature_path + '/'+ vn +'/*'))))
            self.image_index_dict[vn] = np.array(list(map(lambda x: x.split('/')[-1], self.image_path_dict[vn])))

        if type(random_choice) == int:
            choice_indices_dict = {k: np.unique(np.random.choice(range(len(v)), random_choice)) for k, v in self.image_index_dict.items()}
            self.image_path_dict = {k: v[choice_indices_dict[k]] for k, v in self.image_path_dict.items()}
            self.image_index_dict = {k: v[choice_indices_dict[k]] for k, v in self.image_index_dict.items()}
        
        # path로부터 image Load
        feature_instance = Image_load()

        self.data_dict = {}
        self.label_dict = {}
        feature_data_temp = []
        feature_label_temp = []

        progress_bar = ' '*20
        for vi, vn in enumerate(feature_list):
            print(f'Images in folder Loading: {vi+1} / {len(feature_list)}')
            feature_instance.load_data(path=self.image_path_dict[vn], label_dict=self.feature_labels_dict)
            
            self.data_dict[vn] = feature_instance.data
            self.label_dict[vn] = feature_instance.labels
            feature_data_temp.append(feature_instance.data)
            feature_label_temp.append(feature_instance.labels)
            
            clear_output(wait=True)
        self.data = np.vstack(feature_data_temp)
        self.labels = np.hstack(feature_label_temp)
        print('load complete!')
        print(f'data.shape: {self.data.shape},  label.shape: {self.labels.shape}')      # Image Data, Label Data

        # feature count
        feature_data_count = {}
        for vn in feature_list:
            feature_data_count[vn] = len(self.label_dict[vn])

        self.feature_info = pd.concat([pd.DataFrame([self.feature_labels_dict]), pd.DataFrame([feature_data_count])], axis=0).T
        self.feature_info.reset_index(inplace=True)
        self.feature_info.columns = ['label_name','label', 'image_counts']
        self.feature_info.set_index('label', inplace=True)

    def sample_image(self, title_size=12, n_samples=8, sparse=False):
        # sample image
        sample_number = np.random.randint(0,len(self.data), size=n_samples)
        
        sample_label = self.labels[sample_number]
        sample_index = np.array(reduce(lambda x, y: list(x)+ list(y), self.image_index_dict.values()))[sample_number]
        sample_label_name = [self.feature_labels_reverse_dict[i] for i in sample_label]
        sample_label_index = np.array('('+pd.Series(sample_label_name) + ') ' + pd.Series(sample_index))
        
        print(display(HTML(pd.DataFrame(sample_label.astype(int), columns=['label'], index=sample_label_index).T._repr_html_())))
        
        sample_images = show_image(image_data=self.data[sample_number], image_index=sample_index,
                                label=sample_label, label_dict=self.feature_labels_dict, sparse=sparse, title_size=title_size)
        if sparse:
            return sample_images


# CNN Image Classifier Model Evaluate
class EvaluateClassifier():
    def __init__(self, y_pred, y_true, label_dict={}, summary_plot=False, confusion_matrix=False, predict_table=False):
            # predict DataFrame
        true = pd.Series(y_true.astype(int), name='true')
        
        self.label_dict = label_dict
        if label_dict:
            label_reverse_dict = {v:k for k, v in label_dict.items()}
            predict_df = pd.DataFrame(y_pred, columns=label_dict.keys())
            true_predict_proba = pd.concat([predict_df, true], axis=1).apply(lambda x: x[label_reverse_dict[x['true']]], axis=1)
        else:
            predict_df = pd.DataFrame(y_pred)
            true_predict_proba = pd.concat([predict_df, true], axis=1).apply(lambda x: x[int(x['true'])], axis=1)

        true_predict_proba.name = 'true_predict_proba'
        predict_label = predict_df.apply(lambda x: np.argmax(x), axis=1)
        predict_label.name = 'predict'
        predict_proba = predict_df.apply(lambda x: np.max(x), axis=1)
        predict_proba.name = 'predict_proba'
        predict_proba

        correct = (true == predict_label)
        correct.name = 'correct'

        self.predict_summary = pd.concat([correct, true, true_predict_proba, predict_label, predict_proba], axis=1)
        self.predict_overall = pd.concat([self.predict_summary, predict_df], axis=1)
        # self.predict_summary     # ****
        # self.predict_overall     # ****

        # Label_Group true_predict_proba
        label_groupby = self.predict_summary.groupby(['true'])['true_predict_proba']

        if label_dict:
            self.label_proba_dict = {label_reverse_dict[i]: d for i, d in label_groupby}
        else:
            self.label_proba_dict = {i: d for i, d in label_groupby}

        # correct / incorrect index
        self.correct_index = self.predict_summary[self.predict_summary['correct']].index
        self.incorrect_index = self.predict_summary[~self.predict_summary['correct']].index

        # confusion_matrix
        self.confusion_matrix = self.predict_summary.groupby(['true','correct']).count().iloc[:,0].to_frame().unstack('correct').fillna(0)
        self.confusion_matrix.columns = np.array(list(map(lambda x: np.array(x), self.confusion_matrix.columns))).T[1,:]
        self.confusion_proba_matrix = self.confusion_matrix.apply(lambda x: np.round(x/x.sum(),3), axis=1)
        # self.confusion_matrix            # ****
        # self.confusion_proba_matrix     # ****

        # Evaluation-value
        self.accuracy = len(self.correct_index) / len(self.predict_summary)
        self.mean_correct_proba = self.predict_summary['true_predict_proba'].mean()
        self.label_mean_correct_proba = label_groupby.mean().to_frame()
        self.label_mean_accuracy = self.confusion_proba_matrix['True'].mean()
        self.label_correct_proba_mean = self.label_mean_correct_proba['true_predict_proba'].mean()

        # display
        print(f'· Accuracy: {round(self.accuracy,3)},          · Mean_correct_proba: {round(self.mean_correct_proba,3)}')
        print(f'· Label_mean_accuracy: {round(self.label_mean_accuracy,3)},  · Label_mean_correct_proba: {round(self.label_correct_proba_mean,3)}')
        
        if summary_plot:
            print('\n -- [ Test_Data_Prediction:  Summary_Plot ] ---------------')
            self.summary_plot(sparse=False)

        if confusion_matrix == 'counts' or confusion_matrix == True:
            print('\n -- [ Test_Data_Prediction:  Confusion_Matrix ] ---------------')
            print( display(HTML(self.confusion_matrix.T._repr_html_())) )
        elif confusion_matrix == 'proba':
            print('\n -- [ Test_Data_Prediction:  Confusion_Proba_Matrix ] ---------------')
            print( display(HTML(self.confusion_proba_matrix.T._repr_html_())) )

        if predict_table == 'summary' or predict_table == True:
            print('\n -- [ Test_Data_Prediction:  Result_Summary ] ---------------')
            print( display(HTML(self.predict_summary._repr_html_())) )

        elif predict_table == 'all':
            print('\n -- [ Test_Data_Prediction:  Result ] ---------------')
            print( display(HTML(self.predict_overall._repr_html_())) )

    def summary_plot(self, sparse=False):
        # correct_proba plot
        self.result_plot = plt.figure(figsize=(13,3))
        plt.subplot(1,3,1)
        plt.title('Mean_correct_proba: ' + str(round(self.mean_correct_proba,3)) + '\n(total_accuracy:' + str(round(self.accuracy,3)) + ')')
        correct_proba_mean = self.mean_correct_proba
        self.predict_summary['true_predict_proba'].hist(color='skyblue', edgecolor='gray')
        plt.axvline(x=self.accuracy, color='red', linestyle='--', alpha=0.3, label='Model Accuracy')
        plt.axvline(x=self.mean_correct_proba, color='orange', linestyle='--', alpha=0.3, label='Mean correct proba')
        plt.xlim([-0.1,1.1])
        plt.legend()
        plt.grid(alpha=0.1)

        # label단위 전체 정답 맞추는 비율 분포
        plt.subplot(1,3,2)
        plt.title('Label_mean_accuracy: ' +  str(round(self.label_mean_accuracy,3)) )
        self.confusion_proba_matrix['True'].hist(color='skyblue', edgecolor='gray')
        plt.axvline(x=self.label_mean_accuracy, color='orange', linestyle='--', alpha=0.3, label='Label mean accuracy')
        plt.xlim([-0.1,1.1])
        plt.legend()
        plt.grid(alpha=0.1)
        

        # label 단위 정답이라고 예측할 확률 분포
        plt.subplot(1,3,3)
        plt.title('Label_mean_correct_proba: ' +  str(round(self.label_correct_proba_mean,3)) )
        self.label_mean_correct_proba['true_predict_proba'].hist(color='skyblue', edgecolor='gray')
        plt.axvline(x=self.label_correct_proba_mean, color='orange', linestyle='--', alpha=0.3, label='Label mean correct proba')
        plt.xlim([-0.1,1.1])
        plt.legend()
        plt.grid(alpha=0.1)
        
        plt.show()

        if sparse:
            return self.result_plot

    def show_image(self, image_data, kind='MinMax', random_sample=False, image_index=[], label=[], n_samples=48, title_size=12):
        if kind == False:
            show_image(image_data=image_data, image_index=image_index, label=label, title_size=title_size)
        else:
            if self.label_dict:
                label_reverse_dict = {v:k for k, v in self.label_dict.items()}

            if 'minmax' in kind.lower():
                if kind.lower().split('minmax')[1] == '':
                    n_sample = 4
                else:
                    n_sample = int(kind.lower().split('minmax')[1])
                predict_MinMax = self.predict_summary['true_predict_proba'].sort_values()
                sample_index = list(predict_MinMax[:n_sample].index) + list(predict_MinMax[-n_sample:].index)
            else:
                if label:
                    if (self.label_dict) and ('int' in str(np.array(label).dtype)):
                        search_label = [label_reverse_dict[i] for i in label]
                    else:
                        search_label = label
                else:
                    search_label = self.label_proba_dict.keys()

                if kind.lower() == 'incorrect':
                    threshold = 'self.label_proba_dict[k] < 0.5'
                elif kind.lower() == 'correct':
                    threshold = 'self.label_proba_dict[k] >= 0.5'
                elif kind.lower() == 'all':
                    threshold = 'self.label_proba_dict[k] >= 0.0'
                else:
                    threshold = 'self.label_proba_dict[k] ' + kind
                sample_index = []
                for k in search_label:
                    sample_index = sample_index + list(self.label_proba_dict[k][eval(threshold)].index)
                
                if random_sample:
                    sample_index = list(np.random.choice(sample_index, min(len(sample_index), random_sample), replace=False))

            if self.label_dict:
                sample_label = self.predict_summary.iloc[sample_index, :].apply(lambda x: str(x['true']) + ' ' + label_reverse_dict[x['true']] + '\n' + str(np.round(x['true_predict_proba'],3)), axis=1).to_numpy()
            else:
                sample_label = self.predict_summary.iloc[sample_index, :].apply(lambda x: 'Label ' + str(x['true']) + '\n' + str(np.round(x['true_predict_proba'],3)), axis=1).to_numpy()

            # show image
            show_image(image_data=image_data[sample_index,:], image_index=sample_index, label=sample_label, n_samples=n_samples, title_size=title_size)




################################################################################################################################
################################################################################################################################

# # Load Image from list of file
# image_load = Image_load()
# image_load.load_data([f"{target_path}/HRX071660_head.png", f"{target_path}/HRX071670_head.png"])
# image_load.draw_images()
# image_load.data
# image_load.labels


# # path 확인
# current_dir = os.getcwd()
# target_path ='D:/작업방/업무 - 자동차 ★★★/Workspace_Python/DataSet_Image/220804_High_EL_PCM/CT_Head'
# target_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 9) Team Project/Image/Train'
# target_path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 9) Team Project\Image\Train'


# # Load Image from folder
# Train_instance = LoadImage_from_folder()
# Train_instance.load_folder(path=target_path, level='file', random_choice=30)
# # Train_instance.load_folder(path=target_path, level='folder', random_choice=3)
# Train_instance.sample_image(title_size=9)











# # 이미지 불러오기
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2 as cv
# from PIL import Image   # PIL는 이미지를 load 할 때 이용
# from PIL import ImageGrab

# # 경로설정 ###############################################################
# save_path = 'D:/작업방/업무 - 자동차 ★★★/Workspace_Python/DataSet_Image'
# os.getcwd()


# # Clipboard로부터 이미지 불러오기 ###############################################################
# img_pil = ImageGrab.grabclipboard()

# # 파일로부터 이미지 불러오기 ###############################################################
# # (pyplot)  # (H, W, C) # Channel : RGBA
# img_plt = plt.imread(f'{save_path}/Example/example_001.png')
# print(type(img_plt), img_plt.shape) # <class 'numpy.ndarray'> (269, 2020, 4) 

# # show
# plt.imshow(img_plt)

# # (cv2) # (H, W, C) # Channel : BRG
# img_cv = cv.imread(f'./DataSet_Image/Example/example_001.png')  # 한글경로는 Load불가
# print(type(img_cv), img_cv.shape) # <class 'numpy.ndarray'> (269, 2020, 3)  

# # * mode
# # cv.IMREAD_UNCHANGED : 원본 사용
# # cv.IMREAD_GRAYSCALE : 1 채널, 그레이스케일 적용
# # cv.IMREAD_COLOR : 3 채널, BGR 이미지 사용
# # cv.IMREAD_ANYDEPTH : 이미지에 따라 정밀도를 16/32비트 또는 8비트로 사용
# # cv.IMREAD_ANYCOLOR : 가능한 3 채널, 색상 이미지로 사용
# # cv.IMREAD_REDUCED_GRAYSCALE_2 : 1 채널, 1/2 크기, 그레이스케일 적용
# # cv.IMREAD_REDUCED_GRAYSCALE_4 : 1 채널, 1/4 크기, 그레이스케일 적용
# # cv.IMREAD_REDUCED_GRAYSCALE_8 : 1 채널, 1/8 크기, 그레이스케일 적용
# # cv.IMREAD_REDUCED_COLOR_2 : 3 채널, 1/2 크기, BGR 이미지 사용
# # cv.IMREAD_REDUCED_COLOR_4 : 3 채널, 1/4 크기, BGR 이미지 사용
# # cv.IMREAD_REDUCED_COLOR_8 : 3 채널, 1/8 크기, BGR 이미지 사용
# img_cv_origin = cv.imread(f'./DataSet_Image/Example/example_001.png', cv.IMREAD_UNCHANGED)  # 원본
# img_cv_bgr = cv.imread(f'./DataSet_Image/Example/example_001.png', cv.IMREAD_COLOR)     # 3채널 BGR
# img_cv_gray = cv.imread(f'./DataSet_Image/Example/example_001.png', cv.IMREAD_GRAYSCALE)  # 1채널 Grayscale
# print(f"origin: {img_cv_origin.shape} / bgr: {img_cv_bgr.shape} / gray {img_cv_gray.shape}")

# # show
# plt.imshow(img_cv)


# # imshow (별도 윈도우 창에서 실행)
# cv.imshow('Show Image', img_cv)
# cv.waitKey(0)           # time마다 키 입력상태를 받아옵니다. 0일 경우, 지속적으로 검사하여 해당 구문을 넘어가지 않습니다.
# cv.destroyAllWindows()  # 모든 윈도우창을 닫습니다.




# # (PIL)  # (H, W, C) # Channel : RGBA
# img_pil = Image.open(f'{save_path}/Example/example_001.png')
# img_pil = Image.open(f'{save_path}/Example/example_001.png')
# np_pil = np.array(img_pil)

# print(type(img_pil), np_pil.shape)  # <class 'PIL.PngImagePlugin.PngImageFile'> (269, 2020, 4)  

# # show
# img_pil
# plt.imshow(img_pil)


# # 이미지 Library간 Transform ###############################################################
# img_cv2 = cv.cvtColor(img_plt, cv.COLOR_RGB2BGR)
# img_cv2 = cv.cvtColor(np.array(np_pil), cv.COLOR_RGB2BGR) # PIL → cv2


# # 이미지저장 ###############################################################
# # (pyplot)
# plt.imsave(f'{save_path}/Example/example_001_plt.png', img_plt)

# # (cv)
# cv.imwrite(f'./DataSet_Image/Example/example_001_cv.png', img_cv)  # 한글경로는 Save불가

# # (PIL)
# img_pil.save(f'{save_path}/Example/example_001_pil.png')


# # Transpose ###############################################################
# r = np.ones((10,20))* 0.1
# g = np.ones((10,20))* 0.5
# b = np.ones((10,20))* 0.9
# img_sample_chw = np.array([r, g, b])
# print(img_sample_chw.shape)

# # (C, H, W) → (H, W, C)
# img_sampe_hwc = np.transpose(img_sample_chw, [1, 2, 0])
# plt.imshow(img_sampe_hwc)

# print(img_sampe_hwc.shape)

# # (H, W, C) → (C, H, W)
# img_sampe_chw = np.transpose(img_sampe_hwc, [2, 0, 1])
# print(img_sampe_chw.shape)


# # element Split ###############################################################

# # rgb 분리, 결합
# r, g, b, a = cv.split(img_plt)  # pyplot
# b,g,r = cv.split(img_cv)        # cv
# r, g, b, a = cv.split(np.array(img_pil))    # PIL

# img_rgb = cv.merge((r,g,b))
# img_rgb = np.transpose(np.array([r, g, b]), [1,2,0])
# plt.imshow(img_rgb)

# r, g, b = np.split(img_rgb)








# # # 이미지 차원축소
# # img.mean(2).shape


# # # 이미지 해상도 변경(resize)
# # plt.imshow(cv.resize(img, (100,70)))





# # import torch




# ###########################

# r = np.array(
#     [[255, 155, 85, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [0, 100, 120, 0]]
#     )

# g = np.array(
#     [[0, 0, 0, 0],
#     [250, 135, 75, 15],
#     [0, 0, 0, 0],
#     [0, 110, 0, 140]]
#     )

# b = np.array(
#     [[0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [245, 175, 105, 0],
#     [0, 0, 130, 150]]
#     )

# # * matplotlib/Pillow/cv2: (4, 4, 3)


# img_rgb = cv.merge((r,g,b))
# # img_rgb.shape
# img_rgb = np.transpose( np.array([r,g,b]) , [1,2,0])

# img_pillow = Image.fromarray(img_rgb.astype(np.uint8))
# # img_pillow = Image.fromarray(np.array([r,g,b]).astype(np.uint8))
# img_pillow.resize((100,100))


# plt.imshow(img_rgb)
# plt.imshow(np.array([r,g,b]))



# # imshow (별도 윈도우 창에서 실행)
# # cv2.imshow('Show Image', img_rgb)
# # cv2.waitKey(0)           # time마다 키 입력상태를 받아옵니다. 0일 경우, 지속적으로 검사하여 해당 구문을 넘어가지 않습니다.
# # cv2.destroyAllWindows()  # 모든 윈도우창을 닫습니다.