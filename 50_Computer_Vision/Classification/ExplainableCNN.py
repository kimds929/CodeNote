# ! pip install explainable-cnn


from explainable_cnn import CNNExplainer
import pickle, torch
from torchvision import models
import matplotlib.pyplot as plt

from PIL import Image


##############################################################################
data_path = r'D:\DataScience\강의) 강의자료\강의) [FastCampus] 55. 한 번에 끝내는 컴퓨터비전 초격차 패키지 Online\Part 2. 컴퓨터비전 특화 이론과 실습\Part 2. 컴퓨터비전 특화 이론과 실습\Chapter_3. 딥러닝과 컴퓨터비전\Code\data'

# imagenet_class_labels
with open(f"{data_path}/imagenet_class_labels.pkl", "rb") as label_file:
    imagenet_class_labels = pickle.load(label_file)


# model load
model = models.resnet18(pretrained=True)

# explainer load
x_cnn = CNNExplainer(model, imagenet_class_labels)

## Load a sample image
image = Image.open(f'{data_path}/tiger_shark.jpeg').convert('RGB')

plt.figure(figsize=(15,8))
plt.imshow(image)
plt.show()

print(imagenet_class_labels[3])

w,h = image.size
print(w,h)


# saliency_map
saliency_map = x_cnn.get_saliency_map(
    f"{data_path}/tiger_shark.jpeg",
    3, # Label corresponding to Shark. You can pass either 3 or "tiger shark, Galeocerdo cuvieri",
    (224, 224)
)

plt.imshow(saliency_map, cmap="hot")



# guide backprop
guided_backprop = x_cnn.get_guided_back_propagation(
    f"{data_path}/tiger_shark.jpeg",
    3,
    (224, 224)
)

plt.imshow(guided_backprop.astype('uint8'))




# GradCam
grad_cam = x_cnn.get_grad_cam(
    f"{data_path}/tiger_shark.jpeg",
    3,
    (224, 224),
    ["layer1"]  # List of layer names for which you want to generate image.
)

# Note that get_grad_cam() returns list of images (numpy array)
plt.imshow(grad_cam[0].astype('uint8'))



# LayerWise Comparison
model = models.resnet18(pretrained=True)

[name for name, module in model.named_children()]
# for name, module in model.named_children():
#     print(name, ":", module)


# Note that the name of layers should exactly match with model.
layers = ["relu", "layer1", "layer2", "layer3", "layer4"]


# Grad CAMs of the model
grad_cams = x_cnn.get_grad_cam(
    f"{data_path}/tiger_shark.jpeg",
    3,
    (224, 224),
    layers
)

# Guided Grad CAMs of the model
guided_grad_cams = x_cnn.get_guided_grad_cam(
    f"{data_path}/data/tiger_shark.jpeg",
    3,
    (224, 224),
    layers
)


cols = ["Grad CAM", "Guided Grad CAM"]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 24))

for ax, col in zip(axes[0], cols):
    ax.set_title(col)
    
for ax, row in zip(axes[:,0], layers):
    ax.set_ylabel(row, rotation=0, size='large', labelpad=20)
    
for ax, cam in zip(axes[:, 0], grad_cams):
    ax.imshow(cam.astype('uint8'))

for ax, gcam in zip(axes[:, 1], guided_grad_cams):
    ax.imshow(gcam.astype('uint8'))

# plt.xlabel("x",labelpad=10)