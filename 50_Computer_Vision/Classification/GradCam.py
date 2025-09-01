import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------------------------------
# 1. 준비: 모델 & 이미지
# -------------------------------------------------------
data_path = r'D:\DataScience\강의) 강의자료\강의) [FastCampus] 55. 한 번에 끝내는 컴퓨터비전 초격차 패키지 Online\Part 2. 컴퓨터비전 특화 이론과 실습\Part 2. 컴퓨터비전 특화 이론과 실습\Chapter_3. 딥러닝과 컴퓨터비전\Code\data'

# GPU 사용 가능 여부 확인 후 장치 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f"Using device: {device}")

model = models.resnet18(pretrained=True).to(device).eval()

data_path = f"{data_path}/tiger_shark.jpeg"  # 샘플 이미지 경로

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
])

image = Image.open(data_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)  # 1x3x224x224
rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0  # 시각화용

# -------------------------------------------------------
# 2. Saliency Map (직접 구현)
# -------------------------------------------------------
input_tensor.requires_grad_()
scores = model(input_tensor)
target_class = scores.argmax(dim=1).item()

score = scores[0, target_class]
model.zero_grad()
score.backward()

saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
saliency = saliency[0].cpu().numpy()
plt.imshow(saliency, cmap="hot")
plt.title("Saliency Map")
plt.show()

# -------------------------------------------------------
# 3. Grad-CAM
# -------------------------------------------------------
target_layer = model.layer4[-1]

cam = GradCAM(model=model,
              target_layers=[target_layer]
              )

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(target_class)])[0]

cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(cam_image)
plt.title("Grad-CAM")
plt.show()

# -------------------------------------------------------
# 4. Guided Backpropagation
# -------------------------------------------------------
gb_model = GuidedBackpropReLUModel(model=model, device=device)

gb = gb_model(input_tensor, target_category=target_class)

if device != 'cpu':
    gb = gb[0].cpu().numpy().transpose((1, 2, 0)) # cpu로 이동 후 numpy로 변환


plt.imshow(gb)
plt.title("Guided Backpropagation")
plt.show()

# -------------------------------------------------------
# 5. Guided Grad-CAM (GuidedBackprop × CAM)
# -------------------------------------------------------
guided_grad_cam = gb * grayscale_cam[..., np.newaxis]
plt.imshow(guided_grad_cam)
plt.title("Guided Grad-CAM")
plt.show()


# -------------------------------------------------------
# 6. Guided Backpropagation, layer
# -------------------------------------------------------

def get_grad_cam(model, input_tensor, rgb_img, target_class, layer_names):
    cams = []
    
    # 모델의 named_children()를 사용해 정확한 레이어 객체 찾기
    model_layers = {name: module for name, module in model.named_children()}

    for layer_name in layer_names:
        if layer_name in model_layers:
            target_layer = model_layers[layer_name]

            # GradCAM 객체 생성
            cam = GradCAM(model=model, target_layers=[target_layer])

            # Grad-CAM 계산
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(target_class)])[0]

            # 결과를 이미지로 변환
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cams.append(cam_image)
        else:
            print(f"Warning: Layer '{layer_name}' not found. Skipping.")
            
    return cams


layers = ["layer1", "layer2", "layer3"] # 여러 레이어에 대해 Grad-CAM을 생성하고 싶을 때

# Grad-CAM 함수 호출
grad_cams = get_grad_cam(model, input_tensor, rgb_img, target_class, layers)


# 첫 번째 Grad-CAM 이미지 시각화
for i in range(len(layers)):
    plt.subplot(1,len(layers),i+1)
    plt.imshow(grad_cams[i])
    plt.title(f"Grad-CAM on {layers[i]}")
plt.show()
