
import os
os.chdir(r'D:\DataScience\CV & Resume\test_images')
import cv2
import matplotlib.pyplot as plt

# 1. JPG 파일 로드 (컬러로 불러오기)
# img = cv2.imread("0bdfd29ba.jpg")  # 경로에 맞게 수정
img = cv2.imread("0baf676c5.jpg")  # 경로에 맞게 수정

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR이므로 RGB로 변환

# 2. Gaussian Blur 적용
# (커널 크기, sigmaX)
blurred = cv2.GaussianBlur(img_rgb, (3,3), 0)
# blurred = img_rgb

# 방법 1: cv2.convertScaleAbs 사용
alpha = 0.7  # 대비 (1.0이면 원본 유지)
beta = 30    # 명도 (+이면 밝게, -이면 어둡게)
adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

plt.figure(figsize=(30,10))
# 3. 결과 시각화
plt.subplot(3, 1, 1)
plt.imshow(img_rgb)
plt.title("원본 이미지")
plt.axis("off")

plt.subplot(3, 1, 2)
plt.imshow(blurred)
plt.title("Gaussian Blur 적용")
plt.axis("off")

plt.subplot(3, 1, 3)
plt.imshow(adjusted)
plt.title("명도 및 대비 조절")
plt.axis("off")

plt.show()



plt.figure(figsize=(10,10))
plt.imshow(blurred)
plt.title("Gaussian Blur 적용")
plt.axis("off")


# 1. Denoising (Bilateral filter)
denoised = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

# # 2. Contrast Enhancement
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced = clahe.apply(denoised)

# 3. Edge Detection
edges = cv2.Canny(denoised, threshold1=50, threshold2=150)


# 3. 결과 시각화
plt.figure(figsize=(30,10))
plt.subplot(3, 1, 1)
plt.imshow(img_rgb)
plt.title("원본 이미지")
plt.axis("off")

plt.subplot(3, 1, 2)
plt.imshow(blurred)
plt.title("Gaussian")
plt.axis("off")


plt.subplot(3, 1, 3)
plt.imshow(denoised)
plt.title("Transformed")
plt.axis("off")




def lab_l_clahe_unsharp(
    img_bgr,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    do_unsharp=True,
    unsharp_sigma=1.5,     # 가우시안 블러 표준편차
    unsharp_amount=1.0,    # 엣지 강화 강도 (0.5~1.5 권장)
    unsharp_threshold=0    # 저대비 영역 보호(0이면 미사용)
):
    # 1) LAB 변환 후 L채널에만 CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L2 = clahe.apply(L)  # uint8 단일 채널만 허용

    lab2 = cv2.merge([L2, A, B])
    img_enh = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 2) 선택: Unsharp Masking (RGB/BGR 전체에 동일 가중치 → 색왜곡 최소)
    if do_unsharp:
        blur = cv2.GaussianBlur(img_enh, (0, 0), unsharp_sigma)
        if unsharp_threshold > 0:
            # 저대비(평탄) 영역은 보정 약화
            low_contrast_mask = np.abs(img_enh.astype(np.int16) - blur.astype(np.int16)) < unsharp_threshold
            sharp = cv2.addWeighted(img_enh, 1 + unsharp_amount, blur, -unsharp_amount, 0)
            img_enh[~low_contrast_mask] = sharp[~low_contrast_mask]
        else:
            img_enh = cv2.addWeighted(img_enh, 1 + unsharp_amount, blur, -unsharp_amount, 0)

    return img_enh


img_transformed = lab_l_clahe_unsharp(img_rgb)

# 3. 결과 시각화
plt.figure(figsize=(20,10))
plt.subplot(3, 1, 1)
plt.imshow(img_rgb)
plt.title("원본 이미지")
plt.axis("off")

plt.subplot(3, 1, 2)
plt.imshow(img_transformed)
plt.title("Transformed")
plt.axis("off")
