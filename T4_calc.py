import os.path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_uciqe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab_image)
    chroma = np.sqrt(a**2 + b**2)
    contrast_c = np.std(chroma)
    saturation = chroma / (l + 1e-6)
    mean_saturation = np.mean(saturation)
    mean_luminance = np.mean(l)
    uciqe = 0.4680 * contrast_c + 0.2745 * mean_saturation + 0.2576 * mean_luminance
    return uciqe

def calculate_uiqm(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    mean_l = np.mean(l)
    mean_a = np.mean(a)
    mean_b = np.mean(b)

    uiqm = 0.5 * mean_l + 0.3 * mean_a + 0.2 * mean_b
    return uiqm

imgs_path = os.path.join(os.getcwd(), 'imgs_output')
columns = ['Image File Name', 'PSNR', 'UCIQE', 'UIQM']
df_all = pd.DataFrame(columns=columns)
for img in os.listdir(imgs_path):
    if not img.endswith('.png'):
        continue


    image_name = os.path.basename(os.path.join(imgs_path, img))
    original_image = cv2.imread(os.path.join(imgs_path, img))
    original_image = cv2.resize(original_image, (256, 256), interpolation=cv2.INTER_AREA)
    enhanced_image = cv2.imread(os.path.join(imgs_path, 'enhanced', f'enhanced_{img}'))
    enhanced_image = cv2.resize(enhanced_image, (256, 256), interpolation=cv2.INTER_AREA)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)


    def calculate_histogram(image):
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        return hist_r, hist_g, hist_b


    original_hist_r, original_hist_g, original_hist_b = calculate_histogram(original_image)
    enhanced_hist_r, enhanced_hist_g, enhanced_hist_b = calculate_histogram(enhanced_image)

    # Calculate metrics
    psnr = calculate_psnr(original_image, enhanced_image)
    original_uciqe = calculate_uciqe(original_image)
    enhanced_uciqe = calculate_uciqe(enhanced_image)
    original_uiqm = calculate_uiqm(original_image)
    enhanced_uiqm = calculate_uiqm(enhanced_image)

    data = {
        'Image File Name': [f'enhanced_{img}'],
        'PSNR': [psnr],
        'UCIQE': [enhanced_uciqe],
        'UIQM': [enhanced_uiqm]
    }
    df = pd.DataFrame(data)
    df_all = pd.concat([df_all, df], ignore_index=True)

    # Set the super title with metrics
    # visiable
    plt.figure(figsize=(15, 10))
    plt.suptitle(
        f"Comparison for {image_name}\n"
        f"PSNR: {psnr:.2f}, "
        f"UCIQE (Original/Enhanced): {original_uciqe:.2f}/{enhanced_uciqe:.2f}, "
        f"UIQM (Original/Enhanced): {original_uiqm:.2f}/{enhanced_uiqm:.2f}",
        fontsize=16
    )
    # plt.suptitle(f"Comparison for {image_name}", fontsize=16)

    plt.subplot(3, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(3, 2, 2)
    plt.imshow(enhanced_image_rgb)
    plt.title("Powered image")
    plt.axis("off")

    plt.subplot(3, 2, 3)
    plt.plot(original_hist_r, color='r', label='Red')
    plt.plot(original_hist_g, color='g', label='Green')
    plt.plot(original_hist_b, color='b', label='Blue')
    plt.title("Original image - RGB channel distribution")

    plt.subplot(3, 2, 4)
    plt.plot(enhanced_hist_r, color='r', label='Red')
    plt.plot(enhanced_hist_g, color='g', label='Green')
    plt.plot(enhanced_hist_b, color='b', label='Blue')
    plt.title("Powered image - RGB channel distribution")

    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    enhanced_lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)

    plt.subplot(3, 2, 5)
    plt.hist(original_lab[:, :, 0].ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Original image - Brightness distribution")

    plt.subplot(3, 2, 6)
    plt.hist(enhanced_lab[:, :, 0].ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Powered image - Brightness distribution")

    plt.tight_layout()
    if not os.path.exists(os.path.join(imgs_path, 'result')):
        os.mkdir(os.path.join(imgs_path, 'result'))
    plt.savefig(os.path.join(imgs_path, 'result', img))
    #plt.show()
df_all.to_excel(os.path.join(imgs_path, 'answer_updated_T4.xls'), index=False, engine='openpyxl')