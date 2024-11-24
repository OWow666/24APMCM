import cv2
import numpy as np
import os

def classify_images(folder_path, cc, L1, L2, b):
    classifications = {"color_cast":[],
                       "low_light":[],
                       "blur":[]}
    
    a1,b1,c1 = 0,0,0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)
        
        # Color Cast
        B_mean, G_mean, R_mean = cv2.mean(img)[:3]
        color_imbalance = max(abs(R_mean- G_mean), abs(G_mean - B_mean), abs(B_mean - R_mean))
        is_color_cast = color_imbalance > cc
        
        # Low Light
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_low_light = np.mean(gray_img) < L1
        is_low_contrast = np.std(gray_img) < L2
        
        # Blur
        Var_Laplacian = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        is_blur = Var_Laplacian < b
        
        if is_color_cast:
            classifications["color_cast"].append(filename)
            a1 = a1 + 1
        if is_low_light or is_low_contrast:
            classifications["low_light"].append(filename)
            b1 = b1 + 1
        if is_blur:
            classifications["blur"].append(filename)
            c1 = c1 + 1
    print(a1,b1,c1)
    
    return classifications

images_folder_path = os.path.join(os.getcwd(), "D:\shumo\\2024 APMCM\Attachment\Attachment1")
results = classify_images(images_folder_path, 80, 5, 15, 50)

import pandas as pd

df = pd.read_excel('D:\shumo\\2024 APMCM\code\Answer.xls')

for key, value in results.items():
    item_df = pd.DataFrame({
        'Degraded Image Classification': [key] * len(value),
        'image file name': value
    })
    df = pd.concat([df, item_df], ignore_index=True)

df.to_excel('Answer_updated.xlsx', index=False)