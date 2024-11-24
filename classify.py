import os
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def classify_images(path_a, path_b, path_c):
    df = pd.read_excel(path_b)

    if not os.path.exists(path_c):
        os.makedirs(path_c)

    for index, row in df.iterrows():
        image_name = row['image file name']  
        category = row['Degraded Image Classification']    

        category_folder = os.path.join(path_c, str(category))
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        source_image_path = os.path.join(path_a, image_name)
        if os.path.exists(source_image_path):
            destination_image_path = os.path.join(category_folder, image_name)
            shutil.copy(source_image_path, destination_image_path)
            print(f"move {image_name} to {category_folder}")
        else:
            print("source_image_path:",source_image_path)
            print(f"image {image_name} is not existing in {path_a}")
            #image = Image.open(source_image_path)
            #plt.imshow(image)
            #plt.axis('off')
            #plt.show()
            #break

path_a = 'D:\shumo\\2024 APMCM\Attachment\Attachment1'
path_b = r'D:\shumo\2024 APMCM\code\Answer_updated.xlsx'
path_c = r'D:\shumo\2024 APMCM\Attachment\Classify'

classify_images(path_a, path_b, path_c)
