import cv2
import numpy as np
import os


def enhance_underwater_image(image):
    image = image.astype(np.float32) / 255.0

    mean_r = np.mean(image[:, :, 2])  
    mean_g = np.mean(image[:, :, 1])  
    mean_b = np.mean(image[:, :, 0])  

    if mean_g > mean_b:  
        enhanced_r = image[:, :, 2] * 1.4  
        enhanced_g = image[:, :, 1] * 0.7 
        enhanced_b = image[:, :, 0] * 1.05  
    else:  
        enhanced_r = image[:, :, 2] * 1.4  
        enhanced_g = image[:, :, 1] * 1.05 
        enhanced_b = image[:, :, 0] * 0.7  

    image[:, :, 2] = np.clip(enhanced_r, 0, 1)
    image[:, :, 1] = np.clip(enhanced_g, 0, 1)
    image[:, :, 0] = np.clip(enhanced_b, 0, 1)

    image = (image * 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) 
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    image = cv2.convertScaleAbs(image, alpha=1.1, beta=30)

    return image


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Cannot read the file:{input_path}")
                continue

            enhanced_image = enhance_underwater_image(image)

            cv2.imwrite(output_path, enhanced_image)
            print(f"Dealed and solved:{output_path}")

input_folder = os.path.join(os.getcwd(), 'Attachment', 'Classify', 'color_cast')
output_folder = os.path.join(os.getcwd(), 'Attachment', 'Powered', 'color_cast')

process_folder(input_folder, output_folder)
