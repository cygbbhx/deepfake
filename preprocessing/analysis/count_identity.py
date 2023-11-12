import os
import jsbeautifier
import json
from tqdm import tqdm
import csv

from skimage import io, color
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse_metric

def get_metadata(set_num):
    set_path = f'/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge/dfdc_train_part_{set_num}'
    metadata_name = 'metadata.json'
    metadata_path = os.path.join(set_path, metadata_name)
    metadata_file = open(metadata_path, encoding="UTF-8")
    metadata = json.loads(metadata_file.read())

    return metadata

def compare_images(image1_path, image2_path):
    # Load images
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)
    
    # Convert images to grayscale if they are not already
    if image1.shape[-1] == 3:
        image1 = color.rgb2gray(image1)
    if image2.shape[-1] == 3:
        image2 = color.rgb2gray(image2)
    
    # Calculate SSIM and MSE
    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    mse_value = mse_metric(image1, image2)
    
    return ssim_value, mse_value

#data_paths = [f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{i:02}' for i in range(0, 50)]
#data_paths = ['/workspace/ssd1/users/sumin/datasets/celeb/Celeb-real/crop_jpg']
data_paths = ['/workspace/ssd1/users/sumin/datasets/ff/original_sequences/raw/crop_jpg']
#data_paths = ['/media/ssd1/users/sumin/datasets/vfhq/crop_jpg/training']
data = []

for i, data_path in enumerate(data_paths):
    # metadata = get_metadata(i)
    # real_videos = [video for video, info in metadata.items() if info['label'] == "REAL"]
    real_videos = os.listdir(data_path)

    for video in tqdm(real_videos):

        frames_path = os.path.join(data_path, video)
        frames = sorted(os.listdir(frames_path))

        min_sim = 1
        max_mse = -1
        
        out_frame_num1 = -1
        out_frame_num2 = -1

        for j, frame in enumerate(frames):
            frame_path = os.path.join(frames_path, frame)
            
            if (j == 0):
                prev_frame_path = frame_path
                continue
            
            sim, mse = compare_images(prev_frame_path, frame_path)

            if (sim < min_sim):
                out_frame_num1 = j
                min_sim = sim

            if (mse > max_mse):
                out_frame_num2 = j
                max_mse = mse

        data.append([f"{i:02}", f"{video}", f"{min_sim:.2f}", out_frame_num1, f"{max_mse:.2f}", out_frame_num2])
    
    print(f"finished comparison in set_{i}...")

header = ['set', 'video', 'min ssim', 'min ssim frame', 'max mse', 'max mse frame']

with open('results/ff_consecutive_similarity.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)