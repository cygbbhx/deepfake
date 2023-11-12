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

data_paths = [f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{i:02}' for i in range(0, 50)]
data = []

for i, data_path in enumerate(data_paths):
    metadata = get_metadata(i)

    for video, lb in tqdm(metadata.items()):
        if (lb['label'] == "REAL"):
            continue
        else:
            source = lb['original']

        real_path = os.path.join(data_path, source)
        fake_path = os.path.join(data_path, video)

        #compare only first 10 frames
        frames = os.listdir(real_path)[:10]
        sims = []
        mses = []

        for frame in frames:
            real_frame_path = os.path.join(real_path, frame)
            fake_frame_path = os.path.join(fake_path, frame)

            sim, mse = compare_images(real_frame_path, fake_frame_path)
            sims.append(sim)
            mses.append(mse)

        avg_sim = sum(sims) / len(sims)
        avg_mse = sum(mses) / len(mses)
        
        data.append([f"{i:02}", f"{video}", avg_sim, avg_mse])
    
    print(f"finished comparison in set_{i}...")

header = ['set', 'video', 'ssim', 'mse']

with open('results/similarity.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)