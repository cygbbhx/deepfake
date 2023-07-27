import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import json
import glob
import numpy as np
from facenet_pytorch import MTCNN ## pip install facenet-pytorch
from tqdm import tqdm
from utils.FrameDataset import FrameDataset
import jsbeautifier
import torch.nn as nn
import torch
import logging

def video2image():
    data_path = '/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge'
    set_paths = [os.path.join(data_path, f"dfdc_train_part_{i}") for i in range(50)]
    
    metadata_name = 'metadata.json'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    for set_path in tqdm(set_paths):
        metadata_path = os.path.join(set_path, metadata_name)
        metadata_file = open(metadata_path, encoding="UTF-8")
        metadata = json.loads(metadata_file)

        real_videos = [video for video, info in metadata if info['label'] == "REAL"]

        set_name = set_path.split("/")[-1]
        dir_path = os.path.join('bboxes', set_name)
        video_paths = [os.path.join(set_path, video) for video in real_videos]

        count = 0
        os.makedirs(dir_path, exist_ok=True)

        for video_path in tqdm(video_paths):
            video_name = video_path.split("/")[-1][:-4]
            video_dict = {}
        
            dataset = FrameDataset(video_path)
            dataloader = DataLoader(dataset, batch_size=60, shuffle=False, num_workers=4)
            frame_num = 0

            for batch in tqdm(dataloader):
                boxes, _ = mtcnn.detect(batch, landmarks=False)

                for box in boxes:
                    if box is not None:
                        video_dict[f'{frame_num:03}'] = [int(coord) for coord in box[0]]
                        tmp_box = box[0]
                    else:
                        video_dict[f'{frame_num:03}'] = tmp_box
                    frame_num += 1
                    
            count += 1

            json_str = json.dumps(video_dict)
            options = jsbeautifier.default_options()
            
            with open(f"bboxes/{set_name}/{video_name}.json",'w') as outfile:
                outfile.write(jsbeautifier.beautify(json_str, options))

                        

    
    
if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, help='path to video directory')
    # opt = parser.parse_args()

    video2image()