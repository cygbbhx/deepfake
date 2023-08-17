import cv2
from PIL import Image
import os
import json
import glob
import numpy as np
from facenet_pytorch import MTCNN ## pip install facenet-pytorch
from tqdm import tqdm
from utils.face_detector import detect_facenet_pytorch
from utils.video_processing import getFrames

import jsbeautifier
import torch.nn as nn
import torch

def video2image():
    data_path = '/workspace/NAS3/CIPLAB/dataset/DeeperForensics_Dataset/DeeperForensics-1.0/manipulated_videos/end_to_end'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(device=device, post_process=False)

    videos_dict = {}
    video_paths = [os.path.join(data_path, video) for video in os.listdir(data_path)]

    for video in tqdm(video_paths):

        frames = getFrames(video)
        try:
            bboxes = detect_facenet_pytorch(detector, frames, 100)
        except Exception as e:
            print(f"Error while processing {video}", e)

        dir, video_name = os.path.split(video)
        video_boxes_dict = {}

        for frame_index, bbox in enumerate(bboxes):
            frame_key = f"{frame_index:03}"
            video_boxes_dict[frame_key] = bbox

        videos_dict[video_name] = video_boxes_dict
        
    options = jsbeautifier.default_options()
    options.indent_size = 4

    with open(f"bboxes/dff/bboxes.json", "w") as json_file:
        json_file.write(jsbeautifier.beautify(json.dumps(videos_dict), options))
    
    
if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, help='path to video directory')
    # opt = parser.parse_args()

    video2image()