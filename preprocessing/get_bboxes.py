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

def getRealVideos(path):
    metadata_name = 'metadata.json'
    metadata_path = os.path.join(path, metadata_name)
    metadata_file = open(metadata_path, encoding="UTF-8")
    metadata = json.loads(metadata_file.read())

    real_videos = [video for video, info in metadata.items() if info['label'] == "REAL"]

    return real_videos


def video2image():
    data_path = '/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge'
    set_paths = [os.path.join(data_path, f"dfdc_train_part_{i}") for i in range(27, 34)]
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(device=device, post_process=False)

    for set_path in tqdm(set_paths):

        set_name = set_path.split("/")[-1]
        real_videos = getRealVideos(set_path)
        video_paths = [os.path.join(set_path, video) for video in real_videos]

        videos_dict = {}

        for video in tqdm(video_paths):

            frames = getFrames(video)
            bboxes = detect_facenet_pytorch(detector, frames, 60)

            dir, video_name = os.path.split(video)
            video_boxes_dict = {}

            for frame_index, bbox in enumerate(bboxes):
                frame_key = f"{frame_index:03}"
                video_boxes_dict[frame_key] = bbox

            videos_dict[video_name] = video_boxes_dict
            
        options = jsbeautifier.default_options()
        options.indent_size = 4

        with open(f"bboxes/{set_name}.json", "w") as json_file:
            json_file.write(jsbeautifier.beautify(json.dumps(videos_dict), options))
    
    
if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, help='path to video directory')
    # opt = parser.parse_args()

    video2image()