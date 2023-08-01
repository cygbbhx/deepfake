import os 
import json
import cv2
import numpy as np
from tqdm import tqdm
import csv

def getRealVideos(path):
    metadata_name = 'metadata.json'
    metadata_path = os.path.join(path, metadata_name)
    metadata_file = open(metadata_path, encoding="UTF-8")
    metadata = json.loads(metadata_file.read())

    real_videos = [video for video, info in metadata.items() if info['label'] == "REAL"]

    return real_videos


def get_videodim():

    data_path = '/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge'
    set_paths = [os.path.join(data_path, f"dfdc_train_part_{i}") for i in range(0, 50)]
    data = []

    for set_path in tqdm(set_paths):
        set_name = set_path.split("/")[-1]
        real_videos = getRealVideos(set_path)
        video_paths = [os.path.join(set_path, video) for video in real_videos]

        set_widths = []
        set_heights = []

        for video_path in tqdm(video_paths):
            cap = cv2.VideoCapture(video_path)
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH ) 
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            set_widths.append(int(width))
            set_heights.append(int(height))

        avg_width = sum(set_widths) // len(set_widths)
        avg_height = sum(set_heights) // len(set_heights)
        data.append([set_name, avg_width, avg_height])
    
    header = ['set', 'average_width', 'average_height']
    total_avg_w = sum(w for s, w, h in data) // len(data)
    total_avg_h = sum(h for s, w, h in data) // len(data)

    data.append(['total average', total_avg_w, total_avg_h])

    with open('results/video_dimensions.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


if __name__=='__main__':
    get_videodim()