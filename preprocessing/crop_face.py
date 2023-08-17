import os 
import json
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from utils.image_preprocessing import get_metadata, get_bboxdata, make_bbox_square
from datetime import datetime
from pytz import timezone

def process_frame(image, bbox, frame_width, frame_height):
    processed_bbox = make_bbox_square(bbox, frame_width, frame_height)
    x1, y1, x2, y2 = processed_bbox

    cropped = image[y1:y2, x1:x2, :].copy()
    resized = cv2.resize(cropped, (256, 256))
    return resized


def crop_faces(video_dir, bbox_path, output_dir):
    bbox_data = get_bboxdata(bbox_path)
    os.makedirs(output_dir, exist_ok=True)
    total_iterations = len(bbox_data.items())

    for video, info in bbox_data.items():
        os.makedirs(f'{output_dir}/{video}', exist_ok=True)
        
        video_path = os.path.join(video_dir, video)
        cap = cv2.VideoCapture(video_path)
        
        success, image = cap.read()
        frame_height, frame_width, c = image.shape
        count = 0

        while success:
            bbox = bbox_data[video][f"{count:03}"]

            bbox = bbox if bbox != [] else prev_bbox
            prev_bbox = bbox if bbox != [] else prev_bbox
            
            processed_frame = process_frame(image, bbox, frame_width, frame_height)

            cv2.imwrite(f"{output_dir}/{video}/{count:03}.jpg", processed_frame)
            count += 1

            success, image = cap.read()

        cap.release()
            
    

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_dir', type=str, help='path to video directory')
    # parser.add_argument('--bbox_dir', type=str, help='path to bbox data directory')
    # parser.add_argument('--output_dir', type=str, help='path to output directory')
    # args = parser.parse_args()
    data_root ='/workspace/NAS3/CIPLAB/dataset/DeeperForensics_Dataset/DeeperForensics-1.0/manipulated_videos'
    output_root = '/workspace/volume3/sohyun/dff_preprocessed/manipulated_videos/'
    m_folders = os.listdir(data_root)
    b_dir = '/workspace/ssd1/users/sohyun/preprocessing/bboxes/dff/bboxes.json'

    for folder in m_folders:
        now = datetime.now(timezone('Asia/Seoul'))
        print(f"start processing {folder} at {now}...")
        v_dir = os.path.join(data_root, folder)
        o_dir = os.path.join(output_root, folder)

        #skip if already preprocessed
        if (os.path.exists(o_dir) and len(os.listdir(v_dir)) == len(os.listdir(o_dir))):
            continue

        crop_faces(v_dir, b_dir, o_dir)
        print(f"folder {folder} processing finished")
