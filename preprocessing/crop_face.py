import os 
import json
import h5py
import cv2
import numpy as np
from utils.video_processing import getFrames
from tqdm import tqdm
import multiprocessing
from functools import partial

def get_metadata(set_num):
    set_path = f'/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge/dfdc_train_part_{set_num}'
    metadata_name = 'metadata.json'
    metadata_path = os.path.join(set_path, metadata_name)
    metadata_file = open(metadata_path, encoding="UTF-8")
    metadata = json.loads(metadata_file.read())

    return metadata

def get_bboxdata(set_num):
    bbox_path = f"bboxes/dfdc_train_part_{set_num}.json"
    bbox_file = open(bbox_path, encoding="UTF-8")
    bbox_data = json.loads(bbox_file.read())
    
    return bbox_data


def make_bbox_square(bbox, crop_ratio=1.7):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    longest_side = max(width, height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    padding = longest_side * crop_ratio / 2

    new_bbox = [
        center_x - padding,
        center_y - padding,
        center_x + padding,
        center_y + padding
    ]

    return new_bbox


def crop_faces(set_num):
    metadata = get_metadata(set_num)
    bbox_data = get_bboxdata(set_num)

    data_dict = {}
    size = len(metadata.items())

    for video, info in tqdm(metadata.items()):
        video_dict = {}
        video_name = video
        cropped_frames = np.zeros(shape=(300, 224, 224, 3), dtype=np.uint8)

        if (info['label'] != "REAL"):
            video = info['original']
        
        video_path = os.path.join(set_path, video)
        cap = cv2.VideoCapture(video_path)
        
        success, image = cap.read()
        count = 0

        while success:
            bbox = bbox_data[video][f"{count:03}"]

            if (bbox == []):
                bbox = prev_bbox
            else:
                prev_bbox = bbox

            padded_bbox =  make_bbox_square(bbox)
            x1, y1, x2, y2 = padded_bbox

            cropped = image[y1:y2, x1:x2, :].copy()
            resized = cv2.resize(cropped, (256,256))

            cropped_frames[count] = resized
            count += 1
            success, image = cap.read()

        video_dict['frame'] = cropped_frames
        video_dict['label'] = 0 if info['label'] == "REAL" else 1
        
        data_dict[video_name] = video_dict
        cap.release()

    with h5py.File(f'train_{set_num}.h5', 'w') as hf:
        for video_name, data in tqdm(data_dict.items()):
            grp = hf.create_group(video_name)
            frames = grp.create_dataset('frames', data=data['frame'], shape=(300, 224, 224, 3), compression='gzip', chunks=True)
            labels = grp.create_dataset('labels', data=data['label'])
    
    print(f"set_{set_num} conversion finished")

if __name__=='__main__':

    for i in range(1, 7):
        crop_faces(i)

    for i in range(10, 18):
        crop_faces(i)