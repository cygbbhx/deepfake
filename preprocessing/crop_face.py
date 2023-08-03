import os 
import json
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from utils.image_preprocessing import get_metadata, get_bboxdata, make_bbox_square


def crop_faces(set_num):
    set_path = f'/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge/dfdc_train_part_{set_num}'
    metadata = get_metadata(set_num)
    bbox_data = get_bboxdata(set_num)
    
    os.makedirs(f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{set_num:02}', exist_ok=True)
    # data_dict = {}
    size = len(metadata.items())

    for video, info in tqdm(metadata.items()):
        video_dict = {}
        cropped_frames = np.zeros(shape=(300, 256, 256, 3), dtype=np.uint8)
        os.makedirs(f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{set_num:02}/{video}', exist_ok=True)

        if (info['label'] != "REAL"):
            src_video = info['original']
        else:
            src_video = video
        
        video_path = os.path.join(set_path, video)
        cap = cv2.VideoCapture(video_path)
        
        try:
            success, image = cap.read()
            frame_height, frame_width, c = image.shape
            count = 0
        except:
            print(f"Error while processing set{set_num}/{video}")

        while success:
            bbox = bbox_data[src_video][f"{count:03}"]

            if (bbox == []):
                bbox = prev_bbox
            else:
                prev_bbox = bbox
            
            try:
                processed_bbox = make_bbox_square(bbox, frame_width, frame_height)
                x1, y1, x2, y2 = processed_bbox

                cropped = image[y1:y2, x1:x2, :].copy()
                resized = cv2.resize(cropped, (256,256))

                cropped_frames[count] = resized
                cv2.imwrite(f"/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{set_num:02}/{video}/{count:03}.jpg", resized)
            
            except:
                print("Error occured while prcocessing...")
                print(f"video: {video}")
                print(f"processed bbox: {processed_bbox}")
                print(f"width: {frame_width} height: {frame_height} original bbox: {bbox}")

            count += 1
            success, image = cap.read()
        
        cap.release()
    
    print(f"set_{set_num} conversion finished")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--format', 
                        type=str, 
                        default='h5', 
                        choices = ['h5', 'png'],
                        help='dataset type')

    parser.add_argument('--crop_ratio',
                        type=float,
                        default=1.7,
                        help='crop ratio')

    args = parser.parse_args()

    for i in range(50):
        crop_faces(i)