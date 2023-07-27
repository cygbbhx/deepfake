import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
import random
from PIL import Image

class FaceCropVideoDataset(Dataset):        
    def __init__(self, paths, crop_ratio, transform, frame_num):
        self.paths = paths  # [raw/videos/001.mp4, raw/videos/002.mp4, raw/videos/003.mp4, ...]
        self.crop_ratio = crop_ratio
        self.transform = transform
        self.frame_num = frame_num 
        
    def __len__(self):
        return len(self.paths)*self.frame_num
    
    def __getitem__(self, idx):
        video_idx, frame_idx = divmod(idx, self.frame_num)
        path = self.paths[video_idx] # raw/videos/001.mp4
        video_id = path.split("/")[-1][:-4] # 001
        bound_path = path.split("/videos")[0] if path.find("manipulated") >= 0 else path.split("/youtube")[0]+"/raw"
        annotations_path = os.path.join(bound_path, "result", video_id+".json") # raw/result/001.json
        df = pd.read_json(annotations_path)
        label = 1 if path.find('manipulated') < 0 else 0 # original : 1 / manipulated : 0
        
        # 비디오 크롭
        vid, start_idx = self.find_start_idx(path, self.frame_num)
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_idx + frame_idx)
        success, idx_frame = vid.read()
        if not success:
            print('ERROR: failed to read video')
        frame_id = "%04d" %(start_idx + frame_idx)
        bbox = df.loc['bbox', int(frame_id)]
        result = self.crop_video(idx_frame, bbox, self.crop_ratio) # T x H x W x C
        result = self.transform(result)
        return result, label
    
    def find_start_idx(self, path, frame_num):
        vid = cv2.VideoCapture(path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        max_idx = total_frames - frame_num
        if max_idx >=0:
            start_idx = random.randint(0, max_idx)
        else:
            start_idx = 0
            print("WARNING: frame number out of range!")
        return vid, start_idx
    
    def crop_video(self, frame, bbox, ratio):
        w, h= frame.shape[:2]
        
        # Define the four corners of the bounding box
        x1 = int(bbox[0]+bbox[2]*(1-ratio)/2) if bbox else 0
        y1 = int(bbox[1]+bbox[3]*(1-ratio)/2) if bbox else 0
        x2 = int(bbox[0]+bbox[2]*(1-ratio)/2+bbox[2]*ratio) if bbox else w
        y2 = int(bbox[1]+bbox[3]*(1-ratio)/2+bbox[3]*ratio) if bbox else h
        
        bbox = [x1, y1, x2, y2]
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        result = img.crop(bbox)
        return result