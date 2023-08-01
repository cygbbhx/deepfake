import os 
import json
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import csv

def get_facesize():

    bbox_path = "../bboxes/"
    bbox_sets = sorted(os.listdir(bbox_path), key=lambda s: int(s.split('_')[-1][:-5]))
    data = []

    for bbox_set in bbox_sets:
        bbox_filepath = os.path.join(bbox_path, bbox_set)
        bbox_file = open(bbox_filepath, encoding="UTF-8")
        bbox_data = json.loads(bbox_file.read()) 

        set_widths = []
        set_heights = []

        for video in bbox_data:
            for key in bbox_data[video]:
                bbox = bbox_data[video][key]

                if (bbox == []):
                    bbox = prev_bbox
                else:
                    prev_bbox = bbox

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                set_widths.append(width)
                set_heights.append(height)

        avg_width = sum(set_widths) // len(set_widths)
        avg_height = sum(set_heights) // len(set_heights)
        data.append([bbox_set, avg_width, avg_height])

    
    header = ['set', 'average_width', 'average_height']
    total_avg_w = sum(w for s, w, h in data) // len(data)
    total_avg_h = sum(h for s, w, h in data) // len(data)

    data.append(['total average', total_avg_w, total_avg_h])

    with open('results/face_size.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


if __name__=='__main__':
    get_facesize()