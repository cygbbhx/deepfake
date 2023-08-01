import os 
import json
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import csv

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    area_intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union

    return iou

def count_outliers():

    bbox_path = "../bboxes/"
    bbox_sets = sorted(os.listdir(bbox_path), key=lambda s: int(s.split('_')[-1][:-5]))
    data = []
    threshold = 0.15

    for bbox_set in bbox_sets:
        bbox_filepath = os.path.join(bbox_path, bbox_set)
        bbox_file = open(bbox_filepath, encoding="UTF-8")
        bbox_data = json.loads(bbox_file.read()) 

        set_outliers = []

        for video in bbox_data:
            outliers = 0 
            first_bbox = bbox_data[video]["000"]
            if (first_bbox == []):
                continue

            for key in bbox_data[video]:
                bbox = bbox_data[video][key]

                if (bbox == []):
                    bbox = prev_bbox
                else:
                    prev_bbox = bbox

                iou = calculate_iou(first_bbox, bbox)
                if iou < threshold:
                    outliers += 1

            set_outliers.append(outliers)
        
        set_avg_outliers= sum(set_outliers) // len(set_outliers)
        data.append([bbox_set, set_avg_outliers])

    
    header = ['set', 'average_outliers']
    total_avg = sum(a for s, a in data) // len(data)

    data.append(['total average', total_avg])

    with open('results/outliers.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


if __name__=='__main__':
    count_outliers()