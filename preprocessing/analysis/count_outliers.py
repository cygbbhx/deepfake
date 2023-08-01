import os 
import json
import h5py
import cv2
import numpy as np
from utils.video_processing import getFrames
from tqdm import tqdm
import csv

def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # If the intersection is non-positive (no overlap), return 0.0
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    # Calculate the areas of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the intersection
    area_intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate the Union area (area of box1 + area of box2 - area of intersection)
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate the Intersection over Union (IoU)
    iou = area_intersection / area_union

    return iou

def count_outliers():

    bbox_path = "bboxes/"
    bbox_sets = os.listdir(bbox_path)
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

    with open('outliers.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


if __name__=='__main__':

    count_outliers()