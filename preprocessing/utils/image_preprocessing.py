import os 
import json
import cv2
import numpy as np


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


def pad_bbox(bbox, crop_ratio):
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1
    
    longest_side = max(width, height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    padding = longest_side * crop_ratio / 2

    padded_bbox = [
        center_x - padding,
        center_y - padding,
        center_x + padding,
        center_y + padding
    ]

    padded_bbox = list(map(int, padded_bbox))

    return padded_bbox


def shift_bbox(bbox, max_w, max_h):
    x1, y1, x2, y2 = bbox

    x_shift_left = max(0, -x1)
    x_shift_right = max(0, x2 - max_w)

    x1 = x1 + x_shift_left - x_shift_right
    x2 = x2 + x_shift_left - x_shift_right

    y_shift_top = max(0, -y1)
    y_shift_bottom = max(0, y2 - max_h)

    y1 = y1 + y_shift_top - y_shift_bottom
    y2 = y2 + y_shift_top - y_shift_bottom

    final_bbox = [x1, y1, x2, y2]

    return final_bbox


def make_bbox_square(bbox, max_w, max_h, crop_ratio=1.7):
    padded_bbox = pad_bbox(bbox, crop_ratio)
    final_bbox = shift_bbox(padded_bbox, max_w, max_h)

    return final_bbox