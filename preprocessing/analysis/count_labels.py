import os
import jsbeautifier
import json
from tqdm import tqdm
import csv

data_paths = [f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{i:02}' for i in range(0, 50)]
data = []
total_real = 0
total_fake = 0

for i, data_path in tqdm(enumerate(data_paths)):
    label_filename = 'label.json'
    label_path = os.path.join(data_path, label_filename)
    label_file = open(label_path, encoding="UTF-8")
    label = json.loads(label_file.read())

    real = 0
    fake = 0

    for video, lb in label.items():
        if (lb == "REAL"):
            real += 1
        else:
            fake += 1

    total_real += real
    total_fake += fake

    ratio = f"1:{fake // real}"
    data.append([f"set_{i}", real, fake, ratio])

data.append(["total", total_real, total_fake, f"1:{total_fake // total_real}"])
header = ['set', 'real', 'fake', 'R:F']

with open('results/labels.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)