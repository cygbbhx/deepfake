import os
from utils.image_preprocessing import get_metadata
import jsbeautifier
import json
from tqdm import tqdm

data_path = '/workspace/NAS2/CIPLAB/dataset/deepfake-detection-challenge'
set_paths = [os.path.join(data_path, f"dfdc_train_part_{i}") for i in range(0, 50)]
dst_path = [f'/workspace/volume3/sohyun/dfdc_preprocessed/dfdc_{i:02}' for i in range(0, 50)]

for i, set_path in enumerate(tqdm(set_paths)):
    metadata = get_metadata(i)

    videos_dict = {}
    for video, info in metadata.items():
        label = info['label']
        videos_dict[video] = label

    os.makedirs(dst_path[i], exist_ok = True)
    options = jsbeautifier.default_options()
    options.indent_size = 4

    with open(f"{dst_path[i]}/label.json", "w") as json_file:
        json_file.write(jsbeautifier.beautify(json.dumps(videos_dict), options))