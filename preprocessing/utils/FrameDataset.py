import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class FrameDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []
        self._load_frames()

    def _load_frames(self):
        video_path = self.video_path

        reader = cv2.VideoCapture(video_path)
        success, image = reader.read()
        frames = []

        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(image)
            frames.append(frame)
            success, image = reader.read()

        reader.release()
        self.frames = frames
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if index >= len(self.frames):
            raise IndexError("Index out of range.")
        
        return self.frames[index]