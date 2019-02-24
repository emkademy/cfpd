"""
Datasets
"""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import utils


class FacialPartsDataset(Dataset):
    """
    Facial Parts Dataset
    """
    def __init__(self, data_frame, color, transform=None):
        self.df = data_frame
        self.transform = transform
        self.color = color

        left_eyebrow_indices = list(range(17, 22))
        left_eye_indices = list(range(36, 42))
        right_eyebrow_indices = list(range(22, 27))
        right_eye_indices = list(range(42, 48))

        left_eye_part = np.asarray(left_eyebrow_indices + left_eye_indices)
        right_eye_part = np.asarray(right_eyebrow_indices + right_eye_indices)
        nose_part = np.arange(27, 36)
        mouth_part = np.arange(48, 68)

        self.parts_to_detect = (left_eye_part, right_eye_part, nose_part, mouth_part)

    def __getitem__(self, index):
        img = Image.open(self.df["image"].iloc[index])
        if not self.color:
            img = img.convert("L")  # Convert image to black and white

        landmarks = utils.load_pts_file(self.df["landmarks"].iloc[index])
        groundtruth_coordinates = []
        for part in self.parts_to_detect:
            x_min, y_min = np.min(landmarks[part], axis=0)
            x_max, y_max = np.max(landmarks[part], axis=0)
            y_max += 2  # Makes lower boundary a bit larger
            groundtruth_coordinates.extend([x_min, y_min, x_max, y_max])

        if self.transform:
            img = self.transform(img)

        groundtruth_coordinates = torch.tensor(groundtruth_coordinates)
        return img, groundtruth_coordinates

    def __len__(self):
        return len(self.df)
