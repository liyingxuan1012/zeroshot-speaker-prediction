import numpy as np
from PIL import Image
from typing import List, Tuple, Union

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad
from torchvision.transforms import TenCrop


class PatchDataset(Dataset):
    def __init__(self, data: Union[List[np.ndarray], List[Tuple[np.ndarray, int]]], transform=None):
        self.transform = transform
        if isinstance(data[0], tuple):
            # data is a list of (patch, label) tuples
            self.patches, self.labels = zip(*data)
            self.labels = self.labels
        else:
            # data is a list of patches only
            self.patches = data
            self.labels = [None] * len(data)  # create a placeholder list of None

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.transform:
            patch = self.transform(Image.fromarray(patch))
        label = self.labels[idx]
        return patch if label is None else (patch, label)

    def set_label(self, idx, label):
        self.labels[idx] = label


class PadToSquare:
    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, self.fill, self.padding_mode)


def get_transform(is_test: bool, is_danbooru: bool = False, tta: bool = False):
    if is_danbooru:
        if is_test:
            return transforms.Compose(
                [
                    transforms.Resize(360),
                    PadToSquare(),
                    transforms.Resize((360, 360)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(390),
                    transforms.RandomCrop(360),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    PadToSquare(),
                    transforms.Resize((360, 360)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]
                    ),
                ]
            )
    else:
        if is_test:
            if tta:
                return transforms.Compose(
                    [
                        transforms.Resize((270, 270)),
                        TenCrop(256),
                        lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                        lambda crops: torch.stack(
                            [
                                transforms.Normalize(
                                    mean=[0.5125, 0.4667, 0.4110], std=[0.2621, 0.2501, 0.2453]
                                )(crop)
                                for crop in crops
                            ]
                        ),
                    ]
                )
            else:
                return transforms.Compose(
                    [
                        transforms.Resize(size=(256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.5125, 0.4667, 0.4110], std=[0.2621, 0.2501, 0.2453]
                        ),
                    ]
                )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((270, 270)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomCrop(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5125, 0.4667, 0.4110], std=[0.2621, 0.2501, 0.2453]
                    ),
                ]
            )
