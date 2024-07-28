import numpy as np
from typing import List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PatchDataset, get_transform


def apply_classifier(
    patches: List[np.ndarray],
    model: Union[str, torch.nn.Module],
    batch_size: int = 32,
    danbooru: bool = False,
    tta: bool = False,
) -> List[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model loading if model is a str.
    if isinstance(model, str):
        model = torch.load(model)
        model = model.to(device)

    model.eval()

    # Use PatchDataset and DataLoader to handle patches.
    dataset = PatchDataset(
        patches, transform=get_transform(is_test=True, is_danbooru=danbooru, tta=tta)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    labels = []
    confidences = []
    with torch.no_grad():
        for batch_patches in tqdm(dataloader, desc="Apply classifier"):
            batch_patches = batch_patches.to(device)
            if tta:
                bs, ncrops, c, h, w = batch_patches.size()
                temp_output = model(batch_patches.view(-1, c, h, w))  # fuse batch size and ncrops
                outputs = temp_output.view(bs, ncrops, -1).mean(1)  # avg over crops
            else:
                outputs = model(batch_patches)
            confs_batch = nn.Softmax(dim=1)(outputs)
            labels_batch = confs_batch.max(dim=1)[1]
            confidences.extend(confs_batch.tolist())
            labels.extend(labels_batch.tolist())
    return labels, confidences
