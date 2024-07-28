import logging
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import PatchDataset, get_transform


def train_classifier(
    train_data: List[Tuple[np.ndarray, int]],
    pretrained_model_path: str,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    num_epochs: int = 50,
    output_model_path=None,
) -> nn.Module:
    if output_model_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".pt", delete=True)
        output_model_path = tf.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = max([label for _, label in train_data]) + 1

    train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

    is_danbooru = "danbooru" in pretrained_model_path
    train_dataset = PatchDataset(
        train_data, transform=get_transform(is_test=False, is_danbooru=is_danbooru)
    )
    valid_dataset = PatchDataset(
        valid_data, transform=get_transform(is_test=True, is_danbooru=is_danbooru)
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )

    print(f"Load {pretrained_model_path}...")
    if is_danbooru:
        model = torch.hub.load("RF5/danbooru-pretrained", "resnet50")
        model = torch.load(pretrained_model_path)
        model[-1][-1] = nn.Linear(512, num_classes, bias=True)
    elif pretrained_model_path is not None:
        model = torch.load(pretrained_model_path)
        del model.fc[-1]
        model.fc[-1] = nn.Linear(256, num_classes)
    model = model.to(device)

    # loss_func = nn.NLLLoss()
    train_loss_func = nn.CrossEntropyLoss()
    val_loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Start training
    best_acc, best_epoch = 0.0, 0
    print(f"Start training classifier with {len(train_dataset)} samples and {num_epochs} epochs.")
    for epoch in range(num_epochs):
        train_loss, train_acc = _train_epoch(
            model, train_dataloader, train_loss_func, optimizer, device
        )
        valid_loss, valid_acc = _valid_epoch(model, valid_dataloader, val_loss_func, device)
        scheduler.step()

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_acc / len(train_dataset)
        avg_valid_loss = valid_loss / len(valid_dataset)
        avg_valid_acc = valid_acc / len(valid_dataset)

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, output_model_path)
        print(
            "Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100
            )
        )
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        # Early stop training if validation accuracy does not improve for 5 consecutive epochs.
        if epoch > 10 and best_epoch <= epoch - 5:
            break

    print(f"Best Accuracy for validation : {best_acc:.4f} at epoch {best_epoch}")
    print(f"Best model saved at {output_model_path}")

    return torch.load(output_model_path)


def _train_epoch(model, dataloader, loss_func, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)

    return train_loss, train_acc


def _valid_epoch(model, dataloader, loss_func, device):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)

    return valid_loss, valid_acc
