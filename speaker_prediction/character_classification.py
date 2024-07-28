from typing import Any
import random
import numpy as np
from omegaconf import DictConfig
from collections import defaultdict
from tqdm import tqdm
from .utils import crop_img
from .image_classification import train_classifier, apply_classifier


class CharacterClassifier:
    """Character Idenficiation module."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.method = config.method
        self.classifier = config.method.classifier
        self.threshold = config.data.confidence_threshold
        self.tta = config.classifier.tta
        self.split = config.classifier.split
        self.n_ensemble = config.classifier.n_ensemble
        print(f"CharacterClassifier: method={self.method}, threshold={self.threshold}")
        assert self.split >= 1, "Current codes only support split >= 1"

    def __call__(self, images, character_regions, labels, confidences) -> Any:
        patches = [crop_img(images[r.image_index], r.box) for r in character_regions]

        labels, confidences = self.inference_with_classifier(patches, labels, confidences)

        return labels, confidences

    def inference_with_classifier(self, patches, labels, confidences):
        assert len(patches) == len(labels) == len(confidences)

        def train(data):
            return train_classifier(
                data,
                pretrained_model_path=self.config.classifier.pretrained_model_path,
                num_epochs=self.config.classifier.num_epochs,
            )

        train_data_indices = [
            i for i, l in enumerate(labels) if l is not None and confidences[i] >= self.threshold
        ]

        random.shuffle(train_data_indices)
        train_set_indices = np.array_split(train_data_indices, self.split)
        train_sets = [[(patches[i], labels[i]) for i in indices] for indices in train_set_indices]
        for train_set in train_sets:
            describe_train_data([d[1] for d in train_set], f"classifier (thr={self.threshold})")
        train_set_mapping = np.zeros(len(labels), dtype=int)
        for i, indices in enumerate(train_set_indices):
            train_set_mapping[indices] = i + 1

        models = [
            (set_id + 1, train(train_set))
            for set_id, train_set in enumerate(train_sets)
            for _ in range(self.n_ensemble)
        ]

        danbooru = "danbooru" in self.config.classifier.pretrained_model_path

        confs_all = defaultdict(list)
        for train_set_id, model in tqdm(models, desc="apply classifier"):
            confs_all[train_set_id].append(
                apply_classifier(patches, model, danbooru=danbooru, tta=self.tta)[1]
            )

        labels, confidences = [], []
        for i, train_set_id in enumerate(train_set_mapping):
            if self.n_ensemble > 1:
                conf_list = np.array(
                    [
                        confs_all[j][model_idx][i]
                        for j in confs_all.keys()
                        if j != train_set_id
                        for model_idx in range(self.n_ensemble)
                    ]
                )
            else:
                conf_list = np.array(
                    [confs_all[j][0][i] for j in confs_all.keys() if j != train_set_id]
                )
            average_confidence = conf_list.mean(axis=0)
            label = int(np.argmax(average_confidence))
            confidences.append(float(average_confidence[label]))
            labels.append(label)

        return labels, confidences


def describe_train_data(labels, memo):
    label_counts = defaultdict(int)
    for l in labels:
        label_counts[l] += 1
    print(f"{memo}: n={len(labels)}, label_counts={label_counts}")
