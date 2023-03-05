from typing import Tuple

import h5py
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


class ImageDataset(Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        if len(images) != len(labels):
            raise ValueError(
                f"Images and labels should have the same length but got {len(images)} and {len(labels)}."
            )
        self.images = images
        self.labels = labels
        self.tensor_converter = transforms.ToTensor()
        self.resizer = transforms.Resize(224)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.tensor_converter(image)
        image = self.resizer(image)

        label = self.labels[index]
        label = int(label)  # label is originally numpy int

        return image, label

    def __len__(self):
        return len(self.labels)

    def truncate_beginning(self, start_index):
        self.images = self.images[start_index:]
        self.labels = self.labels[start_index:]


def get_dataset(config, allow_sample=True) -> ImageDataset:
    with h5py.File("data/Galaxy10_DECals.h5", "r") as F:
        images = np.array(F["images"])
        labels = np.array(F["ans"])

    if allow_sample and config["dataset_percentage"] < 1.0:
        random.seed(config["seed"])
        selected_index = random.sample(
            [i for i in range(len(images))],
            int(config["dataset_percentage"] * len(images)),
        )
        images = images[selected_index]
        labels = labels[selected_index]

    return ImageDataset(images, labels)


def get_dataloaders(
    config, dataset, shuffle=True, also_get_datasets=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    val_prop = config["validation_dataset_proportion"]
    test_prop = config["test_dataset_proportion"]
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [1 - val_prop - test_prop, val_prop, test_prop],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    batch_size = config["batch_size"]

    if (
        len(train_dataset) == 0
        or len(validation_dataset) == 0
        or len(test_dataset) == 0
    ):
        raise ValueError(
            "The dataset is too small to be split correctly in a training, validation and test \
            dataset. Please consider increasing the value of the script_parsing_model.dataset_percentage parameter."
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    if also_get_datasets :
        return train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader
    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")

    dataset = get_dataset(config)
    train, valid, test = get_dataloaders(config, dataset)
    print(train)
