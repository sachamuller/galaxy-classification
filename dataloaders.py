import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        if len(images) != len(labels):
            raise ValueError(
                f"Images and labels should have the same length but got {len(images)} and {len(labels)}."
            )
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_dataset() -> ImageDataset:
    with h5py.File("data/Galaxy10_DECals.h5", "r") as F:
        images = np.array(F["images"])
        labels = np.array(F["ans"])

    return ImageDataset(images, labels)


def get_dataloaders(config, shuffle=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = get_dataset()

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
    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")

    train, valid, test = get_dataloaders(config)
    print(train)
