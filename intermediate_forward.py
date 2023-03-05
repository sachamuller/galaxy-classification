import os
import h5py
import numpy as np

import torch
import random
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from dataloaders import get_dataset



def get_intermediate_dataset(
    config: dict,
    model: nn.Module,
    device: torch.device,
    save_freq: int = 500,
):
    activation_maps_path = os.path.join(
        config["paths"]["intermediate_dataset_folder"],
        f'{config["paths"]["intermediate_dataset_name"]}_{config["model"]}_{config["last_frozen_layer"]}.h5',
    )

    if os.path.exists(activation_maps_path):
        F = h5py.File(activation_maps_path, "r")
        all_activation_maps = torch.from_numpy(np.array(F["activation_maps"]))
        all_labels = np.array(F["ans"])
        F.close()
        # to find where is the last computed embedding (in case program was stopped in the middle),
        # we suppose that if the sum of the composants of the embedding equals 0, it wasn't computed
        sum_by_activation_map = list(all_activation_maps.sum(axis=(1, 2, 3)))
        if 0.0 not in sum_by_activation_map:
            # means the embeddings matrix is already completely computed
            return get_dataset_sample(config, all_activation_maps, all_labels)

        # If we need to continue from the middle :
        image_dataset = get_dataset(config, allow_sample=False)
        start_index = sum_by_activation_map.index(0.0)
        image_dataset.truncate_beginning(start_index)

    else:
        image_dataset = get_dataset(config, allow_sample=False)
        os.makedirs(config["paths"]["intermediate_dataset_folder"], exist_ok=True)
        all_activation_maps = torch.zeros(
            len(image_dataset), *config["intermediate_activation_map_size"]
        )
        start_index = 0

    batch_size = config["batch_size"]

    loader = DataLoader(
        dataset=image_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_idx, batch in tqdm(
        enumerate(loader), total=len(image_dataset) // batch_size
    ):
        images, labels = batch
        images.to(device)
        labels = labels.to(device)
        current_batch_size = len(labels)

        # compute activation maps
        activation_maps = model.intermediate_forward(images)

        all_activation_maps[
            start_index
            + batch_idx * batch_size : start_index
            + batch_idx * batch_size
            + current_batch_size
        ] = activation_maps

        if batch_idx % save_freq == 0:
            save_intermediate_dataset_as_h5(
                activation_maps_path, all_activation_maps, all_labels
            )

    save_intermediate_dataset_as_h5(
        activation_maps_path, all_activation_maps, all_labels
    )

    return get_dataset_sample(config, all_activation_maps, all_labels)

def get_dataset_sample(config, activation_maps, labels, allow_sample=True):
    if allow_sample and config["dataset_percentage"] < 1.0:
        random.seed(config["seed"])
        selected_index = random.sample(
            [i for i in range(len(activation_maps))],
            int(config["dataset_percentage"] * len(activation_maps)),
        )
        activation_maps = activation_maps[selected_index]
        labels = labels[selected_index]
    return ActivationMapsDataset(activation_maps, labels)


def save_intermediate_dataset_as_h5(
    activation_maps_path, all_activation_maps, all_labels
):
    F = h5py.File(activation_maps_path, "w")
    F.create_dataset("activation_maps", data=all_activation_maps.detach().numpy())
    F.create_dataset("ans", data=all_labels)
    F.close()


class ActivationMapsDataset(Dataset):
    def __init__(
        self,
        activation_maps: np.array,
        labels,
    ):
        self.activation_maps = activation_maps
        self.labels = labels

    def __getitem__(self, index):
        activation_map = self.activation_maps[index]

        label = self.labels[index]
        label = int(label)  # label is originally numpy int

        return activation_map, label

    def __len__(self):
        return len(self.labels)
