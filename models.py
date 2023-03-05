from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50, ResNet18_Weights, resnet18


def get_pretrained_model(config):
    if config["model"] == "resnet50":
        # weights with accuracy 80.858%
        pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return GalaxyResNet(config, pretrained_resnet)
    if config["model"] == "resnet18":
        pretrained_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return GalaxyResNet(config, pretrained_resnet)


class GalaxyResNet(nn.Module):
    def __init__(self, config, pretrained_resnet) -> None:
        super().__init__()

        self.frozen_layers = None
        self.unfrozen_layers = None
        self.split_forward_in_two(config, pretrained_resnet)

        self.fully_connected = build_fully_connected(
            config["model_architecture"]["fully_connected_hidden_layers"],
            pretrained_resnet.fc.in_features,
            config["model_architecture"]["nb_output_classes"],
        )
        self.freeze_beginning()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate_forward(x)
        x = self.end_forward(x)
        return x

    def split_forward_in_two(self, config, pretrained_resnet):
        frozen_half_list = []
        unfrozen_half_list = []
        freezing = True if config["last_frozen_layer"] != "" else False
        # looping on layers except the last one (fully connected)
        for layer_name in list(pretrained_resnet._modules.keys())[:-1]:
            if freezing:
                frozen_half_list.append(pretrained_resnet._modules[layer_name])
            else:
                unfrozen_half_list.append(pretrained_resnet._modules[layer_name])
            if layer_name == config["last_frozen_layer"]:
                freezing = False
                if config["delte_pretrained_model_ending"]:
                    break
        self.frozen_layers = nn.Sequential(*frozen_half_list)
        self.unfrozen_layers = nn.Sequential(*unfrozen_half_list)

    def intermediate_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frozen_layers(x)
        return x

    def end_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfrozen_layers(x)
        x = torch.flatten(x, 1)
        # we only replace resnet's fully connected with ours
        x = self.fully_connected(x)
        return x

    def freeze_beginning(self):
        self.frozen_layers.requires_grad_(False)

    def unfreeze_beginning(self):
        self.frozen_layers.requires_grad_(True)


class CustomModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.conv1 = torch.nn
        self.freeze_beginning()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate_forward(x)
        x = self.end_forward(x)
        return x

    def split_forward_in_two(self, config, pretrained_resnet):
        frozen_half_list = []
        unfrozen_half_list = []
        freezing = True if config["last_frozen_layer"] != "" else False
        # looping on layers except the last one (fully connected)
        for layer_name in list(pretrained_resnet._modules.keys())[:-1]:
            if freezing:
                frozen_half_list.append(pretrained_resnet._modules[layer_name])
            else:
                unfrozen_half_list.append(pretrained_resnet._modules[layer_name])
            if layer_name == config["last_frozen_layer"]:
                freezing = False
        self.frozen_layers = nn.Sequential(*frozen_half_list)
        self.unfrozen_layers = nn.Sequential(*unfrozen_half_list)

    def intermediate_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frozen_layers(x)
        return x

    def end_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfrozen_layers(x)
        x = torch.flatten(x, 1)
        # we only replace resnet's fully connected with ours
        x = self.fully_connected(x)
        return x

    def freeze_beginning(self):
        self.frozen_layers.requires_grad_(False)

    def unfreeze_beginning(self):
        self.frozen_layers.requires_grad_(True)


def build_fully_connected(fully_connected_hidden_layers, input_dim, output_dim):
    dimension_list = [input_dim] + fully_connected_hidden_layers + [output_dim]
    layers = OrderedDict()
    for i in range(len(dimension_list) - 1):
        if dimension_list[i] == "relu":
            layers[f"relu{i}"] = nn.ReLU()
        elif isinstance(dimension_list[i], int):
            j = 1
            while not isinstance(dimension_list[i + j], int):
                j += 1
            layers[f"fc{i}"] = nn.Linear(dimension_list[i], dimension_list[i + j])
        else:
            raise ValueError(
                f"Got invalid value in fully_connected_hidden_layers parameter : only ints \
                and 'relu' are accepted but got {dimension_list[i]} of type {type(dimension_list[i])}."
            )

    fully_connected = nn.Sequential(layers)
    return fully_connected
