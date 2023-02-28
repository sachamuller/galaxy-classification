from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def get_pretrained_model(config):
    if config["model"] == "resnet50":
        # weights with accuracy 80.858%
        pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return GalaxyResNet(config, pretrained_resnet)


class GalaxyResNet(nn.Module):
    def __init__(self, config, pretrained_resnet) -> None:
        super().__init__()

        self.conv1 = pretrained_resnet.conv1
        self.bn1 = pretrained_resnet.bn1
        self.relu = pretrained_resnet.relu
        self.maxpool = pretrained_resnet.maxpool

        self.layer1 = pretrained_resnet.layer1
        self.layer2 = pretrained_resnet.layer2
        self.layer3 = pretrained_resnet.layer3
        self.layer4 = pretrained_resnet.layer4

        self.avgpool = pretrained_resnet.avgpool

        self.fully_connected = build_fully_connected(
            config["model_architecture"]["fully_connected_hidden_layers"],
            pretrained_resnet.fc.in_features,
            config["model_architecture"]["nb_output_classes"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate_forward(x)
        x = self.end_forward(x)
        return x

    def intermediate_forward(self, x: torch.Tensor) -> torch.Tensor:
        # beginning is copy pasted from pytorch's implementation of ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def end_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # we only replace resnet's fully connected with ours
        x = self.fully_connected(x)
        return x

    def freeze_beginning(self, unfreeze=False):
        for module in [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]:
            module.requires_grad_(unfreeze)


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
