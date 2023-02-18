import os
from datetime import datetime
from typing import Dict, Tuple, Union

import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from dataloaders import ImageDataset, get_dataloaders, get_dataset
from models import GalaxyResNet, get_pretrained_model
from monitoring import AverageMeter, Monitor


def fine_tune_parsing_model(
    config: dict,
) -> GalaxyResNet:
    """Creates a model and trains it on a dataset according to the parameters found
    in the config file.

    Args:
        config (dict): config yaml file imported as a dict

    Returns:
        BertClassifier| SentenceTransformerClassifier: the trained model
    """

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    model = get_pretrained_model(config)
    model.to(device)
    if config["load_checkpoint"]:
        model = load_model_from_checkpoint(model, config["checkpoint_path"])
    dataset: ImageDataset = get_dataset()

    train_loader, validation_loader, _ = get_dataloaders(config, dataset)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"]["initial_value"],
        weight_decay=config["weight_decay"],
    )
    if config["learning_rate"]["decrease_on_plateau"]:
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            patience=config["learning_rate"]["patience"],
        )

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    metrics = {
        "top1": MulticlassAccuracy(
            num_classes=config["model_architecture"]["nb_output_classes"],
            top_k=1,
            average="micro",
        ),
        "top3": MulticlassAccuracy(
            num_classes=config["model_architecture"]["nb_output_classes"],
            top_k=3,
            average="micro",
        ),
    }

    experiment_folder_path = get_experiment_folder_name(config)
    monitor = Monitor(
        ["train_loss", "train_top1", "val_loss", "val_top1", "learning_rate"],
        experiment_folder_path,
    )

    nb_epochs = config["nb_epochs"]

    for epoch in range(nb_epochs):
        train_loss, train_top1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            metrics,
            device,
            epoch,
        )
        val_loss, val_top1 = validate(
            model,
            validation_loader,
            criterion,
            metrics,
            device,
            epoch,
        )
        if config["learning_rate"]["decrease_on_plateau"]:
            scheduler.step(val_loss.avg)

        monitor.update_data(
            [
                train_loss.avg,
                train_top1.avg,
                val_loss.avg,
                val_top1.avg,
                optimizer.param_groups[0]["lr"],
            ]
        )
        monitor.plot_data()
        monitor.log_data()
        if val_top1.avg == max(monitor.all_data[3]):
            torch.save(
                model.state_dict(),
                os.path.join(experiment_folder_path, "best_model.pth"),
            )

        torch.save(
            model.state_dict(), os.path.join(experiment_folder_path, "last_model.pth")
        )

    # load best model to return it
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=os.path.join(experiment_folder_path, "best_model.pth"),
    )
    return model


def get_experiment_folder_name(config: dict) -> str:
    """Creates a folder into which the files monitoring the fine-tuning will be kept, and
    returns the path of this folder. The folder's name is the current date (yy-mm-dd_hh:mm:ss).

    Args:
        config (dict): config yaml file imported as a dict

    Returns:
        str: path of the folder meant to contain the fine tuning results
    """
    results_folder = config["paths"]["experiments_folder"]
    os.makedirs(results_folder, exist_ok=True)
    new_folder_path = os.path.join(
        results_folder, datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    )
    os.makedirs(new_folder_path)
    with open(os.path.join(new_folder_path, "config.yaml"), 'w+') as f:
        yaml.dump(config, f)
    return new_folder_path


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.SGD,
    criterion: torch.nn.CrossEntropyLoss,
    metrics: Dict[str, MulticlassAccuracy],
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> Tuple[AverageMeter, AverageMeter]:
    """Trains the model on the training dataset once (one epoch) and returns the metrics
    on this epoch.

    Args:
        model (torch.nn.Module): model to be trained
        train_loader (DataLoader): dataloader containing training data
        optimizer (torch.optim.SGD): optimizer performing the stochastic gradient descent
        criterion (torch.nn.CrossEntropyLoss): criterion that computes the loss
        device (torch.device): can be 'cpu' or 'cuda:i' according to gpus available
        epoch (int): number of epochs already computed
        print_freq (int, optional): printing the metrics every print_freq batches. Defaults to 10.

    Returns:
        Tuple[AverageMeter, AverageMeter]: the loss and the top1 accuracy
    """

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train()  # switch to train mode

    for batch_idx, batch in enumerate(train_loader):
        images, labels = batch
        labels = labels.to(device)
        batch_size = len(images)

        # compute output
        predictions = model(images)

        loss = criterion(predictions, labels)

        # compute accuracy
        prec1 = metrics["top1"](predictions, labels) * 100
        prec3 = metrics["top3"](predictions, labels) * 100

        losses.update(loss.data.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top3.update(prec3.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print(
                "[{epoch}]\t"
                "Train: [{0}/{1}]\t"
                "Loss {loss.avg:.4f}\t"
                "Prec@1 {top1.avg:.3f}\t"
                "Prec@3 {top3.avg:.3f}".format(
                    batch_idx,
                    len(train_loader),
                    epoch=epoch,
                    loss=losses,
                    top1=top1,
                    top3=top3,
                )
            )

    return losses, top1


def validate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    criterion: torch.optim.SGD,
    metrics: Dict[str, MulticlassAccuracy],
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> Tuple[AverageMeter, AverageMeter]:
    """Computes the metrics on the validation dataset.

    Args:
        model (torch.nn.Module): model to be trained
        validation_loader (DataLoader): dataloader containing validation data
        criterion (torch.optim.SGD): criterion that computes the loss
        device (torch.device): can be 'cpu' or 'cuda:i' according to gpus available
        epoch (int): number of epochs already computed
        print_freq (int, optional): printing the metrics every print_freq batches. Defaults to 10.

    Returns:
        Tuple[AverageMeter, AverageMeter]: the loss and the top1 accuracy
    """

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()  # switch to eval mode

    for batch_idx, batch in enumerate(validation_loader):
        images, labels = batch
        labels = labels.to(device)
        batch_size = len(images)

        # compute output
        predictions = model(images)
        loss = criterion(predictions, labels)

        # compute accuracy
        prec1 = metrics["top1"](predictions, labels) * 100
        prec3 = metrics["top3"](predictions, labels) * 100

        losses.update(loss.data.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top3.update(prec3.item(), batch_size)

        if batch_idx % print_freq == 0:
            print(
                "[{epoch}]\t"
                "Validation: [{0}/{1}]\t"
                "Loss {loss.avg:.4f}\t"
                "Prec@1 {top1.avg:.3f}\t"
                "Prec@3 {top3.avg:.3f}".format(
                    batch_idx,
                    len(validation_loader),
                    epoch=epoch,
                    loss=losses,
                    top1=top1,
                    top3=top3,
                )
            )

    return losses, top1


def load_model_from_checkpoint(model: GalaxyResNet, checkpoint_path: str) -> None:
    """Loads the weight of the model from a saved checkpoint file.

    Args:
        model (Union[BertClassifier, SentenceTransformerClassifier]): model onto which the weights will be loaded
        checkpoint_path (str): the path of the saved file, can be a '.pt'
            or a '.pth' file.

    Raises:
        ValueError: the file given in checkpoint_path does not exist
        ValueError: the file given in checkpoint_path is not a '.pt' or a '.pth'
        ValueError: the model loaded in the .pt file is not of the same type as model
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Given checkpoint path does not exist : {checkpoint_path}")
    if checkpoint_path[-4:] == ".pth":
        model.load_state_dict(torch.load(checkpoint_path))
        return model
    elif checkpoint_path[-3:] == ".pt":
        loaded_model = torch.load(checkpoint_path)
        if type(model) != type(loaded_model):
            raise ValueError(
                f"Checkpoint model is not the same as the model indicated \
                in the parameters ({type(loaded_model)} vs. {type(model)}))"
            )
        return loaded_model
    else:
        raise ValueError(
            f"Checkpoint path is not a valid .pth or .pt file : {checkpoint_path}"
        )


if __name__ == "__main__":
    import time

    import configue

    config = configue.load("config.yaml")

    start = time.time()

    fine_tune_parsing_model(config)

    print("Done. Time :", time.time() - start)
