import torch
import os
import matplotlib.pyplot as plt
from dataloaders import ImageDataset, get_dataloaders, get_dataset
from intermediate_forward import ActivationMapsDataset, get_intermediate_dataset
from models import get_pretrained_model
from finetune import load_model_from_checkpoint
from tqdm import tqdm
import configue

def build_confusion_matrix(model, device, dataloader, intermediate_forward, nb_classes):
    
    result = torch.zeros((nb_classes, nb_classes))

    model.eval()  # switch to eval mode

    for batch in tqdm(dataloader):
        if not intermediate_forward:
            images, labels = batch
            labels = labels.to(device)

            # compute output
            predictions = model(images)
        else:
            activation_maps, labels = batch
            labels = labels.to(device)

            # compute output
            predictions = model.end_forward(activation_maps)
        
        predicted_labels = predictions.argmax(dim=1)
        labels = labels.long() 
        for i in range(len(labels)) : 
            result[predicted_labels[i], labels[i]] += 1
    
    return result

def plot_confusion_matrix(confusion_matrix, save:bool=True, folder:str="", show:bool=False, weighted:bool=False):
    plt.figure()
    plt.matshow(confusion_matrix)
    plt.colorbar()
    plt.xlabel("True labels")
    plt.ylabel("Predicted labels")
    if save : 
        plt.savefig(os.path.join(folder, f"confusion_matrix{'_weighted' if weighted else ''}.png"))
    if show : 
        plt.show()

def log_confusion_matrix(confusion_matrix, folder, weighted:bool=False):
    with open(os.path.join(folder, f"confusion_matrix{'_weighted' if weighted else ''}.txt"), "w+") as f : 
        f.write(str(confusion_matrix))

def evaluate_only(model_path:str, which_dataset:str="validation", config:dict=None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    experiment_folder_path = "/".join(folder for folder in model_path.split("/")[:-1])
    
    if config is None : 
        config = configue.load(os.path.join(experiment_folder_path, "config.yaml"))

    model = get_pretrained_model(config)
    model.to(device)
    model = load_model_from_checkpoint(model, model_path)

    intermediate_forward = config["intermediate_forward"]
    if not intermediate_forward:
        dataset: ImageDataset = get_dataset(config, allow_sample=False)
    else:
        dataset: ActivationMapsDataset = get_intermediate_dataset(config, model, device, allow_sample=False)

    train_dataset, validation_dataset, test_dataset, train_loader, validation_loader, test_loader = get_dataloaders(config, dataset, also_get_datasets=True)
    if which_dataset =="train" : 
        loader = train_loader
        subset = train_dataset
    elif which_dataset=="validation" : 
        loader = validation_loader
        subset = validation_dataset
    elif which_dataset=="test" : 
        loader = test_loader
        subset = test_dataset
    else : 
        raise ValueError(f"which dataset can take values 'train', 'validation' or 'test' only, got {which_dataset}")

    confusion_matrix = build_confusion_matrix(model, device, loader, intermediate_forward, config["model_architecture"]["nb_output_classes"])
    plot_confusion_matrix(confusion_matrix, save=True, folder=experiment_folder_path)
    log_confusion_matrix(confusion_matrix, experiment_folder_path)

    labels_frequency = torch.zeros(config["model_architecture"]["nb_output_classes"])
    for i, label in enumerate(dataset.labels) : 
        if i in subset.indices : 
            labels_frequency[label] += 1

    weighted_confusion_matrix = confusion_matrix/labels_frequency

    plot_confusion_matrix(weighted_confusion_matrix, save=True, folder=experiment_folder_path, weighted=True)
    log_confusion_matrix(weighted_confusion_matrix, experiment_folder_path, weighted=True)


if __name__ == "__main__" : 
    # Fine tuning best model
    evaluate_only("experiments/23-03-05_00:58:03/best_model.pth")
