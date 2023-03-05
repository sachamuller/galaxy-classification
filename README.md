# :telescope: Galaxy Classification
---

## :paperclip: Project description

The goal of this project is to build a model trained at classifying galaxy images.

---

## :hammer: Installation
1. Clone the current repository
2. We recommand creating a new virtual environment, then installing the required packages with : 
```
pip install -r requirements.txt
```
3. Download the [Galaxy10 DECals Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html).
4. Place the `Galaxy10_DECals.h5` in the `data` folder.
5. Everything is ready, the program can now be launched !


---
## :ferris_wheel: Usage

To train a model run :
```bash
python finetune.py
```
To compute the confusion matrix of a model and print its accuracy over a given dataste run, modify the end of the `evaluate.py` file and run : 
```bash
python evaluate.py
```

The other parameters of the config file are detailed below. 

---
## :art: Configuration file 

- `seed` : <font color='green'>int</font>, used to get reproducibility

### Dataset parameters
- `validation_dataset_proportion`: <font color='green'>float</font>, the percentage of the dataset used to build the validation set. Should be $<1.0$.
- `test_dataset_proportion`: <font color='green'>float</font>, the percentage of the dataset used to build the test set. Should be $<1.0$.
- `dataset_percentage` : <font color='green'>float</font>, select a random subset of the dataset. Should be $\leq 1.0$.
- `intermediate_forward` : <font color='green'>bool</font>, whether to save the intermediate activation maps and use them as a new dataset. Can only be used for transfer learning, meaning if set to true then `freeze_beginning` has to be true as well. The activation maps saved are the `last_frozen_layer` ones. 
- `freeze_beginning` :  <font color='green'>bool</font>, whether to allow the training of the beggining of the model or not.
- `delte_pretrained_model_ending` : <font color='green'>bool</font>, if true, all the convolutional layers after `last_frozen_layer` are removed.
- `intermediate_activation_map_size` : <font color='green'>List[int]</font>, the dimensions of the activation maps of `last_frozen_layer`

### Model parameters
- `model`:  <font color='green'>str</font>, name of the model
- `model_architecture` : 
    - `fully_connected_hidden_layers`: <font color='green'>int | "relu"</font>, the hidden layers of the fully connected model, currently accepts "relu" and ints only
    - `nb_output_classes` : <font color='green'>int</font>, number of classes in the dataset
- `load_checkpoint`: <font color='green'>bool</font>, whether to start the training from scratch or to load the results of a previous experiment
- `checkpoint_path` : <font color='green'>str</font>, path of the checkpoint

### Training parameters
- `batch_size`:  <font color='green'>int</font>, size of the minibatches used during training
- `learning_rate` : 
    - `initial_value`: <font color='green'>float</font>, the initial value of the learning rate
    - `decrease_on_plateau` : <font color='green'>bool</font>, whether to decrease the value of the learning rate when the validation loss doesn't decrease
    - `patience` :  <font color='green'>int</font>, how many epochs before decreasing the learning rate
- `weight_decay`: <font color='green'>float</font>, value of the weight decay
- `nb_epochs` : <font color='green'>int</font>, how many epochs the training will last

### Paths
- `paths` : 
    - `experiments_folder` : <font color='green'>str</font>, name of the folder containing the training results
    - `intermediate` : <font color='green'>str</font>, name of the folder containing the intermediate activation maps datasets 
    - `intermediate_dataset_name` : <font color='green'>str</font>, name of the intermediate datasets


---
## :rotating_light: Warning

- The intermediate datasets take a significative amount of storage: 7Go for the `layer4`, 14Go for `layer3` and 28Go for `layer2`.

---
## :japanese_ogre: Troubleshooting

- The code to create the intermediate datasets caused some "Out Of Memory" errors, and I couldn't figure out why. If you encounter the same problem, I coded a quick workaround on the open branch `messy-intermediate-forward` that runs the program for a bit, saves the results, then exists and relaunch the function, until all the activation maps have been computed. 
