import logging
import math
import os
from argparse import Namespace
from typing import Callable, List, Union, Tuple

import torch
from torch.optim import Adam, Optimizer
from torch import nn

from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

from chemprop.data import StandardScaler
from utils.model import DownStreamModel

from chemprop.nn_utils import NoamLR



def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def create_logger(name: str, save_dir: str = None) -> logging.Logger:
    logger = logging.getLogger(name)

    # Clear handlers if already added
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create a stream handler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Print DEBUG and above level logs to the console
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        # Create another file handler to save INFO and above level logs to a file info.log
        fh_i = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fh_i.setLevel(logging.INFO)  # Save INFO and above level logs to a file

        # Add the file handler to the logger
        logger.addHandler(fh_i)

    return logger


def save_checkpoint(path: str,
                    model: Union[DownStreamModel],
                    scaler: StandardScaler = None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the dataset.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,

    }
    torch.save(state, path)

def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None) -> DownStreamModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :return: The loaded MoleculeModel.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args.dataset_type = current_args.dataset_type
        args.num_tasks = current_args.num_tasks

    args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = DownStreamModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            pass
            # debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            pass
            # debug(f'Pretrained parameter "{param_name}" '
            #       f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
            #       f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        # debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> StandardScaler:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the dataset scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None

    return scaler


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def heteroscedastic_loss(true, mean, log_var):
    """
    Compute the heteroscedastic loss for regression.

    :param true: A list of true values.
    :param mean: A list of means (output predictions).
    :param mean: A list of logvars (log of predicted variances).
    :return: Computed loss.
    """
    precision = torch.exp(-log_var)
    loss = precision * (true - mean)**2 + log_var
    return loss.mean()


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def get_loss_func(args: Namespace) -> Callable:
    """
    Gets the loss function of regression.

    :return: Heteroscedastic loss function.
    """

    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='mean')
    if args.dataset_type == 'regression':
        return heteroscedastic_loss

def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable:
    """
    Gets the RMSE Loss Function.

    :return: RMSE Loss Function
    """
    if metric == 'roc-auc':
        return roc_auc_score

    if metric == 'rmse':
        return rmse

    if metric == 'accuracy':
        return accuracy

    raise ValueError(f'Metric "{metric}" not supported.')


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, total_epochs: List[int] = None) -> NoamLR:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs],
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )

