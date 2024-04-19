import csv

import os
import random
from argparse import Namespace, ArgumentParser
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.data.utils import get_task_names, get_data, split_data, get_class_sizes
from chemprop.nn_utils import param_count, NoamLR
from chemprop.train.evaluate import evaluate

from chemprop.utils import get_loss_func, get_metric_func, makedirs, build_optimizer, \
    build_lr_scheduler

import warnings

from utils.metric import update_best_score
from utils.model import DownStreamModel

from utils.parsing import add_active_train_args,modify_active_train_args

from rdkit import RDLogger

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def train_step(model: nn.Module,
               data: MoleculeDataset,
               loss_func: Callable,
               optimizer: Optimizer,
               scheduler: NoamLR,
               args: Namespace,) -> float:
    """
    Train a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :return: Train Loss on current epoch.
    """
    model.train()

    data.shuffle()

    # Don't use the last batch if it's small, for stability
    args.batch_size = len(data) if args.batch_size > len(data) else args.batch_size

    num_iters = len(data) // args.batch_size * args.batch_size  # Total number of iter sample
    batch_num = len(data) // args.batch_size

    running_loss = 0.0

    for i in range(0, num_iters, args.batch_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])

        # smiles_batch = (list)[smiles0, smiles1, ..., smilesN]
        # target_batch = (list)[target0, target1, ..., targetN] (Normalized)
        smiles_batch, target_batch = mol_batch.smiles(), mol_batch.targets()

        model.zero_grad()

        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).cuda()

        loss = 0.0

        # Get the loss function for different task type
        if args.dataset_type == 'regression':
            means, logvars = model(smiles_batch)
            loss = loss_func(targets, means, logvars)
        if args.dataset_type == 'classification':
            preds = model(smiles_batch)
            loss = loss_func(preds, targets)

        reg_loss = args.reg_acc.get_sum()  # Regularization term of ELBO

        loss += args.kl_weight * reg_loss

        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item()

    return running_loss / batch_num


def run_training(args: Namespace):
    """
    Active learning on a dataset under a random seed (one time).
    """

    print(f'Training on {args.time} Time...')

    data = get_data(path=args.data_path, args=args)
    args.num_tasks = data.num_tasks()  # Number of tasks of the dataset
    args.task_names = get_task_names(args.data_path)  # Save all task labels to a list

    train_data, val_data, test_data = split_data(data=data, split_type='random', sizes=(0.5, 0.2, 0.3),
                                                 seed=args.seed, args=args)

    args.train_data_size = len(train_data)

    if args.dataset_type == 'regression':
        # Normalize the labels of the regression dataset
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        # Labels after normalization
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Start with a initial training pool, which is selected randomly
    expand_size = int(args.active_ratio * len(train_data))
    train_data, active_data, _ = split_data(data=train_data, split_type='random', sizes= (args.init_ratio, 1 - args.init_ratio, 0),
                                             seed=args.seed, args=args)

    # Get loss and metric function
    loss_func = get_loss_func(args)  # Heteroscedastic Loss for Regression
    metric_func = get_metric_func(metric=args.metric)  # RMSE for Regression

    # Build model
    print(f'Building model...')
    model = DownStreamModel(args)

    print(f'Number of parameters = {param_count(model):,}')

    # Load pretrained model when contrastive learning
    if args.train_strategy == 'cl':
        print(f'Loading pretrain MPNN Model...')
        model.encoder.load_state_dict(torch.load(args.pretrain_encoder_path))

    print('Moving model to cuda')
    model = model.cuda()

    optimizer = build_optimizer(model, args)  # Returns Adam optimizer

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)  # Returns NoamLR learning rate scheduler

    # Best score indicator RMSE the lower the better, AUC-ROC the higher the better
    best_score = float('inf') if args.dataset_type == 'regression' else -float('inf')

    # Train the model with initial training pool
    print(f'Train the model with initial {args.init_ratio} of train data')
    for epoch in range(args.init_train_step):
        train_score = train_step(
            model=model,
            data=train_data,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        # info(f'Epoch {epoch}: Train Loss = {train_score: .4f}')

    # Expand the training pool and Retrain the model
    results = [args.time]

    for expand_step in range(10):
        # Acquire the expanded samples
        _, preds, ale_unc, epi_unc = evaluate(
            model=model,
            data=active_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            sampling_size=args.sampling_size,
            retain_predict_results=True
        )

        # Select Top-k highest epistemic uncertainty samples
        # total_unc = [ale[0] + epi[0] for ale, epi in zip(ale_unc, epi_unc)]
        # ale_unc = [ale[0] for ale in ale_unc]
        epi_unc = [epi[0] for epi in epi_unc]

        selected_indices = None
        # Randomly select samples
        if args.active_learning_type == 'random':
            selected_indices = random.sample(range(len(active_data)), expand_size)
        # Select samples based on epistemic uncertainty
        if args.active_learning_type == 'explorative':
            selected_indices = np.argsort(epi_unc)[-expand_size:]

        # Expend the training pool with expanded samples
        train_data = train_data.expand([active_data[i] for i in selected_indices])
        active_data = active_data.remove(selected_indices)

        current_ratio = args.init_ratio + args.active_ratio * (expand_step + 1)

        # Retrain the model with expanded training pool
        for epoch in range(args.active_train_step):
            expanded_train_score = train_step(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )

        # Test the model with re-trained model
        expanded_test_scores = evaluate(
            model=model,
            data=test_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            sampling_size=args.sampling_size,
        )

        # Update the best score
        best_score = update_best_score(args.dataset_type, expanded_test_scores, best_score)

        print(f'Expand_step {expand_step + 1}: '
             f'Train the model with active expanded {round(current_ratio,3)} of train data. '
             f'Expanded Test {args.metric} = {expanded_test_scores: .4f}, '
             f'Num of Train Pool/Remained Data [{len(train_data)}/{len(active_data)}]'
             )

        results.append(round(expanded_test_scores, 3))

    with open(args.result_path, 'a', newline='') as f:
        csv.writer(f).writerow(results)


def several_times_training(args: Namespace):
    """Active learning on a dataset under several random seeds (several times)."""
    makedirs(args.save_dir)

    save_dir = args.save_dir
    args.result_path = os.path.join(save_dir, 'result.csv')

    with open(args.result_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header of the .csv results file
        header = ['times/expand_ratio'] + [round((i+1) * args.active_ratio + args.init_ratio, 3) for i in range(10)]

        writer.writerow(header)

    for time in range(args.active_train_times):
        args.time = time

        # Run Training on current time
        run_training(args)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_active_train_args(parser)
    args = parser.parse_args()

    modify_active_train_args(args)

    several_times_training(args)
