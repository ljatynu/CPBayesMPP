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

from utils.parsing import add_active_train_args, modify_active_train_args

from rdkit import RDLogger

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def train(model: nn.Module,
          data: MoleculeDataset,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: NoamLR,
          args: Namespace, ) -> float:
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


def active_learning(args: Namespace):
    """
    Active learning on a dataset under a random seed (one time).
    """

    print(f'Training on Fold [{args.fold + 1} / {args.folds}]...')

    data = get_data(path=args.data_path, args=args)
    args.num_tasks = data.num_tasks()  # Number of tasks of the dataset
    args.task_names = get_task_names(args.data_path)  # Save all task labels to a list

    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                 seed=args.fold, args=args)

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

    # Get loss and metric function
    loss_func = get_loss_func(args)  # Heteroscedastic Loss for Regression
    metric_func = get_metric_func(metric=args.metric)  # RMSE for Regression

    # Build model
    print(f'Building model...')
    model = DownStreamModel(args)

    print(f'Number of parameters = {param_count(model):,}')

    # Load pretrained model when contrastive learning
    if args.train_strategy == 'CPBayesMPP+AL':
        print(f'Loading pretrain MPNN Model...')
        model.encoder.load_state_dict(torch.load(args.pretrain_encoder_path))
        # Record the weight of transferred contrastive learning variational parameters for KL divergence calculation.
        model.encoder.record_transfer_weight()
        model.encoder.reset_dropout_rate()

    print('Moving model to cuda')
    model = model.cuda()

    optimizer = build_optimizer(model, args)  # Returns Adam optimizer

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)  # Returns NoamLR learning rate scheduler

    # Best score indicator RMSE the lower the better, AUC-ROC the higher the better
    best_score = float('inf') if args.dataset_type == 'regression' else -float('inf')

    ### Define active learning step variables and subsample the tasks
    n_total = len(train_data)  # 513
    n_start = int(n_total * args.al_init_ratio)

    n_samples_per_run = np.linspace(n_start, args.al_end_ratio * n_total, args.n_loops + 1)
    n_samples_per_run = np.round(n_samples_per_run).astype(int)


    np.random.seed(args.fold)
    train_subset_inds_start = np.random.choice(n_total, n_start, replace=False)

    train_pool = train_data.sample_idxs(train_subset_inds_start)  # Initial training set 20% of the total train data
    train_data = train_data.remove_inds(train_subset_inds_start)  # Initial training pool 80% of the total train data

    best_model = None
    results = [args.fold]

    for lp in range(args.n_loops + 1):

        print(f'Loop [{lp + 1}/{args.n_loops + 1}] with with {n_samples_per_run[lp] / n_total:.2f} % samples')

        for epoch in range(args.epochs):
            train_score = train(
                model=model,
                data=train_pool,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )

            # Validate on the valid set
            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                sampling_size=args.sampling_size,
            )

            if (epoch + 1) % 10 == 0:
                # print(f'Epoch {epoch}: Train Loss = {train_score: .4f}, Validation {args.metric} = {val_scores: .4f}')
                print(
                    f'Fold [{args.fold + 1} / {args.folds}], Loop [{lp + 1}/{args.n_loops + 1}], Epoch [{epoch + 1}/{args.epochs}] Train Loss = {train_score: .5f}, Validation {args.metric} = {val_scores: .5f}')

            # Update best score
            best_score = update_best_score(args.dataset_type, val_scores, best_score)

            # Save the model when the metric is better
            if best_score == val_scores:
                best_model = model

        model = best_model

        test_score = evaluate(
            model=model,
            data=test_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            sampling_size=50,
        )

        print(f'Test {args.metric} = {test_score: .5f}')

        results.append(round(test_score, 5))

        if lp < args.n_loops:
            # Acquire the expanded samples
            _, preds, ale_unc, epi_unc = evaluate(
                model=model,
                data=train_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                sampling_size=50,
                retain_predict_results=True
            )

            epi_unc = [epi[0] for epi in epi_unc]

            selected_indices = None
            expand_size = n_samples_per_run[lp + 1] - n_samples_per_run[lp]

            # Randomly select samples
            if args.al_type == 'random':
                np.random.seed(args.fold)
                selected_indices = random.sample(range(len(train_data)), expand_size)
            # Select samples based on epistemic uncertainty
            elif args.al_type == 'explorative':
                selected_indices = np.argsort(epi_unc)[-expand_size:]
            # Select samples based on true error
            elif args.al_type == 'oracle':
                targets = train_data.targets()
                if args.dataset_type == 'regression':
                    targets = scaler.inverse_transform(targets)
                selected_indices = np.argsort(np.abs(np.array(targets).squeeze() - np.array(preds).squeeze()))[-expand_size:]

            train_pool = train_pool.expand(train_data.sample_idxs(selected_indices))  # Expand the training set
            train_data = train_data.remove_inds(selected_indices)   # Remove the expanded samples from the training pool

        else:
            break

    with open(args.result_path, 'a', newline='') as f:
        csv.writer(f).writerow(results)


def several_folds_al(args: Namespace):
    """Active learning on a dataset under several random seeds (several times)."""
    makedirs(args.save_dir)

    args.result_path = os.path.join(args.save_dir, 'result.csv')

    with open(args.result_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header of the .csv results file
        header = ['Folds/Expand_ratio'] + [ args.al_init_ratio + i * (args.al_end_ratio - args.al_init_ratio) / args.n_loops for i in range(args.n_loops + 1)]

        writer.writerow(header)

    for fold in range(args.folds):
        args.fold = fold

        # Run Training on current time
        active_learning(args)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_active_train_args(parser)
    args = parser.parse_args()

    modify_active_train_args(args)

    several_folds_al(args)
