import csv
import logging
import os

from argparse import Namespace, ArgumentParser
from pprint import pformat
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.data.utils import get_task_names, get_data, split_data, get_class_sizes
from chemprop.nn_utils import param_count, NoamLR
from chemprop.utils import create_logger, get_loss_func, get_metric_func, save_checkpoint, makedirs, build_optimizer, \
    build_lr_scheduler, load_checkpoint
from chemprop.train.evaluate import evaluate
from utils.misc import save_prediction

from utils.parsing import add_train_args, modify_train_args
from rdkit import RDLogger
import warnings

from utils.model import DownStreamModel
from utils.metric import update_best_score

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def train_step(model: nn.Module,
               data: MoleculeDataset,
               loss_func: Callable,
               optimizer: Optimizer,
               scheduler: NoamLR,
               args: Namespace, ) -> float:
    """
    Trains a model for an epoch.

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

    # Use whole data as a batch if it's small, for stability
    args.batch_size = len(data) if args.batch_size > len(data) else args.batch_size

    # Total number of iter samples
    num_iters = len(data) // args.batch_size * args.batch_size  # Number of iter samples and excluding the samples of last batch
    batch_num = len(data) // args.batch_size  # Number of batch

    running_loss = 0.0

    # Start iterating batch to train model
    for i in range(0, num_iters, args.batch_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])

        # smiles_batch = (list)[smiles0, smiles1, ..., smilesN]
        # target_batch = (list)[target0, target1, ..., targetN] (Normalized)
        smiles_batch, target_batch = mol_batch.smiles(), mol_batch.targets()

        # Run training model
        model.zero_grad()

        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).cuda()

        loss = 0.0

        # Output mu and variance of gaussian distribution for regression task
        if args.dataset_type == 'regression':
            means, logvars = model(smiles_batch)
            loss = loss_func(targets, means, logvars)
        # Output logits of categorical distribution for regression task
        elif args.dataset_type == 'classification':
            preds = model(smiles_batch)
            loss = loss_func(preds, targets)

        reg_loss = args.reg_acc.get_sum()  # Regularization Term of ELBO

        loss += args.kl_weight * reg_loss  # ELBO = Data Fitting Term + lamuda * Regularization Term

        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item()

    return running_loss / batch_num


def run_training(args: Namespace, logger: logging.Logger):
    """
    Train on a dataset with a random seed used to split the dataset. (One fold Training)

    :param args: Arguments.
    :param logger: Logger Saver.
    :return: None
    """
    info = logger.info  # info(message) will be save under logger.log file

    # Load dataset
    info('Loading data...')
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()  # Number of labels of dataset
    args.task_names = get_task_names(args.data_path)  # All tasks will be saved on a list

    # Split dataset with current random seed
    info(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                 seed=args.seed, args=args, logger=logger)
    args.train_data_size = len(train_data)

    info(f'Finish Loading and Splitting... '
         f'Total size = {len(data):,} | '
         f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Some classification datasets are multi-labeled,
    # and some are incomplete, therefore we print their detailed information.
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        info('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            info(f'{args.task_names[i]} '
                 f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    # Normalize labels of regression dataset to N(0,1)
    if args.dataset_type == 'regression':
        info('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)  # Heteroscedastic/BCE With Logits Loss for Regression/Classification
    metric_func = get_metric_func(metric=args.metric)  # RMSE/AUC-ROC for Regression/Classification

    args.pred_path = os.path.join(args.save_dir, 'preds.csv')

    # File path to save the model
    save_dir = os.path.join(args.save_dir, f'model')
    makedirs(save_dir)

    # Build model
    info(f'Building model...')
    model = DownStreamModel(args)
    info(f'Number of parameters = {param_count(model):,}')

    # Load pretrained model when contrastive learning
    if args.train_strategy == 'CPBayesMPP':
        info(f'Loading pretrain MPNN Model...')
        model.encoder.load_state_dict(torch.load(args.pretrain_encoder_path))
        # Record the weight of transferred contrastive learning variational parameters for KL divergence calculation.
        model.encoder.record_transfer_weight()
        model.encoder.reset_dropout_rate()

    info('Moving model to cuda')
    model = model.cuda()

    # Save the model befor training
    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, args)

    optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)  # NoamLR learning rate schedulers

    # Best score indicator: RMSE the lower the better, AUC-ROC the higher the better
    best_score = float('inf') if args.dataset_type == 'regression' else -float('inf')

    # Train on current epoch
    for epoch in range(args.epochs):
        # Gradient Decent (Variational Inference) on the train set
        train_score = train_step(
            model=model,
            data=train_data,
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
            logger=logger,
            sampling_size=args.sampling_size,
        )

        info(f'Epoch {epoch}: Train Loss = {train_score: .4f}, Validation {args.metric} = {val_scores: .4f}')

        # Update best score
        best_score = update_best_score(args.dataset_type, val_scores, best_score)

        # Save the model when the metric is better
        if best_score == val_scores:
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, args)

    # Load the best model and evaluate on the test set
    model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda)
    test_score, preds, ale_unc, epi_unc = evaluate(
            model=model,
            data=test_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            logger=logger,
            sampling_size=args.sampling_size,
            retain_predict_results=True
        )
    info(f'Seed {args.seed} test {args.metric} = {test_score:.6f}')

    # Save the results
    save_prediction(args,
                    smiles=test_data.smiles(),
                    labels=test_data.targets(),
                    model_preds=preds,
                    ale_uncs=ale_unc,
                    epi_uncs=epi_unc)

    return test_score


def cross_validate_train(args: Namespace):
    """K-Fold training on a dataset.

    :param args: Arguments.
    :return: None
    """
    save_dir = os.path.join(args.save_dir,
                             f'{args.data_name}_checkpoints')

    # Get Logger
    logger = create_logger(name='train', save_dir=save_dir)
    logger.info(pformat(vars(args)))

    all_scores = []

    for seed in args.seeds:
        args.seed = seed

        # Each seed splitting will be saved in a new folder .../seed_i/
        args.save_dir = os.path.join(save_dir,
                                     f'seed_{args.seed}')



        # Run Training on current seed
        test_score = run_training(args, logger)

        all_scores.append(test_score)

    all_scores = np.array(all_scores)
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)
    logger.info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_train_args(parser)
    args = parser.parse_args()

    modify_train_args(args)

    cross_validate_train(args)
