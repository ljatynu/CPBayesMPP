import pickle

from tqdm import tqdm

from utils.parsing import add_pretrain_args, modify_pretrain_args

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.loss import NTXentLoss

import logging
import os
from argparse import Namespace, ArgumentParser
from pprint import pformat
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data
from chemprop.nn_utils import param_count
from chemprop.utils import create_logger, makedirs
from utils.model import PretrainModel

from chemprop.features import featurization


def pretrain_step(model: nn.Module,
                  pretrain_data: MoleculeDataset,
                  pretrain_loss_func: Callable,
                  optimizer: Optimizer,
                  args: Namespace) -> float:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param pretrain_data: A MoleculeDataset (or a list of MoleculeDatasets).
    :param pretrain_loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param args: Arguments.
    :return: Train Loss on current epoch.
    """
    model.train()

    # Total number of iter samples
    num_iters = len(pretrain_data) // args.batch_size * args.batch_size  # Number of total samples excluding the last batch
    batch_num = len(pretrain_data) // args.batch_size  # Number of batch

    iter_size = args.batch_size

    running_loss = 0.0

    pbar = tqdm(range(0, num_iters, iter_size), desc="Training")

    for i in pbar:
        # Load the data cache of contrastive sample memory to accelerate training
        if args.use_pretrain_data_cache:

            # Upgrade featurization.SMILES_TO_CONTRA_GRAPH
            batch_idx = i // iter_size
            smiles_to_contra_graph_cache_path \
                = os.path.join(args.pretrain_data_cache_path,
                               f'smiles_to_contra_graph_batch_{batch_idx}.pkl')

            with open(smiles_to_contra_graph_cache_path, 'rb') as f:
                featurization.SMILES_TO_CONTRA_GRAPH = pickle.load(f)

        # Prepare batch
        if i + args.batch_size > len(pretrain_data):
            break

        # smiles_batch = (list)[smiles0, smiles1, ..., smilesN]
        pretrain_smiles_batch = MoleculeDataset(pretrain_data[i:i + args.batch_size]).smiles()

        model.zero_grad()

        # feature vector batch of contrastive samples (i, j) and Regularization Term of ELBO
        zis, zjs, lc_reg = model(pretrain_smiles_batch)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = pretrain_loss_func(zis, zjs)  # Contrastive Loss (Data Fitting Term)
        loss += args.kl_weight * lc_reg  # ELBO = Data Fitting Term + lamuda * Regularization Term

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        current_loss = running_loss / (i // iter_size + 1)

        pbar.set_description(f"Epoch: {args.epoch}, Training - Loss: {current_loss:.4f}, Data Fitting: {loss.item():.4f}, Regularization: {lc_reg:.4f}")

    return running_loss / batch_num


def run_pretraining(args: Namespace, logger: logging.Logger):
    """
    Pre-train on a dataset.

    :param args: Arguments.
    :param logger: Logger Saver.
    :return: None
    """
    info = logger.info  # info(message) will be save under logger.log file

    info(pformat(vars(args)))

    info('Loading data...')

    # Load dataset
    pretrain_data = get_data(path=args.pretrain_data_path, args=args, logger=logger)

    # The regularization term of Concrete MC-Dropout
    # needs the length of the dataset as the regularization coefficient
    args.train_data_size = len(pretrain_data)  # (1000000)

    # File path to save the model
    makedirs(args.save_dir)

    # Build model
    info(f'Building model...')
    model = PretrainModel(args)
    info(model)
    info(f'Number of parameters = {param_count(model):,}')

    info('Moving model to cuda')
    model = model.cuda()

    # Contrastive Learning Loss which is calculated on contrastive samples batch (zis, zjs)
    nt_xent_criterion = NTXentLoss(args.batch_size, args.temperature, args.use_cosine_similarity)

    optimizer = torch.optim.Adam(
        model.parameters(), args.init_lr,
        weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warm_up,
        eta_min=0, last_epoch=-1
    )

    best_score = float('inf')

    for epoch in range(args.epochs):
        args.epoch = epoch

        # Gradient Decent (Variational Inference) on the train set
        train_score = pretrain_step(
            model=model,
            pretrain_data=pretrain_data,
            pretrain_loss_func=nt_xent_criterion,
            optimizer=optimizer,
            args=args,
        )

        info(f'Epoch {epoch}: Train Loss = {train_score}')

        # After few warm-up epochs,
        if args.epoch >= args.warm_up:
            scheduler.step()

        # Save the model when the metric (ELBO L_C for the contrastive learning phase) is better
        if train_score < best_score:
            best_score, best_epoch = train_score, epoch
            torch.save(model.encoder.state_dict(), os.path.join(args.save_dir, 'pretrain_encoder.pt'))
            torch.save(model.header.state_dict(), os.path.join(args.save_dir, 'pretrain_header.pt'))


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_pretrain_args(parser)
    args = parser.parse_args()

    modify_pretrain_args(args)

    # Get Logger
    logger = create_logger(name='pretrain', save_dir=args.save_dir)

    run_pretraining(args, logger)

