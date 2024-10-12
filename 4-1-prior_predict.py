import csv
import os.path
import random

import pandas as pd

import torch
from sklearn.manifold import TSNE

from argparse import Namespace, ArgumentParser

import numpy as np

from chemprop.data.utils import get_data, split_data
from chemprop.features.featurization import MolGraph, BatchMolGraph
from chemprop.train.predict import predict_latent

from utils.model import PretrainModel
from utils.parsing import add_pretrain_args, add_prior_predict_args, modify_prior_predict_args


def predict_prior_similarity(args: Namespace):
    """
    Predict the feature similarity between different sample pairs.
    The results will be saved as dataname_prior_similarity.csv.
    """

    args.seed = 111

    args.save_dir = os.path.join('results',
                                 f'{args.prior}',
                                 f'{args.data_name}_checkpoints')

    args.prior_similarity_output_path = os.path.join(args.save_dir,
                                       f'similarity.csv')

    os.makedirs(args.save_dir, exist_ok=True)

    print(args)
    args.train_data_size = 128

    print(f'Running Predict Prior Predictive Testing... Seed {args.seed}')

    model = PretrainModel(args)

    # Load the contrastive prior
    if args.prior == 'CPBayesMPP+Prior':
        print(f'Loading pretrain MPNN Model...')
        model.encoder.load_state_dict(torch.load(args.pretrain_encoder_path))
        # model.header.load_state_dict(torch.load(args.pretrain_header_path))

    model = model.cuda()
    model.eval()

    data = get_data(path=args.data_path, args=args, logger=None)
    train_data, valid_data, test_data = split_data(data=data, balance=False,
                                                   split_type='scaffold', sizes=(0.5, 0.2, 0.3),
                                                   seed=111, args=args)

    # Define 3 types of sample pairs
    whole_aug_similarities = []  # Augmented pairs
    whole_same_similarities = []  # In-distribution pairs
    whole_diff_similarities = []  # Out-of-distribution pairs

    # Generate 10,000 samples for each pairs
    for base_data in train_data[:200]:
        base_samples = []
        augmented_samples = []
        same_samples = []
        diff_samples = []

        base_smiles = base_data.smiles
        m_base = MolGraph(base_smiles, args)

        for _ in range(50):
            base_samples.append(m_base)

            # Generate Augmented Samples
            m_aug = m_base.contra_sample()
            augmented_samples.append(m_aug)

            # Create a list excluding the base_data
            other_train_data = [data for data in train_data if data != base_data]

            # Generate In-Distribution Samples, ensuring base_data is not included
            same_smiles = random.choice(other_train_data).smiles
            m_same = MolGraph(same_smiles)
            same_samples.append(m_same)

            # Generate Out-of-Distribution Samples
            diff_smiles = random.choice(test_data.data).smiles
            m_diff = MolGraph(diff_smiles)
            diff_samples.append(m_diff)

        base_samples = BatchMolGraph(base_samples, args)
        augmented_samples = BatchMolGraph(augmented_samples, args)
        same_samples = BatchMolGraph(same_samples, args)
        diff_samples = BatchMolGraph(diff_samples, args)

        # Get the latent features of the samples
        z_base = model.header(model.encoder(base_samples))
        z_aug = model.header(model.encoder(augmented_samples))
        z_same = model.header(model.encoder(same_samples))
        z_diff = model.header(model.encoder(diff_samples))

        # Calculate the cosine similarity between different pairs of samples
        aug_similarities = torch.nn.functional.cosine_similarity(z_base, z_aug)
        same_similarities = torch.nn.functional.cosine_similarity(z_base, z_same)
        diff_similarities = torch.nn.functional.cosine_similarity(z_base, z_diff)

        # Transform the tensor result to list
        whole_aug_similarities.extend(aug_similarities.tolist())
        whole_same_similarities.extend(same_similarities.tolist())
        whole_diff_similarities.extend(diff_similarities.tolist())

    # Incoporate the list into a dictionary, where the keys will become the column names in the csv file
    data = {
        'Augmented': whole_aug_similarities,
        'Same Class': whole_same_similarities,
        'Different Class': whole_diff_similarities
    }

    # Transfer the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a csv file
    df.to_csv(args.prior_similarity_output_path, index=False)

    pass


def predict_prior_latent(args: Namespace):
    """
    Predict latent feature using t-SNE.
    """
    args.save_dir = os.path.join('results',
                                 f'{args.prior}',
                                 f'{args.data_name}_checkpoints')

    args.prior_latent_output_path = os.path.join(args.save_dir,
                                       f'latent.csv')

    args.seed = 123
    args.train_data_size = 128

    print(args)

    print(f'Runing Predict UC Latent Testing...')

    # print('Loading data')
    data = get_data(path=args.data_path, args=args, logger=None)

    model = PretrainModel(args)
    model = model.cuda()
    model.eval()

    if args.prior == 'CPBayesMPP+Prior':
        print(f'Loading pretrain MPNN Model...')
        model.encoder.load_state_dict(torch.load(args.pretrain_encoder_path))

    # Predict the latent features
    latent_fs = predict_latent(model=model,
                               data=data,
                               batch_size=args.batch_size)
    latent_fs = np.array(latent_fs)

    # Reduce the dimension of the latent features to 2D using t-SNE
    t_sne = TSNE(n_components=2)
    latent_fs = t_sne.fit_transform(latent_fs)

    smiles, labels = data.smiles(), data.targets()
    smiles, labels = np.array(smiles), np.array(labels).squeeze()

    # Write predictions
    with open(args.prior_latent_output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['smiles',
                  f'label',
                  f'T-SNE-x',
                  f'T-SNE-y',
                  ]

        writer.writerow(header)

        for i in range(len(labels)):
            row = []

            row.append(smiles[i])
            row.append(labels[i])
            row.append(latent_fs[i][0])
            row.append(latent_fs[i][1])

            writer.writerow(row)

    return 0


def predict_prior(args: Namespace):
    if args.predict_type == 'similarity':
        predict_prior_similarity(args)
    elif args.predict_type == 'latent':
        predict_prior_latent(args)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_pretrain_args(parser)
    add_prior_predict_args(parser)

    args = parser.parse_args()

    modify_prior_predict_args(args)

    predict_prior(args)