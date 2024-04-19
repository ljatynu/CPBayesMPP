import csv
import os.path

import warnings

from argparse import Namespace, ArgumentParser

import numpy as np

from chemprop.data.utils import get_data, split_data
from chemprop.train.predict import predict
from chemprop.utils import load_scalers, load_args, load_checkpoint

from rdkit import RDLogger

from utils.parsing import add_predict_args, modify_predict_args

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def cross_validate_uncertainty_predict(args: Namespace):
    """
    Evaluate the predict uncertainty for Regreesion / Classification datasets.
    The results will be saved as uncertainty_pred.csv in the save_dir.

    :param args: Arguments.
    :return: None
    """
    save_dir = args.save_dir

    for seed in args.seeds:
        print(f'Runing Uncertainty Predict... Seed {seed}')

        args.seed = seed

        # Working Directory
        args.save_dir = os.path.join(save_dir,
                                     f'{args.train_strategy}',
                                     f'{args.split_type}_split',
                                     f'{args.data_name}_checkpoints',
                                     f'seed_{args.seed}')

        # Trained model
        args.checkpoint_path = os.path.join(args.save_dir,
                                            f'model',
                                            f'model.pt')

        args.output_path = os.path.join(args.save_dir,
                                            f'uncertainty_pred.csv')

        scaler = load_scalers(args.checkpoint_path)

        train_args = load_args(args.checkpoint_path)

        # Update args with training arguments
        for key, value in vars(train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

        data = get_data(path=args.data_path, args=args, logger=None)

        # Get test data under the same random seed
        _, _, data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed,
                                args=args)

        model = load_checkpoint(args.checkpoint_path, current_args=args, cuda=args.cuda)

        # Calculate the prediction uncertainty
        model_preds, ale_uncs, epi_uncs = predict(
            model=model,
            data=data,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size
        )

        smiles, labels = data.smiles(), data.targets()

        # When task number > 1, we use only the first task to evaluate uncertainty
        # (Especially on classification multi-label datasets)
        if args.num_tasks == 1:
            model_preds, ale_uncs, epi_uncs = \
                np.array(model_preds).squeeze(), np.array(ale_uncs).squeeze(), np.array(epi_uncs).squeeze()

            smiles, labels = np.array(smiles), np.array(labels).squeeze()
        else:
            model_preds, ale_uncs, epi_uncs = \
                np.array(model_preds)[:, 0], np.array(ale_uncs)[:, 0], np.array(epi_uncs)[:, 0]

            smiles, labels = np.array(smiles), np.array(labels)[:, 0]

        # Save the results
        with open(args.output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['smiles',
                      f'pred',
                      f'label',
                      f'ale_unc',
                      f'epi_unc',
                      ]

            writer.writerow(header)

            for i in range(len(labels)):
                row = []

                row.append(smiles[i])
                row.append(model_preds[i])
                row.append(labels[i])
                row.append(ale_uncs[i])
                row.append(epi_uncs[i])

                writer.writerow(row)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_predict_args(parser)
    args = parser.parse_args()

    modify_predict_args(args)

    cross_validate_uncertainty_predict(args)