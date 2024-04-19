import csv
import os.path

import warnings

from argparse import Namespace, ArgumentParser

import numpy as np

from chemprop.data.utils import get_data, split_data
from chemprop.train.evaluate import evaluate_predictions
from chemprop.train.predict import predict
from chemprop.utils import load_scalers, load_args, load_checkpoint

from utils.metric import rmse

from sklearn.metrics import r2_score, roc_auc_score
from rdkit import RDLogger

from utils.parsing import add_predict_args, modify_predict_args

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def cross_validate_predict(args: Namespace):
    """
    Evaluate the predict performance (RMSE / AUC-ROC) for Regreesion / Classification dataset.
    The results will be saved as data_name_metric.csv in the save_dir.

    :param args: Arguments.
    :return: None
    """
    save_dir = args.save_dir

    # Cross validation
    for seed in args.seeds:
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

        # Output Directory
        args.metric_path = os.path.join(args.save_dir,
                                        f'{args.data_name}_metric.csv')

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

        model_preds, ale_uncs, epi_uncs = predict(
            model=model,
            data=data,
            batch_size=args.batch_size,
            scaler=scaler,
            sampling_size=args.sampling_size
        )

        if args.dataset_type == 'regression':
            model_preds, ale_uncs, epi_uncs = \
                np.array(model_preds).squeeze(), np.array(ale_uncs).squeeze(), np.array(epi_uncs).squeeze()

            smiles, labels = data.smiles(), data.targets()
            smiles, labels = np.array(smiles), np.array(labels).squeeze()

            # Calculate regression metric RMSE
            test_rmse, test_r_2 = rmse(model_preds, labels), r2_score(model_preds, labels)
            print_info = f'Seed <{args.seed}>. ' \
                         f'Test Results: ' \
                         f'Train Strategy <{args.train_strategy}> ' \
                         f'Dataset <{args.task_names}>; RMSE <{test_rmse}>; R_2 <{test_r_2}>; '

            print(print_info)

            # Save the results
            with open(args.metric_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['RMSE', 'R_2']
                writer.writerow(header)
                writer.writerow([test_rmse, test_r_2])

        if args.dataset_type == 'classification':
            smiles, labels = data.smiles(), data.targets()

            test_roc_auc = evaluate_predictions(
                preds=model_preds,
                targets=labels,
                num_tasks=args.num_tasks,
                metric_func=roc_auc_score,
                dataset_type='classification',
                logger=None,
            )

            # Calculate classification metric AUC-ROC
            test_roc_auc = np.nanmean(test_roc_auc)

            print_info = f'Seed <{args.seed}>' \
                         f'Test Results: ' \
                         f'Train Strategy <{args.train_strategy}>' \
                         f'Dataset <{args.task_names}>; ROC-AUC <{test_roc_auc}>; '

            print(print_info)

            # Save the results
            with open(args.metric_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['ROC-AUC']
                writer.writerow(header)
                writer.writerow([test_roc_auc])


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_predict_args(parser)
    args = parser.parse_args()

    modify_predict_args(args)

    cross_validate_predict(args)