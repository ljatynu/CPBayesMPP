import csv
import os.path

import warnings

from argparse import Namespace, ArgumentParser

import numpy as np

from chemprop.data.utils import get_data, split_data
from chemprop.train.evaluate import evaluate
from chemprop.train.predict import predict
from chemprop.utils import load_scalers, load_args, load_checkpoint, save_checkpoint, get_metric_func

from rdkit import RDLogger

from utils.metric import calculate_uncertainty_scaler
from utils.misc import save_prediction
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

    all_scores = []

    for seed in args.seeds:
        # print(f'Runing Uncertainty Predict... Seed {seed}')

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
                                            f'preds.csv')

        scaler = load_scalers(args.checkpoint_path)

        train_args = load_args(args.checkpoint_path)

        # Update args with training arguments
        for key, value in vars(train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

        test_data = get_data(path=args.data_path, args=args, logger=None)

        # args.split_sizes = [0.5, 0.2, 0.3]

        # Get test data under the same random seed
        _, val_data, test_data = split_data(data=test_data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed,
                                args=args)

        model = load_checkpoint(args.checkpoint_path, current_args=args, cuda=args.cuda)

        test_score, preds, ale_uncs, epi_uncs = evaluate(
            model=model,
            data=test_data,
            num_tasks=args.num_tasks,
            metric_func=get_metric_func(metric=args.metric),
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            logger=None,
            sampling_size=args.sampling_size,
            retain_predict_results=True
        )

        all_scores.append(test_score)
        print(f'Seed {args.seed} test {args.metric} = {test_score:.6f}')

        # Save the results
        save_prediction(args,
                        smiles=test_data.smiles(),
                        labels=test_data.targets(),
                        model_preds=preds,
                        ale_uncs=ale_uncs,
                        epi_uncs=epi_uncs)

    all_scores = np.array(all_scores)
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)
    print(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_predict_args(parser)
    args = parser.parse_args()

    # For debugging
    # args.data_name = 'sider'  # delaney, freesolv
    # args.train_strategy = 'BayesMPP'  # BayesMPP, CPBayesMPP
    # args.sampling_size = 100
    # args.split_type = 'scaffold'  # random, scaffold

    modify_predict_args(args)

    cross_validate_uncertainty_predict(args)