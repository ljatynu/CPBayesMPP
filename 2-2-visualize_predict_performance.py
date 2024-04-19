import pandas as pd

from argparse import Namespace, ArgumentParser

import statistics

from utils.parsing import add_visualize_metric_args, modify_visualize_metric_args


def visualize_metric(args: Namespace):
    """
    Calculate mean and standard variance of metrics.

    :param args: Arguments.
    :return: None
    """
    print(f'Beginning Testing {args.data_name} {args.train_strategy}')

    # Calculate statistics of RMSE for regression task
    if args.dataset_type == 'regression':
        rmse_list = []
        r_2_list = []

        for seed, metric_pred_path in zip(args.seeds, args.metric_path_list):
            df = pd.read_csv(metric_pred_path)
            rmse = df['RMSE'].values.astype(float)[0]
            r_2 = df['R_2'].values.astype(float)[0]

            print(f'Seed: {seed}, RMSE: {rmse: .2f}, r_2:{r_2: .2f}')
            rmse_list.append(rmse)
            r_2_list.append(r_2)

        mean_rmse = statistics.mean(rmse_list)
        std_rmse = statistics.stdev(rmse_list)

        mean_r_2 = statistics.mean(r_2_list)
        std_r_2 = statistics.stdev(r_2_list)

        print(f'RMSE mean = {mean_rmse: .2f}, std = {std_rmse: .2f}\n'
              f'R_2 mean = {mean_r_2: .2f}, std = {std_r_2: .2f}')

    # Calculate ROC-AUC of RMSE for regression task
    elif args.dataset_type == 'classification':
        roc_auc_list = []

        for seed, metric_pred_path in zip(args.seeds, args.metric_path_list):
            df = pd.read_csv(metric_pred_path)
            roc_auc = df['ROC-AUC'].values.astype(float)[0] * 100

            print(f'Seed: {seed}, ROC_AUC: {roc_auc: .1f}')
            roc_auc_list.append(roc_auc)

        mean_roc_auc = statistics.mean(roc_auc_list)
        std_roc_auc = statistics.stdev(roc_auc_list)

        print(f'ROC_AUC mean = {mean_roc_auc: .1f}, std = {std_roc_auc: .1f}')


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_metric_args(parser)
    args = parser.parse_args()

    modify_visualize_metric_args(args)

    visualize_metric(args)