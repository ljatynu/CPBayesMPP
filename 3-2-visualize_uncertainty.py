import os

import pandas as pd

from argparse import Namespace, ArgumentParser

from sklearn.utils import column_or_1d, check_consistent_length
from typing import Tuple

from utils.metric import rmse
from utils.parsing import add_visualize_uncertainty_args, modify_visualize_uncertainty_args

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

green = sns.color_palette("deep")[2]
blue = sns.color_palette("deep")[0]


def calculate_threshold_rmse(preds: np.array,
                             labels: np.array,
                             filter_indicator: np.array,
                             confidence_percentile: np.array) -> list:
    """
    Calculate and return the RMSE value list at different confidence percentiles.

    :param preds: Array of predictions.
    :param labels: Array of actual labels.
    :param filter_indicator: Indicator used to filter sample points.
    :param confidence_percentile: Which confidence percentiles to use to filter sample points.
    :return: List of RMSE values under different confidence percentiles.
    """
    sorted_indices = np.argsort(filter_indicator)

    sorted_preds = preds[sorted_indices]
    sorted_labels = labels[sorted_indices]

    rmse_values = []

    for cp in confidence_percentile:
        # Select data points with top (100 - p) confidence percentile
        ct_idx = int(len(sorted_indices) * ((100 - cp) / 100))

        selected_preds = sorted_preds[:ct_idx]
        selected_labels = sorted_labels[:ct_idx]
        top_k_percent_rmse = rmse(selected_preds, selected_labels)

        # Return RMSE of selected data points
        rmse_values.append(top_k_percent_rmse)

    return rmse_values


def calculate_auco_curve(cd_pred_path: str,
                         cl_pred_path: str,
                         confidence_percentile,
                         args: Namespace) -> Tuple[list, list, list]:
    """
    Calculate the uncertainty comparison curve for a unique seed (Regression task).

    :param cd_pred_path: File path of the prediction results from BayesMPP.
    :param cl_pred_path: File path of the prediction results from CPBayesMPP (our method).
    :param confidence_percentile: Confidence percentiles used to filter sample points.
    :param args: Namespace parameters, including data name, uncertainty type, etc.
    :return: Returns a list of RMSEs values for BayesMPP, a list of RMSEs values for CPBayesMPP, and a list of RMSEs values for the Oracle curve.
    """
    pred_paths = [cd_pred_path, cl_pred_path]
    rmse_curves = [[], []]

    for i, pp in enumerate(pred_paths):
        df = pd.read_csv(pp)

        # Extract columns from DataFrame to numpy arrays
        preds = df['pred'].values.astype(float)
        labels = df['label'].values.astype(float)
        ale_unc = df['ale_unc'].values.astype(float)
        epi_unc = df['epi_unc'].values.astype(float)
        tol_unc = ale_unc + epi_unc

        if args.uncertainty_type == 'aleatoric':
            filter_indicator = ale_unc
        elif args.uncertainty_type == 'epistemic':
            filter_indicator = epi_unc
        elif args.uncertainty_type == 'total':
            filter_indicator = tol_unc
        else:
            raise ValueError(f'Unsupported Uncertainty type {args.uc_type}')

        rmse_values = calculate_threshold_rmse(preds, labels, filter_indicator, confidence_percentile)

        rmse_curves[i] = rmse_values

    # Calculate Oracle Curve using CPBayesMPP output
    df = pd.read_csv(cl_pred_path)
    preds = df['pred'].values.astype(float)
    labels = df['label'].values.astype(float)
    true_error = abs(preds - labels)

    oracle_curve = calculate_threshold_rmse(preds, labels, true_error, confidence_percentile)

    return rmse_curves[0], rmse_curves[1], oracle_curve


def calculate_ece_curve(
        pred_path,
        n_bins=5,
        strategy="uniform",
) -> Tuple[list, list, list]:
    """
    Compute MPVs, FOPs, and ECE for a calibration curve under a unique seed (classification task).
    Implemented by scikit-learn library (https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    :param pred_path : str
        File path of the prediction results.
    :param n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.
    :param strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
    :return: MPVs, FOPs, ECE of the current prediction result.
    """
    df = pd.read_csv(pred_path)

    # Extract columns from DataFrame to numpy arrays
    y_true = df['label'].values.astype(float)
    y_prob = df['pred'].values.astype(float)

    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    fops = bin_true[nonzero] / bin_total[nonzero]
    mpvs = bin_sums[nonzero] / bin_total[nonzero]
    bin_total = bin_total[nonzero]

    ece = np.sum(abs(fops - mpvs) * bin_total) / np.sum(bin_total) * 100

    return fops, mpvs, ece


def visualize_auco_curve(args: Namespace):
    """
    Visualize the Area Under the Confidence-Ordered (AUCO) calibration curve.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """
    confidence_percentile = np.arange(0, 100, 1)

    cd_curves = []
    cl_curves = []
    oracle_curves = []

    plt.figure(figsize=(8, 6))

    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.xlim(0, 100)

    for cd_pred_path, cl_pred_path in zip(args.cd_path_list, args.cl_path_list):
        nc, cc, oc = calculate_auco_curve(cd_pred_path, cl_pred_path, confidence_percentile, args)
        cd_curves.append(nc)
        cl_curves.append(cc)
        oracle_curves.append(oc)

    # Calculate mean and standard deviation of the BayesMPP curves
    cd_mean = np.mean(cd_curves, axis=0)
    cd_std = np.std(cd_curves, axis=0)

    # Calculate mean and standard deviation of the CPBayesMPP curves
    cl_mean = np.mean(cl_curves, axis=0)
    cl_std = np.std(cl_curves, axis=0)

    # Calculate mean of the oracle curves
    oracle_mean = np.mean(oracle_curves, axis=0)

    # Calculate mean and standard deviation of BayesMPP and CPBayesMPP AUCO
    cd_auco = np.sum(cd_mean - oracle_mean)
    cl_auco = np.sum(cl_mean - oracle_mean)
    cd_auco_std = np.std(np.sum(np.array(cd_curves) - np.array(oracle_curves), axis=1), axis=0)
    cl_auco_std = np.std(np.sum(np.array(cl_curves) - np.array(oracle_curves), axis=1), axis=0)

    print(f'BayesMPP AUCO = {cd_auco: .2f}, std = {cd_auco_std: .2f}\n'
          f'CPBayesMPP AUCO = {cl_auco: .2f}, std = {cl_auco_std: .2f}')

    print(f'Perforamce Improvement {round((cd_auco - cl_auco) / cd_auco * 100, 2)}%')

    # Plot BayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, cd_mean, color=green, label='BayesMPP', linewidth=5)
    plt.fill_between(confidence_percentile, cd_mean - cd_std, cd_mean + cd_std, color=green, alpha=0.3)

    # Plot CPBayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, cl_mean, color=blue, label='CPBayesMPP (Ours)', linewidth=5)
    plt.fill_between(confidence_percentile, cl_mean - cl_std, cl_mean + cl_std, color=blue, alpha=0.3)

    # Plot Oracle uncertainty calibration curve
    plt.plot(confidence_percentile, oracle_mean, color='black', linestyle='dashed', label='Oracle Calibration', linewidth=4)

    plt.xlabel('Confidence Percentile (%)', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    title = f'{args.print_name[args.data_name]} {args.uncertainty_type.title()} Uncertainty'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=23, loc=args.legend_loc)

    args.fig_output_path = os.path.join(f'figures',
                                        f'Uncertainty calibration curves for regression datasets',
                                        f'{title}.JPG')

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


def visualize_ece_curve(args: Namespace):
    """
    Visualize the Expected Calibration Error (ECE) curve.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """
    cd_eces = []
    cl_eces = []

    plt.figure(figsize=(8, 6.5))

    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)

    # cd_path_list and cl_path_list are the prediction result paths of BayesMPP and CPBayesMPP, respectively
    for i, (cd_pred_path, cl_pred_path) in enumerate(zip(args.cd_path_list, args.cl_path_list)):
        cd_fops, cd_mpvs, cd_ece = calculate_ece_curve(cd_pred_path)
        cl_fops, cl_mpvs, cl_ece = calculate_ece_curve(cl_pred_path)

        plt.plot(cd_mpvs, cd_fops, color=green,
                 label='BayesMPP' if i == 0 else None, marker='s', markersize=8, linewidth=3, alpha=0.7)
        plt.plot(cl_mpvs, cl_fops, color=blue,
                 label='CPBayesMPP (Ours)' if i == 0 else None, marker='s', markersize=8, linewidth=3, alpha=0.7)

        cd_eces.append(cd_ece)
        cl_eces.append(cl_ece)

    # Calculate and print the mean of ECE for BayesMPP and CPBayesMPP
    cd_eces_mean = sum(cd_eces) / len(cd_eces) if cd_eces else 0
    cl_eces_mean = sum(cl_eces) / len(cl_eces) if cl_eces else 0
    print(f'Mean of cd_eces: {cd_eces_mean}')
    print(f'Mean of cl_eces: {cl_eces_mean}')

    # Plot the ideal calibration line
    oracle = np.arange(0, 1.1, 0.1)
    plt.plot(oracle, oracle, color='black', linestyle='dashed', label='Oracle Calibration', linewidth=3)

    legend = plt.legend(frameon=True, fontsize=22, loc='upper left')

    for handle in legend.legendHandles:
        handle.set_linewidth(4)
        handle.set_alpha(1)

    plt.xlabel('Mean Predicted Value (MPV)', fontsize=20)
    plt.ylabel('Fraction of Positives (FOP)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    title = f'{args.data_name}'.upper()

    plt.title(title, fontsize=20)

    args.fig_output_path = os.path.join(f'figures'
                                        f'Uncertainty calibration curves for classification datasets',
                                        args.visualize_type,
                                        f'{title}.JPG')

    plt.show()


def visualize_uncertainty(args: Namespace):
    """
    Visualize the uncertainty calibration curve for regression (AUCO) or classification (ECE) tasks.

    :param args: Arguments.
    :return: None
    """
    if args.visualize_type == 'auco':
        visualize_auco_curve(args)
    elif args.visualize_type == 'ece':
        visualize_ece_curve(args)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_uncertainty_args(parser)
    args = parser.parse_args()

    modify_visualize_uncertainty_args(args)

    visualize_uncertainty(args)
