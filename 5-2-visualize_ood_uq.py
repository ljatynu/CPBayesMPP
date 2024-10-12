import os

import pandas as pd

from argparse import Namespace, ArgumentParser

from sklearn.utils import column_or_1d, check_consistent_length
from typing import Tuple

from scipy.special import erfinv

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from utils.metric import calculate_auco_curve
from utils.metric import calculate_auce_curve
from utils.metric import calculate_ence_points
from utils.metric import calculate_coefficient_of_variation
from utils.metric import calculate_ece_curve
from utils.parsing import add_visualize_uncertainty_args, modify_visualize_uncertainty_args, \
    add_visualize_ood_uncertainty_args, modify_visualize_ood_uncertainty_args

green = sns.color_palette("deep")[2]
blue = sns.color_palette("deep")[0]


def visualize_auco_curve(args: Namespace):
    """
    Visualize the Area Under the Confidence-Ordered (AUCO) calibration curve.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """
    confidence_percentile = np.arange(0, 100, 1)

    BayesMPP_curves = []
    CPBayesMPP_curves = []

    BayesMPP_Oracle_curves = []
    CPBayesMPP_Oracle_curves = []

    plt.figure(figsize=(8, 6))

    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.xlim(0, 100)

    for BayesMPP_pred_path, CPBayesMPP_pred_path in zip(args.BayesMPP_pred_list, args.CPBayesMPP_pred_list):
        BayesMPP_curve, BayesMPP_Oracle_curve = calculate_auco_curve(BayesMPP_pred_path, confidence_percentile, args)
        CPBayesMPP_curve, CPBayesMPP_Oracle_curve = calculate_auco_curve(CPBayesMPP_pred_path, confidence_percentile,
                                                                         args)

        BayesMPP_curves.append(BayesMPP_curve)
        CPBayesMPP_curves.append(CPBayesMPP_curve)

        BayesMPP_Oracle_curves.append(BayesMPP_Oracle_curve)
        CPBayesMPP_Oracle_curves.append(CPBayesMPP_Oracle_curve)

    # Calculate mean and standard deviation of the BayesMPP curves
    BayesMPP_mean = np.mean(BayesMPP_curves, axis=0)
    BayesMPP_std = np.std(BayesMPP_curves, axis=0)

    # Calculate mean and standard deviation of the CPBayesMPP curves
    CPBayesMPP_mean = np.mean(CPBayesMPP_curves, axis=0)
    CPBayesMPP_std = np.std(CPBayesMPP_curves, axis=0)

    # Calculate mean of the oracle curves
    BayesMPP_Oracle_mean = np.mean(BayesMPP_Oracle_curves, axis=0)
    CPBayesMPP_Oracle_mean = np.mean(CPBayesMPP_Oracle_curves, axis=0)

    # Calculate mean and standard deviation of BayesMPP and CPBayesMPP AUCO
    BayesMPP_auco = np.sum(BayesMPP_mean - BayesMPP_Oracle_mean)
    CPBayesMPP_auco = np.sum(CPBayesMPP_mean - CPBayesMPP_Oracle_mean)

    BayesMPP_auco_std = np.std(np.sum(np.array(BayesMPP_curves) - np.array(BayesMPP_Oracle_curves), axis=1), axis=0)
    CPBayesMPP_auco_std = np.std(np.sum(np.array(CPBayesMPP_curves) - np.array(CPBayesMPP_Oracle_curves), axis=1),
                                 axis=0)

    # AUCO metric
    print(
        f'BayesMPP AUCO = {BayesMPP_auco: .{args.decimal_places}f}, std = {BayesMPP_auco_std: .{args.decimal_places}f}\n'
        f'CPBayesMPP AUCO = {CPBayesMPP_auco: .{args.decimal_places}f}, std = {CPBayesMPP_auco_std: .{args.decimal_places}f}')

    BayesMPP_auco = round(BayesMPP_auco, args.decimal_places)
    CPBayesMPP_auco = round(CPBayesMPP_auco, args.decimal_places)

    # Plot Oracle uncertainty calibration curve
    plt.plot(confidence_percentile, BayesMPP_Oracle_mean, color=green, linestyle='dashed', linewidth=4, alpha=0.7)
    plt.plot(confidence_percentile, CPBayesMPP_Oracle_mean, color=blue, linestyle='dashed', linewidth=4)

    # Plot BayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, BayesMPP_mean, color=green, label='BayesMPP', linewidth=5)
    plt.fill_between(confidence_percentile, BayesMPP_mean - BayesMPP_std, BayesMPP_mean + BayesMPP_std, color=green,
                     alpha=0.3)

    # Plot CPBayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, CPBayesMPP_mean, color=blue, label='CPBayesMPP (Ours)', linewidth=5)
    plt.fill_between(confidence_percentile, CPBayesMPP_mean - CPBayesMPP_std, CPBayesMPP_mean + CPBayesMPP_std,
                     color=blue, alpha=0.3)

    plt.xlabel('Confidence Percentile (%)', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    title = f'{args.print_name[args.data_name]} {args.uncertainty_type.title()} Uncertainty'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=23, loc='lower left')

    args.fig_output_path = os.path.join(f'figures',
                                        f'OOD_detection_Performance_Improvement',
                                        f'{args.print_name[args.data_name]}_{args.uncertainty_type.title()}_Uncertainty.JPG')

    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()

    print(
        f'Performance Improvement {round((BayesMPP_auco - CPBayesMPP_auco) / BayesMPP_auco * 100, args.decimal_places)}%')


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_ood_uncertainty_args(parser)
    args = parser.parse_args()

    modify_visualize_ood_uncertainty_args(args)

    visualize_auco_curve(args)
