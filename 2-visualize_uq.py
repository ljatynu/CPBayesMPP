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
from utils.parsing import add_visualize_uncertainty_args, modify_visualize_uncertainty_args

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
        BayesMPP_curve, BayesMPP_Oracle_curve= calculate_auco_curve(BayesMPP_pred_path, confidence_percentile, args)
        CPBayesMPP_curve, CPBayesMPP_Oracle_curve = calculate_auco_curve(CPBayesMPP_pred_path, confidence_percentile, args)


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
    CPBayesMPP_auco_std = np.std(np.sum(np.array(CPBayesMPP_curves) - np.array(CPBayesMPP_Oracle_curves), axis=1), axis=0)

    # AUCO metric
    print(f'BayesMPP AUCO = {BayesMPP_auco: .{args.decimal_places}f}, std = {BayesMPP_auco_std: .{args.decimal_places}f}\n'
          f'CPBayesMPP AUCO = {CPBayesMPP_auco: .{args.decimal_places}f}, std = {CPBayesMPP_auco_std: .{args.decimal_places}f}')

    BayesMPP_auco = round(BayesMPP_auco, args.decimal_places)
    CPBayesMPP_auco = round(CPBayesMPP_auco, args.decimal_places)

    # Plot Oracle uncertainty calibration curve
    plt.plot(confidence_percentile, BayesMPP_Oracle_mean, color=green, linestyle='dashed',linewidth=4, alpha=0.7)
    plt.plot(confidence_percentile, CPBayesMPP_Oracle_mean, color=blue, linestyle='dashed',linewidth=4)

    # Plot BayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, BayesMPP_mean, color=green, label='BayesMPP', linewidth=5)
    plt.fill_between(confidence_percentile, BayesMPP_mean - BayesMPP_std, BayesMPP_mean + BayesMPP_std, color=green, alpha=0.3)

    # Plot CPBayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(confidence_percentile, CPBayesMPP_mean, color=blue, label='CPBayesMPP (Ours)', linewidth=5)
    plt.fill_between(confidence_percentile, CPBayesMPP_mean - CPBayesMPP_std, CPBayesMPP_mean + CPBayesMPP_std, color=blue, alpha=0.3)

    plt.xlabel('Confidence Percentile (%)', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    title = f'{args.print_name[args.data_name]} {args.uncertainty_type.title()} Uncertainty'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=23, loc='lower left')

    args.fig_output_path = os.path.join(f'figures',
                                        f'Uncertainty_calibration_curves_for_regression_datasets',
                                        f'AUCO',
                                        f'{args.print_name[args.data_name]}_{args.uncertainty_type.title()}_Uncertainty.JPG')

    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()

    print(f'Perforamce Improvement {round((BayesMPP_auco - CPBayesMPP_auco) / BayesMPP_auco * 100, args.decimal_places)}%')


def visualize_auce_curve(args: Namespace):
    """
    Visualize the area under the calibration error (AUCE) curve.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """

    BayesMPP_curves = []
    CPBayesMPP_curves = []
    oracle_curves = []

    plt.figure(figsize=(8, 6))
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)

    for BayesMPP_pred_path, CPBayesMPP_pred_path in zip(args.BayesMPP_pred_list, args.CPBayesMPP_pred_list):
        BayesMPP_curve, Oracle_curve = calculate_auce_curve(BayesMPP_pred_path, args)
        CPBayesMPP_curve, _ = calculate_auce_curve(CPBayesMPP_pred_path, args)

        BayesMPP_curves.append(BayesMPP_curve)
        CPBayesMPP_curves.append(CPBayesMPP_curve)
        oracle_curves.append(Oracle_curve)

    # Calculate mean and standard deviation of the BayesMPP curves
    BayesMPP_mean = np.mean(BayesMPP_curves, axis=0)
    BayesMPP_std = np.std(BayesMPP_curves, axis=0)

    # Calculate mean and standard deviation of the CPBayesMPP curves
    CPBayesMPP_mean = np.mean(CPBayesMPP_curves, axis=0)
    CPBayesMPP_std = np.std(CPBayesMPP_curves, axis=0)

    # Calculate mean of the oracle curves
    oracle_mean = np.mean(oracle_curves, axis=0)

    # Calculate mean and standard deviation of BayesMPP and CPBayesMPP AUCO
    cd_auce = np.sum(abs(BayesMPP_mean - oracle_mean))
    cl_auce = np.sum(abs(CPBayesMPP_mean - oracle_mean))

    cd_auce_std = np.std(np.sum(np.array(BayesMPP_curves) - np.array(oracle_curves), axis=1), axis=0)
    cl_auce_std = np.std(np.sum(np.array(CPBayesMPP_curves) - np.array(oracle_curves), axis=1), axis=0)

    print(f'BayesMPP AUCE = {cd_auce: .{args.decimal_places}f}, std = {cd_auce_std: .{args.decimal_places}f}\n'
            f'CPBayesMPP AUCE = {cl_auce: .{args.decimal_places}f}, std = {cl_auce_std: .{args.decimal_places}f}')

    cd_auce = round(cd_auce, args.decimal_places)
    cl_auce = round(cl_auce, args.decimal_places)

    print(f'Perforamce Improvement {round((cd_auce - cl_auce) / cd_auce * 100, args.decimal_places)}%')

    # Plot BayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(oracle_mean, BayesMPP_mean, color=green, label='BayesMPP', linewidth=5)
    plt.fill_between(oracle_mean, BayesMPP_mean - BayesMPP_std, BayesMPP_mean + BayesMPP_std, color=green, alpha=0.3)

    # Plot CPBayesMPP uncertainty calibration curve and its uncertainty shadow
    plt.plot(oracle_mean, CPBayesMPP_mean, color=blue, label='CPBayesMPP (Ours)', linewidth=5)
    plt.fill_between(oracle_mean, CPBayesMPP_mean - CPBayesMPP_std, CPBayesMPP_mean + CPBayesMPP_std, color=blue, alpha=0.3)

    # Plot Oracle uncertainty calibration curve
    plt.plot(oracle_mean, oracle_mean, color='black', linestyle='dashed', label='Oracle Calibration',
             linewidth=4)

    plt.xlabel('Confidence Interval (%)', fontsize=20)
    plt.ylabel('Proportion of datapoints', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    title = f'{args.print_name[args.data_name]} {args.uncertainty_type.title()} Uncertainty'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=23)

    args.fig_output_path = os.path.join(f'figures',
                                        f'Uncertainty_calibration_curves_for_regression_datasets',
                                        f'AUCE',
                                        f'{args.print_name[args.data_name]}_{args.uncertainty_type.title()}_Uncertainty.JPG')
    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


def visualize_ence_points(args: Namespace):
    """
    Visualize the Expected Normalized Calibration Error (ENCE) points

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """
    K = args.bins_num

    BayesMPP_ences = []
    CPBayesMPP_ences = []

    plt.figure(figsize=(8, 6))

    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.xlim(0, args.max_uc)
    plt.ylim(0, args.max_uc)

    # Draw diagonal line from bottom-left to top-right
    plt.plot([0, args.max_uc], [0, args.max_uc], color='black', linestyle='--')

    for i, (BayesMPP_pred_path, CPBayesMPP_pred_path) in enumerate(zip(args.BayesMPP_pred_list, args.CPBayesMPP_pred_list)):
        BayesMPP_rmses, BayesMPP_mvars, BayesMPP_ence = calculate_ence_points(BayesMPP_pred_path, K, args)
        CPBayesMPP_rmses, CPBayesMPP_mvars, CPBayesMPP_ence = calculate_ence_points(CPBayesMPP_pred_path, K, args)

        plt.scatter(BayesMPP_mvars, BayesMPP_rmses, color=green, marker='o', label='BayesMPP' if i == 0 else None, s=80)
        plt.scatter(CPBayesMPP_mvars, CPBayesMPP_rmses, color=blue, marker='x', label='CPBayesMPP (Ours)' if i == 0 else None, s=80, zorder=10)

        BayesMPP_ences.append(BayesMPP_ence)
        CPBayesMPP_ences.append(CPBayesMPP_ence)

    # cd_pred_path, cl_pred_path = args.cd_path_list[-1], args.cl_path_list[-1]
    #
    # cd_rmses, cd_mvars, cd_ence = calculate_ence_points(cd_pred_path, K, args)
    # cl_rmses, cl_mvars, cl_ence = calculate_ence_points(cl_pred_path, K, args)
    #
    # plt.scatter(cd_mvars, cd_rmses, color=green, marker='o', label='BayesMPP')
    # plt.scatter(cl_mvars, cl_rmses, color=blue, marker='x', label='CPBayesMPP (Ours)')

    BayesMPP_ences.append(BayesMPP_ence)
    CPBayesMPP_ences.append(CPBayesMPP_ence)


    plt.xlabel('RMU', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.xticks(fontsize=15 if args.data_name == 'qm8' else 20)
    plt.yticks(fontsize=15 if args.data_name == 'qm8' else 20)

    title = f'{args.print_name[args.data_name]} {args.uncertainty_type.title()} Uncertainty'

    plt.title(title, fontsize=20)

    plt.legend(fontsize=23, loc='best')

    # print(f'Mean of cd_ence: {np.mean(cd_ences)}')
    # print(f'Mean of cl_ence: {np.mean(cl_ences)}')

    BayesMPP_ence = np.mean(BayesMPP_ences)
    CPBayesMPP_ence = np.mean(CPBayesMPP_ences)
    BayesMPP_ence_std = np.std(BayesMPP_ences)
    CPBayesMPP_ence_std = np.std(CPBayesMPP_ences)

    print(f'BayesMPP ENCE = {BayesMPP_ence: .{args.decimal_places}f}, std = {BayesMPP_ence_std: .{args.decimal_places}f}\n'
          f'CPBayesMPP ENCE = {CPBayesMPP_ence: .{args.decimal_places}f}, std = {CPBayesMPP_ence_std: .{args.decimal_places}f}')

    BayesMPP_ence = round(BayesMPP_ence, args.decimal_places)
    CPBayesMPP_ence = round(CPBayesMPP_ence, args.decimal_places)

    print(f'Perforamce Improvement {round((BayesMPP_ence - CPBayesMPP_ence) / BayesMPP_ence * 100, args.decimal_places)}%')

    args.fig_output_path = os.path.join(f'figures',
                                        f'Uncertainty_calibration_curves_for_regression_datasets',
                                        f'ENCE',
                                        f'{args.print_name[args.data_name]}_{args.uncertainty_type.title()}_Uncertainty.JPG')
    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


def visualize_coefficient_of_variation(args: Namespace):
    """
    Visualize the Coefficient of Variation (C_v) for the predicted uncertainty which measures their dispersion.
    See http://arxiv.org/abs/1905.11659 for details.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """

    BayesMPP_Cvs = []
    CPBayesMPP_Cvs = []

    for i, (BayesMPP_pred_path, CPBayesMPP_pred_path) in enumerate(zip(args.BayesMPP_pred_list, args.CPBayesMPP_pred_list)):
        BayesMPP_Cv = calculate_coefficient_of_variation(BayesMPP_pred_path, args)
        CPBayesMPP_Cv = calculate_coefficient_of_variation(CPBayesMPP_pred_path, args)

        BayesMPP_Cvs.append(BayesMPP_Cv)
        CPBayesMPP_Cvs.append(CPBayesMPP_Cv)

    BayesMPP_Cv_mean = np.mean(BayesMPP_Cvs)
    BayesMPP_Cv_std = np.std(BayesMPP_Cvs)

    CPBayesMPP_Cv_mean = np.mean(CPBayesMPP_Cvs)
    CPBayesMPP_Cv_std = np.std(CPBayesMPP_Cvs)


    print(f'BayesMPP C_v = {BayesMPP_Cv_mean: .{args.decimal_places}f}, std = {BayesMPP_Cv_std: .{args.decimal_places}f}\n'
            f'CPBayesMPP C_v = {CPBayesMPP_Cv_mean: .{args.decimal_places}f}, std = {CPBayesMPP_Cv_std: .{args.decimal_places}f}')

    BayesMPP_Cv_mean = round(BayesMPP_Cv_mean, args.decimal_places)
    CPBayesMPP_Cv_mean = round(CPBayesMPP_Cv_mean, args.decimal_places)

    print(f'Perforamce Improvement {round((CPBayesMPP_Cv_mean - BayesMPP_Cv_mean) / BayesMPP_Cv_mean * 100, args.decimal_places)}%')


def visualize_ece_curve(args: Namespace):
    """
    Visualize the Expected Calibration Error (ECE) curve.

    :param args: Namespace parameters, including data name, uncertainty type,
                 prediction result paths of BayesMPP and CPBayesMPP, etc.
    :return: None
    """
    BayesMPP_eces = []
    CPBayesMPP_eces = []

    plt.figure(figsize=(8, 6.5))

    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)

    # cd_path_list and cl_path_list are the prediction result paths of BayesMPP and CPBayesMPP, respectively
    for i, (BayesMPP_pred_path, CPBayesMPP_pred_path) in enumerate(zip(args.BayesMPP_pred_list, args.CPBayesMPP_pred_list)):
        BayesMPP_fops, BayesMPP_mpvs, BayesMPP_ece = calculate_ece_curve(BayesMPP_pred_path)
        CPBayesMPP_fops, CPBayesMPP_mpvs, CPBayesMPP_ece = calculate_ece_curve(CPBayesMPP_pred_path)

        plt.plot(BayesMPP_mpvs, BayesMPP_fops, color=green,
                 label='BayesMPP' if i == 0 else None, marker='s', markersize=8, linewidth=3, alpha=0.7)
        plt.plot(CPBayesMPP_mpvs, CPBayesMPP_fops, color=blue,
                 label='CPBayesMPP (Ours)' if i == 0 else None, marker='s', markersize=8, linewidth=3, alpha=0.7)

        BayesMPP_eces.append(BayesMPP_ece)
        CPBayesMPP_eces.append(CPBayesMPP_ece)

    # Calculate and print the mean of ECE for BayesMPP and CPBayesMPP
    BayesMPP_eces_mean = sum(BayesMPP_eces) / len(BayesMPP_eces) if BayesMPP_eces else 0
    CPBayesMPP_eces_mean = sum(CPBayesMPP_eces) / len(CPBayesMPP_eces) if CPBayesMPP_eces else 0
    print(f'Mean of BayesMPP ECE: {BayesMPP_eces_mean}')
    print(f'Mean of CPBayesMPP ECE: {CPBayesMPP_eces_mean}')
    print(f'Perforamce Improvement {round((BayesMPP_eces_mean - CPBayesMPP_eces_mean) / BayesMPP_eces_mean * 100, args.decimal_places)}%')


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

    args.fig_output_path = os.path.join(f'figures',
                                        f'Uncertainty_calibration_curves_for_classification_datasets',
                                        f'{title}.JPG')
    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)

    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_uncertainty_args(parser)
    args = parser.parse_args()

    modify_visualize_uncertainty_args(args)

    if args.visualize_type == 'auco':
        visualize_auco_curve(args)
    elif args.visualize_type == 'auce':
        visualize_auce_curve(args)
    elif args.visualize_type == 'ence':
        visualize_ence_points(args)
    elif args.visualize_type == 'Cv':
        visualize_coefficient_of_variation(args)
    elif args.visualize_type == 'ece':
        visualize_ece_curve(args)
