import os
from argparse import ArgumentParser, Namespace

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from utils.parsing import add_visualize_active_args

cd_color = sns.color_palette("deep")[2]
cl_color = sns.color_palette("deep")[0]


def load_curves(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, 12))
    return data


def compute_mean_std(curves):
    """
    Compute the mean and standard deviation of the curve
    """
    mean_curve = np.mean(curves, axis=0)
    std_curve = 1.0 * np.std(curves, axis=0)
    return mean_curve, std_curve


def plot_with_std(ax, data_ratio, mean, std, color, label, linestyle, linewidth=4, markersize=20, draw_std=True):
    """
    Plot the curve and shadow.
    """
    ax.plot(data_ratio, mean, label=label, color=color,
            linestyle=linestyle, linewidth=linewidth, marker='.', markersize=markersize,
            markeredgecolor='white')
    if draw_std:
        ax.fill_between(data_ratio, mean - std, mean + std, color=color, alpha=0.1)


def visualize_active_learning(args: Namespace):
    """
    Visualize the Active Learning curve.

    :param args: Namespace parameters
    :return: None
    """
    data_type = 'regression' if args.data_name in ['delaney', 'freesolv', 'qm8'] else 'classification'

    # Load the results of 4 training strategies
    BayesMPP_active_random_path = os.path.join(f'results',
                                         f'BayesMPP+AL',
                                         f'{args.data_name}_checkpoints',
                                         f'random',
                                         f'result.csv')
    BayesMPP_active_explorative_path = os.path.join(f'results',
                                              f'BayesMPP+AL',
                                              f'{args.data_name}_checkpoints',
                                              f'explorative',
                                              f'result.csv')
    CPBayesMPP_active_random_path = os.path.join(f'results',
                                         f'CPBayesMPP+AL',
                                         f'{args.data_name}_checkpoints',
                                         f'random',
                                         f'result.csv')
    CPBayesMPP_active_explorative_path = os.path.join(f'results',
                                              f'CPBayesMPP+AL',
                                              f'{args.data_name}_checkpoints',
                                              f'explorative',
                                              f'result.csv')
    CPBayesMPP_active_oracle_path = os.path.join(f'results',
                                          f'CPBayesMPP+AL',
                                          f'{args.data_name}_checkpoints',
                                          f'oracle',
                                          f'result.csv')

    BayesMPP_random_curves = load_curves(BayesMPP_active_random_path)
    BayesMPP_explorative_curves = load_curves(BayesMPP_active_explorative_path)
    CPBayesMPP_random_curves = load_curves(CPBayesMPP_active_random_path)
    CPBayesMPP_explorative_curves = load_curves(CPBayesMPP_active_explorative_path)
    CPBayesMPP_oracle_curves = load_curves(CPBayesMPP_active_oracle_path)

    if data_type == 'classification':
        BayesMPP_random_curves = BayesMPP_random_curves * 100
        BayesMPP_explorative_curves = BayesMPP_explorative_curves * 100
        CPBayesMPP_random_curves = CPBayesMPP_random_curves * 100
        CPBayesMPP_explorative_curves = CPBayesMPP_explorative_curves * 100
        CPBayesMPP_oracle_curves = CPBayesMPP_oracle_curves * 100

    data_ratio = [0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')

    # Plot each curve and its shadow
    plot_with_std(ax, data_ratio, *compute_mean_std(BayesMPP_random_curves), sns.color_palette("deep")[2], 'BayesMPP + Random Learning', 'dashed')
    plot_with_std(ax, data_ratio, *compute_mean_std(BayesMPP_explorative_curves), sns.color_palette("deep")[2], 'BayesMPP + Explorative Learning', '-')
    plot_with_std(ax, data_ratio, *compute_mean_std(CPBayesMPP_random_curves), sns.color_palette("deep")[0], 'CPBayesMPP + Random Learning (Ours)', '--')
    plot_with_std(ax, data_ratio, *compute_mean_std(CPBayesMPP_explorative_curves), sns.color_palette("deep")[0], 'CPBayesMPP + Explorative Learning (Ours)', '-')
    plot_with_std(ax, data_ratio, *compute_mean_std(CPBayesMPP_oracle_curves), 'brown', 'CPBayesMPP + Oracle', '-', draw_std=False)

    # legend = plt.legend(fontsize=15, loc='best', handlelength=3.0)
    # legend.get_frame().set_alpha(0.8)
    # for handle in legend.legendHandles:
    #     handle.set_linewidth(2.5)

    ax.set_xlabel('Proportion of full training set', fontsize=20)
    ax.set_ylabel('RMSE' if data_type == 'regression' else 'ROC-AUC (%)', fontsize=20)

    percent_labels = [f'{(float(value) * 100) :.0f}%' for value in data_ratio]
    plt.xticks(data_ratio[::2], percent_labels[::2], fontsize=20)
    plt.yticks(fontsize=20)

    if args.data_name == "delaney":
        title = "ESOL (Regression dataset)"
    elif args.data_name == "freesolv":
        title = "FreeSolv (Regression dataset)"
    elif args.data_name == "qm7":
        title = "QM7 (Regression dataset)"
    elif args.data_name == "qm8":
        title = "QM8 (Regression dataset)"
    elif args.data_name == "bace":
        title = "BACE (Classification dataset)"
    else:
        title = "BBBP (Classification dataset)"

    plt.title(title, fontsize=20)

    fig_output_path = os.path.join(f'figures',
                                   f'Performance_changes_in_Active_Learning',
                                   f'{args.data_name}.JPG')

    os.makedirs(os.path.dirname(fig_output_path), exist_ok=True)

    plt.savefig(fig_output_path, bbox_inches='tight', dpi=600)

    linewidth=2.5
    markersize=15

    # Create zoomed inset
    axins = ax.inset_axes([0.4, 0.4, 0.57, 0.57])
    plot_with_std(axins, data_ratio, *compute_mean_std(BayesMPP_random_curves), sns.color_palette("deep")[2], '', linestyle='dashed', linewidth=linewidth, markersize=markersize)
    plot_with_std(axins, data_ratio, *compute_mean_std(BayesMPP_explorative_curves), sns.color_palette("deep")[2], '',  linestyle='-', linewidth=linewidth, markersize=markersize)
    plot_with_std(axins, data_ratio, *compute_mean_std(CPBayesMPP_random_curves), sns.color_palette("deep")[0], '', linestyle='--', linewidth=linewidth, markersize=markersize)
    plot_with_std(axins, data_ratio, *compute_mean_std(CPBayesMPP_explorative_curves), sns.color_palette("deep")[0], '', linestyle='-', linewidth=linewidth, markersize=markersize)
    plot_with_std(axins, data_ratio, *compute_mean_std(CPBayesMPP_oracle_curves), 'brown', '', linestyle='-', linewidth=linewidth, markersize=markersize, draw_std=False)

    axins.set_xlim(0.19, 0.41)
    axins.set_ylim(ax.get_ylim())
    # axins.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')

    # Set x-axis ticks and labels
    zoom_data_ratio = [0.2, 0.25, 0.3, 0.35, 0.4]
    zoom_percent_labels = [f'{(float(value) * 100) :.0f}%' for value in zoom_data_ratio]
    axins.set_xticks(zoom_data_ratio)
    axins.set_xticklabels(zoom_percent_labels)

    plt.savefig(os.path.join(f'figures',
                                  f'Performance_changes_in_Active_Learning',
                                  f'{args.data_name}_zoom.JPG'), bbox_inches='tight', dpi=600)

    # plt.show()


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_active_args(parser)

    args = parser.parse_args()

    visualize_active_learning(args)