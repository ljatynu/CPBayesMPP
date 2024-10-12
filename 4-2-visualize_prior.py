import os
from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils.parsing import add_visualize_prior_args, modify_visualize_prior_args
from argparse import Namespace, ArgumentParser

import csv


def visualize_prior_latent(args: Namespace):
    """
    Visualize the feature distinctiveness in 2D plane.
    """
    args.prior_latent_path = f'results/' \
                           f'{args.prior}/' \
                           f'{args.data_name}_checkpoints/' \
                           f'latent.csv'
    args.fig_output_path = f'figures/' \
                           f'prior_latent/' \
                           f'{args.data_name}-{args.prior}.jpg'

    smileses = []
    labels = []
    t_sne_x = []
    t_sne_y = []

    with open(args.prior_latent_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            smileses.append(row[0])
            labels.append(float(row[1]))
            t_sne_x.append(float(row[2]))
            t_sne_y.append(float(row[3]))

    plt.figure(figsize=(8, 6))

    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='-', alpha=0.3)

    # Clip the uncertainty values to the 5th and 95th percentiles
    labels = np.clip(labels, np.percentile(labels, 5), np.percentile(labels, 95))

    scatter = plt.scatter(t_sne_x, t_sne_y, c=labels, cmap='viridis', alpha=0.5)

    colorbar = plt.colorbar(scatter)
    colorbar.ax.tick_params(labelsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()

    title = f'{args.print_name[args.data_name]} ' \
            f'Uninformative Prior' if args.prior == 'BayesMPP+Prior' \
        else f'{args.print_name[args.data_name]} Contrastive Prior'

    plt.title(title, fontsize=20)

    args.fig_output_path = os.path.join(f'figures',
                                        f'Prior_prediction_latent',
                                        f'{args.print_name[args.data_name]}_Uninformative_Prior.JPG' if args.prior == 'BayesMPP+Prior' else f'{args.print_name[args.data_name]}_Contrastive_Prior.JPG')
    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)
    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


def visualize_prior_similarity(args: Namespace):
    """
    Visualize the feature similarity in 2D plane.
    """
    args.prior_pred_path = f'results/' \
                           f'{args.prior}/' \
                           f'{args.data_name}_checkpoints/' \
                           f'similarity.csv'
    args.fig_output_path = f'figures/' \
                           f'prior_sim/' \
                           f'{args.data_name}-{args.prior}.jpg'

    df_loaded = pd.read_csv(args.prior_pred_path)

    # Convert each column of the DataFrame back to a list
    aug_similarities = df_loaded['Augmented'].tolist()
    same_similarities = df_loaded['Same Class'].tolist()
    diff_similarities = df_loaded['Different Class'].tolist()

    print(f'Mean Aug = {mean(aug_similarities)}\n'
          f'Mean In-Distribution = {mean(same_similarities)}\n'
          f'Mean Out-of-Distribution = {mean(diff_similarities)}')

    blue = (64 / 255, 149 / 255, 196 / 255)
    orange = (233 / 255, 134 / 255, 62 / 255)
    green = (64 / 255, 181 / 255, 149 / 255)

    plt.figure(figsize=(8, 6))

    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='-', alpha=0.3)

    plt.hist(aug_similarities, bins=100, alpha=0.6, label='Augmented Pairs', color=blue)
    plt.hist(same_similarities, bins=100, alpha=0.6, label='In-Distribution Pairs', color=orange)
    plt.hist(diff_similarities, bins=100, alpha=0.6, label='Out-of-Distribution Pairs', color=green)

    # Compute the normalized histogram values for each histogram
    counts_aug, bin_edges_aug = np.histogram(aug_similarities, bins=100)
    counts_same, bin_edges_same = np.histogram(same_similarities, bins=100)
    counts_diff, bin_edges_diff = np.histogram(diff_similarities, bins=100)

    # Plot the edges of the histograms using the midpoints
    bin_centers_aug = 0.5 * (bin_edges_aug[1:] + bin_edges_aug[:-1])
    bin_centers_same = 0.5 * (bin_edges_same[1:] + bin_edges_same[:-1])
    bin_centers_diff = 0.5 * (bin_edges_diff[1:] + bin_edges_diff[:-1])

    line_width = 4

    # Smooth the counts of the histograms using a Gaussian filter
    smooth_counts_aug = gaussian_filter1d(counts_aug, sigma=1.5)
    smooth_counts_same = gaussian_filter1d(counts_same, sigma=1.5)
    smooth_counts_diff = gaussian_filter1d(counts_diff, sigma=1.5)

    smooth_counts_aug = np.append(smooth_counts_aug, 0)
    bin_centers_aug = np.append(bin_centers_aug, 1.01)

    # Plot the smoothed lines
    plt.plot(bin_centers_aug, smooth_counts_aug, linewidth=line_width, color=blue)
    plt.plot(bin_centers_same, smooth_counts_same, linewidth=line_width, color=orange)
    plt.plot(bin_centers_diff, smooth_counts_diff, linewidth=line_width, color=green)

    plt.legend()

    plt.xlabel('Cosine Similarity', fontsize=20)
    plt.ylabel('Number of Pairs', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    title = f'{args.print_name[args.data_name]} Uninformative Prior' if args.prior == 'BayesMPP+Prior' else f'{args.print_name[args.data_name]} Contrastive Prior'
    plt.title(title, fontsize=20)

    args.fig_output_path = os.path.join(f'figures',
                                        f'Prior_prediction_similarity',
                                        f'{args.print_name[args.data_name]}_Uninformative_Prior.JPG' if args.prior == 'BayesMPP+Prior' else f'{args.print_name[args.data_name]}_Contrastive_Prior.JPG')
    os.makedirs(os.path.dirname(args.fig_output_path), exist_ok=True)
    plt.savefig(args.fig_output_path, bbox_inches='tight', dpi=600)

    plt.show()


def visualize_prior(args: Namespace):
    if args.visualize_type == 'similarity':
        visualize_prior_similarity(args)
    elif args.visualize_type == 'latent':
        visualize_prior_latent(args)


if __name__ == '__main__':
    # Get Hyperparameters
    parser = ArgumentParser()

    add_visualize_prior_args(parser)
    args = parser.parse_args()

    modify_visualize_prior_args(args)

    visualize_prior(args)
