from math import sqrt

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.special import erfinv

from argparse import Namespace

import pandas as pd
from sklearn.utils import column_or_1d, check_consistent_length
from typing import Tuple
import numpy as np


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def update_best_score(dataset_type, current_score, best_score):
    '''
    Compare the best score and current score
    Regression: rmse the lower the better
    Classification: auc-roc the higher the better

    return: better score
    '''
    is_better = (dataset_type == 'regression' and current_score < best_score) or \
                (dataset_type == 'classification' and current_score > best_score)
    if is_better:
        return current_score
    return best_score


def calculate_uncertainty_scaler(labels, preds, uncertainties):
    '''
    Calculate uncertainty scaler to avoid over-confident predictions
    See http://arxiv.org/abs/1905.11659 Eqn (12) for details
    '''
    T = len(labels)

    # def calibration_obj(s):
    #     if s <= 0:
    #         return np.inf
    #
    #     term1 = T / 2 * np.log(s)
    #     term2 = np.sum((labels - perds) ** 2 / (2 * s ** 2 * uncertainties))
    #     return term1 - term2

    def calibration_obj(s):
        error = np.abs(preds - labels)

        bin_scaling = [0]
        obs_emp_cov = np.zeros([100])  # shape(tasks, 101)
        exp_emp_cov = np.arange(100) / 100

        for i in range(1, 100):
            bin_scaling.append(erfinv(i / 100) * np.sqrt(2))

        for i in range(1, 100):
            bin_unc = np.sqrt(uncertainties * s) * bin_scaling[i]
            bin_fraction = np.mean(bin_unc >= error)
            obs_emp_cov[i] = bin_fraction

        auce = abs(np.sum(obs_emp_cov - exp_emp_cov))

        return auce

    # Initialize "s" to 1
    s = 1

    # Use scipy to minimize the calibration_obj
    result = minimize(calibration_obj, s, bounds=[(1, 10)])

    s = result.x[0]

    return s ** 2


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
        try:
            top_k_percent_rmse = rmse(selected_preds, selected_labels)
        except Exception:
            top_k_percent_rmse = 0

        # Return RMSE of selected data points
        rmse_values.append(top_k_percent_rmse)

    return rmse_values


def calculate_auco_curve(pred_paths: str,
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

    rmse_curves = []
    oracle_curves = []

    df = pd.read_csv(pred_paths)

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

    rmse_curve = calculate_threshold_rmse(preds, labels, filter_indicator, confidence_percentile)

    # Calculate Oracle Curve
    true_error = abs(preds - labels)
    oracle_curve = calculate_threshold_rmse(preds, labels, true_error, confidence_percentile)

    return rmse_curve, oracle_curve


def calculate_auce_curve(pred_path: str, args, uc_scaler: float = None):
    """
    :param pred_path : File path of the prediction results.
    :param args : Namespace
    """
    df = pd.read_csv(pred_path)

    preds = df['pred'].values.astype(float)
    labels = df['label'].values.astype(float)
    error = np.abs(preds - labels)
    ale_unc = df['ale_unc'].values.astype(float)
    epi_unc = df['epi_unc'].values.astype(float)
    tol_unc = ale_unc + epi_unc

    if args.uncertainty_type == 'aleatoric':
        uc = ale_unc
    elif args.uncertainty_type == 'epistemic':
        uc = epi_unc
    elif args.uncertainty_type == 'total':
        uc = tol_unc
    else:
        raise ValueError(f'Unsupported Uncertainty type {args.uc_type}')

    if uc_scaler is not None:
        uc = uc * uc_scaler

    bin_scaling = [0]
    obs_emp_cov = np.zeros([100])  # shape(tasks, 101)
    exp_emp_cov = np.arange(100) / 100

    # Calculate Z-values for each quartile for scaling standard deviation
    # For example, Z-values of 95 percentile is 1.645, then we use ± 1.645 * std to calculate the interval
    for i in range(1, 100):
        bin_scaling.append(erfinv(i / 100) * np.sqrt(2))

    for i in range(1, 100):
        bin_unc = np.sqrt(uc) * bin_scaling[i]
        bin_fraction = np.mean(bin_unc >= error)
        obs_emp_cov[i] = bin_fraction

    return obs_emp_cov, exp_emp_cov


def calculate_ence_points(pred_path: str, K: int, args, uc_scaler: float = None) -> list:
    """
    Calculate the Expected Normalized Calibration Error (ENCE) points for one seed.

    :param pred_path: prediction result path
    :param K: K bins
    :param uc_scaler: uncertainty scaler factor
    :param args: Namespace
    :return: ECE points for K bins
    """
    df = pd.read_csv(pred_path)

    preds = df['pred'].values.astype(float)
    labels = df['label'].values.astype(float)
    ale_unc = df['ale_unc'].values.astype(float)
    epi_unc = df['epi_unc'].values.astype(float)
    tol_unc = ale_unc + epi_unc

    if args.uncertainty_type == 'aleatoric':
        uncertainty = ale_unc
    elif args.uncertainty_type == 'epistemic':
        uncertainty = epi_unc
    elif args.uncertainty_type == 'total':
        uncertainty = tol_unc
    else:
        raise ValueError(f'Unsupported Uncertainty type {args.uncertainty_type}')

    if uc_scaler is not None:
        uncertainty = uncertainty * uc_scaler

    rmses = []
    mvars = []
    ence_values = []
    # bins = np.linspace(0, args.max_uc, K+1)
    lower_bound = np.percentile(uncertainty, 5)
    upper_bound = np.percentile(uncertainty, 95)
    bins = np.linspace(lower_bound, upper_bound, K + 1)

    for i in range(K):
        # Select data points that fall within the current bin
        bin_mask = (uncertainty >= bins[i]) & (uncertainty < bins[i + 1])

        if np.sum(bin_mask) > 0:
            # Calculate the RMSE for predictions and labels within the current bin
            bin_rmse = np.sqrt(np.mean((preds[bin_mask] - labels[bin_mask]) ** 2))
            rmses.append(bin_rmse)

            # Calculate the mean of uncertainties within the current bin, then take the square root to get mVAR(i)
            bin_mvar = np.sqrt(np.mean(uncertainty[bin_mask]))
            # bin_mvar = np.mean(uncertainty[bin_mask])

            # bin_mvar = np.mean(uncertainty[bin_mask])
            mvars.append(bin_mvar)

            # Compute ENCE using the formula: ENCE = |mVAR(i) - RMSE(i)| / mVAR(i)
            ence = np.abs(bin_mvar - bin_rmse) / bin_mvar
            ence_values.append(ence)
        else:
            # If the current bin has no data points, set ENCE value to 0
            ence_values.append(0)

    return rmses, mvars, np.mean(ence_values)


def calculate_coefficient_of_variation(pred_path: str, args):
    """
    Calculate Coefficient of Variation (C_v) for the predicted uncertainty which measures their dispersion.
    See http://arxiv.org/abs/1905.11659 for details.
    """
    df = pd.read_csv(pred_path)

    preds = df['pred'].values.astype(float)
    labels = df['label'].values.astype(float)
    ale_unc = df['ale_unc'].values.astype(float)
    epi_unc = df['epi_unc'].values.astype(float)
    tol_unc = ale_unc + epi_unc

    if args.uncertainty_type == 'aleatoric':
        uncertainty = ale_unc
    elif args.uncertainty_type == 'epistemic':
        uncertainty = epi_unc
    elif args.uncertainty_type == 'total':
        uncertainty = tol_unc
    else:
        raise ValueError(f'Unsupported Uncertainty type {args.uncertainty_type}')

    # Calculate Cv
    sigma = np.sqrt(uncertainty)
    mu_sigma = np.mean(sigma)

    Cv = np.sqrt(np.sum((sigma - mu_sigma)**2) / (len(sigma) - 1) ) / mu_sigma

    return Cv


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
