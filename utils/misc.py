import csv
from argparse import Namespace
from typing import List

import numpy as np


def save_prediction(args: Namespace,
                    smiles: List[str],
                    labels: List[List[float]],
                    model_preds: List[float],
                    ale_uncs: List[float],
                    epi_uncs: List[float],
                    ):
    """
    Save the prediction results to a CSV file.
    """
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

    with open(args.pred_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['smiles',
                  f'pred',
                  f'label',
                  f'ale_unc',
                  f'epi_unc']

        writer.writerow(header)

        for i in range(len(labels)):
            row = []

            row.append(smiles[i])
            row.append(model_preds[i])
            row.append(labels[i])
            row.append(ale_uncs[i])
            row.append(epi_uncs[i])

            writer.writerow(row)