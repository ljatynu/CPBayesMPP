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
    if args.num_tasks == 1:
        model_preds, ale_uncs, epi_uncs = \
            np.array(model_preds).squeeze(), np.array(ale_uncs).squeeze(), np.array(epi_uncs).squeeze()

        smiles, labels = np.array(smiles), np.array(labels).squeeze()

        # Save the results
        with open(args.output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['smiles',
                      f'pred',
                      f'label',
                      f'ale_unc',
                      f'epi_unc'
                      ]

            writer.writerow(header)

            for i in range(len(labels)):
                row = []

                row.append(smiles[i])
                row.append(model_preds[i])
                row.append(labels[i])
                row.append(ale_uncs[i])
                row.append(epi_uncs[i])

                writer.writerow(row)

    else:
        with open(args.output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['smiles']
            for task in range(args.num_tasks):
                header.append(f'task{task}_pred')
                header.append(f'task{task}_label')
                header.append(f'task{task}_ale_unc')
                header.append(f'task{task}_epi_unc')

            writer.writerow(header)

            for i in range(len(smiles)):
                row = [smiles[i]]
                for task in range(args.num_tasks):
                    row.append(model_preds[i][task])
                    row.append(labels[i][task])
                    row.append(ale_uncs[i][task])
                    row.append(epi_uncs[i][task])
                writer.writerow(row)