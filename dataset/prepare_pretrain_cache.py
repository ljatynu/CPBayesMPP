import pickle
import os

from tqdm import tqdm

from chemprop.features.featurization import MolGraph

SMILES_TO_CONTRA_GRAPH = {}

import csv


def save_batch_to_disk(data, directory, batch_number):
    filename = os.path.join(directory, f'smiles_to_contra_graph_batch_{batch_number}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def read_csv_in_batches(filename: str, batch_size: int):
    """
    Read a CSV file and return the data in specified batch sizes.

    :param filename: The path to the CSV file.
    :param batch_size: The size of each batch.
    :return: A generator that returns a batch of data each time.
    """

    batch = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            batch.append(row[0])
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Return the last batch if there are any remaining
        if batch:
            yield batch


def get_total_batches(filename: str, batch_size: int) -> int:
    """
    Calculate the total number of batches in a given CSV file.

    :param filename: The path to the CSV file.
    :param batch_size: The size of each batch.
    :return: The total number of batches.
    """
    with open(filename, 'r') as file:
        total_lines = sum(1 for line in file) - 1  # Remove the title line
    total_batches = -(-total_lines // batch_size)  # Round up
    return total_batches


filename = 'pubchem-10K-clean.csv'
output_directory = 'dataset/pubchem-10K-clean-cache'
batch_size = 512
total_batches = get_total_batches(filename, batch_size)

os.makedirs(output_directory)

for batch_number, batch in enumerate(tqdm(read_csv_in_batches(filename, batch_size), total=total_batches)):
    temp_smiles_to_contra_graph = {}
    for smiles in batch:
        mol_graph = MolGraph(smiles, None)
        m_i = mol_graph.contra_sample()
        m_j = mol_graph.contra_sample()
        temp_smiles_to_contra_graph[smiles] = (m_i, m_j)
    # Save the temporary dictionary to disk
    save_batch_to_disk(temp_smiles_to_contra_graph, output_directory, batch_number)
