import csv
import math
import os
import pickle
import random
from argparse import Namespace
from copy import deepcopy
from typing import List, Tuple, Union

from matplotlib import pyplot as plt
from rdkit import Chem
import torch
from rdkit.Chem import Draw

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}
SMILES_TO_CONTRA_GRAPH = {}

def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace=None):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        # Suppose an undirected molecule graph containing 4 atoms and 6 bonds
        # Atoms: x_1, x_2, x_3, x_4
        # Bonds: e_12, e_21, e_13, e_31, e_14, e_41
        self.smiles = smiles
        self.n_atoms = 0  # (4) number of atoms
        self.n_bonds = 0  # (3) number of bonds
        self.f_atoms = []  # [[x_1], [x_2], [x_3], [x_4]] mapping from atom index to atom features
        self.f_bonds = []  # [[e_12], [e_21], [e_13], [e_31], [e_14], [e_41]]
                           # The bonds above are numbered: [0, 1, 2, 3, 4, 5]
                           # mapping from bond index to concat(in_atom, bond) features
        self.bond_list = []  # [[1, 2], [1, 3],  [1, 4]]
                             # Save the information of all bonds in the molecule
        self.a2b = []  # [[1,3,5], [0], [2], [4]]
                       # Save the ending bonds of each atom, for example, [1,3,5] means the ending bonds of x_1 are [e_12], [e_31], [e_41]
        self.b2a = []  # [1, 2, 1, 3, 1, 4]
                       # Save the starting atom of each bond, for example, the last "4" means the starting atom of e_41 is x_4
        self.b2revb = []  # [1, 0, 3, 2, 5, 4]
                          # Save the reverse bond of each bond, for example, the first "1" means the reverse bond of e_12 is e_21

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                # Get the bond between the two atoms using the atom indices
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)  # Get the features of the edge

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings, we only discuss the first time we traverse to edge e_12, where a1 = 1 and a2 = 2
                b1 = self.n_bonds  # 0 represents the current edge is the 0th edge
                b2 = b1 + 1  # 1 represents the current edge is the 1st edge (the reverse edge of the 0th edge)
                self.a2b[a2].append(b1)  # b1 = a1 --> a2, add the 0th edge to the incoming edge of atom 2
                self.b2a.append(a1)  # Inform the current 0th edge that it starts from node 1
                self.a2b[a1].append(b2)  # b2 = a2 --> a1, add the reverse edge of the 0th edge to the incoming edge of atom 1
                self.b2a.append(a2)  # Inform the reverse edge of the 0th edge that it starts from node 1
                self.b2revb.append(b2)  # Inform the current "0th edge[e_12]" that its reverse edge is "1st edge[e_21]"
                self.b2revb.append(b1)  # Inform the current "1st edge[e_21]" that its reverse edge is "0th edge[e_12]"

                self.bond_list.append([a1, a2])

                self.n_bonds += 2

        pass

    def reset(self):
        """
        Resets the MolGraph's attributes to their initial state.
        """
        self.n_atoms = 0  # number of atoms is reset to 0
        self.n_bonds = 0  # number of bonds is reset to 0
        self.f_atoms = []  # clear atom features list
        self.f_bonds = []  # clear bond features list
        self.bond_list = []  # clear the bond list
        self.a2b = []  # clear the incoming edge list of each atom
        self.b2a = []  # clear the starting atom list of each edge
        self.b2revb = []  # clear the reverse edge list of each edge

    def contra_sample(self):
        m_i = deepcopy(self)  # molgraph_i
        m_i.reset()

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(m_i.smiles)

        # fake the number of "atoms" if we are collapsing substructures
        m_i.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            m_i.f_atoms.append(atom_features(atom))

        # Randomly mask 25% of the nodes
        num_atoms_to_mask = int(0.25 * m_i.n_atoms)
        mask_atoms_indices = random.sample(range(m_i.n_atoms), num_atoms_to_mask)

        for atom_idx in mask_atoms_indices:
            m_i.f_atoms[atom_idx] = [0] * len(m_i.f_atoms[0])

        # Get bond features
        for _ in range(self.n_atoms):
            m_i.a2b.append([])

        for a1 in range(m_i.n_atoms):
            for a2 in range(a1 + 1, m_i.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)  # Get the bond between the two atoms using the atom indices

                if bond is None:
                    continue

                m_i.bond_list.append([a1, a2])

        # Randomly mask 25% of the edges
        num_bonds_to_mask = int(0.25 * (len(m_i.bond_list)))
        mask_bond_pairs_indices = random.sample(range(0, len(m_i.bond_list)), num_bonds_to_mask)

        m_i.bond_list = [m_i.bond_list[i] for i in range(len(m_i.bond_list)) if i not in mask_bond_pairs_indices]

        for bond in m_i.bond_list:
            a1, a2 = bond

            bond = mol.GetBondBetweenAtoms(a1, a2)  # Get the bond between the two atoms using the atom indices

            if bond is None:
                continue

            f_bond = bond_features(bond)  # Get the features of the edge

            m_i.f_bonds.append(m_i.f_atoms[a1] + f_bond)
            m_i.f_bonds.append(m_i.f_atoms[a2] + f_bond)

            # Update index mappings, we only discuss the first time we traverse to edge e_12, where a1 = 1 and a2 = 2
            b1 = m_i.n_bonds  # 0 represents the current edge is the 0th edge
            b2 = b1 + 1  # 1 represents the current edge is the 1st edge (the reverse edge of the 0th edge)
            m_i.a2b[a2].append(b1)  # b1 = a1 --> a2, add the 0th edge to the incoming edge of atom 2
            m_i.b2a.append(a1)  # Inform the current 0th edge that it starts from node 1
            m_i.a2b[a1].append(b2)  # b2 = a2 --> a1, add the reverse edge of the 0th edge to the incoming edge of atom 1
            m_i.b2a.append(a2)  # Inform the reverse edge of the 0th edge that it starts from node 1
            m_i.b2revb.append(b2)  # Inform the current "0th edge[e_12]" that its reverse edge is "1st edge[e_21]"
            m_i.b2revb.append(b1)  # Inform the current "1st edge[e_21]" that its reverse edge is "0th edge[e_12]"

            m_i.n_bonds += 2

        return m_i


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)  # 133
        self.bond_fdim = get_bond_fdim(args) + self.atom_fdim  # 147

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)


def mol2contra_graph(smiles_batch: List[str],
              args: Namespace) -> Tuple['BatchMolGraph', 'BatchMolGraph']:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs_i = []
    mol_graphs_j = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_CONTRA_GRAPH:
            m_i, m_j = SMILES_TO_CONTRA_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            m_i = mol_graph.contra_sample()
            m_j = mol_graph.contra_sample()
            SMILES_TO_CONTRA_GRAPH[smiles] = (m_i, m_j)

        mol_graphs_i.append(m_i)
        mol_graphs_j.append(m_j)

    return BatchMolGraph(mol_graphs_i, args), BatchMolGraph(mol_graphs_i, args)
