from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + self.atom_fdim
        self.hidden_size = args.hidden_size
        self.depth = args.depth  # Number of message passing steps. args.depth = 3

        self.args = args

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Concrete Dropout for Bayesian NN
        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Input
        input_dim = self.bond_fdim

        # Input matrix
        self.W_i = ConcreteDropout(layer=nn.Linear(input_dim, self.hidden_size, bias=False), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd, train_strategy=args.train_strategy)

        # Shared weight matrix across depths (default)
        self.W_h = ConcreteDropout(layer=nn.Linear(self.hidden_size, self.hidden_size, bias=False), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd, depth=self.depth - 1, train_strategy=args.train_strategy)
        self.W_o = ConcreteDropout(layer=nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd, train_strategy=args.train_strategy)

    def forward(self,
                mol_graph: BatchMolGraph) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size Get (h_vw)^0
        message = self.act_func(input)  # num_bonds x hidden_size Get (h_vw)^0

        # Message passing
        for depth in range(self.depth - 1):
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size

        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden

    def record_transfer_weight(self):
        # Record the weight of transferred contrastive variational parameters
        self.W_i.record_transfer_layer(self.W_i.layer)
        self.W_h.record_transfer_layer(self.W_h.layer)
        self.W_o.record_transfer_layer(self.W_o.layer)

    def reset_dropout_rate(self):
        self.W_i.reset_dropout_rate()
        self.W_h.reset_dropout_rate()
        self.W_o.reset_dropout_rate()

