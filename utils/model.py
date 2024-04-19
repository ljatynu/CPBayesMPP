from typing import List

from argparse import Namespace

import torch.nn as nn

from chemprop.models.concrete_dropout import RegularizationAccumulator
from chemprop.nn_utils import initialize_weights
from chemprop.models.mpn import MPNEncoder
from chemprop.models.header import Header
from chemprop.features.featurization import mol2graph, mol2contra_graph


################# Pretrain Model ###################
class PretrainModel(nn.Module):
    def __init__(self, args: Namespace):
        super(PretrainModel, self).__init__()

        self.args = args

        args.reg_acc = RegularizationAccumulator()

        self.encoder = MPNEncoder(args)
        self.header = nn.Linear(args.hidden_size, args.ffn_hidden_size)

        initialize_weights(self)

        args.reg_acc.initialize(cuda=args.cuda)

    def forward(self,
                batch: List[str]):
        # smiles -> molgraph_i, molgraph_j
        batch_i, batch_j = mol2contra_graph(batch, self.args)

        zis = self.encoder(batch_i)
        zis = self.header(zis)

        lc_reg = self.args.reg_acc.get_sum()  # Clear the cache of regularization terms for subsequent forward propagation.

        zjs = self.encoder(batch_j)
        zjs = self.header(zjs)

        lc_reg = self.args.reg_acc.get_sum()

        return zis, zjs, lc_reg

    def encoder_forward(self,
                        batch: List[str]):
        # smiles -> molgraph
        batch = mol2graph(batch, self.args)
        x = self.encoder(batch)

        return x


################# DownStream Model ###################
class DownStreamModel(nn.Module):
    def __init__(self, args: Namespace):
        super(DownStreamModel, self).__init__()

        self.args = args
        self.dataset_type = args.dataset_type

        args.reg_acc = RegularizationAccumulator()

        self.encoder = MPNEncoder(args)
        self.header = Header(args)

        initialize_weights(self)

        args.reg_acc.initialize(cuda=args.cuda)

    def forward(self,
                batch: List[str]):
        # smiles -> molgraph
        batch = mol2graph(batch, self.args)
        x = self.encoder(batch)
        x = self.header(x)
        return x

    def encoder_forward(self,
                        batch: List[str]):
        # smiles -> molgraph
        batch = mol2graph(batch, self.args)
        x = self.encoder(batch)

        return x
