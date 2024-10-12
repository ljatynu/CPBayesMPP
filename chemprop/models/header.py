from argparse import Namespace

import torch.nn as nn
import torch

from chemprop.nn_utils import get_activation_function, initialize_weights, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout, RegularizationAccumulator

class Header(nn.Module):
    def __init__(self, args: Namespace):
        super(Header, self).__init__()

        first_linear_dim = args.hidden_size

        activation = get_activation_function(args.activation)

        output_size = args.num_tasks

        self.dataset_type = args.dataset_type


        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = []
            last_linear_dim = first_linear_dim
        else:
            ffn = [
                ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                reg_acc=args.reg_acc, weight_regularizer=wd,
                                dropout_regularizer=dd)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    ConcreteDropout(layer=nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                                    reg_acc=args.reg_acc, weight_regularizer=wd,
                                    dropout_regularizer=dd)
                ])
            ffn.extend([
                activation,
            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self._ffn = nn.Sequential(*ffn)

        if args.dataset_type == 'regression':
            self.output_layer = nn.Linear(last_linear_dim, output_size)  # 1-dim mean output for Regression Task
            self.logvar_layer = nn.Linear(last_linear_dim, output_size)  # 1-dim logvar output for Regression Task

        if args.dataset_type == 'classification':
            self.output_layer = nn.Linear(last_linear_dim, output_size)  # K-dim mean output for Multi Classification Task
            self.sigmoid = nn.Sigmoid()


    def forward(self,
                batch: torch.FloatTensor):

        _output = self._ffn(batch)

        if self.dataset_type == 'regression':
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)
            # Gaussian log-variance logvar only for regression
            return output, logvar

        if self.dataset_type == 'classification' and self.training:
            output = self.output_layer(_output)
            return output

        # Apply sigmoid during Classification Testing
        if self.dataset_type == 'classification' and not self.training:
            _output = self.output_layer(_output)
            output = self.sigmoid(_output)
            return output

