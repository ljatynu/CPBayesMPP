from argparse import ArgumentParser, Namespace

import os
import torch

from chemprop.data.utils import get_task_names


def add_pretrain_args(parser: ArgumentParser):
    """
    Add pretraining arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--pretrain_data_name', type=str, default='pubchem-10K-clean',
                        help='name of pre-training data')
    parser.add_argument('--train_strategy', type=str, default='pretrain')
    parser.add_argument('--save_dir', type=str, default='results/pretrain',
                        help='Directory where pretrained model checkpoints will be saved')
    parser.add_argument('--use_pretrain_data_cache', type=bool,
                        default=True,
                        help='Whether use the datacache to accelerate pre-training.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='dataloader number of workers')
    parser.add_argument('--valid_size', type=int, default=0.05,
                        help='ratio of validation data')
    parser.add_argument('--temperature', type=int, default=0.1,
                        help='temperature of NT-Xent loss')
    parser.add_argument('--init_lr', type=int, default=0.0005,
                        help='# initial learning rate for Adam')
    parser.add_argument('--weight_decay', type=int, default=1e-5,
                        help='# weight decay for Adam')
    parser.add_argument('--warm_up', type=int, default=10,
                        help='# warm-up epochs')
    parser.add_argument('--use_cosine_similarity', type=bool, default=True,
                        help='whether to use cosine similarity in NT-Xent loss (i.e. True/False)')
    parser.add_argument('--kl_weight', type=int, default=2,
                        help='The regularization weight of ELBO KL term in pre-training '
                             'is determined jointly with the number of samples in the dataset.')

    # Model arguments (Make sure the MPNN parameters here same as the train args)
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='Hidden dim for contrastive learning output')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in downstream FFN after MPN encoding')

    parser.add_argument('--regularization_scale', type=float, default=1e-4,
                        help='Concrete dropout regularization scale')


def modify_pretrain_args(args: Namespace):
    """
    Modify and validate pre-training arguments in place.

    :param args: Arguments.
    """
    args.pretrain_data_path = os.path.join(f'dataset', f'{args.pretrain_data_name}.csv')

    if args.use_pretrain_data_cache:
        args.pretrain_data_cache_path = os.path.join('dataset',
                                                     f'{args.pretrain_data_name}-cache')

    args.task_names = get_task_names(args.pretrain_data_path)

    args.cuda = torch.cuda.is_available()


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_name', type=str, default='freesolv',
                        help='Downstream task name')
    parser.add_argument('--train_strategy', type=str, default='CPBayesMPP',
                        choices=['BayesMPP', 'CPBayesMPP'],
                        help='Training strategy of downstream task.'
                             'BayesMPP means training with uninformative prior.'
                             'CPBayesMPP means training with contrastive prior.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--kl_weight', type=float, default=1,
                        help='The regularization weight of ELBO KL term in downstream task '
                             'is determined jointly with the number of samples in the dataset.')

    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.5, 0.2, 0.3],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='List of random seeds to use when splitting dataset into train/val/test sets.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')

    parser.add_argument('--regularization_scale', type=float, default=1e-4,
                        help='Concrete dropout regularization scale')
    parser.add_argument('--sampling_size', type=int, default=20,
                        help='Sampling size for MC-Dropout (Used in validation step during training)')


def modify_train_args(args: Namespace):
    """
    Modify and validate training arguments in place.

    :param args: Arguments.
    """
    args.data_path = os.path.join(f'dataset',
                                  f'{args.data_name}.csv')

    if args.train_strategy == 'BayesMPP':
        args.save_dir = os.path.join(args.save_dir,
                                     f'{args.train_strategy}',
                                     f'{args.split_type}_split')
    elif args.train_strategy == 'CPBayesMPP':
        args.pretrain_encoder_path = os.path.join(args.save_dir,
                                                  f'pretrain',
                                                  f'pretrain_encoder.pt')
        args.save_dir = os.path.join(args.save_dir,
                                     f'{args.train_strategy}',
                                     f'{args.split_type}_split', )

    else:
        raise ValueError(f'Unsupported train strategy {args.train_strategy}')

    if args.data_name in ['delaney', 'freesolv', 'lipo', 'qm7', 'qm8', 'pdbbind']:
        args.dataset_type = 'regression'
        args.metric = 'rmse'
    elif args.data_name in ['bbbp', 'tox21', 'clintox', 'hiv', 'bace', 'sider']:
        args.dataset_type = 'classification'
        args.metric = 'roc-auc'
    else:
        raise ValueError(f'No supported data_name {args.data_name}')

    if not args.seeds:
        if args.dataset_type == 'regression':
            args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds for regression
        if args.dataset_type == 'classification':
            args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds for regression
            # args.seeds = [111, 222, 333]  # Use 3 random seeds for regression

    args.cuda = torch.cuda.is_available()


def add_predict_args(parser: ArgumentParser):
    """
    Add predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='freesolv',
                        help='Downstream task name')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--train_strategy', type=str, default='CPBayesMPP',
                        choices=['BayesMPP', 'CPBayesMPP'],
                        help='Training strategy of downstream task.'
                             'cd means training with uninformative prior.'
                             'cl means training with contrastive prior.')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--sampling_size', type=int, default=50,
                        help='Sampling size for MC-Dropout')


def modify_predict_args(args: Namespace):
    """
    Modify and validate predict arguments in place.

    :param args: Arguments.
    """
    if args.data_name in ['delaney', 'freesolv', 'lipo', 'qm7', 'qm8', 'pdbbind']:
        args.dataset_type = 'regression'
        args.metric = 'rmse'
    if args.data_name in ['bbbp', 'tox21', 'hiv', 'bace', 'clintox', 'sider']:
        args.dataset_type = 'classification'
        args.metric = 'roc-auc'

    args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds

    if args.dataset_type == 'regression':
        args.num_tasks = 1

    args.cuda = torch.cuda.is_available()


def add_visualize_metric_args(parser: ArgumentParser):
    """
    Adds metric visualization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='freesolv',
                        help='name of dataset visualized')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints are be saved')
    parser.add_argument('--train_strategy', type=str, default='cl',
                        choices=['cd', 'cl'],
                        help='Training strategy of downstream task.'
                             'cd means training with uninformative prior.'
                             'cl means training with contrastive prior.')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')


def modify_visualize_metric_args(args: Namespace):
    """
    Modify and validate visualization arguments in place.

    :param args: Arguments.
    """
    assert args.data_name is not None

    _temp_main_save_dir = args.save_dir

    if args.data_name in ['delaney', 'freesolv', 'lipo', 'qm7', 'qm8', 'pdbbind']:
        args.dataset_type = 'regression'
        args.metric = 'rmse'
    if args.data_name in ['bbbp', 'tox21', 'hiv', 'bace', 'clintox', 'sider']:
        args.dataset_type = 'classification'
        args.metric = 'roc-auc'

    args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds for regression

    args.cuda = torch.cuda.is_available()

    args.metric_path_list = [os.path.join(f'results',
                                          f'{args.train_strategy}',
                                          f'{args.split_type}_split',
                                          f'{args.data_name}_checkpoints',
                                          f'seed_{seed}',
                                          f'{args.data_name}_metric.csv')
                             for seed in args.seeds]


def add_visualize_uncertainty_args(parser: ArgumentParser):
    """
    Adds uncertainty visualization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='freesolv',
                        help='name of dataset to plot')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints are be saved')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')

    parser.add_argument('--visualize_type', type=str, default='ece',
                        choices=['auco', 'auce', 'ence', 'Cv', 'ece'],
                        help='Methods of uncertainty calibration.'
                             'auco for regression tasks, and ece for classification tasks.')
    parser.add_argument('--uncertainty_type', type=str, default='epistemic',
                        choices=['aleatoric', 'epistemic', 'total'],
                        help='The uncertainty type which calibration is based')
    parser.add_argument('--legend_loc', type=str, default='lower left',
                        choices=['lower left', 'lower right', 'upper left', 'upper right'],
                        help='Location of the plot legend')
    parser.add_argument('--decimal_places', type=int, default=2,
                        help='Number of decimal places to round to')

    # parameters for ENCE plot
    parser.add_argument('--bins_num', type=int, default=10,
                        help='K-bins for ENCE calculation')
    parser.add_argument('--max_uc', type=float, default=2,
                        help='The max value of uncertainty interval')
    parser.add_argument('--uc_scaler', type=float, default=1,
                        help='The max value of uncertainty interval')

def modify_visualize_uncertainty_args(args: Namespace):
    """
    Modify and validate uncertainty visualization arguments in place.

    :param args: Arguments.
    """

    args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]

    args.BayesMPP_pred_list = [os.path.join(
                                            f'results',
                                            f'BayesMPP',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.CPBayesMPP_pred_list = [os.path.join(
                                            f'results',
                                            f'CPBayesMPP',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.Ensemble_pred_list = [os.path.join(
                                            f'results',
                                            f'Ensemble',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.CLEnsemble_pred_list = [os.path.join(
                                            f'results',
                                            f'CLEnsemble',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.print_name = {
        'delaney': 'ESOL',
        'freesolv': 'FreeSolv',
        'lipo': 'Lipo',
        'qm7': 'QM7',
        'qm8': 'QM8',
        'pdbbind': 'PDBbind',
    }

    if args.visualize_type == 'ence':
        dataset_max_uc_values = {
            'delaney': 2,
            'freesolv': 4,
            'lipo': 1.4,
            'qm7': 300,
            'qm8': 0.02,
            'pdbbind': 3.0,
        }
        args.max_uc = dataset_max_uc_values.get(args.data_name)


def add_active_train_args(parser: ArgumentParser):
    """
    Adds active learning arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_name', type=str, default='bace',
                        help='Downstream task name')
    parser.add_argument('--train_strategy', type=str, default='CPBayesMPP+AL',
                        choices=['BayesMPP+AL', 'CPBayesMPP+AL'],
                        help='Training strategy of downstream task.'
                             'BayesMPP means training with uninformative prior.'
                             'CPBayesMPP means training with contrastive prior.')
    parser.add_argument('--al_type', type=str, default='oracle',
                        choices=['random', 'explorative', 'oracle'],
                        help='random means randomly select top-k samples when retraining the model'
                             'uncertainty means select top-k highest epistemic uncertainty samples when retraining the model')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--kl_weight', type=float, default=5.0,
                        help='The regularization weight of ELBO KL term in downstream task '
                             'is determined jointly with the number of samples in the dataset.')

    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.5, 0.2, 0.3],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--seed', type=int, default=123,
                        help='List of random seeds to use when splitting dataset into train/val/test sets.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')

    parser.add_argument('--regularization_scale', type=float, default=1e-4,
                        help='Concrete dropout regularization scale')
    parser.add_argument('--sampling_size', type=int, default=10,
                        help='Sampling size for MC-Dropout (USed in validation step during training)')

    # Active Learning arguments
    parser.add_argument('--al_init_ratio', type=float, default=0.25,
                        help='The ratio of initial training pool')
    parser.add_argument('--al_end_ratio', type=float, default=0.75,
                        help='The ratio of ending training pool')
    parser.add_argument('--n_loops', type=float, default=10,
                        help='The ratio of expanded set')
    # parser.add_argument('--init_train_step', type=int, default=100,
    #                     help='Epoch number for initial training on the initial training pool')
    # parser.add_argument('--active_train_step', type=int, default=10,
    #                     help='Epoch number of re-training')
    parser.add_argument('--folds', type=int, default=6,
                        help='Cross-validation times for active learning')


def modify_active_train_args(args: Namespace):
    """
    Modify and validate active learning arguments in place.

    :param args: Arguments.
    """
    args.data_path = os.path.join(f'dataset',
                                  f'{args.data_name}.csv')

    args.save_dir = os.path.join(args.save_dir,
                                 f'{args.train_strategy}',
                                 f'{args.data_name}_checkpoints',
                                 f'{args.al_type}',
                                 )


    if args.train_strategy == 'CPBayesMPP+AL':
        args.pretrain_encoder_path = os.path.join(f'results',
                                                  f'pretrain',
                                                  f'pretrain_encoder.pt')

    if args.data_name in ['delaney', 'freesolv', 'lipo', 'qm7', 'qm8', 'pdbbind']:
        args.dataset_type = 'regression'
        args.metric = 'rmse'
    elif args.data_name in ['bbbp', 'tox21', 'clintox', 'hiv', 'bace', 'sider']:
        args.dataset_type = 'classification'
        args.metric = 'roc-auc'
    else: raise ValueError(f'No supported data_name {args.data_name}')

    args.cuda = torch.cuda.is_available()


def add_visualize_active_args(parser: ArgumentParser):
    """
    Add active learning visualization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='delaney',
                        help='name of dataset to plot')


def modify_visualize_active_args(args: Namespace):
    """
    Modify and validate active learning visualization arguments in place.

    :param args: Arguments.
    """
    pass


def add_prior_predict_args(parser: ArgumentParser):
    """
    Add prior prediction arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='delaney',
                        help='name of dataset to plot')
    parser.add_argument('--prior', type=str, default='CPBayesMPP+Prior',
                        choices=['BayesMPP+Prior', 'CPBayesMPP+Prior'],
                        help='Training strategy of downstream task.'
                             'cd means training with uninformative prior.'
                             'cl means training with contrastive prior.')
    parser.add_argument('--predict_type', type=str, default='latent',
                        choices=['similarity', 'latent'],
                        help='Methods of uncertainty calibration.'
                             'prior_similarity means visualize the feature similarity between 3 types of sample pairs.'
                             '(Augmented pairs / In-distribution pairs / Out-of-distribution pairs)'
                             'prior_latent means using t-sne to visualize the 2D latent feature distribution colored by their labels.')


def modify_prior_predict_args(args: Namespace):
    """
    Modify and validate prior prediction arguments in place.

    :param args: Arguments.
    """
    assert args.data_name is not None

    args.pretrain_encoder_path = os.path.join(f'results',
                                              f'pretrain',
                                              f'pretrain_encoder.pt')

    args.pretrain_header_path = os.path.join(f'results',
                                              f'pretrain',
                                              f'pretrain_header.pt')

    args.data_path = os.path.join(f'dataset',
                                  f'{args.data_name}.csv')

    args.cuda = torch.cuda.is_available()


def add_visualize_prior_args(parser: ArgumentParser):
    """
    Add prior visualization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='delaney',
                        help='name of dataset to plot')
    parser.add_argument('--prior', type=str, default='BayesMPP',
                        choices=['BayesMPP+Prior', 'CPBayesMPP+Prior'],
                        help='Training strategy of downstream task.'
                             'cd means training with uninformative prior.'
                             'cl means training with contrastive prior.')
    parser.add_argument('--visualize_type', type=str, default='prior_latent',
                        choices=['similarity', 'latent'],
                        help='Methods of uncertainty calibration.'
                             'prior_similarity means visualize the feature similarity between 3 types of samples.'
                             '(In distribution / Out distribution / OOD samples)'
                             'prior_latent means visualize the latent feature distribution colored by their labels.')


def modify_visualize_prior_args(args: Namespace):
    """
    Modify and validate prior visualization arguments in place.

    :param args: Arguments.
    """

    args.print_name = {
        'delaney': 'ESOL',
        'freesolv': 'FreeSolv',
        'lipo': 'Lipo',
        'qm7': 'QM7',
        'qm8': 'QM8',
        'pdbbind': 'PDBbind',
    }


def add_ood_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_name', type=str, default='freesolv',
                        help='Downstream task name')
    parser.add_argument('--train_strategy', type=str, default='CPBayesMPP',
                        choices=['BayesMPP+OOD', 'CPBayesMPP+OOD'],
                        help='Training strategy of downstream task.'
                             'BayesMPP means training with uninformative prior.'
                             'CPBayesMPP means training with contrastive prior.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--kl_weight', type=float, default=10,
                        help='The regularization weight of ELBO KL term in downstream task '
                             'is determined jointly with the number of samples in the dataset.')

    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.5, 0.2, 0.3],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='List of random seeds to use when splitting dataset into train/val/test sets.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')

    parser.add_argument('--regularization_scale', type=float, default=1e-4,
                        help='Concrete dropout regularization scale')
    parser.add_argument('--sampling_size', type=int, default=20,
                        help='Sampling size for MC-Dropout (Used in validation step during training)')


def modify_ood_train_args(args: Namespace):
    """
    Modify and validate training arguments in place.

    :param args: Arguments.
    """
    args.data_path = os.path.join(f'dataset',
                                  f'{args.data_name}.csv')

    if args.train_strategy == 'BayesMPP+OOD':
        args.save_dir = os.path.join(args.save_dir,
                                     f'{args.train_strategy}',
                                     f'{args.split_type}_split')
    elif args.train_strategy == 'CPBayesMPP+OOD':
        args.pretrain_encoder_path = os.path.join(args.save_dir,
                                                  f'pretrain',
                                                  f'pretrain_encoder.pt')
        args.save_dir = os.path.join(args.save_dir,
                                     f'{args.train_strategy}',
                                     f'{args.split_type}_split', )

    else:
        raise ValueError(f'Unsupported train strategy {args.train_strategy}')

    if args.data_name in ['delaney', 'freesolv', 'lipo', 'qm7', 'qm8', 'pdbbind']:
        args.dataset_type = 'regression'
        args.metric = 'rmse'
    elif args.data_name in ['bbbp', 'tox21', 'clintox', 'hiv', 'bace', 'sider']:
        args.dataset_type = 'classification'
        args.metric = 'roc-auc'
    else:
        raise ValueError(f'No supported data_name {args.data_name}')

    if not args.seeds:
        if args.dataset_type == 'regression':
            args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds for regression
        if args.dataset_type == 'classification':
            args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]  # Use 8 random seeds for regression
            # args.seeds = [111, 222, 333]  # Use 3 random seeds for regression

    args.cuda = torch.cuda.is_available()


def add_visualize_ood_uncertainty_args(parser: ArgumentParser):
    """
    Adds uncertainty visualization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--data_name', type=str, default='delaney',
                        help='name of dataset to plot')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory where model checkpoints are be saved')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        choices=['random', 'scaffold'],
                        help='Method of splitting the dataset into train/val/test')

    parser.add_argument('--visualize_type', type=str, default='auco',
                        choices=['auco'],
                        help='Methods of uncertainty calibration.'
                             'auco for regression tasks, and ece for classification tasks.')
    parser.add_argument('--uncertainty_type', type=str, default='aleatoric',
                        choices=['aleatoric', 'epistemic', 'total'],
                        help='The uncertainty type which calibration is based')
    parser.add_argument('--legend_loc', type=str, default='lower left',
                        choices=['lower left', 'lower right', 'upper left', 'upper right'],
                        help='Location of the plot legend')
    parser.add_argument('--decimal_places', type=int, default=2,
                        help='Number of decimal places to round to')

    # parameters for ENCE plot
    parser.add_argument('--bins_num', type=int, default=10,
                        help='K-bins for ENCE calculation')
    parser.add_argument('--max_uc', type=float, default=2,
                        help='The max value of uncertainty interval')
    parser.add_argument('--uc_scaler', type=float, default=1,
                        help='The max value of uncertainty interval')


def modify_visualize_ood_uncertainty_args(args: Namespace):
    """
    Modify and validate uncertainty visualization arguments in place.

    :param args: Arguments.
    """

    args.seeds = [123, 234, 345, 456, 567, 678, 789, 890]

    args.BayesMPP_pred_list = [os.path.join(
                                            f'results',
                                            f'BayesMPP+OOD',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.CPBayesMPP_pred_list = [os.path.join(
                                            f'results',
                                            f'CPBayesMPP+OOD',
                                            f'{args.split_type}_split',
                                            f'{args.data_name}_checkpoints',
                                            f'seed_{seed}',
                                            f'preds.csv'
                                            )
                                for seed in args.seeds]

    args.print_name = {
        'delaney': 'ESOL',
        'freesolv': 'FreeSolv',
        'lipo': 'Lipo',
        'qm7': 'QM7',
        'qm8': 'QM8',
        'pdbbind': 'PDBbind',
    }

    if args.visualize_type == 'ence':
        dataset_max_uc_values = {
            'delaney': 2,
            'freesolv': 4,
            'lipo': 1.4,
            'qm7': 300,
            'qm8': 0.02,
            'pdbbind': 3.0,
        }
        args.max_uc = dataset_max_uc_values.get(args.data_name)