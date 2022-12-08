import argparse

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def magic_func(args):
    """
    The default function to run classification.
    Args:
        args: args from parser

    Returns:

    """
    from magiccluster.magic_clustering import clustering
    clustering(
        args.participant_tsv,
        args.opnmf_dir,
        args.output_dir,
        args.k_min,
        args.k_max,
        args.num_components_min,
        args.num_components_max,
        args.num_components_step,
        args.cv_repetition,
        args.covariate_tsv,
        args.cv_strategy,
        args.save_models,
        args.cluster_predefined_c,
        args.class_weight_balanced,
        args.weight_initialization_type,
        args.num_iteration,
        args.num_consensus,
        args.tol,
        args.multiscale_tol,
        args.n_threads,
        args.verbose
    )

def parse_command_line():
    """
    Definition for the commandline parser
    Returns:

    """

    parser = argparse.ArgumentParser(
        prog='magiccluster-cluster',
        description='Perform multi-scale semi-supervised clustering using MAGIC...')

    subparser = parser.add_subparsers(
        title='''Task to perform...''',
        description='''We now only allow to use MAGIC for clustering''',
        dest='task',
        help='''****** Tasks proposed by MAGIC ******''')

    subparser.required = True

########################################################################################################################

    ## Add arguments for ADML ROI classification
    clustering_parser = subparser.add_parser(
        'cluster',
        help='Perform clustering with MAGIC.')

    clustering_parser.add_argument(
        'participant_tsv',
        help="Path to the tsv containing the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the diagnosis. ",
        default=None
    )

    clustering_parser.add_argument(
        'opnmf_dir',
        help='Path to the directory of where SOPNMF was run (the voxel-wise images should be run first with SOPNMF).',
        default=None
    )

    clustering_parser.add_argument(
        'output_dir',
        help='Path to the directory of where to store the final output.',
        default=None
    )
    
    clustering_parser.add_argument(
        'k_min',
        help='Number of cluster (k) minimum value.',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'k_max',
        help='Number of cluster (k) maximum value.',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'num_components_min',
        help='Number of the min PSC for the SOPNMF',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'num_components_max',
        help='Number of the max PSC for the SOPNMF',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'num_components_step',
        help='The step size between the min and the max PSC for the SOPNMF',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'cv_repetition',
        help='Number of repetitions for the chosen cross-validation (CV).',
        default=None, type=int
    )

    clustering_parser.add_argument(
        '--covariate_tsv',
        help="Path to the tsv containing covariates, following the BIDS convention. The first 3 columns is the same as feature_tsv",
        default=None,
        type=str
    )

    clustering_parser.add_argument(
        '-cs', '--cv_strategy',
        help='Chosen CV strategy, default is hold_out. ',
        type=str, default='hold_out',
        choices=['k_fold', 'hold_out'],
    )

    clustering_parser.add_argument(
        '-sm', '--save_models',
        help='If save modles during all repetitions of CV. ',
        default=False, action="store_true"
    )

    clustering_parser.add_argument(
        '--cluster_predefined_c',
        type=float,
        default=0.25,
        help="Predefined hyperparameter C of SVM. Default is 0.25. "
             "Better choice may be guided by HYDRA global classification with nested CV for optimal C searching. "
    )

    clustering_parser.add_argument(
        '-cwb', '--class_weight_balanced',
        help='If group samples are balanced, default is True. ',
        default=False, action="store_true"
    )

    clustering_parser.add_argument(
        '-wit', '--weight_initialization_type',
        help='Strategy for initializing the weighted sample matrix of the polytope. ',
        type=str, default='DPP',
        choices=['DPP', 'random_assign'],
    )

    clustering_parser.add_argument(
        '--num_iteration',
        help='Number of iteration to converge each SVM.',
        default=50, type=int
    )

    clustering_parser.add_argument(
        '--num_consensus',
        help='Number of iteration for inner consensus clusetering.',
        default=20, type=int
    )

    clustering_parser.add_argument(
        '--tol',
        help='Clustering stopping criterion, until the polytope becomes stable',
        default=1e-8, type=float
    )

    clustering_parser.add_argument(
        '--multiscale_tol',
        help='Clustering stopping criterion, until the multi-scale clustering solution stable',
        default=0.85, type=float
    )

    clustering_parser.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    clustering_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    clustering_parser.set_defaults(func=magic_func)
    
    

