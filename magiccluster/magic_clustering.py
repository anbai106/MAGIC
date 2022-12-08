from .clustering import DualSVM_Subtype, DualSVM_Subtype_transfer_learning
from .base import OPNMF_Input
import os, pickle
from .utils import cluster_stability_across_resolution, summary_clustering_result_multiscale, shift_list, consensus_clustering_across_c, make_cv_partition
import numpy as np

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def clustering(participant_tsv, opnmf_dir, output_dir, k_min, k_max, num_components_min, num_components_max, num_components_step, cv_repetition, covariate_tsv=None, cv_strategy='hold_out', save_models=False,
            cluster_predefined_c=0.25, class_weight_balanced=True, weight_initialization_type='DPP', num_iteration=50,
            num_consensus=20, tol=1e-8, multiscale_tol=0.85, n_threads=8, verbose=False):
    """
    pyhydra core function for clustering
    Args:
        participant_tsv:str, path to the participant_tsv tsv, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
        opnmf_dir: str, path to store the OPNMF results
        output_dir: str, path to store the clustering results
        k_min: int, minimum k (number of clusters)
        k_max: int, maximum k (number of clusters)
        cv_repetition: int, number of repetitions for cross-validation (CV)
        covariate_tsv: str, path to the tsv containing the covaria`tes, eg., age or sex. The header (first 3 columns) of
                     the tsv file is the same as the feature_tsv, following the BIDS convention.
        cv_strategy: str, cross validation strategy used. Default is hold_out. choices=['k_fold', 'hold_out']
        save_models: Bool, if save all models during CV. Default is False to save space.
                      Set true only if you are going to apply the trained model to unseen data.
        cluster_predefined_c: Float, default is 0.25. The predefined best c if you do not want to perform a nested CV to
                             find it. If used, it should be a float number
        class_weight_balanced: Bool, default is True. If the two groups are balanced.
        weight_initialization_type: str, default is DPP. The strategy for initializing the weight to control the
                                    hyperplances and the subpopulation of patients. choices=["random_hyperplane", "random_assign", "k_means", "DPP"]
        num_iteration: int, default is 50. The number of iterations to iteratively optimize the polytope.
        num_consensus: int, default is 20. The number of repeats for consensus clustering to eliminate the unstable clustering.
        tol: float, default is 1e-8. Clustering stopping criterion.
        multiscale_tol: float, default is 0.85. Double cyclic optimization stopping criterion.
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: clustering outputs.

    """
    ### For voxel approach
    print('MAGIC for semi-supervised clustering...')
    if covariate_tsv == None:
        input_data = OPNMF_Input(opnmf_dir, participant_tsv, covariate_tsv=None)
    else:
        input_data = OPNMF_Input(opnmf_dir, participant_tsv, covariate_tsv=covariate_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    if cv_strategy == "hold_out":
        ## check if data split has been done, if yes, the pickle file is there
        if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
            split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
        else:
            split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)
    elif cv_strategy == "k_fold":
        ## check if data split has been done, if yes, the pickle file is there
        if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl')):
            split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl'), 'rb'))
        else:
            split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)

    print('Data split has been done!\n')

    print('Starts semi-supervised clustering...')
    ### Here, semi-supervised clustering with multi-scale feature reduction learning
    if (num_components_max - num_components_min) % num_components_step != 0:
        raise Exception('Number of componnets step should be divisible!')

    ## C lists
    C_list = list(range(num_components_min, num_components_max+num_components_step, num_components_step))
    ## first loop on different initial C.
    for i in range(len(C_list)):
        c_list = shift_list(C_list, i)
        num_run = 0
        loop = True
        print('Initialize C == %d\n' % C_list[i])
        while loop:
            for j in range(len(c_list)):
                if num_run == 0:
                    num_run += 1
                    k_continuing = np.arange(k_min, k_max+1).tolist()
                    print('First C == %d\n' % c_list[j])
                    output_dir_loop = os.path.join(output_dir, 'initialization_c_' + str(C_list[i]), 'clustering_run' + str(num_run))
                    wf_clustering = DualSVM_Subtype(input_data,
                                                    participant_tsv,
                                                    split_index,
                                                    cv_repetition,
                                                    k_min,
                                                    k_max,
                                                    output_dir_loop,
                                                    opnmf_dir,
                                                    balanced=class_weight_balanced,
                                                    num_consensus=num_consensus,
                                                    num_iteration=num_iteration,
                                                    tol=tol,
                                                    predefined_c=cluster_predefined_c,
                                                    weight_initialization_type=weight_initialization_type,
                                                    n_threads=n_threads,
                                                    num_components_min=c_list[j],
                                                    num_components_max=c_list[j],
                                                    num_components_step=num_components_step,
                                                    save_models=save_models,
                                                    verbose=verbose)

                    wf_clustering.run()
                else: ## initialize the model from the former resolution
                    num_run += 1
                    print('Transfer learning on resolution C == %d for run == %d\n' % (c_list[j], num_run))
                    output_dir_tl = os.path.join(output_dir, 'initialization_c_' + str(C_list[i]))
                    wf_clustering = DualSVM_Subtype_transfer_learning(input_data,
                                                                      participant_tsv,
                                                                      split_index,
                                                                      cv_repetition,
                                                                      k_continuing,
                                                                      output_dir_tl,
                                                                      opnmf_dir,
                                                                      balanced=class_weight_balanced,
                                                                      num_iteration=num_iteration,
                                                                      tol=tol,
                                                                      predefined_c=cluster_predefined_c,
                                                                      weight_initialization_type=weight_initialization_type,
                                                                      n_threads=n_threads,
                                                                      num_component=c_list[j],
                                                                      num_component_former=c_list[j-1],
                                                                      num_run=num_run)

                    wf_clustering.run()

                    ### check the clustering stability between the current C and former C
                    k_continuing, k_converged = cluster_stability_across_resolution(c_list[j], c_list[j-1], os.path.join(output_dir, 'initialization_c_' + str(C_list[i])), k_continuing, num_run, stop_tol=multiscale_tol)

                    if not k_continuing:
                        loop = False
                        break

        ## After cross validate the hyperparameter k & num_components, summarize the results into a single tsv file.
        if not k_continuing:
            summary_clustering_result_multiscale(os.path.join(output_dir, 'initialization_c_' + str(C_list[i])), k_min, k_max)

    ## consensus learning based on different initialization Cs
    print('Computing the final consensus group membership!\n')
    consensus_clustering_across_c(output_dir, C_list, k_min, k_max)
    print('Finish...')