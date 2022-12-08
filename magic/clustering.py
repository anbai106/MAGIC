import os
import numpy as np
import pandas as pd
from .utils import consensus_clustering, cv_cluster_stability, hydra_solver_svm_tl
from .base import WorkFlow
from utils import hydra_solver_svm

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"
class DualSVM_Subtype(WorkFlow):

    def __init__(self, input, participant_tsv, split_index, cv_repetition, k_min, k_max, output_dir, opnmf_dir, balanced=True,
                test_size=0.2, num_consensus=20, num_iteration=50, tol=1e-6, predefined_c=None, weight_initialization_type='DPP',
                 n_threads=8, num_components_min=10, num_components_max=100, num_components_step=10, save_models=False,
                 verbose=True):

        self._input = input
        self._participant_tsv = participant_tsv
        self._split_index = split_index
        self._cv_repetition = cv_repetition
        self._output_dir = output_dir
        self._opnmf_dir = opnmf_dir
        self._k_min = k_min
        self._k_max = k_max
        self._balanced = balanced
        self._test_size = test_size
        self._num_consensus = num_consensus
        self._num_iteration = num_iteration
        self._tol = tol
        self._predefined_c = predefined_c
        self._weight_initialization_type = weight_initialization_type
        self._k_range_list = list(range(k_min, k_max + 1))
        self._n_threads = n_threads
        self._num_components_min = num_components_min
        self._num_components_max = num_components_max
        self._num_components_step = num_components_step
        self._save_models = save_models
        self._verbose = verbose


    def run(self):

        ## by default, we solve the problem using dual solver with a linear kernel.
        for num_component in range(self._num_components_min, self._num_components_max + self._num_components_step, self._num_components_step):

            if os.path.exists(os.path.join(self._output_dir, 'component_' + str(num_component), "adjusted_rand_index.tsv")):
                print("This number of component have been trained and converged: %d" % num_component)
            else:
                x = self._input.get_x(num_component, self._opnmf_dir)
                y = self._input.get_y_raw()
                data_label_folds_ks = np.zeros((y.shape[0], self._cv_repetition, self._k_max - self._k_min + 1)).astype(int)

                for i in range(self._cv_repetition):
                        for j in self._k_range_list:
                            print('Applying pyHRDRA for finding %d clusters. Repetition: %d / %d...\n' % (j, i+1, self._cv_repetition))
                            training_final_prediction = hydra_solver_svm(i, x[self._split_index[i][0]], y[self._split_index[i][0]], j, self._output_dir,
                                                                     self._num_consensus, self._num_iteration, self._tol, self._balanced, self._predefined_c,
                                                                     self._weight_initialization_type, self._n_threads, self._save_models, self._verbose)


                            # change the final prediction's label: test data to be 0, the rest training data will b e updated by the model's prediction
                            data_label_fold = y.copy()
                            data_label_fold[self._split_index[i][1]] = 0 # all test data to be 0
                            data_label_fold[self._split_index[i][0]] = training_final_prediction ## assign the training prediction
                            data_label_folds_ks[:, i, j - self._k_min] = data_label_fold

                print('Estimating clustering stability...\n')
                ## for the adjusted rand index, only consider the PT results
                adjusted_rand_index_results = np.zeros(self._k_max - self._k_min + 1)
                index_pt = np.where(y == 1)[0]  # index for PTs
                for m in range(self._k_max - self._k_min + 1):
                    result = data_label_folds_ks[:, :, m][index_pt]
                    adjusted_rand_index_result = cv_cluster_stability(result, self._k_range_list[m])
                    # saving each k result into the final adjusted_rand_index_results
                    adjusted_rand_index_results[m] = adjusted_rand_index_result

                print('Computing the final consensus group membership...\n')
                final_assignment_ks = -np.ones((self._input.get_y_raw().shape[0], self._k_max - self._k_min + 1)).astype(int)
                for n in range(self._k_max - self._k_min + 1):
                    result = data_label_folds_ks[:, :, n][index_pt]
                    final_assignment_ks_pt = consensus_clustering(result, n + self._k_min)
                    final_assignment_ks[index_pt, n] = final_assignment_ks_pt + 1

                print('Saving the final results...\n')
                # save_cluster_results(adjusted_rand_index_results, final_assignment_ks)
                columns = ['ari_' + str(i) + '_subtypes' for i in self._k_range_list]
                ari_df = pd.DataFrame(adjusted_rand_index_results[:, np.newaxis].transpose(), columns=columns)
                ari_df.to_csv(os.path.join(self._output_dir, 'adjusted_rand_index.tsv'), index=False, sep='\t',
                              encoding='utf-8')

                # save the final assignment for consensus clustering across different folds
                participant_df = pd.read_csv(self._participant_tsv, sep='\t')
                columns = ['assignment_' + str(i) for i in self._k_range_list]
                cluster_df = pd.DataFrame(final_assignment_ks, columns=columns)
                all_df = pd.concat([participant_df, cluster_df], axis=1)
                all_df.to_csv(os.path.join(self._output_dir, 'clustering_assignment.tsv'), index=False,
                              sep='\t', encoding='utf-8')

class DualSVM_Subtype_transfer_learning(WorkFlow):
    """
    Instead of training from scratch, we initialize the polytope from the former C
    """
    def __init__(self, input, participant_tsv, split_index, cv_repetition, k_list, output_dir, opnmf_output, balanced=True,
                 test_size=0.2, num_iteration=50, tol=1e-6, predefined_c=None,
                 weight_initialization_type='DPP', n_threads=8, num_component=10, num_component_former=10, num_run=None):

        self._input = input
        self._participant_tsv = participant_tsv
        self._split_index = split_index
        self._cv_repetition = cv_repetition
        self._output_dir = output_dir
        self._opnmf_output = opnmf_output
        self._k_list = k_list
        self._balanced = balanced
        self._test_size = test_size
        self._num_iteration = num_iteration
        self._tol = tol
        self._predefined_c = predefined_c
        self._weight_initialization_type = weight_initialization_type
        self._n_threads = n_threads
        self._num_component = num_component
        self._num_component_former = num_component_former
        self._num_run = num_run

    def run(self):

        if os.path.exists(os.path.join(self._output_dir, 'clustering_run' + str(self._num_run), 'component_' + str(self._num_component), "adjusted_rand_index.tsv")):
            print("This number of component have been trained and converged: %d" % self._num_component)
        else:
            print("cross validate for num_component, running for %d components for feature selection" % self._num_component)
            x = self._input.get_x(self._num_component, self._opnmf_output)

            y = self._input.get_y_raw()
            data_label_folds_ks = np.zeros((y.shape[0], self._cv_repetition, len(self._k_list))).astype(int)

            for i in range(self._cv_repetition):
                for j in range(len(self._k_list)):
                    print('Applying HRDRA for finding %d clusters. Repetition: %d / %d...\n' % (self._k_list[j], i+1, self._cv_repetition))
                    training_final_prediction = hydra_solver_svm_tl(self._num_component, self._num_component_former, i, x[self._split_index[i][0]], y[self._split_index[i][0]], self._k_list[j], self._output_dir,
                                                            self._num_iteration, self._tol, self._balanced, self._predefined_c,
                                                            self._n_threads, self._num_run)


                    # change the final prediction's label: test data to be 0, the rest training data will be updated by the model's prediction
                    data_label_fold = y.copy()
                    data_label_fold[self._split_index[i][1]] = 0 # all test data to be 0
                    data_label_fold[self._split_index[i][0]] = training_final_prediction ## assign the training prediction
                    data_label_folds_ks[:, i, j] = data_label_fold

            print('Finish the clustering procedure!\n')

            print('Estimating clustering stability!\n')
            ## for the adjusted rand index, only consider the PT results
            adjusted_rand_index_results = np.zeros(len(self._k_list))
            index_pt = np.where(y == 1)[0]  # index for PTs
            for m in range(len(self._k_list)):
                result = data_label_folds_ks[:, :, m][index_pt] ## the result of each K during all runs of CV
                adjusted_rand_index_result = cv_cluster_stability(result, self._k_list[m])

                # saving each k result into the final adjusted_rand_index_results
                adjusted_rand_index_results[m] = adjusted_rand_index_result
            print('Done!\n')

            print('Computing the final consensus group membership!\n')
            final_assignment_ks = -np.ones((self._input.get_y_raw().shape[0], len(self._k_list))).astype(int)
            for n in range(len(self._k_list)):
                result = data_label_folds_ks[:, :, n][index_pt]
                final_assignment_ks_pt = consensus_clustering(result, n + self._k_list[0]) ## the final subtype assignment is performed with consensus clustering with KMeans
                final_assignment_ks[index_pt, n] = final_assignment_ks_pt + 1
            print('Done!\n')

            print('Saving the final results!\n')
            # save_cluster_results(adjusted_rand_index_results, final_assignment_ks)
            columns = ['ari_' + str(i) + '_subtypes' for i in self._k_list]
            ari_df = pd.DataFrame(adjusted_rand_index_results[:, np.newaxis].transpose(), columns=columns)
            ari_df.to_csv(os.path.join(self._output_dir, 'clustering_run' + str(self._num_run), 'component_' + str(self._num_component), 'adjusted_rand_index.tsv'), index=False, sep='\t',
                          encoding='utf-8')

            # save the final assignment for consensus clustering across different folds
            df_feature = pd.read_csv(self._participant_tsv, sep='\t')
            columns = ['assignment_' + str(i) for i in self._k_list]
            participant_df = df_feature.iloc[:, :3]
            cluster_df = pd.DataFrame(final_assignment_ks, columns=columns)
            all_df = pd.concat([participant_df, cluster_df], axis=1)
            all_df.to_csv(os.path.join(self._output_dir, 'clustering_run' + str(self._num_run), 'component_' + str(self._num_component), 'clustering_assignment.tsv'), index=False,
                          sep='\t', encoding='utf-8')

            print('Done!\n')