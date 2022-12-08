import glob
import numpy as np
import scipy
import os, pickle
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit
from joblib import dump
import pandas as pd
from multiprocessing.pool import ThreadPool
from sklearn.svm import SVC

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def elem_sym_poly(lambda_value, k):
    """
    given a vector of lambdas and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    :param lambda_value: the corresponding eigenvalues
    :param k: number of clusters
    :return:
    """
    N = lambda_value.shape[0]
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1

    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i, j] = E[i, j - 1] + lambda_value[j-1] * E[i - 1, j - 1]

    return E


def sample_k(lambda_value, k):
    """
    Pick k lambdas according to p(S) \propto prod(lambda \in S)
    :param lambda_value: the corresponding eigenvalues
    :param k: the number of clusters
    :return:
    """

    ## compute elementary symmetric polynomials
    E = elem_sym_poly(lambda_value, k)

    ## ietrate over the lambda value
    num = lambda_value.shape[0]
    remaining = k
    S = np.zeros(k)
    while remaining > 0:
        #compute marginal of num given that we choose remaining values from 0:num-1
        if num == remaining:
            marg = 1
        else:
            marg = lambda_value[num-1] * E[remaining-1, num-1] / E[remaining, num]

        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining-1] = num
            remaining = remaining - 1
        num = num - 1
    return S

def sample_dpp(evalue, evector, k=None):
    """
    sample a set Y from a dpp.  evalue, evector are a decomposed kernel, and k is (optionally) the size of the set to return
    :param evalue: eigenvalue
    :param evector: normalized eigenvector
    :param k: number of cluster
    :return:
    """
    if k == None:
        # choose eigenvectors randomly
        evalue = np.divide(evalue, (1 + evalue))
        evector = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
    else:
        v = sample_k(evalue, k) ## v here is a 1d array with size: k

    k = v.shape[0]
    v = v.astype(int)
    v = [i - 1 for i in v.tolist()]  ## due to the index difference between matlab & python, here, the element of v is for matlab
    V = evector[:, v]

    ## iterate
    y = np.zeros(k)
    for i in range(k, 0, -1):
        ## compute probabilities for each item
        P = np.sum(np.square(V), axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        y[i-1] = np.where(np.random.rand(1) < np.cumsum(P))[0][0]
        y = y.astype(int)

        # choose a vector to eliminate
        j = np.where(V[y[i-1], :])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, 1)

        ## Update V
        if V.size == 0:
           pass
        else:
            V = np.subtract(V, np.multiply(Vj, (V[y[i-1], :] / Vj[y[i-1]])[:, np.newaxis]).transpose())  ## watch out the dimension here

        ## orthogonalize
        for m in range(i - 1):
            for n in range(m):
                V[:, m] = np.subtract(V[:, m], np.matmul(V[:, m].transpose(), V[:, n]) * V[:, n])

            V[:, m] = V[:, m] / np.linalg.norm(V[:, m])

    y = np.sort(y)

    return y

def proportional_assign(l, d):
    """
    Proportional assignment based on margin
    :param l: int
    :param d: int
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    invL = np.divide(1, l)
    idx = np.isinf(invL)
    invL[idx] = d[idx]

    for i in range(l.shape[0]):
        pos = np.where(invL[i, :] > 0)[0]
        neg = np.where(invL[i, :] < 0)[0]
        if pos.size != 0:
            invL[i, neg] = 0
        else:
            invL[i, :] = np.divide(invL[i, :], np.amin(invL[i, :]))
            invL[i, invL[i, :] < 1] = 0

    S = np.multiply(invL, np.divide(1, np.sum(invL, axis=1))[:, np.newaxis])

    return S

def random_init_dirichlet(k, num_pt):
    """
    a sample from a dirichlet distribution
    :param k: number of clusters
    :param num_pt: number of PT
    :return:
    """
    a = np.ones(k)
    s = np.random.dirichlet(a, num_pt)

    return s

def hydra_init_weight(X, y, k, index_pt, index_cn, weight_initialization_type):
    """
    Function performs initialization for the polytope of mlni
    Args:
        X: the input features
        y: the label
        k: number of predefined clusters
        index_pt: list, the index for patient subjects
        index_cn: list, the index for control subjects
        weight_initialization_type: the type of chosen initialization method
    Returns:

    """
    if weight_initialization_type == "DPP":  ##
        num_subject = y.shape[0]
        W = np.zeros((num_subject, X.shape[1]))
        for j in range(num_subject):
            ipt = np.random.randint(index_pt.shape[0])
            icn = np.random.randint(index_cn.shape[0])
            W[j, :] = X[index_pt[ipt], :] - X[index_cn[icn], :]

        KW = np.matmul(W, W.transpose())
        KW = np.divide(KW, np.sqrt(np.multiply(np.diag(KW)[:, np.newaxis], np.diag(KW)[:, np.newaxis].transpose())))
        evalue, evector = np.linalg.eig(KW)
        Widx = sample_dpp(np.real(evalue), np.real(evector), k)
        prob = np.zeros((len(index_pt), k))  # only consider the PTs

        for i in range(k):
            prob[:, i] = np.matmul(np.multiply(X[index_pt, :], np.divide(1, np.linalg.norm(X[index_pt, :], axis=1))[:, np.newaxis]), W[Widx[i], :].transpose())

        l = np.minimum(prob - 1, 0)
        d = prob - 1
        S = proportional_assign(l, d)

    elif weight_initialization_type == "random_hyperplane":
        print("TODO")

    elif weight_initialization_type == "random_assign":
        S = random_init_dirichlet(k, len(index_pt))

    elif weight_initialization_type == "k_means":
        print("TODO")
    else:
        raise Exception("Not implemented yet!")

    return S

def hydra_solver_svm(num_repetition, X, y, k, output_dir, num_consensus, num_iteration, tol, balanced, predefined_c,
                     weight_initialization_type, n_threads, save_models, verbose):
    """
    This is the main function of HYDRA, which find the convex polytope using a supervised classification fashion.
    Args:
        num_repetition: int, number of repetitions for CV
        X: input matrix for features
        y: input for group label
        k: number of clusters
        output_dir: the path for output
        num_consensus: int, number of runs for consensus clustering
        num_iteration: int, number of maximum iterations for running HYDRA
        tol: float, tolerance value for model convergence
        balanced: if sample imbalance should be considered during model optimization
        predefined_c: predefined c for SVM for clustering
        weight_initialization_type: the type of initialization of the weighted sample matrix
        n_threads: number of threads used
        save_models: if save all models during CV
        verbose: if output is verbose

    Returns:

    """
    censensus_assignment = np.zeros((y[y == 1].shape[0], num_consensus))  ## only consider the PTs

    index_pt = np.where(y == 1)[0]  # index for PTs
    index_cn = np.where(y == -1)[0]  # index for CNs

    for i in range(num_consensus):
        weight_sample = np.ones((y.shape[0], k)) / k
        ## depending on the weight initialization strategy, random hyperplanes were initialized with maximum diversity to constitute the convex polytope
        weight_sample_pt = hydra_init_weight(X, y, k, index_pt, index_cn, weight_initialization_type)
        weight_sample[index_pt] = weight_sample_pt  ## only replace the sample weight of the PT group
        ## cluster assignment is based on this svm scores across different SVM/hyperplanes
        svm_scores = np.zeros((weight_sample.shape[0], weight_sample.shape[1]))
        update_weights_pool = ThreadPool(processes=n_threads)

        for j in range(num_iteration):
            for m in range(k):
                sample_weight = np.ascontiguousarray(weight_sample[:, m])
                if np.count_nonzero(sample_weight[index_pt]) == 0:
                    if verbose == True:
                        print(
                            "Cluster dropped, meaning that all PT has been assigned to one single hyperplane in iteration: %d" % (
                                        j - 1))
                        print(
                            "Be careful, this could cause problem because of the ill-posed solution. Especially when k==2")
                else:
                    results = update_weights_pool.apply_async(launch_svc,
                                                              args=(X, y, predefined_c, sample_weight, balanced))
                    weight_coef = results.get()[0]
                    intesept = results.get()[1]
                    ## Apply the data again the trained model to get the final SVM scores
                    svm_scores[:, m] = (np.matmul(weight_coef, X.transpose()) + intesept).transpose().squeeze()

            cluster_index = np.argmax(svm_scores[index_pt], axis=1)

            ## decide the converge of the polytope based on the toleration
            weight_sample_hold = weight_sample.copy()
            # after each iteration, first set the weight of patient rows to be 0
            weight_sample[index_pt, :] = 0
            # then set the pt's weight to be 1 for the assigned hyperplane
            for n in range(len(index_pt)):
                weight_sample[index_pt[n], cluster_index[n]] = 1

            ## check the loss comparted to the tolorence for stopping criteria
            loss = np.linalg.norm(np.subtract(weight_sample, weight_sample_hold), ord='fro')
            if verbose == True:
                print("The loss is: %f" % loss)
            if loss < tol:
                if verbose == True:
                    print(
                        "The polytope has been converged for iteration %d in finding %d clusters in consensus running: %d" % (
                        j, k, i))
                break
        update_weights_pool.close()
        update_weights_pool.join()

        ## update the cluster index for the consensus clustering
        censensus_assignment[:, i] = cluster_index + 1

    ## do censensus clustering
    final_predict = consensus_clustering(censensus_assignment.astype(int), k)

    ## after deciding the final convex polytope, we refit the training data once to save the best model
    weight_sample_final = np.zeros((y.shape[0], k))
    ## change the weight of PTs to be 1, CNs to be 1/k

    # then set the pt's weight to be 1 for the assigned hyperplane
    for n in range(len(index_pt)):
        weight_sample_final[index_pt[n], final_predict[n]] = 1

    weight_sample_final[index_cn] = 1 / k
    update_weights_pool_final = ThreadPool(processes=n_threads)
    ## create the final polytope by applying all weighted subjects
    for o in range(k):
        sample_weight = np.ascontiguousarray(weight_sample_final[:, o])
        results = update_weights_pool_final.apply_async(launch_svc, args=(X, y, predefined_c, sample_weight, balanced))

        if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'models')):
            os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'models'))

        ## save the final model for the k SVMs/hyperplanes
        if save_models == True:
            if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'models')):
                os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'models'))

            dump(results.get()[2], os.path.join(output_dir, str(k) + '_clusters', 'models',
                                                'svm-' + str(o) + '_cv_' + str(num_repetition) + '.joblib'))
        else:
            ## only save the last repetition
            if not os.path.isfile(os.path.join(output_dir, str(k) + '_clusters', 'models',
                                               'svm-' + str(o) + '_last_repetition.joblib')):
                dump(results.get()[2], os.path.join(output_dir, str(k) + '_clusters', 'models',
                                                    'svm-' + str(o) + '_last_repetition.joblib'))
    update_weights_pool_final.close()
    update_weights_pool_final.join()

    y[index_pt] = final_predict + 1

    if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'tsv')):
        os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'tsv'))

    ### save results also in tsv file for each repetition
    ## save the assigned weight for each subject across k-fold
    columns = ['hyperplane' + str(i) for i in range(k)]
    weight_sample_df = pd.DataFrame(weight_sample_final, columns=columns)
    weight_sample_df.to_csv(
        os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'weight_sample_cv_' + str(num_repetition) + '.tsv'),
        index=False, sep='\t', encoding='utf-8')

    ## save the final_predict_all
    columns = ['y_hat']
    y_hat_df = pd.DataFrame(y, columns=columns)
    y_hat_df.to_csv(os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'y_hat_cv_' + str(num_repetition) + '.tsv'),
                    index=False, sep='\t', encoding='utf-8')

    ## save the pt index
    columns = ['pt_index']
    pt_df = pd.DataFrame(index_pt, columns=columns)
    pt_df.to_csv(os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'pt_index_cv_' + str(num_repetition) + '.tsv'),
                 index=False, sep='\t', encoding='utf-8')

    return y

def GLMcorrection(X_train, Y_train, covar_train, X_test, covar_test):
    """
    Eliminate the confound of covariate, such as age and sex, from the disease-based changes.
    Ref: "Age Correction in Dementia Matching to a Healthy Brain"
    :param X_train: array, training features
    :param Y_train: array, training labels
    :param covar_train: array, ttraining covariate data
    :param X_test: array, test labels
    :param covar_test: array, ttest covariate data
    :return: corrected training & test feature data
    """
    Yc = X_train[Y_train == -1]
    Xc = covar_train[Y_train == -1]
    Xc = np.concatenate((Xc, np.ones((Xc.shape[0], 1))), axis=1)
    beta = np.matmul(np.matmul(Yc.transpose(), Xc), np.linalg.inv(np.matmul(Xc.transpose(), Xc)))
    num_col = beta.shape[1]
    X_train_cor = (X_train.transpose() - np.matmul(beta[:, : num_col - 1], covar_train.transpose())).transpose()
    X_test_cor = (X_test.transpose() - np.matmul(beta[:, : num_col - 1], covar_test.transpose())).transpose()

    return X_train_cor, X_test_cor

def launch_svc(X, y, predefined_c, sample_weight, balanced):
    """
    Lauch svc classifier of sklearn
    Args:
        X: input matrix for features
        y: input matrix for label
        predefined_c: predefined C
        sample_weight: the weighted sample matrix
        balanced:

    Returns:

    """
    if not balanced:
        model = SVC(kernel='linear', C=predefined_c)
    else:
        model = SVC(kernel='linear', C=predefined_c, class_weight='balanced')

    ## fit the different SVM/hyperplanes
    model.fit(X, y, sample_weight=sample_weight)

    weight_coef = model.coef_
    intesept = model.intercept_

    return weight_coef, intesept, model

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the numpy array is symmetric or not
    Args:
        a:
        rtol:
        atol:

    Returns:

    """
    result = np.allclose(a, a.T, rtol=rtol, atol=atol)
    return result

def make_cv_partition(diagnosis, cv_strategy, output_dir, cv_repetition, seed=None):
    """
    Randomly generate the data split index for different CV strategy.

    :param diagnosis: the list for labels
    :param cv_repetition: the number of repetitions or folds
    :param output_dir: the output folder path
    :param cv_repetition: the number of repetitions for CV
    :param seed: random seed for sklearn split generator. Default is None
    :return:
    """
    unique = list(set(diagnosis))
    y = np.array(diagnosis)
    if len(unique) == 2: ### CV for classification and clustering
        if cv_strategy == 'k_fold':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = StratifiedKFold(n_splits=cv_repetition, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        elif cv_strategy == 'hold_out':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = StratifiedShuffleSplit(n_splits=cv_repetition, test_size=0.2, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            raise Exception("this cross validation strategy has not been implemented!")
    elif len(unique) == 1:
        raise Exception("Diagnosis cannot be the same for all participants...")
    else: ### CV for regression, no need to be stratified
        if cv_strategy == 'k_fold':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_' + str(cv_repetition) + '-fold.pkl')

            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = KFold(n_splits=cv_repetition, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        elif cv_strategy == 'hold_out':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_' + str(cv_repetition) + '-holdout.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = ShuffleSplit(n_splits=cv_repetition, test_size=0.2, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            raise Exception("this cross validation strategy has not been implemented!")

    with open(splits_indices_pickle, 'wb') as s:
        pickle.dump(splits_indices, s)

    return splits_indices, splits_indices_pickle

def consensus_clustering(clustering_results, k):
    """
    This function performs consensus clustering on a co-occurence matrix
    :param clustering_results: an array containing all the clustering results across different iterations, in order to
    perform
    :param k:
    :return:
    """

    num_pt = clustering_results.shape[0]
    cooccurence_matrix = np.zeros((num_pt, num_pt))

    for i in range(num_pt - 1):
        for j in range(i + 1, num_pt):
            cooccurence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

    cooccurence_matrix = np.add(cooccurence_matrix, cooccurence_matrix.transpose())
    ## here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(cooccurence_matrix, axis=1)), cooccurence_matrix)

    Laplacian_norm = np.subtract(np.eye(num_pt), np.matmul(np.matmul(np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1))), cooccurence_matrix), np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1)))))
    ## replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    ## check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    if check_symmetric(Laplacian_norm):
        ## extract the eigen value and vector
        ## matlab eig equivalence is eigh, not eig from numpy or scipy, see this post: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        ## Note, the eigenvector is not unique, thus the matlab and python eigenvector may be different, but this will not affect the results.
        evalue, evector = scipy.linalg.eigh(Laplacian_norm)
    else:
        # evalue, evector = np.linalg.eig(Laplacian_norm)
        raise Exception("The Laplacian matrix should be symmetric here...")

    ## check if the eigen vector is complex
    if np.any(np.iscomplex(evector)):
        evalue, evector = scipy.linalg.eigh(Laplacian)

    ## create the kmean algorithm with sklearn
    kmeans = KMeans(n_clusters=k, n_init=20).fit(evector.real[:, 0: k])
    final_predict = kmeans.labels_

    return final_predict

def cv_cluster_stability(result, k):
    """
    To compute the adjusted rand index across different pair of 2 folds cross CV
    :param result:
    :return:
    """

    num_pair = 0
    aris = []
    if k == 1:
        adjusted_rand_index = 0  ## note, here, we manually set it to be 0, because it does not make sense when k==1. TODO, need to clarify if there is really heterogeneity in the data, i.e., k == 1 or k>1
    else:
        for i in range(result.shape[1] - 1):
            for j in range(i+1, result.shape[1]):
                num_pair += 1
                non_zero_index = np.all(result[:, [i, j]], axis=1)
                pair_result = result[:, [i, j]][non_zero_index]
                ari = adjusted_rand_score(pair_result[:, 0], pair_result[:, 1])
                aris.append(ari)

        adjusted_rand_index = np.mean(np.asarray(aris))

    return adjusted_rand_index

def hydra_solver_svm_tl(num_component, num_component_former, num_repetition, X, y, k, output_dir, num_iteration, tol, balanced, predefined_c, n_threads, num_run):
    """
    This is the main function of HYDRA, which find the convex polytope using a supervised classification fashion.
    :param num_repetition: the number of iteration of CV currently. This is helpful to reconstruct the model and also moniter the processing
    :param X: corrected training data feature
    :param y: traing data label
    :param k: hyperparameter for desired number of clusters in patients
    :param options: commandline parameters
    :return: the optimal model
    """
    index_pt = np.where(y == 1)[0]  # index for PTs
    index_cn = np.where(y == -1)[0]  # index for CNs

    ### initialize the final weight for the polytope from the former C
    weight_file = os.path.join(output_dir, 'clustering_run' + str(num_run-1), 'component_' + str(num_component_former), str(k) + '_clusters', 'tsv', 'weight_sample_cv_' + str(num_repetition) + '.tsv')
    weight_sample = pd.read_csv(weight_file, sep='\t').to_numpy()

    ## cluster assignment is based on this svm scores across different SVM/hyperplanes
    svm_scores = np.zeros((weight_sample.shape[0], weight_sample.shape[1]))
    update_weights_pool = ThreadPool(n_threads)
    for j in range(num_iteration):
        for m in range(k):
            sample_weight = np.ascontiguousarray(weight_sample[:, m])

            if np.count_nonzero(sample_weight[index_pt]) == 0:
                print("Cluster dropped, meaning that all PT has been assigned to one single hyperplane in iteration: %d" % (j-1))
                svm_scores[:, m] = np.asarray([np.NINF] * (y.shape[0]))
            else:

                results = update_weights_pool.apply_async(launch_svc, args=(X, y, predefined_c, sample_weight, balanced))
                weight_coef = results.get()[0]
                intesept = results.get()[1]
                ## Apply the data again the trained model to get the final SVM scores
                svm_scores[:, m] = (np.matmul(weight_coef, X.transpose()) + intesept).transpose().squeeze()


        final_predict = np.argmax(svm_scores[index_pt], axis=1)

        ## decide the converge of the polytope based on the toleration
        weight_sample_hold = weight_sample.copy()
        # after each iteration, first set the weight of patient rows to be 0
        weight_sample[index_pt, :] = 0
        # then set the pt's weight to be 1 for the assigned hyperplane
        for n in range(len(index_pt)):
            weight_sample[index_pt[n], final_predict[n]] = 1

        ## check the loss comparted to the tolorence for stopping criteria
        loss = np.linalg.norm(np.subtract(weight_sample, weight_sample_hold), ord='fro')
        print("The loss is: %f" % loss)
        if loss < tol:
            print("The polytope has been converged for iteration %d in finding %d clusters" % (j, k))
            break
    update_weights_pool.close()
    update_weights_pool.join()

    ## after deciding the final convex polytope, we refit the training data once to save the best model
    weight_sample_final = np.zeros((y.shape[0], k))
    ## change the weight of PTs to be 1, CNs to be 1/k

    # then set the pt's weight to be 1 for the assigned hyperplane
    for n in range(len(index_pt)):
        weight_sample_final[index_pt[n], final_predict[n]] = 1

    weight_sample_final[index_cn] = 1 / k
    update_weights_pool_final = ThreadPool(n_threads)

    for o in range(k):
        sample_weight = np.ascontiguousarray(weight_sample_final[:, o])
        if np.count_nonzero(sample_weight[index_pt]) == 0:
            print("Cluster dropped, meaning that the %d th hyperplane is useless!" % (o))
        else:
            results = update_weights_pool_final.apply_async(launch_svc, args=(X, y, predefined_c, sample_weight, balanced))

            ## save the final model for the k SVMs/hyperplanes
            if not os.path.exists(
                    os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component),
                                 str(k) + '_clusters', 'models')):
                os.makedirs(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component),
                                         str(k) + '_clusters', 'models'))

            dump(results.get()[2],
                 os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component),
                              str(k) + '_clusters', 'models',
                              'svm-' + str(o) + '_last_repetition.joblib'))

    update_weights_pool_final.close()
    update_weights_pool_final.join()

    y[index_pt] = final_predict + 1

    if not os.path.exists(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component), str(k) + '_clusters', 'tsv')):
        os.makedirs(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component), str(k) + '_clusters', 'tsv'))

    ## save the assigned weight for each subject across k-fold
    columns = ['hyperplane' + str(i) for i in range(k)]
    weight_sample_df = pd.DataFrame(weight_sample_final, columns=columns)
    weight_sample_df.to_csv(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component), str(k) + '_clusters', 'tsv', 'weight_sample_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    ## save the final_predict_all
    columns = ['y_hat']
    y_hat_df = pd.DataFrame(y, columns=columns)
    y_hat_df.to_csv(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component), str(k) + '_clusters', 'tsv', 'y_hat_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    ## save the pt index
    columns = ['pt_index']
    pt_df = pd.DataFrame(index_pt, columns=columns)
    pt_df.to_csv(os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(num_component), str(k) + '_clusters', 'tsv', 'pt_index_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    return y

def cluster_stability_across_resolution(c, c_former, output_dir, k_continuing, num_run, stop_tol=0.98):
    """
    To evaluate the stability of clustering across two different C for stopping criterion.
    Args:
        c:
        c_former:
        output_dir:
        k_continuing:
        num_run:
        stop_tol:
        max_num_iter:

    Returns:

    """
    ## read the output of current C and former Cs
    cluster_ass1 = os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(c), 'clustering_assignment.tsv')
    ass1_df = pd.read_csv(cluster_ass1, sep='\t')
    ass1_df = ass1_df.loc[ass1_df['diagnosis'] == 1]

    cluster_ass2 = os.path.join(output_dir, 'clustering_run' + str(num_run-1), 'component_' + str(c_former), 'clustering_assignment.tsv')
    ass2_df = pd.read_csv(cluster_ass2, sep='\t')
    ass2_df = ass2_df.loc[ass2_df['diagnosis'] == 1]

    df_final = pd.DataFrame(columns=['C', 'K', 'num_run'])

    k_continuing_update = []
    k_converged = []
    for i in k_continuing:
        ari = adjusted_rand_score(ass1_df['assignment_' + str(i)], ass2_df['assignment_' + str(i)])
        print("For k == %d, run %d got ARI == %f compared to former run" % (i, num_run, ari))
        if ari < stop_tol and num_run:
            k_continuing_update.append(i)
        else:
            print("Model has been converged or stop at the max iteration: C == %d, K == %d and run == %d" % (c, i, num_run))
            k_converged.append(i)
            df_row = pd.DataFrame(columns=['C', 'K', 'num_run'])
            df_row.loc[len(['C', 'K', 'num_run'])] = [c, i, num_run]
            df_final = df_final.append(df_row)

    if len(k_converged) != 0:
        df_final.to_csv(os.path.join(output_dir, 'results_convergence_run' + str(num_run) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    return k_continuing_update, k_converged

def summary_clustering_result_multiscale(output_dir, k_min, k_max):
    """
    This is a function to summarize the clustering results
    :param num_components_min:
    :param num_components_max:
    :param num_components_step:
    :param output_dir:
    :return:
    """
    clu_col_list = ['assignment_' + str(e) for e in range(k_min, k_max)]
    df_clusters = pd.DataFrame(columns=clu_col_list)

    ## read the convergence tsv
    convergence_tsvs = [f for f in glob.glob(output_dir + "/results_convergence_*.tsv", recursive=True)]

    for tsv in convergence_tsvs:
        df_convergence = pd.read_csv(tsv, sep='\t')

        ## sorf by K
        df_convergence = df_convergence.sort_values(by=['K'])

        for i in range(df_convergence.shape[0]):
            k = df_convergence['K'].tolist()[i]
            num_run = df_convergence['num_run'].tolist()[i]
            C = df_convergence['C'].tolist()[i]
            cluster_file = os.path.join(output_dir, 'clustering_run' + str(num_run), 'component_' + str(C), 'clustering_assignment.tsv')

            df_cluster = pd.read_csv(cluster_file, sep='\t')
            if i == 0:
                df_header = df_cluster.iloc[:, 0:3]
            assign = df_cluster['assignment_' + str(k)]
            df_clusters['assignment_' + str(k)] = assign

    ## concatenqte the header
    df_assignment = pd.concat((df_header, df_clusters), axis=1)

    ## save the result
    df_assignment.to_csv(os.path.join(output_dir, 'results_cluster_assignment_final.tsv'), index=False, sep='\t', encoding='utf-8')

def shift_list(c_list, index):
    """
    This is a function to reorder a list to have all posibility by putting each element in the first place
    Args:
        c_list: list to shift
        index: the index of which element to shift

    Returns:

    """
    new_list = c_list[index:] + c_list[:index]

    return new_list

def consensus_clustering_across_c(output_dir, c_list, k_min, k_max):
    """
    This is for consensus learning at the end across different Cs
    Args:
        output_dir:
        c_list:

    Returns:

    """
    k_list = list(range(k_min, k_max+1))
    for k in k_list:
        for i in c_list:
            clu_col_list = ['c_' + str(i) + '_assignment_' + str(e) for e in k_list]
            df_clusters = pd.DataFrame(columns=clu_col_list)

            tsv = os.path.join(output_dir, 'initialization_c_' + str(i), 'results_cluster_assignment_final.tsv')
            df = pd.read_csv(tsv, sep='\t')

            if i == c_list[0]:
                df_header = df.iloc[:, 0:3]
            df_clusters['c_' + str(i) + '_assignment_' + str(k)] = df['assignment_' + str(k)]
            if i == c_list[0]:
                df_final = df_clusters
            else:
                df_final = pd.concat([df_final, df_clusters], axis=1)

    ## concatenate the header and the results
    df_final = pd.concat([df_header, df_final], axis=1)
    df_final_pt = df_final.loc[df_final['diagnosis'] == 1]
    df_final_cn = df_final.loc[df_final['diagnosis'] == -1]
    num_cn = df_final_cn.shape[0]

    ## create the final dataframe to store the final assignment
    col_list = ['assignment_' + str(e) for e in k_list]
    df_final_assign = pd.DataFrame(columns=col_list)

    ## read the final clustering assignment for each C
    for m in k_list:
        columns_names = ['c_' + str(e) + '_assignment_' + str(m) for e in c_list]
        assignment_pt = df_final_pt[columns_names]
        final_predict_pt = consensus_clustering(assignment_pt.to_numpy(), m)
        final_predict_cn = -2 * np.ones(num_cn)
        final_predict = np.concatenate((final_predict_cn, final_predict_pt)).astype(int)
        df_final_assign['assignment_' + str(m)] = final_predict + 1

    df_final_assign = pd.concat([df_header, df_final_assign], axis=1)
    ## save the final results into tsv file.
    df_final_assign.to_csv(os.path.join(output_dir, 'results_cluster_assignment_final.tsv'), index=False, sep='\t',
                         encoding='utf-8')