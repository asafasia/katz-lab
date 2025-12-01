import collections.abc
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import viridis
from scipy.stats import zscore
from experiment_utils.configuration import *
# from qualang_tools.analysis import two_state_discriminator
from scipy.spatial import distance
# from scipy.stats import chi2
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.mixture import GaussianMixture
# import scipy
import cvxpy as cp


from experiment_utils.change_args import modify_json


def rotate_data(X, angle_degrees):
    # Convert angle to radians
    angle = np.radians(angle_degrees)

    # Rotation matrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Apply rotation
    X_rotated = X @ R
    return X_rotated

def analyze_distribution(data):
    """
    Analyze the distribution of the data and return the mean, covariance
    data should be a 2D numpy array where each row is a data point.
    """
    # Compute mean and covariance
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)
    return mean, covariance

def get_cov_ellipse(cov, mean, threshold, ax, **kwargs):
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals*threshold)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def state_discrimination(I,Q, gef_centers=None, gef_covs=None, n_states=2):
    """
    Perform state discrimination using the provided I Q data , based on the Mahalanobis distance from the gef centers defined in args file.
     the Mahalanobis distance is calculated based on the 3 covariance matrices stored in the args file.
    :param I: array of I quadrature data
    :param Q: array of Q quadrature data
    :param gef_centers: list of centers for the three states, if None, will use the centers from the args file
    :param gef_covs: list of covariance matrices for the three states, if None, will use the covariance matrices from the args file
    :param n_states: number of states to discriminate, either 2 or 3.  default is 2 (g and e)
    :return: state: int
    """
    if gef_centers is None:
        gef_centers = resonator_args['gef_centers']
    if gef_covs is None:
        gef_covs = resonator_args['gef_covariance_mats']

    IQ_data = np.column_stack((I, Q))
    distances2=[]
    for i in range(n_states):
        mean = np.array(gef_centers[i])
        cov = np.array(gef_covs[i])
        cov_inv  = np.linalg.inv(cov)
        d2 = np.array([distance.mahalanobis(x, mean, cov_inv) ** 2 for x in IQ_data])
        distances2.append(d2)
    state = np.argmin(distances2, axis=0)
    return state




def discard_outliers(X, threshold=95):
    """ Discard outliers from the dataset X using the Mahalanobis distance.
    Args:
        X (np.ndarray): The input data, shape (n_samples, n_features).
        threshold (float): The percentile threshold for outlier detection (default is 95).
    Returns:
        X_filtered (np.ndarray): The filtered data without outliers.
        threshold (float): The threshold distance value used for filtering.
        """
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    # Mahalanobis distances
    d_squared = np.array([distance.mahalanobis(x, mean, inv_cov) ** 2 for x in X])

    threshold_d = np.percentile(d_squared, threshold)

    mask = d_squared < threshold_d
    X_filtered = X[mask]
    return X_filtered, threshold_d

def three_state_discrimination_calib(data1, data2, data3, threshold=[97,95,94], plot_blobs=True, prints = True, update_args=True):
    """
    Perform three state discrimination calibration using the provided data.
    This function filters outliers using z-score and Mahalanobis distance,
    then computes mean values and covariance matrices for the three IQ distributions (after filtering).
    It also computes the fidelity matrix for the three states, assuming that state discrimination is based on the Mahalanobis distance
    with the computed means and covariance matrices.
    the 2x2 fidelity matrix for the g and e states is also computed.
    The results are updated in the args file.
    the function also plots the IQ distributions
    :param data1: array of I Q data for state g
    :param data2: array of I Q data for state e
    :param data3: array of I Q data for state f
    :param threshold: threshold for Mahalanobis distance filtering, defined as a percentile. can be a single value or a list of three values for each state
    :param plot_blobs = True: whether to plot the IQ blobs of the three states
    :param prints: whether to print the fidelty matices
    :param update_args: whether to update the args file with the new means and covariance matrices
    :return means: list of means for the three states
    :return covs: list of covariance matrices for the three states
    :return fidelity_mat: 3x3 matrix with fidelity values for g, e and f states
    :return fidelity_mat_ge: 2x2 matrix with fidelity values for g and e states

    """
    if not isinstance(threshold, collections.abc.Iterable):
        thresholds_mahalanobis = [threshold, threshold, threshold]
    else:
        thresholds_mahalanobis = threshold

    # filter outliers using z-score
    data1_filt= data1[(np.abs(zscore(data1)) < 3).all(axis=1)]
    data2_filt= data2[(np.abs(zscore(data2)) < 1.8).all(axis=1)]
    data3_filt= data3[(np.abs(zscore(data3)) < 1.8).all(axis=1)]

    # # filter outliers using z-score
    # data1_filt= data1
    # data2_filt= data2
    # data3_filt= data3

    # filter outliers using Mahalanobis distance
    data1_filt, thresh1 = discard_outliers(data1_filt, threshold=thresholds_mahalanobis[0])
    data2_filt, thresh2 = discard_outliers(data2_filt,threshold=thresholds_mahalanobis[1])
    data3_filt, thresh3 = discard_outliers(data3_filt,threshold=thresholds_mahalanobis[2])
     # repreat on purpose
    data1_filt, thresh1 = discard_outliers(data1_filt, threshold=thresholds_mahalanobis[0])
    data2_filt, thresh2 = discard_outliers(data2_filt,threshold=thresholds_mahalanobis[1])
    data3_filt, thresh3 = discard_outliers(data3_filt,threshold=thresholds_mahalanobis[2])

    data = [data1,data2,data3]
    data_filt = [data1_filt, data2_filt, data3_filt]
    covs = []
    inv_covs = []
    means = []


    for i,X in enumerate(data_filt):
        mean, cov = analyze_distribution(X)
        means.append(mean)
        covs.append(cov)
        inv_covs.append(np.linalg.inv(cov))

    if update_args:
        modify_json(qubit, 'resonator', "gef_centers", [mean.tolist() for mean in means])
        modify_json(qubit, 'resonator', "gef_covariance_mats", [cov.tolist() for cov in covs])

    fidelity_mat = get_fidelity_matrix(data1, data2, data3, gef_centers=means, gef_covs=covs)
    fidelity_mat_ge = get_fidelity_matrix(data1, data2, gef_centers=means, gef_covs=covs)
    if prints:
        print("full fidelity matrix (from post analysis with updated means and covariances):")
        print(fidelity_mat)

        print("fidelity matrix ge (from post analysis with updated means and covariances):")
        print(fidelity_mat_ge)

    if update_args:
        modify_json(qubit, 'resonator', "fidelity_matrix_3", fidelity_mat.tolist())
        modify_json(qubit, 'resonator', "fidelity_matrix_2", fidelity_mat_ge.tolist())
    # print(f"Fidelity matrix: {fidelity_mat_ge}")

    e1, e2, e3 = means

    # Plot the results
    # e1 = gmm_means[0, :]
    if plot_blobs:
        plt.figure(figsize=(8, 6))
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        plt.scatter(data1[:,0],data1[:,1], color=clrs[0], s=1, alpha=0.35)
        plt.scatter(data1_filt[:, 0], data1_filt[:, 1],color=clrs[0], s=2, alpha=0.7)
        # plt.axline(matched_means[0], slope = slope_gmm1, color='blue', label='fit g_gmm', linewidth=2)
        # plt.axline(e1, slope=slope_pca1, linestyle="--",color ='blue', label='fit g pca', linewidth=2)
        plt.scatter(data2[:,0],data2[:,1], color=clrs[1], s=1, alpha=0.35)
        plt.scatter(data2_filt[:, 0], data2_filt[:, 1],color=clrs[1], s=2, alpha=0.7)
        #
        plt.scatter(data3[:,0],data3[:,1], color=clrs[2], s=1, alpha=0.35)
        plt.scatter(data3_filt[:, 0], data3_filt[:, 1],color=clrs[2], s=2, alpha=0.7)


        ax = plt.gca()
        thresholds_ellipse = [thresh1, thresh2, thresh3]

        for i in range(3):
            mean, cov = analyze_distribution(data_filt[i])
            get_cov_ellipse(cov, mean, threshold=thresholds_ellipse[i], ax=ax, edgecolor=clrs[i], facecolor='none', linestyle='--', linewidth=1.5)


        # Plot means
        plt.plot(e1[0], e1[1], 'o', color='blue', label='Cluster g', markersize=10)
        # plt.plot(*matched_means[0], '^', color='blue', label='Cluster g_gmm', markersize=10)
        plt.plot(e2[0], e2[1], 'o', color='red', label='Cluster e', markersize=10)
        # plt.plot(*matched_means[1], '^', color='red', label='Cluster e_gmm', markersize=10)
        plt.plot(e3[0], e3[1], 'o', color='#006400', label='Cluster f', markersize=10)
        # plt.plot(*matched_means[2], '^', color='#006400', label='Cluster f_gmm', markersize=10)

        plt.title('3-state discrimination')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # c = np.array([e1,e2,e3])

    return means, covs, fidelity_mat, fidelity_mat_ge

def two_state_discrimination_calib(data1, data2,  threshold=[95, 97], plot_blobs=True, prints = True, update_args = True, e2f=False):
    """
    Perform two state discrimination calibration using the provided data.
    This function filters outliers using z-score and Mahalanobis distance,
    then computes mean values and covariance matrices for the two IQ distributions (after filtering).
    It also computes the fidelity matrix for the two states, assuming that state discrimination is based on the Mahalanobis distance.
    with the computed means and covariance matrices.
    The results are updated in the args file.
    the function also plots the IQ distributions
    :param data1: array of I Q data for state g
    :param data2: array of I Q data for state e
    :param threshold: threshold for Mahalanobis distance filtering, defined as a percentile. can be a single value or a list of 2 values (for each state)
    :param plot_blobs = True: whether to plot the IQ blobs of the two states
    :param prints: whether to print the fidelty matices
    :param update_args: whether to update the args file with the new means and covariance matrices
    :return means: list of means for the two states
    :return covs: list of covariance matrices for the two states
    :return fidelity_mat: 2x2 matrix with fidelity values for g and e states
    """
    if not isinstance(threshold, collections.abc.Iterable):
        thresholds_mahalanobis = [threshold, threshold, threshold]
    else:
        thresholds_mahalanobis = threshold

    # filter outliers using z-score
    data1_filt= data1[(np.abs(zscore(data1)) < 3).all(axis=1)]
    data2_filt= data2[(np.abs(zscore(data2)) < 2.5).all(axis=1)]

    # filter outliers using Mahalanobis distance
    data1_filt, thresh1 = discard_outliers(data1_filt, threshold=thresholds_mahalanobis[0])
    data2_filt, thresh2 = discard_outliers(data2_filt,threshold=thresholds_mahalanobis[1])
     # repreat on purpose
    data1_filt, thresh1 = discard_outliers(data1_filt, threshold=thresholds_mahalanobis[0])
    data2_filt, thresh2 = discard_outliers(data2_filt,threshold=thresholds_mahalanobis[1])

    data = [data1,data2]
    data_filt = [data1_filt, data2_filt]
    covs = []
    inv_covs = []
    means = []


    for i,X in enumerate(data_filt):
        mean, cov = analyze_distribution(X)
        means.append(mean)
        covs.append(cov)
        inv_covs.append(np.linalg.inv(cov))

    # save the means and covariance matrices in the args file, but make sure not to touch the values for the f state:
    current_means = gef_centers
    current_covs = gef_covs
    if e2f:
        means_to_save = [means[0], np.array(current_means[1]), means[1]]  # store the e results in f
        covs_to_save = [covs[0], current_covs[1], covs[1]]  # store e resutls in f
    else:
        means_to_save = [means[0], means[1], np.array(current_means[2])]  # keep the f state mean
        covs_to_save = [covs[0], covs[1], current_covs[2]]  # keep the f state covariance matrix
    if update_args:
        modify_json(qubit, 'resonator', "gef_centers", [mean.tolist() for mean in means_to_save])
        modify_json(qubit, 'resonator', "gef_covariance_mats", [cov.tolist() for cov in covs_to_save])


    fidelity_mat = get_fidelity_matrix(data1, data2, gef_centers=means, gef_covs=covs)
    if prints:
        print("fidelity matrix (from post analysis with updated means and covariances):")
        print(fidelity_mat)

    if update_args:
        modify_json(qubit, 'resonator', "fidelity_matrix_2", fidelity_mat.tolist())
    # print(f"Fidelity matrix: {fidelity_mat_ge}")

    e1, e2 = means



    # Plot the results

    if plot_blobs:
        # e1 = gmm_means[0, :]
        plt.figure(figsize=(8, 6))
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        plt.scatter(data1[:,0],data1[:,1], color=clrs[0], s=1, alpha=0.35)
        plt.scatter(data1_filt[:, 0], data1_filt[:, 1],color=clrs[0], s=2, alpha=0.7)
        # plt.axline(matched_means[0], slope = slope_gmm1, color='blue', label='fit g_gmm', linewidth=2)
        # plt.axline(e1, slope=slope_pca1, linestyle="--",color ='blue', label='fit g pca', linewidth=2)
        plt.scatter(data2[:,0],data2[:,1], color=clrs[1], s=1, alpha=0.35)
        plt.scatter(data2_filt[:, 0], data2_filt[:, 1],color=clrs[1], s=2, alpha=0.7)
        #


        ax = plt.gca()
        thresholds_ellipse = [thresh1, thresh2]

        for i in range(2):
            mean, cov = analyze_distribution(data_filt[i])
            get_cov_ellipse(cov, mean, threshold=thresholds_ellipse[i], ax=ax, edgecolor=clrs[i], facecolor='none', linestyle='--', linewidth=1.5)


        # Plot means
        plt.plot(e1[0], e1[1], 'o', color='blue', label='Cluster g', markersize=10)
        # plt.plot(*matched_means[0], '^', color='blue', label='Cluster g_gmm', markersize=10)
        plt.plot(e2[0], e2[1], 'o', color='red', label='Cluster e', markersize=10)
        # plt.plot(*matched_means[1], '^', color='red', label='Cluster e_gmm', markersize=10)

        plt.title('2-state discrimination')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')


    return means, covs, fidelity_mat

def readout_correction_single(pop_vec, regularization=True, lambda_reg = 0.05):
    if len(pop_vec) ==2:
        fid_mat  = fid_matrix_2
    if len(pop_vec) == 3:
        fid_mat  =  fid_matrix_3

    if regularization:
        # pop_true = cp.Variable(len(pop_vec))
        p_true = cp.Variable(len(pop_vec))

        # Objective: minimize ||A*p_true - p_measured||^2 + lambda * ||p_true||^2
        objective = cp.Minimize(cp.sum_squares(fid_mat @ p_true - pop_vec) + lambda_reg * cp.sum_squares(p_true))

        # Constraints: probabilities must be non-negative and sum to 1
        constraints = [p_true >= 0, cp.sum(p_true) == 1]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return p_true.value
    else:
        inv_fid = np.linalg.inv(fid_mat)
        return np.dot(inv_fid.T, pop_vec.T).T

def readout_correction(pop_vecs):
    pop_vecs_cor = pop_vecs.copy()
    if len(pop_vecs.shape)==1:
        return readout_correction_single(pop_vecs)
    else:
        for idx in itertools.product(*[range(pop_vecs.shape[i]) for i in range(len(pop_vecs.shape) -1)]):
            pop_vecs_cor[idx] = readout_correction_single(pop_vecs[idx])
    return pop_vecs_cor


def state_measurement_stretch(states, regularization=True, lambda_reg=0.05):
    # fid_matrix = resonator_args['fidelity_matrix']
    inverse_fid_matrix = np.linalg.inv(fid_matrix_2)
    # bias = (fid_matrix[0][0] + fid_matrix[1][1]) / 2 - 0.5
    bias = 0
    # p = 0.95
    if isinstance(states, (int, float)):
        vec = np.array([1 - states, states])
        new_vec = readout_correction_single(vec, regularization=regularization, lambda_reg=lambda_reg)
        # new_vec = vec.T @ inverse_fid_matrix - bias
        return new_vec[1]
    else:
        new_vec = []
        for state in states:
            vec = np.array([1 - state, state])
            # new_vec.append(vec.T @ inverse_fid_matrix - bias)
            new_vec.append(readout_correction_single(vec, regularization=regularization, lambda_reg=lambda_reg))

        new_vec = np.array(new_vec)

        return new_vec.T[1]



def get_fidelity_matrix(IQ_g, IQ_e, IQ_f=None, gef_centers=None, gef_covs=None):
    """
    Calculate the fidelity matrix for two states based on IQ data.
    :param IQ_g: array of I Q data for state g
    :param IQ_e: array of I Q data for state e
    :param gef_centers: list of centers for the two states, if None, will use the centers from the args file
    :param gef_covs: list of covariance matrices for the two states, if None, will use the covariance matrices from the args file

    :return: fidelity_matrix: 2x2 matrix with fidelity values for g and e states
    """
    if gef_centers is None:
        gef_centers = resonator_args['gef_centers']
    if gef_covs is None:
        gef_covs = [np.array(cov) for cov in  resonator_args['gef_covariance_mats']]


    if IQ_f is None:
        n_states=2
    else:
        n_states = 3

    data = [IQ_g, IQ_e, IQ_f]

    fidelity_matrix = np.zeros((n_states,n_states))
    for i in range(n_states):
        X = data[i]
        state = state_discrimination(X[:, 0], X[:, 1], gef_centers=gef_centers, gef_covs=gef_covs, n_states=n_states)
        for j in range(n_states):
            fidelity_matrix[i, j] = np.sum(state == j) / len(X)
    return fidelity_matrix

if __name__ == "__main__":
    # Set random seed for reproducibility

    # Number of samples per class
    n_samples = 10000

    # Define means and covariances for the three Gaussian distributions
    mean1 = [0.0, 0.0]
    cov1 = [[0.5, 0.0],
            [0.0, 0.5]]

    mean2 = [3.0, 2.0]
    cov2 = [[0.5, 0.0],
            [0.0, 0.5]]

    mean3 = [-2.0, 1.0]
    cov3 = [[0.5, 0.0],
            [0.0, 0.5]]

    # Generate data
    data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    data2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    data3 = np.random.multivariate_normal(mean3, cov3, n_samples)

    three_state_discrimination(data1, data2, data3)
    plt.show()

