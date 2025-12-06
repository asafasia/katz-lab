import numpy as np

def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


# def state_measurement_stretch(states, regularization=True, lambda_reg=0.05):
#     # fid_matrix = resonator_args['fidelity_matrix']
#     inverse_fid_matrix = np.linalg.inv(fid_matrix_2)
#     # bias = (fid_matrix[0][0] + fid_matrix[1][1]) / 2 - 0.5
#     bias = 0
#     # p = 0.95
#     if isinstance(states, (int, float)):
#         vec = np.array([1 - states, states])
#         new_vec = readout_correction_single(vec, regularization=regularization, lambda_reg=lambda_reg)
#         # new_vec = vec.T @ inverse_fid_matrix - bias
#         return new_vec[1]
#     else:
#         new_vec = []
#         for state in states:
#             vec = np.array([1 - state, state])
#             # new_vec.append(vec.T @ inverse_fid_matrix - bias)
#             new_vec.append(readout_correction_single(vec, regularization=regularization, lambda_reg=lambda_reg))
#
#         new_vec = np.array(new_vec)
#
#         return new_vec.T[1]