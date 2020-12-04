'''
Many of the principles implemented in this module were learned from the textbook "Probablistic Robotics"
A link to this book can be found at https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf
Table 3.1 on page 36 is a good jumping off location
'''

import numpy as np


def discrete_propagate(x, P, A, Rt, B=None, u=None):
    """
    Performs the Discrete Propogate step (lines 2 and 3 in table 3.1)
    :param x: The state of the filter
    :type x: (N, 1) ndarray
    :param P: The covariance of the current state
    :type P: (N, N) ndarray
    :param A: The propagation model matrix
    :type A: (N, N) ndarray
    :param Rt: The Covariance Matrix of the propagation model noise
    :type Rt: (N, N) ndarray
    :param B: The motion model input matrix
    :type B: (N, P) ndarray where p is the number of inputs
    :param u: The inputs to the motion model
    :type u: (P, 1) ndarray
    :return: x_out, P_out: the predicted state and predicted covariance
    :rtype: (N, 1) ndarray and (N, N) ndarray
    """
    x_out = A @ x
    if B is not None:
        x_out += B @ u
    P_out = A @ P @ A.conj().T + Rt
    return x_out, P_out


def runge_kutta4(f, del_t, x):
    """
    The runge kutta 4 method for propagating nonlinear continuous models
    :param f:
    :type f:
    :param del_t:
    :type del_t:
    :param x:
    :type x:
    :return:
    :rtype:
    """
    k1 = f(0, x)
    k2 = f(del_t / 2, x + del_t * k1 / 2)
    k3 = f(del_t / 2, x + del_t * k2 / 2)
    k4 = f(del_t, x + del_t * k3)
    y_out = x + del_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


def continuous_propagate(x, P, A, Rt, B=None, u=None, del_t=.1):
    """
    The propogation function for if the model is a continuous motion model
    :param x:
    :type x:
    :param P:
    :type P:
    :param A:
    :type A:
    :param Rt:
    :type Rt:
    :param B:
    :type B:
    :param u:
    :type u:
    :param del_t:
    :type del_t:
    :return:
    :rtype:
    """
    if B is not None:
        f = lambda t, xx: A @ xx + B @ u
        x_out = runge_kutta4(f, del_t, x)
        f = lambda t, xx: A @ xx + xx @ A.T + Rt
        P_out = runge_kutta4(f, del_t, P)
    else:
        A_tmp = continuous2discrete(A, del_t)
        x_out, P_out = discrete_propagate(x, P, A=A_tmp, Rt=del_t * Rt)
    return x_out, P_out


def continuous2discrete(A, del_t):
    """
    Converts a continuous model matrix to a discrete model matrix
    :param A:
    :type A:
    :param del_t:
    :type del_t:
    :return:
    :rtype:
    """
    from scipy.linalg import expm
    return expm(A * del_t)


def discrete_sensor_update(x, P, C, z, Qt, outlier_thresh=None):
    """
    Performs the Discrete sensor update (Lines 4-6 of table 2.1)
    :param x: The state of the filter
    :type x: (N, 1) ndarray
    :param P: The covariance of the current state
    :type P: (N, N) ndarray
    :param C: The sensor model Matrix
    :type C: (l, N) ndarray where L is the dimensionality of the measurement (usually 1)
    :param z: The measurement
    :type z: (l, 1) ndarray
    :param Qt: The measurement noise covariance matrix
    :type Qt: (l, l) ndarray
    :param outlier_thresh: The number of standard deviations away a measurement can be before being thrown out
    :type outlier_thresh: float
    :return: x_out, P_out  The aposteri estimates of the mean and covariance
    :rtype: (N, 1) ndarray, (N, N) ndarray
    """
    resid = (z - C @ x)
    sig_i = np.linalg.inv(C @ P @ C.T + Qt)
    if outlier_thresh is not None:
        mahalanobis = np.sqrt(resid.conj().T @ sig_i @ resid)
        print(mahalanobis)
        if mahalanobis > outlier_thresh:
            print('OUTLIER')
            return x, P
    K = P @ C.conj().T @ sig_i
    x_out = x + K @ resid
    P_out = (np.eye(len(P)) - K @ C) @ P
    return x_out, P_out


def create_synthetic_data(A, B, C, Rt, Qt, x, P, num_samples=100):
    x_list = []
    z_list = []
    for i in range(num_samples):
        x = A @ x + np.random.multivariate_normal(np.zeros(len(A)), Rt)[:, np.newaxis]
        x_list.append(x)
        z = C @ x + np.random.multivariate_normal(np.zeros(1), Qt)
        z_list.append(z.squeeze())
    return x_list, z_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    del_t = .1
    A = np.array([[1, del_t, del_t ** 2 / 2],
                  [0, 1, del_t],
                  [0, 0, 1]])
    B = None
    C = np.array([[1, 0, 0]])

    Rt = np.diag([0, 0, 2])

    Qt = np.array([[2]])

    x = np.array([[500], [0], [0]])
    P = np.zeros(1)

    x_list, z_list = create_synthetic_data(A, B, C, Rt, Qt, x, P, num_samples=100)

    plt.plot(z_list, '+')
    plt.show()

    # del_t = .1
    # A_cont = np.array([[0, 1, 0],
    #                    [0, 0, 1],
    #                    [0, 0, 0]])
    # A_disc = np.array([[1, del_t, del_t ** 2 / 2],
    #                    [0, 1, del_t],
    #                    [0, 0, 1]])
    #
    # x = np.array([[0, 0, 9.8]]).T
    # P = np.eye(3)
    # Rt = np.zeros((3, 3))
    # Rt[2, 2] = 1E-5
    # for i in range(10):
    #     x, P = continuous_propagate(x, P, A_cont, Rt, del_t=.1)
    # print(x)
    # print(P)
    # Qt = np.eye(3)
    # x, P = discrete_sensor_update(x, P, np.eye(3), x, Qt, 3)
