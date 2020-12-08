'''
Many of the principles implemented in this module were learned from the textbook "Probablistic Robotics"
A link to this book can be found at https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf
Table 3.1 on page 36 is a good jumping off location
'''

import numpy as np
import matplotlib.pyplot as plt

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


def get_transformed_covariance(C, P):
    """

    :param C:
    :type C:
    :param P:
    :type P:
    :return:
    :rtype:
    """
    return C @ P @ C.T


def create_synthetic_data(A_cont, C, Rt, Qt, x, num_samples=100):
    time_list = np.linspace(0,30,num_samples)
    delta = time_list[1]-time_list[0]
    time_list += np.random.random(num_samples)/5*delta
    x_list, z_list = [], []
    for idx in range(len(time_list)-1):
        delta_t = time_list[idx+1]-time_list[idx]
        A = continuous2discrete(A_cont,delta_t)
        x = A @ x + np.random.multivariate_normal(np.zeros(len(A)), delta_t*Rt)[:, np.newaxis]
        x_list.append(x)
        z = C @ x + np.random.multivariate_normal(np.zeros(1), Qt)
        z_list.append(z.squeeze())
    return time_list, x_list, z_list

# def kalman_filter(x0,P0,A,B,C,Rt,Qt,u_list,z_list):


def kalman_josh_main(A_cont,C,Rt,Qt,time_list,z_list):
    x = np.array([[z_list[0],0,0]]).T
    P = np.diag([1,10,10])


    x_list, P_list = [], []

    for idx, measurement in enumerate(z_list):
        delta_t = time_list[idx + 1] - time_list[idx]
        A = continuous2discrete(A_cont, delta_t)
        x, P = discrete_propagate(x, P, A, delta_t*Rt)
        x, P = discrete_sensor_update(x, P, C, measurement, Qt)
        x_list.append(x)
        P_list.append(P)

    x_list = np.atleast_3d(x_list)
    P_list = np.atleast_3d(P_list)
    fig, ax_list = plt.subplots(3,1, sharex=True, figsize=(10,10))
    plt.sca(ax_list[0])
    plt.plot(time_list[1:], z_list, 'r+', label='Stock Prices')
    label_list = ['Price', 'Velocity', 'Acceleration']

    A_1day =continuous2discrete(A_cont,1)
    x_pred, P_pred = [x_list[-1,:,:]], [P_list[-1,:,:]]
    for i in range(7):
        x, P = discrete_propagate(x_pred[-1],P_pred[-1],A_1day,Rt)
        x_pred.append(x)
        P_pred.append(P)

    t_pred = time_list[-1]+np.arange(8)
    x_pred = np.atleast_3d(x_pred)
    P_pred = np.atleast_3d(P_pred)

    for i, label in enumerate(label_list):
        plt.sca(ax_list[i])
        plt.ylabel(label)
        plt.plot(time_list[1:],x_list[:,i,0])
        mean_pred = x_pred[:,i,0]
        plt.plot(t_pred,mean_pred,'c2')
        e0 = np.array([[0, 0, 0]]).T
        e0[i,0] = 1
        e0 = e0[np.newaxis,...]
        conf_interval = 2*np.sqrt(e0.swapaxes(-1,-2)@P_pred@e0).squeeze()
        plt.fill_between(t_pred,mean_pred+conf_interval,mean_pred-conf_interval,alpha = .2)





    plt.xlabel('Time (days)')
    plt.show()

if __name__ == '__main__':


    A_cont = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])

    C = np.array([[1, 0, 0]])

    Rt = np.diag([6, 0, .005])

    Qt = np.array([[2]])

    x = np.array([[500], [0], [0]])


    time, x_list, z_list = create_synthetic_data(A_cont, C, Rt, Qt, x, num_samples=1000)
    kalman_josh_main(A_cont, C, Rt, Qt, time, z_list)

    # plt.plot(z_list, '+')
    # plt.show()

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
