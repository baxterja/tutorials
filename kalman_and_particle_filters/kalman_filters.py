import numpy as np


def discrete_propagate(x, P, A, Rt, B=None, u=None):
    x_out = A @ x
    if B is not None:
        x_out += B @ u
    P_out = A @ P @ A.conj().T + Rt
    return x_out, P_out


def runge_kutta4(f, del_t, x):
    k1 = f(0, x)
    k2 = f(del_t / 2, x + del_t * k1 / 2)
    k3 = f(del_t / 2, x + del_t * k2 / 2)
    k4 = f(del_t, x + del_t * k3)
    y_out = x + del_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


def continuous_propagate(x, P, A, Rt, B=None, u=None, del_t=.1):
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
    from scipy.linalg import expm
    return expm(A * del_t)


def discrete_sensor_update(x, P, C, z, Qt, outlier_thresh=None):
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

discrete_kalman(x, P, A, Rt, )

if __name__ == '__main__':
    del_t = 4
    A_cont = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    A_disc = np.array([[1, del_t, del_t ** 2 / 2],
                       [0, 1, del_t],
                       [0, 0, 1]])

    x = np.array([[0, 0, 9.8]]).T
    P = np.eye(3)
    Rt = np.zeros((3, 3))
    Rt[2, 2] = 1E-5
    for i in range(10):
        x, P = continuous_propagate(x, P, A_cont, Rt, del_t=.1)
    print(x)
    print(P)
    Qt = np.eye(3)
    x, P = discrete_sensor_update(x, P, np.eye(3),x,Qt,3)
