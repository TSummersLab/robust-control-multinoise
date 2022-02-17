import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt
import cvxpy as cp
import itertools

from matrixmath import is_pos_def, mdot, specrad, positive_definite_part, dlyap, dare


def gen_rand_pd(n):
    Qdiag = np.diag(npr.rand(n))
    Qvecs = la.qr(npr.randn(n, n))[0]
    Q = np.dot(Qvecs, np.dot(Qdiag, Qvecs.T))
    return Q


def sim(A, T, x0):
    n = x0.size
    x_hist = np.zeros([T, n])
    x_hist[0] = x0
    for t in range(T-1):
        x_hist[t+1] = np.dot(A, x_hist[t])
    return x_hist


def calc_MN_i(A, Delta, theta, R, S):
    M = dlyap((A + theta*Delta).T, R)
    N = dlyap((A - theta*Delta).T, S)
    return M, N


def calc_MN(A, Delta_all, theta_all, R_all, S_all):
    M_all = np.zeros([p, n, n])
    N_all = np.zeros([p, n, n])
    for i in range(p):
        M_all[i], N_all[i] = calc_MN_i(A, Delta_all[i], theta_all[i], R_all[i], S_all[i])
    return M_all, N_all


def calc_PD(M_all, N_all):
    return (M_all+N_all) / 2, (M_all-N_all) / 2


def calc_Q(R_all, S_all):
    return (R_all+S_all) / 2


def check_assumption1_i(A, Delta, Q, D, theta):
    A_D_Delta = np.dot(A.T, np.dot(D, Delta))
    LHS = Q+theta * (A_D_Delta+A_D_Delta.T)
    RHS = np.zeros(n)
    return is_pos_def(LHS-RHS)


def check_assumption1(A, Delta_all, R_all, S_all, theta_all, tol=1e-12):
    M_all, N_all = calc_MN(A, Delta_all, theta_all, R_all, S_all)
    P_all, D_all = calc_PD(M_all, N_all)
    Q_all = calc_Q(R_all, S_all)
    P = np.sum([weight_sq_all[i] * P_all[i] for i in range(p)], axis=0)

    for i in range(p):
        Delta = Delta_all[i]
        Q = Q_all[i]
        D = D_all[i]
        theta = theta_all[i]
        A_D_Delta = np.dot(A.T, np.dot(D, Delta))
        LHS = Q+theta * (A_D_Delta+A_D_Delta.T)
        RHS = np.zeros(n)
        # print(LHS-RHS)
        if not is_pos_def(LHS - RHS + tol*np.eye(n)):
            return P, False
    return P, True


def check_assumption2(A, Delta_all, R_all, S_all, theta_all, psi_sq_all, tol=1e-12):
    M_all, N_all = calc_MN(A, Delta_all, theta_all, R_all, S_all)
    P_all, D_all = calc_PD(M_all, N_all)
    P = np.sum([weight_sq_all[i] * P_all[i] for i in range(p)], axis=0)

    for i in range(p):
        Pi = P_all[i]
        psi_sq = psi_sq_all[i]
        LHS = Pi
        RHS = psi_sq * P
        # print(LHS-RHS)
        if not is_pos_def(LHS - RHS + tol*np.eye(n)):
            return P, False
    return P, True


def bisection1_i(R, S, Delta, eta, t_lwr=0.0, t_upr=1.0, bisection_epsilon=1e-6):
    while t_upr-t_lwr > bisection_epsilon:
        # print(t_lwr, t_upr)
        t_mid = (t_upr+t_lwr) / 2

        M, N = calc_MN_i(A, Delta, t_mid*eta, R, S)
        D = (M-N) / 2
        Q = (R+S) / 2
        good = check_assumption1_i(A, Delta, Q, D, t_mid*eta)
        if good:
            t_lwr = t_mid
        else:
            t_upr = t_mid
    return t_lwr


def make_noise_stds(weight_sq_all, phi_all, psi_sq_all):
    noise_stds = np.zeros(p)
    for i in range(p):
        weight = weight_sq_all[i]**0.5
        phi = phi_all[i]
        psi = psi_sq_all[i]**0.5
        noise_stds[i] = weight*phi*psi
    return noise_stds


def check_claim(P, Delta_all, noise_stds):
    if P is None:
        return None
    LHS = P
    RHS = np.dot(A.T, np.dot(P, A))
    for i in range(p):
        Delta = Delta_all[i]
        RHS += (noise_stds[i]**2)*np.dot(Delta.T, np.dot(P, Delta))
    # print('claim diff eigs %s' % str(sla.eigh(LHS-RHS, eigvals_only=True)))
    flag = is_pos_def(LHS - RHS)
    if flag:
        return True
    else:
        raise ValueError('The claim is not satisfied - system is not ms-stable with specified noise levels!')


def algorithm_thm1(A, Delta_all, eta_all, RS_scale=1.0, share_phi=False, check=True):
    # Set the penalty matrices
    R_all = RS_scale*np.array([np.eye(n), np.eye(n)])
    S_all = (1/RS_scale)*np.array([np.eye(n), np.eye(n)])
    # Compute phi scalars via bisection
    phi_all = np.zeros(p)
    t_all = np.zeros(p)
    for i in range(p):
        R, S = R_all[i], S_all[i]
        Delta = Delta_all[i]
        eta = eta_all[i]
        t_all[i] = bisection1_i(R, S, Delta, eta)

    if share_phi:
        phi_all = np.min(t_all) * np.ones(p) * eta_all
    else:
        for i in range(p):
            phi_all[i] = t_all[i]*eta_all[i]

    # Compute intermediate quantities
    M_all, N_all = calc_MN(A, Delta_all, phi_all, R_all, S_all)
    P_all, D_all = calc_PD(M_all, N_all)
    P = np.sum([weight_sq_all[i] * P_all[i] for i in range(p)], axis=0)

    # Compute psi scalars via generalized eigenvalues
    psi_sq_all = np.zeros(p)
    for i in range(p):
        Pi = P_all[i]
        e = np.max(sla.eigh(P, Pi, eigvals_only=True))
        psi_sq_all[i] = 1.0 / e

    beta_all = make_noise_stds(weight_sq_all, phi_all, psi_sq_all)
    if check:
        check_claim(P, Delta_all, beta_all)
    return beta_all, P


def algorithm_sdp4(A, Delta_all, eta_all, check=True):
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> np.eye(n)]
    signs = list(itertools.product([1, -1], repeat=p))
    for sign in signs:
        A_shifted = np.copy(A)
        for i in range(p):
            A_shifted += sign[i] * eta_all[i] * Delta_all[i]
        constraints.append(X - A_shifted.T @ X @ A_shifted >> 0)
    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    prob.solve()
    P = X.value
    if P is None:
        beta_all = np.zeros(p)
    else:
        phi_all = np.copy(eta_all)
        psi_sq_all = np.ones(p)
        beta_all = make_noise_stds(weight_sq_all, phi_all, psi_sq_all)
    if check:
        check_claim(P, Delta_all, beta_all)
    return beta_all, P


if __name__ == "__main__":
    plt.close('all')
    plt.style.use('./conlab.mplstyle')

    # This avoids the use of Type 3 fonts which are unacceptable in IEEE PaperPlaza
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Pendulum with friction
    n = 2     # number of states
    p = 2     # number of perturbation directions
    m = 1.0   # mass (kilograms)
    l = 2.0   # length (meters)
    g = 9.81  # gravity (meters/second/second)
    c = 0.10   # damping coefficient ()
    dt = 0.1  # sampling time (seconds)

    # Continuous-time system matrix
    Ac = np.array([[0.0, 1.0],
                   [-(g/l), -(c/m*l*l)]])
    # Discrete-time system matrix using exact discretization (sampled-data system)
    A = sla.expm(Ac*dt)

    # # Simulate the system (sanity check)
    # x0 = np.array([1.0, 0.0])
    # T = 100
    #
    # t_hist = np.arange(T)
    # x_hist2 = sim(A, T, x0)
    # plt.figure()
    # plt.plot(t_hist, x_hist2)
    # plt.show()

    # Assumed robust stability levels and directions
    eta_all_max = np.array([0.8*(g/l)*dt,
                            1.0*(c/m*l*l)*dt])

    Delta_all = np.array([[[0.0, 0.0],
                           [1.0, 0.0]],
                          [[0.0, 0.0],
                           [0.0, 1.0]]
                          ])

    # Check robust stability by gridding
    plt.figure()
    for i in range(p):
        eta_frac_hist = np.linspace(-1, 1, 1000)
        s = np.array([specrad(A + theta*eta_all_max[i]*Delta_all[i]) for theta in eta_frac_hist])

        if np.any(s > 1):
            raise ValueError('System is not robustly stable!')

        plt.plot(eta_frac_hist, s)
    plt.axhline(1.0, linestyle='--', color='k', alpha=0.5)

    # User-defined parameters for weights
    weight_sq_all = np.ones(p) / p

    ns = 30  # Number of samples of robustness margins
    t_hist = np.linspace(1e-3, 1, ns+1)
    beta_all_hist_thm1 = np.zeros([ns+1, p])
    beta_all_hist_thm1_1 = np.zeros([ns+1, p])
    beta_all_hist_thm1_2 = np.zeros([ns+1, p])
    beta_all_hist_sdp4 = np.zeros([ns+1, p])
    RS_scale_1 = 5.0
    RS_scale_2 = 0.2
    for k, t in enumerate(t_hist):
        eta_all = t*eta_all_max
        # Compute the noise scalars
        beta_all_hist_thm1[k], P = algorithm_thm1(A, Delta_all, eta_all)
        beta_all_hist_thm1_1[k], P = algorithm_thm1(A, Delta_all, eta_all, RS_scale=RS_scale_1)
        beta_all_hist_thm1_2[k], P = algorithm_thm1(A, Delta_all, eta_all, RS_scale=RS_scale_2)
        beta_all_hist_sdp4[k], P = algorithm_sdp4(A, Delta_all, eta_all)
        # print(' max noise stds, thm1 %s' % str(beta_all))
        # print(' max noise stds, sdp4 %s' % str(beta_all))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(beta_all_hist_thm1[:, 0], beta_all_hist_thm1[:, 1], marker='o', markersize=8, label=r'thm1, $\zeta=1.0$')
    ax.plot(beta_all_hist_thm1_1[:, 0], beta_all_hist_thm1_1[:, 1], marker='^', markersize=8, label=r'thm1, $\zeta=%.1f$' % RS_scale_1)
    ax.plot(beta_all_hist_thm1_2[:, 0], beta_all_hist_thm1_2[:, 1], marker='v', markersize=8, label=r'thm1, $\zeta=%.1f$' % RS_scale_2)
    ax.plot(beta_all_hist_sdp4[:, 0], beta_all_hist_sdp4[:, 1], marker='x', markersize=16, markeredgewidth=3, label='sdp4')
    ax.legend()
    # # Annotations
    # for k, t in enumerate(t_hist):
    #     if k in [0, 5, 10, 15, 20]:
    #         txt = 't = %.2f' % t
    #         xx = beta_all_hist_thm1[k, 0] + 0.001
    #         yy = beta_all_hist_thm1[k, 1] + 0.001
    #         ax.annotate(txt, (xx, yy))
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_xlim([-0.01, 0.2])
    ax.set_ylim([-0.02, 0.04])
    ax.axvline(0, lw=2, color=[0.5, 0.5, 0.5], zorder=1)
    ax.axhline(0, lw=2, color=[0.5, 0.5, 0.5], zorder=1)
    axins = ax.inset_axes([0.5, 0.14, 0.45, 0.30])
    axins.plot(beta_all_hist_thm1[:, 0], beta_all_hist_thm1[:, 1], marker='o', markersize=8)
    axins.plot(beta_all_hist_thm1_1[:, 0], beta_all_hist_thm1_1[:, 1], marker='^', markersize=8)
    axins.plot(beta_all_hist_thm1_2[:, 0], beta_all_hist_thm1_2[:, 1], marker='v', markersize=8)
    axins.plot(beta_all_hist_sdp4[:, 0], beta_all_hist_sdp4[:, 1], marker='x', markersize=16, markeredgewidth=3)
    axins.axvline(0, lw=2, color=[0.5, 0.5, 0.5], zorder=1)
    axins.axhline(0, lw=2, color=[0.5, 0.5, 0.5], zorder=1)
    # sub region of the original image
    x1, x2, y1, y2 = -0.005, 0.04, -0.001, 0.005
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins, edgecolor='k')
    fig.tight_layout()
    fig.savefig('margins.pdf')
    fig.savefig('margins.png', dpi=600)


    # # Assumed scalars for the ms-stability claim
    # weight_sq_all = np.ones(p) / p
    # R_all = np.array([np.eye(n), np.eye(n)])
    # S_all = np.array([np.eye(n), np.eye(n)])
    # # phi_all = 1.0*eta_all
    # # psi_sq_all = 1.0*np.ones(p)
    #
    # # REF
    # phi_all = 0.6*eta_all
    # psi_sq_all = 0.9*np.ones(p)
    #
    #
    # # Sanity check, should be satisfied by construction
    # P, assm1 = check_assumption1(A, Delta_all, R_all, S_all, phi_all)
    # P, assm2 = check_assumption2(A, Delta_all, R_all, S_all, phi_all, psi_sq_all)
    #
    # if not assm1:
    #     raise ValueError('Assumption 1 not satisfied! Try smaller scalars phi_all.')
    # if not assm2:
    #     raise ValueError('Assumption 2 not satisfied! Try smaller scalars phi_all, psi_sq_all.')


    # print(' max noise stds %s' % str(beta_all))
