import numpy as np
import numpy.random as npr
import numpy.linalg as la
from matrixmath import is_pos_def,mdot,specrad,positive_definite_part,dare

# Generalized discrete-time algebraic Riccati equation (GARE) for systems with multiplicative noise on A only
def gare(A, B, a, Aa, Q, R):
    # Options
    max_iters = 1000
    epsilon = 1e-6
    Pelmax = 1e20
    n = A.shape[1]
    p = len(a)
    # Initialize
    P = Q
    iterc = 0
    stop_early = False
    converged = False
    stop = False

    while not stop:
        # Record previous iterate
        P_prev = P
        # Certain part
        APAcer = mdot(A.T,P,A)
        BPBcer = mdot(B.T,P,B)
        # Uncertain part
        APAunc = np.zeros([n,n])
        for i in range(p):
            APAunc += a[i]*mdot(Aa[i].T,P,Aa[i])
        APAsum = APAcer+APAunc
        BPBsum = np.copy(BPBcer)
        # Recurse
        P = Q + APAsum - mdot(A.T,P,B,la.solve(R+BPBsum,B.T),P,A)
        # Check for stopping condition
        if la.norm(P-P_prev,'fro')/la.norm(P,'fro') < epsilon:
            converged = True
        if iterc >= max_iters or np.any(np.abs(P)>Pelmax):
            stop_early = True
        else:
            iterc += 1
        stop = converged or stop_early
    # Compute the gains
    if stop_early:
        P,K = None,None
    else:
        K = -mdot(la.solve(R+BPBsum,B.T),P,A)
    return P,K


# Feasibility bisection
def bisection(y_lwr_0, y_upr_0, objective, tol=1e-3):
    # Initialize
    y_lwr, y_upr = y_lwr_0, y_upr_0
    while objective(y_lwr) is None:
        y_lwr /= 2
    while objective(y_upr) is not None:
        y_upr *= 2
    # Bisect
    while y_upr - y_lwr > tol:
        y_mid = (y_lwr + y_upr) / 2
        if objective(y_mid) is None:
            y_upr = y_mid
        else:
            y_lwr = y_mid
    return y_lwr


# Algorithm 1: Robustness via shared Lyapunov functions
def algo1(A, B, Aa, Q, R, theta):

    def calc_control(z):
        a = theta * z
        return gare(A, B, a, Aa, Q, R)

    def func(y):
        return calc_control(y)[0]

    z = bisection(y_lwr_0=0, y_upr_0=1, objective=func)
    a = theta * z
    P,K = calc_control(z)

    ABK = A+mdot(B,K)

    def func2(y):
        eta = theta * y
        LHS = Q + mdot(K.T,R,K)
        for i in range(p):
            LHS += a[i]*mdot(Aa[i].T,P,Aa[i])
        RHS = np.zeros_like(LHS)
        for i in range(p):
            RHS += eta[i]*positive_definite_part(mdot(Aa[i].T,P,ABK)+mdot(ABK.T,P,Aa[i]))
            for j in range(p):
                RHS += eta[i]*eta[j]*positive_definite_part(mdot(Aa[i].T,P,Aa[j])+mdot(Aa[j].T,P,Aa[i]))
        if is_pos_def(LHS-RHS):
            return True
        else:
            return None

    y = bisection(y_lwr_0=0, y_upr_0=1, objective=func2)
    eta = theta * y

    return K, eta


# Algorithm 2: Robustness via mean-square stability of an auxiliary system
def algo2(A,B,Aa,Q,R,eta_bar):

    def calc_control(y):
        eta = eta_bar*y
        a = eta*(1+np.sum(eta))
        z = (1+np.sum(eta))**0.5
        Ag = z * A
        Bg = z * B
        return gare(Ag, Bg, a, Aa, Q, R)

    def func(y):
        return calc_control(y)[0]

    y = bisection(y_lwr_0=0, y_upr_0=1, objective=func)
    eta = eta_bar*y
    P,K = calc_control(y)

    return K, eta


def system_inverted_pendulum():
    n = 2
    m = 1
    dt = 0.1
    mass_true = 10
    A_true = np.array([[1, dt], [mass_true * dt, 1]])
    B_true = np.array([[0], [dt]])
    Q = np.eye(n)
    R = np.eye(m)
    mass = 5
    A = np.array([[1, dt], [mass * dt, 1]])
    B = np.copy(B_true)
    return n, m, A_true, B_true, Q, R, A, B


def system_random(n=2, m=1, seed=1):
    npr.seed(seed)
    A_true = 2*npr.randn(n,n)
    B_true = npr.randn(n,m)
    Q = np.eye(n)
    R = np.eye(m)
    A = A_true + 0.5*npr.randn(n,n)
    B = np.copy(B_true)
    return n, m, A_true, B_true, Q, R, A, B


def system_example():
    # Example where Algorithm 2 performs better than Algorithm 1
    n = 2
    m = 1
    A_true = np.array([[ 0.1, 1.0],
                       [-2.0, 1.4]])
    B_true = np.array([[ 0.3],
                       [-1.2]])
    Q = np.eye(n)
    R = np.eye(m)
    A = np.array([[-0.1, 0.2],
                  [-2.3, 1.7]])
    B = np.copy(B_true)
    return n, m, A_true, B_true, Q, R, A, B


if __name__ == "__main__":
    # Define the true and assumed dynamics and LQR costs
    n, m, A_true, B_true, Q, R, A, B = system_inverted_pendulum()
    # n, m, A_true, B_true, Q, R, A, B = system_random()
    # n, m, A_true, B_true, Q, R, A, B  = system_example()

    # Define uncertainty directions and magnitudes
    p = 1
    Ai = np.zeros([p, n, n])
    Ai[0, 1, 0] = 1
    eta_bar = np.ones(p)

    K1, eta1 = algo1(A, B, Ai, Q, R, eta_bar)
    K2, eta2 = algo2(A, B, Ai, Q, R, eta_bar)

    # Compute the certainty-equivalent control
    Pce = dare(A, B, Q, R)
    Kce = -la.solve((R + mdot(B.T, Pce, B)), mdot(B.T, Pce, A))

    def check_random_systems(K, eta, R, bound_type):
        s = np.zeros(R)
        if bound_type is "unidirectional":
            r = np.linspace(0, 1, R)
        elif bound_type is "bidirectional":
            r = np.linspace(-1,1,R)
        for j in range(R):
            Arand = np.copy(A)
            for i in range(p):
                Arand += eta[i]*r[j]*Ai[i]
            s[j] = specrad(Arand+B_true@K)
        return s

    s1 = check_random_systems(K1, eta1, 10000, "unidirectional")
    s2 = check_random_systems(K2, eta2, 10000, "unidirectional")

    def stablestr(r):
        r_str = ("%f " % r)
        s_str = "(  stable)" if r < 1 else "(unstable)"
        return r_str + s_str

    print("")
    print("Stability report")
    print("-----------------------------------------------")
    print("Closed-loop gain matrices")
    print("K_certainty_equivalent = %s" % Kce)
    print("K_robust_algo1         = %s" % K1)
    print("K_robust_algo2         = %s" % K2)
    print("")
    print("Spectral radii of true system, closed-loop")
    print("Open-loop              = %s" % stablestr(specrad(A_true)))
    print("K_certainty_equivalent = %s" % stablestr(specrad(A_true + mdot(B_true, Kce))))
    print("K_robust_algo1         = %s" % stablestr(specrad(A_true + mdot(B_true,K1))))
    print("K_robust_algo2         = %s" % stablestr(specrad(A_true + mdot(B_true,K2))))
    print("")
    print("Spectral radii of assumed system, closed-loop")
    print("Open-loop              = %s" % stablestr(specrad(A)))
    print("K_certainty_equivalent = %s" % stablestr(specrad(A + mdot(B_true, Kce))))
    print("K_robust_algo1         = %s" % stablestr(specrad(A + mdot(B_true,K1))))
    print("K_robust_algo2         = %s" % stablestr(specrad(A + mdot(B_true,K2))))
    print("")
    print("Spectral radii of random systems in robustness region, closed-loop")
    print("K_robust_algo1         = %s" % stablestr(s1.max()))
    print("K_robust_algo2         = %s" % stablestr(s2.max()))
    print("")
    print("Robustness regions, closed-loop")
    print("eta_algo1              = %f" % eta1)
    print("eta_algo2              = %f" % eta2)
    print("")