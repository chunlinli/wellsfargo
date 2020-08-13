import numpy as np

def prob_safe(x, b):
    """ Calculate the probability of Y = 1 given x.

    Parameters
    ----------
    x: array
        Features. d-dimensional.
    b: array 
        Coefficients. d-dimensional. 

    Returns
    -------
    Conditional probability of Y = 1 given x.
    """
    return 1.0 / (1.0 + np.exp(-np.matmul(x, b)))


def soft_thresh(b, thresh):
    """ Soft-thresholding function.

    Parameters
    ----------
    b: float
        Coefficient. 
    thresh: float 
        Threshold. 

    Returns
    -------
    Soft-thresholded coefficient.
    """
    if (b > thresh):
        return b - thresh
    elif (b < -thresh):
        return b + thresh
    else:
        return 0.0


def LogitGroupingPursuit(x,
                         y,
                         lambda_par,
                         tau_par=0.1,
                         rho=10.0,
                         max_iter=1000,
                         tol=1e-4):
    """ Compute the minimizer of Sparse Grouping Pursuit. 

        This implementation combines MM-LLA and ADMM. 
        Given an estimate beta(t) at t-th iteration.

        (1) Constructing majorization function. 
            Pen_0(beta) <= |beta| if |beta(t)| <= tau.
                        <= 1      if |beta(t)| >= tau.
            Pen_G(beta) <= |beta(t)_j - beta(t)_j'| if |beta(t)_j - beta(t)_j'| <= tau
                        <= 1                        if |beta(t)_j - beta(t)_j'| >= tau
            At (t+1)-th iteration, we solve:

            minimize
                Loss(beta) + Weighted Lasso_0(beta) + Weighted Lasso_G(beta)

        (2) For ADMM, we solve an equivalent minimization:

            minimize
                Loss(beta) + Weighted Lasso(gamma1) + Weighted Lasso(gamma2)
            subject to
                gamma1 = beta
                gamma2 = A * beta
            
            where gamma1 is ADMM copy of beta, gamma2 is difference between beta,
            A is a matrix defines gamma2. 
        
        (3) Iterate until convergence. 
            Zou (2008) proves MM-LLA only need to iterate for 2-3 times. 

        References:
            MM: Hunter (2000) Quantile regression via an MM algorithm. JCGS. 
            LLA: Zou (2008) One-step sparse estimates in nonconcave penalized likelihood models. AoS.
            ADMM: Boyd (2011) Distributed optimization and statistical learning via ADMM.

    Parameters
    ----------
    x: array 
        Predictor matrix. n by d dimensional.

    y: array 
        Response. Takes values 0, 1. n-dimensional.

    lambda_par: float
        Penalty parameter. 

    tau_par: float
        Threshold in truncated lasso penalty. 

    rho: float
        ADMM penalty parameter. For computation only.

    max_iter: integer
        Maximum iteration number. 

    tol: float
        Numerical precision. 

    Returns
    -------
    A tuple of objects: (beta, gamma1, gamma2, prob_safe(x, beta))

        beta: array
            Estimated coefficients. d-dimensional.

        gamma1: array
            A copy of beta in ADMM. d-dimensional. For convergence check only. 

        gamma2: array
            Estimated coefficients difference. (d-1)-dimensional. 

        prob_safe(x, beta): array
            Estimated probability Y = 1 given x for each observation. 
    """

    n, d = x.shape[0], x.shape[1]

    beta = np.zeros(d)

    for _ in range(3):  # LLA loop

        # A is matrix that defines linear constraint:
        # gamma1 = beta
        # gamma2 = A * beta 
        gamma1 = beta.copy()
        gamma2 = np.zeros(d - 1)
        A = np.zeros((d - 1, d))

        idx = np.argsort(beta)
        for i in range(d - 1): 
            j, l = idx[i], idx[i + 1]
            gamma2[i] = beta[j] - beta[l]
            A[i, j], A[i, l] = 1.0, -1.0

        # defines weights in weighted lasso
        nonzero1 = np.abs(beta) >= tau_par
        nonzero2 = np.abs(gamma2) >= tau_par

        # u is dual variable for gamma1 in ADMM
        # w is dual variable for gamma2 in ADMM
        u = np.zeros(d)
        w = np.zeros(d - 1)

        for iter in range(max_iter):  # ADMM loop

            # minimize beta: Newton-Raphson (IRLS). 
            for iter_newton in range(max_iter):

                p = prob_safe(x, beta)

                # gradient
                grad = -np.matmul(
                    x.T, y - p) + rho * (beta - gamma1 + u) + rho * np.matmul(
                        A.T, (np.matmul(A, beta) - gamma2 + w))

                # hessian matrix
                hess = np.matmul(np.matmul(x.T, np.diag(p * (1.0 - p))),
                                 x) + rho * (np.eye(d) + np.matmul(A.T, A))

                beta_tmp = beta - np.linalg.solve(hess, grad)

                # for numerical stability, beta is constrained ||beta|| <= 20. 
                if (np.linalg.norm(beta_tmp) > 20.0):
                    beta_tmp = beta_tmp / np.linalg.norm(beta_tmp)

                change = beta_tmp - beta

                # termination criteria
                if (np.linalg.norm(change) < d * tol):
                    break

                # update beta
                beta = beta_tmp.copy()

            # min gamma1: Proximal gradient, soft-thresholding
            gamma1 = beta + u
            for j in range(d):
                if (not nonzero1[j]):
                    gamma1[j] = soft_thresh(gamma1[j],
                                            lambda_par / (tau_par * rho))

            # min gamma2: Proximal gradient, soft-thresholding
            gamma2 = np.matmul(A, beta) + w
            for j in range(d - 1):
                if (not nonzero2[j]):
                    gamma2[j] = soft_thresh(gamma2[j],
                                            lambda_par / (tau_par * rho))

            # update u
            r1 = beta - gamma1
            u += r1

            # update w
            r2 = np.matmul(A, beta) - gamma2
            w += r2

            # termination criteria
            if (np.dot(r1, r1) + np.dot(r2, r2) <= d * tol * tol):
                break

    # Grouping the coefficients.
    # Calculate grouped coefficients by average.
    for _ in range(3):
        group = np.zeros(d)
        cutoff = np.abs(gamma2) > tau_par
        k = 0
        idx = np.argsort(beta)
        for i in range(d - 1):
            j, l = idx[i], idx[i + 1]
            if (cutoff[i]):
                group[j], group[l] = k, k + 1
                k = k + 1
            else:
                group[j], group[l] = k, k
        for i in range(k + 1):
            beta[group == i] = beta[group == i].mean()
        beta = np.where(np.abs(beta) <= tau_par, 0, beta)

        idx = np.argsort(beta)
        for i in range(d - 1):
            j, l = idx[i], idx[i + 1]
            gamma2[i] = beta[j] - beta[l]

    return (beta, gamma1, gamma2, prob_safe(x, beta))