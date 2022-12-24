# ===================================================================
# Description:
#   This function solves the linear system Ax=b using the
#   following algorithm:
#       x^{0}   = A\b
#       S^{k}   = {j: |x^{k}_{j}| >= lambda}
#       x^{k+1} = argmin || Ax-b ||_2 s.t. supp(x) in S^{k}
#
# Inputs:
#   A = matrix of size m*n
#   b = vector of size m*1
#   lambda = thresholding parameter
#
# Outputs:
#   X = the solution at each iteration
#   Sk_sorted = the ordered support set of each x^k
#   F = the value of the objective function at each step
#
# References:
#   [1] Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.
#       Discovering governing equations from data by sparse
#       identification of nonlinear dynamical systems. Proceedings of
#       the National Academy of Sciences, 113(15):3932-3937, 2016.
#   [2] Linan Zhang and Hayden Schaeffer.
#       On the Convergence of the SINDy Algorithm. Multiscale Modeling
#       & Simulation 17(3), 948–972, 2019.
#
# Authors: Linan Zhang and Hayden Schaeffer
# Date: May 16, 2018
# ===================================================================
import numpy as np

def alg21(A, b, LAMBDA):    
    # make the inputs compatible
    b = b.reshape((-1, 1))
    
    # Set iteration parameters.
    n = np.size(A, 1)
    MaxIt = n  # maximum number of iterations needed
    err = True  # flag for error
    k = 1  # iteration index
    
    # Initialize x^{k} and S^{k} for each k.
    X = np.zeros((n, MaxIt))  # Xk(:,k) = x^{k-1}
    Sk = np.zeros((n, MaxIt))  # Sk(:,k) = S^{k-1}
    
    # Sort Sk such that | x ^ {k}_{S ^ {k}(j+1)} | >= | x ^ {k}_{S ^ {k}(j)}|
    Sk_sorted = np.zeros((n, MaxIt))
    
    
    # Perform the initial step.
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # x = x ^ {0}
    
    X[:, 0] = x[:, 0]
    S = np.where(np.abs(x) >= LAMBDA)[0] # S = S ^ {0}
    # Sort S such that | x_{S(j+1)} | >= | x_{S(j)} | .
    ind = np.argsort(x[S, 0])[::-1]
    Sk[0: len(ind), 0] = S
    Sk_sorted[0: len(ind), 0] = S[ind]
    
    while k<=MaxIt-1 and  err:
        
        # xnew = argmin ||Ay-b||_2 subject to supp(y) in S
        y = np.linalg.lstsq(A[:,S], b, rcond=None)[0]
        x = np.zeros((n, 1))    
        x[S] = y; # x = x^{k-1}
        X[:,k] = x[:, 0]
        
        S = np.where(abs(x)>=LAMBDA)[0] # S = S^{k-1}
        # Sort S such that |x_{S(j+1)}| >= |x_{S(j)}|.
        ind = np.argsort(x[S, 0])[::-1]
        Sk[0:len(ind), k] = S
        Sk_sorted[0:len(ind), k] = S[ind]
    
        # Stopping criterion: S^{k-1} = S^{k-2}.
        err = not (Sk[:, k-1] == Sk[:, k]).all();
        
        k = k+1;
        
    # Deleting all-zero columns and rows
    X = np.delete(X, np.argwhere(np.all(X[..., :] == 0, axis=0)), axis=1)
    
    # Compute the objective function.
    k = np.size(X, 1); # number of iterations (including the 0th step)
    normA = np.linalg.norm(A, 2); # l2 norm of A
    F = np.zeros((k, 1));
    
    def obj_func(x):
        f = np.linalg.norm(np.matmul(A, x)-b, ord='fro')**2 / (
            normA**2) + LAMBDA**2 * np.count_nonzero(x)
        return f
    
    for i in range(k):
        F[i, 0] = obj_func(X[:, i].reshape(-1, 1));
        
    return X, Sk_sorted, F




