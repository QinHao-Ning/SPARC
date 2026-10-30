"""
Utilities for maximizing concave functions over simple feasible sets with Frank-Wolfe
"""

import numpy as np
import logging

def naive_stepsize(k):
    return 2.0 / (k + 2.0)

def frank_wolfe(initial,
                problem,
                solve_lp,
                stepsize=None,
                maxiter=50,
                relative_duality_gap_tol=1e-5,
                grad_norm_tol=1e-10,
                verbose=False):
    """Frank-Wolfe algorithm for maximizing a concave function.

    Parameters
    ----------
    initial : array-like
        Initial guess for the minimizer.
    problem : callable
        Function returning a tuple (f, gradf) where f is the objective function.
    solve_lp : callable
        Solve the Frank-Wolfe LP subproblem over the feasible set.
    stepsize : callable, optional
        Step size function.
    maxiter : int, optional
        Maximum number of iterations to perform.
    relative_duality_gap_tol : float, optional
        Tolerance for the *relative* duality gap (i.e., the duality gap divided
        by the objective value). If the relative duality gap is less than this
        tolerance, the algorithm terminates.
    grad_norm_tol : float, optional
        Tolerance for the norm of the gradient. If the norm of the gradient is
        less than this tolerance, the algorithm terminates.
    verbose : bool, optional
        Whether to print the iteration number and the objective value.

    Returns
    -------
    x : array-like
        The minimizer.

    """
    if stepsize is None:
        stepsize = lambda x, g, s, k: naive_stepsize(k)

    x = initial
    u = float("inf")
    
    # 初始化best_f和best_x
    best_f = -float("inf")
    best_x = None
    
    for i in range(maxiter):
        # 计算目标值和梯度
        f, gradf = problem(x)

        # 更新最优解
        if f > best_f:
            best_f = f
            best_x = x.copy()
            
        # 计算当前解的线性近似
        # 求解线性规划子问题
        s = solve_lp(gradf)

        # 计算当前解的目标值
        u = min(u, f + gradf @ (s - x))
        
        if verbose:
            logging.info(f"迭代 {i+1}/{maxiter}: 目标值={f:.6f}, 对偶间隙={u-f:.6f}，梯度范数={np.linalg.norm(gradf):.6f}")
            
        # If the gradient norm is sufficiently small, we are done.
        if np.linalg.norm(gradf) < grad_norm_tol:
            if verbose:
                logging.info("梯度范数接近0，已找到最优解")
            return x, u

        # 如果对偶间隙足够小，则我们已经找到最优解
        if (u - f) < relative_duality_gap_tol * abs(f):
            if verbose:
                logging.info("相对对偶间隙足够小，已找到最优解")
            return x, u

        x = x + stepsize(x, gradf, s, i) * (s - x)
        
    if verbose:
        logging.info(f"达到最大迭代次数，返回最优解 (目标值={best_f:.6f})")
    return best_x, u   # 返回最优解而非当前解

