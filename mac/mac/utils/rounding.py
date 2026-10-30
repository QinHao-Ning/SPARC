"""
Utilities for rounding solutions to convex optimization problems onto simple constraint sets.
"""

import numpy as np
import logging

def round_nearest(w, k, weights=None, break_ties_decimal_tol=None):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to a
    solution to the original problem with w_i \in {0,1}. Ties between edges
    are broken based on the original weight (all else being equal, we
    prefer edges with larger weight).

    w: A solution in the feasible set for the relaxed problem
    weights: The original weights of the edges to use as a tiebreaker
    k: The number of edges to select
    break_ties_decimal_tol: tolerance for determining floating point equality of weights w. If two selection weights are equal to this many decimal places, we break the tie based on the original weight.

    returns w': A solution in the feasible set for the original problem
    """
    if weights is None or break_ties_decimal_tol is None:
        # If there are no tiebreakers, just set the top k elements of w to 1,
        # and the rest to 0
        idx = np.argpartition(w, -k)[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    # If there are tiebreakers, we truncate the selection weights to the
    # specified number of decimal places, and then break ties based on the
    # original weights
    truncated_w = w.round(decimals=break_ties_decimal_tol)
    zipped_vals = np.array(
        [(truncated_w[i], weights[i]) for i in range(len(w))],
        dtype=[("w", "float"), ("weight", "float")],
    )
    idx = np.argpartition(zipped_vals, -k, order=["w", "weight"])[-k:]
    rounded = np.zeros(len(w))
    if k > 0:
        rounded[idx] = 1.0
    return rounded

def round_random(w, k):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to
    one with hard edge constraints and satisfying the constraint that the
    expected number of selected edges is equal to k.

    w: A solution in the feasible set for the relaxed problem
    k: The number of edges to select _in expectation_

    returns w': A solution containing hard edge selections with an expected
    number of selected edges equal to k.
    """
    x = np.zeros(len(w))
    for i in range(len(w)):
        r = np.random.rand()
        if w[i] > r:
            x[i] = 1.0
    return x

def round_madow(w, k, seed=None, value_fn=None, max_iters=1, verbose=False):
    if value_fn is None or max_iters == 1:
        return round_madow_base_optimized(w, k, seed, verbose)

    best_x = None
    best_val = -np.inf
    for i in range(max_iters):
        x = round_madow_base_optimized(w, k, seed, verbose)
        val = value_fn(x)
        if val > best_val:
            best_val = val
            best_x = x
    return best_x


def round_madow_base(w, k, seed=None, verbose=False):
    """
    Use Madow rounding
    """
    if verbose:
        logging.info(f"开始Madow随机舍入，向量长度={len(w)}，选择边数k={k}")
        
    if seed is None:
        u = np.random.rand()
        if verbose:
            logging.info(f"使用随机种子生成随机数u={u:.6f}")
    else:
        u = seed.rand()
        if verbose:
            logging.info(f"使用指定种子生成随机数u={u:.6f}")
            
    # 初始化结果向量和辅助向量 
    x = np.zeros(len(w))
    pi = np.zeros(len(w))
    sumw = np.cumsum(w)
    pi[1:] = sumw[:-1]
    
    # 计算舍入边
    if verbose and k > 1000:
        # 只有当k很大时才记录进度
        logging.info(f"开始选择边，总计需要选择{k}条...")
        progress_step = max(1, k // 10)
        
    for i in range(k):
        total = u + i
        x[np.where((pi <= total) & (total < sumw))] = 1.0
        
        if verbose and k > 1000 and (i+1) % progress_step == 0:
            logging.info(f"已完成 {i+1}/{k} 条边的选择 ({(i+1)/k*100:.1f}%)")
            
    # 验证选择的边数是否正确
    selected_count = np.sum(x)
    if verbose:
        logging.info(f"舍入完成，选择了 {selected_count} 条边")
        
    assert selected_count == k, f"错误: 选择了 {selected_count} 条边，而不是要求的 {k} 条"
    return x

def round_madow_base_optimized(w, k, seed=None, verbose=False):
    """优化版的Madow随机舍入算法"""
    if verbose:
        logging.info(f"开始优化版Madow随机舍入，向量长度={len(w)}，选择边数k={k}")
        
    if seed is None:
        u = np.random.rand()
    else:
        u = seed.rand()
        
    if verbose:
        logging.info(f"使用随机数u={u:.6f}")
        
    # 计算累积和
    sumw = np.cumsum(w)
    
    # 一次性生成所有需要的total值
    if verbose:
        logging.info("生成查找值...")
    
    totals = u + np.arange(k)
    
    # 一次性为每个total找到对应的边
    if verbose:
        logging.info("一次性选择所有边...")
    
    # 初始化结果数组
    x = np.zeros(len(w))
    
    # 使用二分查找找到每个total对应的索引
    indices = np.searchsorted(sumw, totals)
    x[indices] = 1.0
    
    selected_count = np.sum(x)
    if verbose:
        logging.info(f"舍入完成，选择了 {selected_count} 条边")
    
    # 确保选择了正确数量的边
    if selected_count != k:
        if verbose:
            logging.warning(f"选择了 {selected_count} 条边，但需要 {k} 条，进行调整...")
        # 处理边缘情况
        if selected_count < k:
            zeros = np.where(x < 0.5)[0]
            np.random.shuffle(zeros)
            x[zeros[:k-selected_count]] = 1.0
        else:
            ones = np.where(x > 0.5)[0]
            np.random.shuffle(ones)
            x[ones[:selected_count-k]] = 0.0
            
    return x