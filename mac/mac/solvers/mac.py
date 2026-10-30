import numpy as np
import networkx as nx
import networkx.linalg as la

from dataclasses import dataclass
from typing import Optional
from timeit import default_timer as timer

from mac.utils.graphs import *
from mac.utils.rounding import *

import mac.utils.fiedler as fiedler
import mac.optimization.frankwolfe as fw
import mac.optimization.constraints as constraints

class MAC:
    @dataclass
    class Cache:
        """问题数据缓存"""
        Q: Optional[np.ndarray] = None

    def __init__(self, fixed_edges, candidate_edges, num_nodes,
                fiedler_method='tracemin_lu', fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10):
        """MAC算法初始化
        
        参数
        ----------
        fixed_edges : Edge对象列表
            图中固定的边列表。
        candidate_edges : Edge对象列表
            可以添加到图中的候选边列表。
        num_nodes : int
            图中节点的数量。
        fiedler_method : str, 可选
            计算Fiedler向量的方法。可选项为'tracemin_lu'和'tracemin_cholesky'。
            默认为'tracemin_lu'。使用'tracemin_cholesky'方法更快，
            但需要安装SuiteSparse。
        fiedler_tol : float, 可选
            计算Fiedler向量和对应特征值的容差。
        min_selection_weight_tol : float, 可选
            边选择权重的最小容差。默认为1e-10。
        """
        # 检查{fixed_edges ∪ candidate_edges}集合中至少可能有一个生成树
        # 这不能保证生成树存在，但这是一个基本测试
        num_edges = len(fixed_edges) + len(candidate_edges)
        assert (num_nodes - 1) <= num_edges

        # 还需检查边的数量不会"过多"。完全图K(n)中边的数量等于n * (n - 1) / 2，
        # 所以不可能有超过这个数量的边
        assert num_edges <= 0.5 * num_nodes * (num_nodes - 1)

        # 预先计算仅包含"固定边"的子图的拉普拉斯矩阵
        self.L_fixed = weight_graph_lap_from_edge_list(fixed_edges, num_nodes)
        self.num_nodes = num_nodes

        self.weights = []
        self.edge_list = []
        for edge in candidate_edges:
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))

        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

        # Fiedler向量计算的配置
        self.fiedler_method = fiedler_method
        self.fiedler_tol = fiedler_tol

        # 截断选择权重低于此阈值的边
        self.min_selection_weight_tol = min_selection_weight_tol

    def laplacian(self, x):
        """构建组合拉普拉斯矩阵(固定边加上由x加权的候选边)
        
        参数:
        x: [0,1]^m上的元素；这是要使用的边选择权重
        
        参数`min_selection_weight_tol`用于删除数值接近零的边。
        这提高了速度，特别是在边不是*完全*为零但接近于零时，
        因为这些边对图的影响几乎可以忽略不计。
        
        返回矩阵L(x)
        """
        idx = np.where(x > self.min_selection_weight_tol)
        prod = x[idx]*self.weights[idx]
        L_candidate = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_nodes)
        L_x = self.L_fixed + L_candidate
        return L_x

    def evaluate_objective(self, x):
        """
        计算L(x)的第二小特征值lambda_2，其中L(x)是边i由x_i*weight_i加权的拉普拉斯矩阵，
        lambda_2是第二小特征值(即代数连通性)。
        
        参数:
        x: 每个候选边的权重(不包括固定边)
        
        返回F(x) = lambda_2(L(x))。
        """
        return fiedler.find_fiedler_pair(L=self.laplacian(x),
                                         method=self.fiedler_method, tol=self.fiedler_tol)[0]

    def problem(self, x, cache=None):
        """计算L(x)的代数连通性以及代数连通性相对于x的(超)梯度。
        
        参数:
        x: 每个候选边的权重(不包括固定边)
        cache: 可变的`Cache`对象。如果在`cache`字段中提供了`Cache`对象，它将被使用
               并更新，但不会显式返回。相反，它将被直接更新。
        
        返回值: x, grad F(x)。
        """
        Q = None if cache is None else cache.Q
        f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x), X=Q)

        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # 获取边(i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Qnew
        return f, gradf

    def solve(self, k, x_init=None, rounding="nearest", fallback=False,
              max_iters=5, relative_duality_gap_tol=1e-4,
              grad_norm_tol=1e-8, random_rounding_max_iters=1,
              verbose=False, return_rounding_time=False, use_cache=False):
        """使用Frank-Wolfe方法求解子集选择问题。
        
        参数
        ----------
        k : int
            要选择的边数量。
        x_init : 可选, 类数组
            候选边的初始权重，必须满足0 <= w_i <= 1且|w| <= k。
            这是Frank-Wolfe算法的起点。
        rounding : str, 可选
            使用的舍入方法。选项为"nearest"(默认)和"madow"(随机舍入程序)。
        fallback : bool, 可选
            如果为True，当舍入解比初始解更差时，回退到初始化。
        max_iters: int, 可选
            Frank-Wolfe算法的最大迭代次数。
        relative_duality_gap_tol: float, 可选
            相对对偶间隙的容差，表示为函数值的分数。即，若(upper - f)/f < 
            relative_duality_gap_tol，其中"upper"是'f'最优值的上界，则算法终止。
        grad_norm_tol: float, 可选
            梯度范数的容差。如果梯度范数小于此值，则算法终止。
        random_rounding_max_iters: int, 可选
            随机舍入程序的最大迭代次数。仅当rounding="madow"时使用。
            若大于1，将多次随机舍入并返回最佳解(就代数连通性而言)。
        verbose: bool, 可选
            如为True，打印有关算法进度的信息。
        
        返回元组(solution, unrounded, upper_bound)，其中
        solution: (舍入的)解w \in {0,1}^m |w| = k
        unrounded: 舍入前获得的解
        upper_bound: 最后一次迭代的对偶值
        """

        if k >= len(self.weights):
            # 如果预算大于候选边的数量，则保留所有边
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        # TODO 处理x为none的情况
        assert(len(x_init) == len(self.weights))

        # 方向查找子问题的解决方案
        solve_lp = lambda g: constraints.solve_subset_box_lp(g, k)

        # 设置问题以使用缓存(或不使用)
        cache = None
        if use_cache:
            cache = MAC.Cache()
        problem = lambda x: self.problem(x, cache=cache)

        # 运行Frank-Wolfe算法求解子集约束代数连通性最大化的松弛问题
        w, u = fw.frank_wolfe(initial=x_init, problem=problem,
                                solve_lp=solve_lp, maxiter=max_iters,
                                relative_duality_gap_tol=relative_duality_gap_tol,
                                grad_norm_tol=grad_norm_tol,
                                verbose=verbose)

        start = timer()
        if verbose:
            logging.info(f"Frank-Wolfe 算法完成，开始舍入过程 ({rounding})...")

        if rounding == "madow":
            # 添加计时和进度指示
            madow_start = timer()
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters, verbose=verbose)
            madow_end = timer()
            if verbose:
                logging.info(f"随机舍入完成，用时: {madow_end - madow_start:.2f}秒")
        else:
            # rounding == "nearest"
            nearest_start = timer()
            if verbose:
                logging.info("开始最近点舍入操作...")
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
            nearest_end = timer()
            if verbose:
                logging.info(f"最近点舍入完成，用时: {nearest_end - nearest_start:.2f}秒")
                
        end = timer()
        rounding_time = end - start

        if verbose:
            try:
                rounded_obj = self.evaluate_objective(rounded)
                logging.info(f"舍入后目标值: {rounded_obj:.6f}，总舍入用时: {rounding_time:.2f}秒")
            except Exception as e:
                logging.warning(f"计算舍入后目标值失败: {e}，但舍入结果仍然有效")
            
        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)
            
            if verbose:
                logging.info(f"比较舍入结果与初始解: 舍入={rounded_f:.6f}, 初始={init_f:.6f}")
                
            # 如果舍入解比初始解更差，则返回初始解
            if rounded_f < init_f:
                if verbose:
                    logging.info(f"舍入解比初始解差，回退到初始解")
                rounded = x_init
        if verbose:
            logging.info("MAC 求解过程完成，返回最终结果")
            
        # 返回舍入解及未舍入解和对偶上界
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u
