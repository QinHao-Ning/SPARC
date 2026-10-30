"""
使用MAC算法对相似度矩阵进行稀疏化处理 (简化版)
通过动态调整所选边数量来确保图连通性
"""

import os
import argparse
import torch
import numpy as np
import logging
import traceback
import sys
from mac.solvers.mac import MAC
from mac.utils.graphs import Edge, select_edges
from timeit import default_timer as timer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_graph_connectivity(candidate_edges, n_nodes):
    """检查由候选边构成的图是否连通
    
    参数:
        candidate_edges: 候选边列表
        n_nodes: 节点总数
        
    返回:
        is_connected: 图是否连通
        components: 连通分量数量
        component_sizes: 各连通分量大小列表
        isolated_nodes: 孤立节点列表
        component_nodes: 连通分量节点列表
    """
    # 创建一个邻接表来表示图
    adj_list = [[] for _ in range(n_nodes)]
    
    # 填充邻接表
    for edge in candidate_edges:
        i, j = edge.i, edge.j
        adj_list[i].append(j)
        adj_list[j].append(i)  # 无向图
    
    # 使用BFS检查连通性
    visited = [False] * n_nodes
    components = 0
    component_sizes = []
    component_nodes = []  # 存储每个连通分量的节点
    
    for i in range(n_nodes):
        if not visited[i]:
            components += 1
            size = 0
            current_component = []
            
            # BFS遍历一个连通分量
            queue = [i]
            visited[i] = True
            while queue:
                node = queue.pop(0)
                size += 1
                current_component.append(node)
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            component_sizes.append(size)
            component_nodes.append(current_component)
    
    # 孤立节点是那些没有任何边的节点
    isolated_nodes = [i for i in range(n_nodes) if not adj_list[i]]
    
    # 是否所有节点都被访问过（图是连通的）
    is_connected = components == 1
    
    return is_connected, components, component_sizes, isolated_nodes, component_nodes

def similarity_matrix_to_edges(sim_matrix, similarity_threshold=0.2, max_edges=30000000):
    """将相似度矩阵转换为边列表，限制最大边数量"""
    n_items = sim_matrix.shape[0]
    edge_data = []
    
    logger.info(f"开始从相似度矩阵构建边列表(形状: {sim_matrix.shape})...")
    
    # 仅考虑上三角矩阵以避免重复边，收集所有合格边
    for i in range(n_items):
        for j in range(i+1, n_items):
            sim_val = sim_matrix[i, j].item()
            if sim_val > similarity_threshold:
                edge_data.append((i, j, sim_val))
    
    logger.info(f"初步筛选出 {len(edge_data)} 条候选边")
    
    # 按相似度排序并截取前max_edges个
    if len(edge_data) > max_edges:
        logger.info(f"候选边数量过多，按相似度排序并截取前 {max_edges} 条")
        edge_data.sort(key=lambda x: x[2], reverse=True)
        edge_data = edge_data[:max_edges]
    
    # 转换为Edge对象
    all_potential_edges = [Edge(i, j, float(sim)) for i, j, sim in edge_data]
    
    logger.info(f"最终构建了 {len(all_potential_edges)} 条候选边")
    return all_potential_edges

def get_symmetric_knn_edges(sim_matrix, k, device='cpu'):
    """获取对称的KNN边"""
    n_items = sim_matrix.shape[0]
    logger.info(f"计算对称 KNN 边 (k={k})...")
    
    # 计算 Top-K 邻居
    sim_matrix_gpu = sim_matrix.to(device) # 使用指定设备加速topk
    _, knn_indices = torch.topk(sim_matrix_gpu, k, dim=-1)
    knn_indices = knn_indices.cpu() # 结果移回CPU
    del sim_matrix_gpu

    fixed_edge_set = set()
    fixed_edges_list = []

    # 创建对称边集合 (无向)
    for i in range(n_items):
        for j_idx in range(k):
            j = knn_indices[i, j_idx].item()
            if i == j: continue # 跳过自环
            # 添加无向边 (排序以保证唯一性)
            u, v = tuple(sorted((i, j)))
            if (u, v) not in fixed_edge_set:
                fixed_edge_set.add((u, v))
                # 从原始 sim_matrix 获取权重
                weight = sim_matrix[u, v].item() 
                fixed_edges_list.append(Edge(u, v, float(weight)))

    logger.info(f"找到 {len(fixed_edges_list)} 条对称 KNN 边作为固定边")
    return fixed_edges_list, fixed_edge_set


def apply_mac_to_similarity_matrix(
    sim_matrix_path,
    output_dir=None,
    knn_k=50, # 总目标平均度数的一半 (近似)
    knn_k_fixed=20, # 强制保留的 KNN 边的 K 值
    similarity_threshold=0.1,
    max_edges=30000000,
    device='cpu',
    fiedler_method='lanczos',
    check_connectivity=True
):
    """使用MAC算法对相似度矩阵进行稀疏化处理 (强制保留KNN边)"""
    try:
        # 加载相似度矩阵
        logger.info(f"从 {sim_matrix_path} 加载相似度矩阵...")
        sim_matrix = torch.load(sim_matrix_path, map_location='cpu') # 先加载到CPU
        
        n_items = sim_matrix.shape[0]
        logger.info(f"相似度矩阵形状: {sim_matrix.shape}")

        # 1. 获取需要固定的对称 KNN 边
        fixed_knn_edges, fixed_knn_edge_set = get_symmetric_knn_edges(sim_matrix, knn_k_fixed, device=device)

        # 2. 获取所有潜在的候选边 (高于阈值)
        all_potential_edges = similarity_matrix_to_edges(sim_matrix, similarity_threshold, max_edges)

        # 3. 分离出 MAC 可以选择的候选边
        candidate_edges_for_solver = []
        fixed_edge_indices_in_all = set() # 记录固定边在 all_potential_edges 中的索引，方便后续查找
        
        # 为了方便查找，创建一个字典 {(u,v): index_in_all_potential}
        potential_edge_dict = {tuple(sorted((edge.i, edge.j))): idx 
                               for idx, edge in enumerate(all_potential_edges)}

        for edge in fixed_knn_edges:
             u, v = tuple(sorted((edge.i, edge.j)))
             if (u, v) in potential_edge_dict:
                 fixed_edge_indices_in_all.add(potential_edge_dict[(u, v)])
        
        for idx, edge in enumerate(all_potential_edges):
            if idx not in fixed_edge_indices_in_all:
                 candidate_edges_for_solver.append(edge)

        logger.info(f"固定边数量: {len(fixed_knn_edges)}")
        logger.info(f"MAC 可选的候选边数量: {len(candidate_edges_for_solver)}")

        # 4. 创建MAC求解器
        logger.info("初始化MAC求解器...")
        mac_solver = MAC(fixed_knn_edges, candidate_edges_for_solver, n_items,
                         fiedler_method=fiedler_method,
                         fiedler_tol=1e-4)

        # 5. 计算目标边数和需要从候选中选择的边数
        initial_knn_k = knn_k
        current_knn_k = initial_knn_k
        max_knn_k = 500  # 最大k值，避免无限增加
        
        num_fixed_edges = len(fixed_knn_edges)
        
        # --- 移除连通性检查循环和 min_node_k 逻辑 ---
        # 假设 fixed_knn_edges 已经提供了足够的基础连通性
        # 如果后续 MAC 执行失败或结果不佳，可以考虑重新引入或修改连通性保证逻辑
        
        total_target_edges = current_knn_k * n_items // 2 # 目标总边数 (无向)
        edges_to_select_from_candidates = max(0, total_target_edges - num_fixed_edges)
        
        # 确保选择的数量不超过可用候选边
        edges_to_select_from_candidates = min(edges_to_select_from_candidates, len(candidate_edges_for_solver))

        logger.info(f"目标总边数 (无向): {total_target_edges}")
        logger.info(f"需要从候选中选择的边数: {edges_to_select_from_candidates}")

        # 6. 创建初始权重向量 x_init (针对 candidate_edges_for_solver)
        # 简单地基于权重选择 top K
        if edges_to_select_from_candidates > 0 and len(candidate_edges_for_solver) > 0:
            candidate_edges_for_solver.sort(key=lambda x: x.weight, reverse=True)
            x_init = np.zeros(len(candidate_edges_for_solver))
            for i in range(edges_to_select_from_candidates):
                x_init[i] = 1.0
            # 恢复原始顺序（如果 MAC 库要求）- 假设不需要
            # candidate_edges_for_solver.sort(key=lambda x: original_index) # 如果需要恢复顺序
        else:
             x_init = np.zeros(len(candidate_edges_for_solver))


        # 7. 使用MAC算法选择边
        logger.info("开始执行MAC算法...")
        start = timer()

        try:
            # 注意：solve 的第一个参数是 *从候选边中选择的数量*
            selected_candidate_weights, unrounded, upper_bound = mac_solver.solve(
                edges_to_select_from_candidates,
                x_init=x_init,            # 提供初始化向量
                rounding="madow",         # 使用随机舍入方法
                max_iters=10,            # 迭代次数
                use_cache=True,            # 启用缓存
                verbose=True,            # 启用详细输出
            )
            logger.info(f"MAC算法求解成功")
            # 目标值评估可能需要调整，因为它现在只基于候选边
            # logger.info(f"初始候选边目标值: {mac_solver.evaluate_objective(x_init):.6f}, "
            #           f"最终候选边目标值: {mac_solver.evaluate_objective(selected_candidate_weights):.6f}")
        except RuntimeError as e:
            error_message = f"MAC算法执行失败: {e}，程序终止"
            logger.error(error_message)
            
            # 记录失败信息
            base_name = os.path.basename(sim_matrix_path)
            name_parts = base_name.split('.')
            failure_info = {
                "initial_knn_k": initial_knn_k,
                "final_knn_k": current_knn_k,
                "error": f"MAC算法失败: {e}",
                "edges_selected": edges_to_select_from_candidates
            }
            
            info_path = os.path.join(output_dir or os.path.dirname(sim_matrix_path),
                                    f"{name_parts[0]}_mac_algorithm_failed_{current_knn_k}_info.txt")
            
            with open(info_path, 'w') as f:
                for k, v in failure_info.items():
                    f.write(f"{k}: {v}\n")
            logger.info(f"保存失败信息至 {info_path}")
            
            # 终止程序执行
            sys.exit(1)

        end = timer()
        logger.info(f"MAC算法完成，用时: {end - start:.2f}秒")

        # 8. 合并固定边和选中的候选边
        selected_candidate_edges = select_edges(candidate_edges_for_solver, selected_candidate_weights)
        final_selected_edges = fixed_knn_edges + selected_candidate_edges
        logger.info(f"最终选择了 {len(final_selected_edges)} 条边 (固定: {len(fixed_knn_edges)}, MAC选择: {len(selected_candidate_edges)})")

        # 9. 检查最终图的连通性 (可选)
        if check_connectivity:
            logger.info("检查最终图的连通性...")
            is_connected, components, component_sizes, isolated_nodes, component_nodes = check_graph_connectivity(final_selected_edges, n_items)
            
            if not is_connected:
                message = f"最终图不连通! 发现{components}个连通分量，最大分量大小: {max(component_sizes)}，有{len(isolated_nodes)}个孤立节点。"
                logger.warning(message)
            else:
                logger.info("最终图是连通的。")

        # 10. 构建稀疏邻接矩阵 (使用原始相似度作为权重)
        logger.info("构建稀疏邻接矩阵...")
        indices = []
        values = []

        for edge in final_selected_edges:
            # 添加双向边
            indices.append([edge.i, edge.j])
            indices.append([edge.j, edge.i])
            # 使用边的原始权重 (来自 sim_matrix)
            values.append(edge.weight) 
            values.append(edge.weight)

        indices = torch.tensor(indices).t()
        values = torch.tensor(values)

        # 创建稀疏张量 (合并重复项)
        sparse_adj = torch.sparse_coo_tensor(
            indices,
            values,
            (n_items, n_items)
        ).coalesce() # 合并重复边 (例如，如果一条边既是固定的又是MAC选的)
        
        # 获取合并后的索引和值
        final_indices = sparse_adj.indices()
        final_values = sparse_adj.values()


        # 11. 保存结果
        if output_dir is None:
            output_dir = os.path.dirname(sim_matrix_path)

        # 创建输出文件名 (包含 knn_k 和 knn_k_fixed)
        base_name = os.path.basename(sim_matrix_path)
        name_parts = base_name.split('.')
        # 使用 fk 表示 fixed_k
        output_name = f"{name_parts[0]}_mac_k{current_knn_k}_fk{knn_k_fixed}.pt"
        output_path = os.path.join(output_dir, output_name)

        logger.info(f"保存稀疏邻接矩阵至 {output_path}")
        # 保存合并后的稀疏张量
        torch.save(sparse_adj, output_path) 

        # 12. 计算并保存归一化的拉普拉斯矩阵
        logger.info("计算归一化拉普拉斯矩阵...")
        # 使用合并后的索引和值
        norm_adj = compute_normalized_laplacian(final_indices, sparse_adj.size(), final_values) 
        norm_output_name = f"{name_parts[0]}_mac_k{current_knn_k}_fk{knn_k_fixed}_norm.pt"
        norm_output_path = os.path.join(output_dir, norm_output_name)
        logger.info(f"保存归一化拉普拉斯矩阵至 {norm_output_path}")
        torch.save(norm_adj, norm_output_path)

        # 生成连通性报告
        connectivity_info = {
            "target_knn_k": initial_knn_k,
            "fixed_knn_k": knn_k_fixed,
            "final_total_edges": len(final_selected_edges),
            "is_connected": is_connected if check_connectivity else "Not Checked",
            "components": components if not is_connected else 1,
            "largest_component_size": max(component_sizes) if not is_connected else n_items,
            "isolated_nodes_count": len(isolated_nodes) if not is_connected else 0
        }
        info_path = os.path.join(output_dir, f"{name_parts[0]}_mac_k{current_knn_k}_fk{knn_k_fixed}_info.txt")
        with open(info_path, 'w') as f:
            for k, v in connectivity_info.items():
                f.write(f"{k}: {v}\n")
        logger.info(f"保存图信息至 {info_path}")

        return sparse_adj, norm_adj, current_knn_k

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        traceback.print_exc()
        # 终止程序
        sys.exit(1)

def compute_normalized_laplacian(indices, adj_size, values=None):
    """计算归一化拉普拉斯矩阵"""
    if values is None:
        # 如果没有提供值，则假设为1 (但在新逻辑中应该总是有值的)
        values = torch.ones(indices.shape[1], device=indices.device) 
        
    # 确保 adj 是 coalesced
    adj = torch.sparse_coo_tensor(indices, values, adj_size).coalesce()
    indices = adj.indices() # 使用合并后的索引
    values = adj.values()   # 使用合并后的值
    
    # 计算度矩阵
    row_sum = 1e-7 + torch.sparse.sum(adj, dim=1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # 处理孤立节点
    
    # D^(-1/2) A D^(-1/2)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    cols_inv_sqrt = r_inv_sqrt[indices[1]]
    norm_values = values * rows_inv_sqrt * cols_inv_sqrt
    
    return torch.sparse_coo_tensor(indices, norm_values, adj_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用MAC算法对相似度矩阵进行稀疏化 (强制保留KNN边)")
    parser.add_argument('--input', '-i', type=str, required=True, help='输入相似度矩阵文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--knn_k', '-k', type=int, default=50, help='目标总边数的近似 K 值 (控制总密度)')
    parser.add_argument('--knn_k_fixed', '-fk', type=int, default=20, help='强制保留的 KNN 边的 K 值') # <-- 新增参数
    parser.add_argument('--sim_threshold', '-t', type=float, default=0, help='相似度阈值 (用于初始候选边)')
    parser.add_argument('--max_edges', '-m', type=int, default=30000000, help='最大初始候选边数量')
    parser.add_argument('--fiedler_method', '-fm', type=str, default='tracemin_lu',
                       choices=['lanczos', 'lobpcg', 'tracemin_lu', 'tracemin_cholesky'],
                       help='计算Fiedler向量的方法')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='计算设备 (用于加速KNN查找等)')
    parser.add_argument('--check_connectivity', '-c', action='store_true', help='检查最终图的连通性')

    args = parser.parse_args()

    apply_mac_to_similarity_matrix(
        args.input,
        args.output_dir,
        args.knn_k,
        args.knn_k_fixed, # <-- 传递新参数
        args.sim_threshold,
        args.max_edges,
        args.device,
        args.fiedler_method,
        args.check_connectivity
    )

'''
# 示例命令:
# 目标总边数近似 k=50，强制保留 k=20 的 KNN 边
python mac/sim2mac_fixed.py --input data/baby/image_sim_matrix.pt --output_dir data/baby -k 50 -fk 20 --check_connectivity --device cuda:0
python mac/sim2mac_fixed.py --input data/baby/text_sim_matrix.pt --output_dir data/baby -k 50 -fk 20 --check_connectivity --device cuda:0

# 目标总边数近似 k=100，强制保留 k=20 的 KNN 边
python mac/sim2mac_fixed.py --input data/baby/image_sim_matrix.pt --output_dir data/baby -k 100 -fk 20 --check_connectivity --device cuda:0
python mac/sim2mac_fixed.py --input data/baby/text_sim_matrix.pt --output_dir data/baby -k 100 -fk 20 --check_connectivity --device cuda:0
'''