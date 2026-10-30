r"""
fr1_single_gcn.py
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian
from mac.solvers.mac import MAC
from mac.utils.graphs import Edge, select_edges

class SPARC(GeneralRecommender):
    def _compute_normalized_laplacian_weighted(self, indices, values, adj_size):
        """
        计算加权图的归一化拉普拉斯矩阵 (D^-0.5 * A * D^-0.5).
        """
        # 确保 adj 是 coalesced 以正确处理度和权重
        adj = torch.sparse_coo_tensor(indices.to(self.device), values.to(self.device), adj_size).coalesce()
        
        # 使用合并后的索引和值
        final_indices = adj.indices()
        final_values = adj.values()
        
        # 计算度矩阵 (基于权重)
        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7 # 加 epsilon 防止除零
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        r_inv_sqrt[torch.isinf(r_inv_sqrt) | torch.isnan(r_inv_sqrt)] = 0. # 处理孤立节点或无效值
        
        # D^(-1/2) A D^(-1/2)
        rows_inv_sqrt = r_inv_sqrt[final_indices[0]]
        cols_inv_sqrt = r_inv_sqrt[final_indices[1]]
        
        norm_values = final_values * rows_inv_sqrt * cols_inv_sqrt
        
        return torch.sparse_coo_tensor(final_indices, norm_values, adj_size).to(self.device)

    # 模型初始化方法
    def __init__(self, config, dataset):
        super(SPARC, self).__init__(config, dataset)

        # 初始化模型参数
        self.embedding_dim = config['embedding_size'] # This now refers to the size of each component before concatenation
        self.final_embedding_dim = self.embedding_dim * 2 # The actual dimension used throughout the GCNs and for loss
        
        self.lambda_coeff = config['lambda_coeff']
        # self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout'] # For UI graph edge dropout
        # self.degree_ratio = config['degree_ratio']
        self.mac_knn_k = config['mac_knn_k']
        self.mac_fixed_k = config['mac_fixed_k']
        self.n_nodes = self.n_users + self.n_items
        self.mm_dropout_rate = config['mm_dropout_rate'] # Dropout rate for MM graph

        # Contrastive Loss parameters
        self.cl_temperature = config['cl_temperature'] # 0.2
        self.cl_weight = config['cl_weight'] # 0.1
        self.dropout_for_cl = nn.Dropout(p=config['cl_dropout_rate']) # 0.5

        # 加载数据集信息，获取交互矩阵
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device) # UI graph (normalized)
        
        self.masked_adj = None # For UI graph dropout
        self.raw_weighted_mm_adj = None # Fused, unnormalized, weighted MM graph
        self.mm_adj_normalized_no_dropout = None # Normalized version of raw_weighted_mm_adj if no dropout
        self.masked_mm_adj = None # MM graph to be used in forward pass

        self.edge_indices, self.edge_values = self.get_edge_info() # For UI graph
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        
        self.user_embedding1 = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_embedding2 = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding1.weight)
        nn.init.xavier_uniform_(self.user_embedding2.weight)
        # No user_concat_trs needed, concatenated features will be self.final_embedding_dim

        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        # No item_concat_trs needed, concatenated features will be self.final_embedding_dim

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mac_knn_k_val = self.mac_knn_k # Use a distinct variable for f-string
        mac_fixed_k_val = self.mac_fixed_k # Use a distinct variable for f-string

        # --- Load or Construct Raw Weighted MM Graph ---
        # Define filenames for UNNORMALIZED MAC outputs
        unnorm_mm_adj_file = os.path.join(dataset_path, f'mm_adj_fr1_mac_k{mac_knn_k_val}_fk{mac_fixed_k_val}_{int(10*self.mm_image_weight)}_RAW.pt') # Suffix to distinguish

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim) 
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        if os.path.exists(unnorm_mm_adj_file):
            self.raw_weighted_mm_adj = torch.load(unnorm_mm_adj_file, map_location='cpu')
            logging.info(f"从 {unnorm_mm_adj_file} 加载预存的原始带权多模态邻接矩阵")
        else:
            image_adj_raw, text_adj_raw = None, None
            # Load UNNORMALIZED image_sim_matrix_mac_k{...}_fk{...}.pt
            image_mac_raw_file = os.path.join(dataset_path, f'image_sim_matrix_mac_k{mac_knn_k_val}_fk{mac_fixed_k_val}.pt')
            text_mac_raw_file = os.path.join(dataset_path, f'text_sim_matrix_mac_k{mac_knn_k_val}_fk{mac_fixed_k_val}.pt')

            if os.path.exists(image_mac_raw_file) and self.v_feat is not None:
                logging.info(f"加载未归一化的图像邻接矩阵: {image_mac_raw_file}")
                image_adj_raw = torch.load(image_mac_raw_file, map_location='cpu')
            if os.path.exists(text_mac_raw_file) and self.t_feat is not None:
                logging.info(f"加载未归一化的文本邻接矩阵: {text_mac_raw_file}")
                text_adj_raw = torch.load(text_mac_raw_file, map_location='cpu')

            if image_adj_raw is not None and text_adj_raw is not None:
                logging.info(f"融合图像和文本邻接矩阵 (原始权重), 图像权重: {self.mm_image_weight}")
                self.raw_weighted_mm_adj = self.mm_image_weight * image_adj_raw + (1.0 - self.mm_image_weight) * text_adj_raw
            elif image_adj_raw is not None:
                self.raw_weighted_mm_adj = image_adj_raw
            elif text_adj_raw is not None:
                self.raw_weighted_mm_adj = text_adj_raw
            
            if self.raw_weighted_mm_adj is None:
                error_message = f"未找到MAC处理后的原始带权邻接矩阵！请确保 sim2mac_fixed.py 已生成不带 _norm 后缀的文件。"
                logging.error(error_message)
                raise RuntimeError(error_message)
            
            self.raw_weighted_mm_adj = self.raw_weighted_mm_adj.coalesce() # Ensure it's coalesced
            torch.save(self.raw_weighted_mm_adj, unnorm_mm_adj_file)
            logging.info(f"保存融合的原始带权多模态邻接矩阵到 {unnorm_mm_adj_file}")

        self.mm_sampling_weights_cache = None
        self.mm_original_indices_cache = None
        self.mm_original_weights_cache = None # To store raw weights of edges for reconstruction

        if self.raw_weighted_mm_adj is not None:
            if self.raw_weighted_mm_adj.shape[0] != self.n_items or self.raw_weighted_mm_adj.shape[1] != self.n_items:
                 raise ValueError(f"原始带权MM邻接矩阵维度 ({self.raw_weighted_mm_adj.shape}) 与物品数量 ({self.n_items}) 不匹配!")
            self.raw_weighted_mm_adj = self.raw_weighted_mm_adj.to(self.device).coalesce()
            
            # Always compute the normalized version for no-dropout case or as base for sampling weights
            self.mm_adj_normalized_no_dropout = self._compute_normalized_laplacian_weighted(
                self.raw_weighted_mm_adj.indices(), 
                self.raw_weighted_mm_adj.values(), 
                self.raw_weighted_mm_adj.shape
            ).coalesce() # Ensure it's coalesced

            if self.mm_dropout_rate > 0.0:
                # For dropout, we sample based on the values of the *normalized* version of raw_weighted_mm_adj
                # The indices for sampling will correspond to the edges in mm_adj_normalized_no_dropout
                # However, we need to map these sampled edges back to the *original raw weights*
                # So, we need a mapping or ensure raw_weighted_mm_adj and mm_adj_normalized_no_dropout
                # have the same edge ordering after coalescing if their structures are identical.
                # A safer way: sample from raw_weighted_mm_adj's structure, using normalized values as weights.

                # Coalesce raw_weighted_mm_adj to get its definitive structure and original weights
                self.raw_weighted_mm_adj = self.raw_weighted_mm_adj.coalesce()
                self.mm_original_indices_cache = self.raw_weighted_mm_adj.indices().clone()
                self.mm_original_weights_cache = self.raw_weighted_mm_adj.values().clone()

                # Compute temporary normalized version of raw_weighted_mm_adj *just for getting sampling weights*
                # This assumes _compute_normalized_laplacian_weighted returns a graph
                # with the same .indices() order as its input if the input is already coalesced.
                # This is generally true for sparse.mm-like operations if structure doesn't change.
                temp_normalized_for_sampling_weights = self._compute_normalized_laplacian_weighted(
                    self.mm_original_indices_cache, # Use original structure
                    self.mm_original_weights_cache, # Use original weights
                    self.raw_weighted_mm_adj.shape
                ).coalesce()
                
                # Ensure the indices match up. This is a strong assumption.
                # A check:
                if not torch.equal(temp_normalized_for_sampling_weights.indices(), self.mm_original_indices_cache):
                    logging.warning("Indices mismatch between raw and temp normalized for sampling weights. This might lead to incorrect sampling probabilities for edges.")
                    # Fallback: use absolute raw weights for sampling if mismatch
                    self.mm_sampling_weights_cache = torch.abs(self.mm_original_weights_cache) + 1e-12
                else:
                    self.mm_sampling_weights_cache = torch.abs(temp_normalized_for_sampling_weights.values()) + 1e-12
                
                if self.mm_sampling_weights_cache.numel() != self.mm_original_indices_cache.shape[1]:
                     logging.error(f"Sampling weights cache size ({self.mm_sampling_weights_cache.numel()}) "
                                   f"does not match number of original edges ({self.mm_original_indices_cache.shape[1]}).")
                     # Fallback or raise error
                     self.mm_sampling_weights_cache = torch.abs(self.mm_original_weights_cache) + 1e-12


            logging.info(f"原始带权MM邻接矩阵已处理并移至设备: {self.device}")
        else: # raw_weighted_mm_adj is None
            self.mm_adj_normalized_no_dropout = None


    # compute_normalized_laplacian (old one for unweighted) is no longer needed if not used elsewhere.
    # If it was used by get_norm_adj_mat or _normalize_adj_m, those need to be checked.
    # For now, assuming it's replaced by _compute_normalized_laplacian_weighted for MM graph.

    # 构造规范化邻接矩阵 (for UI graph, this is fine)
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # 计算每个节点度数，并进行归一化处理
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # 转换为tensor形式的稀疏矩阵
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    # 训练前的预处理
    def pre_epoch_processing(self):
        # --- UI Graph Edge Dropout (Degree Sensitive) ---
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
        else:
            degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
            if degree_len > 0 and self.edge_values.numel() > 0 : # Ensure there are edges and values to sample from
                # Add epsilon for stability if edge_values can be zero
                degree_idx = torch.multinomial(self.edge_values + 1e-12, degree_len, replacement=False)
                keep_indices = self.edge_indices[:, degree_idx]
                keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
                all_values = torch.cat((keep_values, keep_values))
                
                ui_graph_kept_indices_for_sparse = keep_indices.clone()
                ui_graph_kept_indices_for_sparse[1] += self.n_users 
                all_indices = torch.cat((ui_graph_kept_indices_for_sparse, torch.flip(ui_graph_kept_indices_for_sparse, [0])), 1)
                self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)
            else: # No edges to sample or all edges dropped
                 self.masked_adj = torch.sparse.FloatTensor(
                    torch.empty((2,0), dtype=torch.long, device=self.device),
                    torch.empty((0), dtype=torch.float, device=self.device),
                    self.norm_adj.shape
                ).to(self.device)


        # --- MM Graph Edge Dropout (Using sampling weights derived from normalized raw_weighted_mm_adj) ---
        if self.raw_weighted_mm_adj is not None:
            if self.mm_dropout_rate > 0.0 and \
               self.mm_sampling_weights_cache is not None and \
               self.mm_original_indices_cache is not None and \
               self.mm_original_weights_cache is not None:

                num_original_mm_edges = self.mm_original_indices_cache.size(1)
                
                if num_original_mm_edges == 0:
                    self.masked_mm_adj = self.mm_adj_normalized_no_dropout # Should be an empty normalized graph
                else:
                    num_edges_to_keep_mm = int(num_original_mm_edges * (1. - self.mm_dropout_rate))
                    
                    if num_edges_to_keep_mm > 0:
                        if self.mm_sampling_weights_cache.numel() == num_original_mm_edges:
                            try:
                                perm_indices_for_sampling = torch.multinomial(
                                    self.mm_sampling_weights_cache, 
                                    num_edges_to_keep_mm, 
                                    replacement=False
                                )
                                
                                # Get the indices and *original raw weights* of the sampled edges
                                mm_kept_indices = self.mm_original_indices_cache[:, perm_indices_for_sampling]
                                mm_kept_original_weights = self.mm_original_weights_cache[perm_indices_for_sampling]
                                
                                # Normalize the pruned graph using its original (kept) weights
                                self.masked_mm_adj = self._compute_normalized_laplacian_weighted(
                                    mm_kept_indices, 
                                    mm_kept_original_weights, 
                                    torch.Size((self.n_items, self.n_items))
                                )
                            except RuntimeError as e:
                                logging.error(f"Error during multinomial sampling for mm_adj: {e}. Falling back to non-dropout normalized.")
                                self.masked_mm_adj = self.mm_adj_normalized_no_dropout
                        else:
                            logging.error("Mismatch between sampling weights and original edges for mm_adj. Falling back.")
                            self.masked_mm_adj = self.mm_adj_normalized_no_dropout
                    else: # All edges are dropped
                        self.masked_mm_adj = torch.sparse.FloatTensor(
                            torch.empty((2,0), dtype=torch.long, device=self.device),
                            torch.empty((0), dtype=torch.float, device=self.device),
                            torch.Size((self.n_items, self.n_items))
                        ).to(self.device)
            else: # No dropout for MM graph, use pre-normalized version
                self.masked_mm_adj = self.mm_adj_normalized_no_dropout
        else: 
            self.masked_mm_adj = None

    # 对邻接矩阵进行归一化处理 (for UI graph, this is fine)
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0], device=indices.device), adj_size) # ensure ones_like is on correct device
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    # 获取边信息 (for UI graph, this is fine)
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # self._normalize_adj_m expects indices on CPU if it creates new tensors without device specified
        # or ensure its internal operations handle devices correctly.
        # For safety, ensure edges are on CPU if _normalize_adj_m doesn't manage device internally for new tensors.
        # However, self.edge_values is moved to self.device later.
        values = self._normalize_adj_m(edges.cpu(), torch.Size((self.n_users, self.n_items)))
        return edges, values # edges will be moved to device later with self.edge_values

    # _info_nce_loss (no changes)
    def _info_nce_loss(self, view1, view2, temperature):
        """
        Calculates InfoNCE loss between two views of a batch of embeddings.
        Assumes view1[i] and view2[i] are positive pairs.
        Other elements in view2 (view2[j] where j!=i) serve as negative samples for view1[i].
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        batch_size = view1.shape[0]
        
        # Calculate cosine similarity
        # sim_matrix[i, j] = sim(view1[i], view2[j])
        sim_matrix = torch.matmul(view1, view2.t()) / temperature
        
        # Labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size).to(view1.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    # 前向传播函数
    def forward(self, adj): # adj is self.masked_adj (UI graph)
        # --- Determine initial user features ---
        user_emb1 = self.user_embedding1.weight
        user_emb2 = self.user_embedding2.weight
        user_initial_features = torch.cat((user_emb1, user_emb2), dim=1) # Dim: self.final_embedding_dim

        # --- Determine initial item features based on modalities ---
        image_f, text_f = None, None
        
        if hasattr(self, 'image_embedding') and self.image_embedding is not None and hasattr(self, 'image_trs'):
            image_f = self.image_trs(self.image_embedding.weight) # Dim: self.embedding_dim
            image_f = F.normalize(image_f, dim=1)
        if hasattr(self, 'text_embedding') and self.text_embedding is not None and hasattr(self, 'text_trs'):
            text_f = self.text_trs(self.text_embedding.weight) # Dim: self.embedding_dim
            text_f = F.normalize(text_f, dim=1)

        if image_f is not None and text_f is not None:
            item_initial_features = torch.cat((image_f, text_f), dim=1) # Dim: self.final_embedding_dim
        elif image_f is not None:
            zeros_for_text = torch.zeros_like(image_f) # zeros_like image_f (self.embedding_dim)
            item_initial_features = torch.cat((image_f, zeros_for_text), dim=1) # Dim: self.final_embedding_dim
        elif text_f is not None:
            zeros_for_image = torch.zeros_like(text_f) # zeros_like text_f (self.embedding_dim)
            item_initial_features = torch.cat((zeros_for_image, text_f), dim=1) # Dim: self.final_embedding_dim
        else:
            # Fallback to ID embedding: concatenate with zeros
            id_based_item_features = self.item_id_embedding.weight # Dim: self.embedding_dim
            zeros_for_modality = torch.zeros_like(id_based_item_features)
            item_initial_features = torch.cat((id_based_item_features, zeros_for_modality), dim=1) # Dim: self.final_embedding_dim
        
        h = item_initial_features 
        
        # Use self.masked_mm_adj for MM graph propagation
        current_mm_graph_to_use = self.masked_mm_adj 

        if current_mm_graph_to_use is not None and self.n_layers > 0 :
            item_gcn_embs = [h] # Store embeddings from each layer
            for _ in range(self.n_layers):
                h_next = torch.sparse.mm(current_mm_graph_to_use, h)
                item_gcn_embs.append(h_next)
                h = h_next # Update h for the next iteration
            # Aggregate embeddings from all layers (e.g., mean or sum)
            # Using mean as in the original forward logic
            h = torch.mean(torch.stack(item_gcn_embs, dim=1), dim=1)
        else: 
            pass # h remains item_initial_features if no MM GCN

        ego_embeddings = torch.cat((user_initial_features, h), dim=0)
        
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            # 'adj' is the UI graph (self.masked_adj from calculate_loss)
            # The GCN update rule was: ego_embeddings = side_embeddings
            # Let's follow a more standard LightGCN-like aggregation:
            side_embeddings = torch.sparse.mm(adj, all_embeddings[-1]) # Propagate from previous layer's full embedding
            all_embeddings.append(side_embeddings) # Append the new propagated state
        
        # Aggregate embeddings from all UI layers
        final_ego_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        u_g_embeddings, i_g_embeddings_ui = torch.split(final_ego_embeddings, [self.n_users, self.n_items], dim=0)
        
        final_i_embeddings = i_g_embeddings_ui + h # Combine UI influence with MM influence
        
        return u_g_embeddings, final_i_embeddings

    # 计算BPR损失
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    # 计算总的损失，包括BPR损失及正则化项
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # ua_embeddings and ia_embeddings are the final embeddings from the forward pass
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj) # self.masked_adj is prepared in pre_epoch

        u_g_embeddings_batch = ua_embeddings[users]
        pos_i_g_embeddings_batch = ia_embeddings[pos_items]
        neg_i_g_embeddings_batch = ia_embeddings[neg_items]

        # BPR Loss (main recommendation task loss)
        batch_mf_loss = self.bpr_loss(u_g_embeddings_batch, pos_i_g_embeddings_batch, neg_i_g_embeddings_batch)
        
        # --- Contrastive Loss Calculation ---
        # Create augmented views using dropout
        u_view1 = self.dropout_for_cl(u_g_embeddings_batch)
        u_view2 = self.dropout_for_cl(u_g_embeddings_batch) # Another dropout pass for a different view
        
        # For items, use positive items from BPR for contrastive learning
        i_view1 = self.dropout_for_cl(pos_i_g_embeddings_batch)
        i_view2 = self.dropout_for_cl(pos_i_g_embeddings_batch)
        
        user_cl_loss = self._info_nce_loss(u_view1, u_view2, self.cl_temperature)
        item_cl_loss = self._info_nce_loss(i_view1, i_view2, self.cl_temperature)
        
        total_cl_loss = user_cl_loss + item_cl_loss
        # --- End Contrastive Loss ---

        # Previous auxiliary losses (mf_t_loss, mf_v_loss) are now replaced by CL loss.
        # If L2 regularization on embeddings is desired, it should be handled by self.reg_loss or similar.
        # The self.reg_weight from config is now effectively self.cl_weight if we rename.
        # For clarity, we use self.cl_weight as defined in __init__.

        # L2 Regularization on embeddings (if needed, can be added via EmbLoss)
        # reg_loss = self.reg_loss(u_g_embeddings_batch, pos_i_g_embeddings_batch, neg_i_g_embeddings_batch) # Example
        
        final_loss = batch_mf_loss + self.cl_weight * total_cl_loss # + self.reg_weight * reg_loss (if L2 reg is separate)
        
        return final_loss

    # 针对全物品的预测，返回用户对所有物品的评分
    def full_sort_predict(self, interaction):
        user = interaction[0]

        # For prediction, use the non-dropout version of UI graph (self.norm_adj)
        # and the non-dropout version of MM graph (self.mm_adj_normalized_no_dropout)
        
        # Temporarily set masked_mm_adj for prediction if it's different
        original_masked_mm_adj = self.masked_mm_adj
        if self.mm_adj_normalized_no_dropout is not None:
            self.masked_mm_adj = self.mm_adj_normalized_no_dropout
        elif self.raw_weighted_mm_adj is not None: # Fallback if no dropout was ever configured
             self.masked_mm_adj = self._compute_normalized_laplacian_weighted(
                self.raw_weighted_mm_adj.indices(),
                self.raw_weighted_mm_adj.values(),
                self.raw_weighted_mm_adj.shape
            )
        else:
            self.masked_mm_adj = None # No MM graph available

        restore_user_e, restore_item_e = self.forward(self.norm_adj) # Pass the original UI graph
        
        # Restore masked_mm_adj if it was changed
        self.masked_mm_adj = original_masked_mm_adj
            
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


