
"""
PaiNN (Polarizable Atom Interaction Neural Network) 层实现。
包含 SE(3) 等变的消息传递和更新机制，同时处理标量和向量特征。

关键特性：
- 严格保证 SE(3) 等变性
- 标量特征 s: [N, hidden_dim]
- 向量特征 v: [N, hidden_dim, 3]
- 旋转输入 → 输出同样旋转
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_sum



class CosineCutoff(nn.Module):
    """
    余弦截断函数，用于平滑地将远距离相互作用衰减为0。
    f(d) = 0.5 * [cos(pi * d / cutoff) + 1], d &lt;= cutoff
    f(d) = 0, d &gt; cutoff
    """
    
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        
    def forward(self, d):
        """
        参数:
            d: [E] 边距离张量
        返回:
            [E] 截断后的权重
        """
        d_scaled = d / self.cutoff
        mask = (d_scaled <= 1.0).float()
        cos_val = 0.5 * (torch.cos(d_scaled * torch.pi) + 1.0)
        return cos_val * mask


class PaiNNMessage(MessagePassing):
    """
    PaiNN 消息传递层。
    从邻居节点收集信息，生成消息用于更新当前节点。
    
    输入：
        s: [N, hidden_dim] 标量特征
        v: [N, hidden_dim, 3] 向量特征
        edge_index: [2, E] 边索引
        edge_attr: [E, edge_dim] 边特征
        edge_vec: [E, 3] 边向量 (r_j - r_i)
    
    输出：
        ds: [N, hidden_dim] 标量消息
        dv: [N, hidden_dim, 3] 向量消息
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        
        # 输入投影层：将 s_i 和 s_j 合并投影
        self.s_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
        # 边特征投影层
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
        # 输出层的可学习参数
        self.final_s = nn.Linear(hidden_dim, hidden_dim)
        self.final_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.s_proj[0].weight)
        nn.init.xavier_uniform_(self.s_proj[2].weight)
        nn.init.xavier_uniform_(self.edge_proj[0].weight)
        nn.init.xavier_uniform_(self.edge_proj[2].weight)
        nn.init.xavier_uniform_(self.final_s.weight)
        nn.init.xavier_uniform_(self.final_v.weight)
        
    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_vec: torch.Tensor
    ):
        return self.propagate(
            edge_index,
            s=s,
            v=v,
            edge_attr=edge_attr,
            edge_vec=edge_vec
        )
    
    def message(
        self,
        s_i,
        s_j,
        v_j,
        edge_attr,
        edge_vec
    ):
        """
        计算边 (j -> i) 的消息。
        """
        # s_i: [E, hidden_dim], s_j: [E, hidden_dim]
        s_combined = torch.cat([s_i, s_j], dim=-1)  # [E, 2*hidden_dim]
        
        # 投影标量部分
        phi = self.s_proj(s_combined)  # [E, 3*hidden_dim]
        phi_edge = self.edge_proj(edge_attr)  # [E, 3*hidden_dim]
        phi = phi * phi_edge  # [E, 3*hidden_dim]
        
        # 分割为三个部分
        phi1, phi2, phi3 = torch.chunk(phi, 3, dim=-1)  # 每个 [E, hidden_dim]
        
        # 计算向量消息：phi2 * v_j + phi3 * (edge_vec方向)
        # edge_vec: [E, 3] -&gt; [E, 1, 3] -&gt; [E, hidden_dim, 3]
        edge_vec_expanded = edge_vec.unsqueeze(1)  # [E, 1, 3]
        dv_msg = phi2.unsqueeze(-1) * v_j + phi3.unsqueeze(-1) * edge_vec_expanded
        
        # 计算标量消息
        ds_msg = phi1
        
        return ds_msg, dv_msg
    
    def aggregate(
        self,
        inputs,
        index,
        ptr,
        dim_size
    ):
        ds_msg, dv_msg = inputs
        ds = scatter_sum(ds_msg, index, dim=0, dim_size=dim_size)
        dv = scatter_sum(dv_msg, index, dim=0, dim_size=dim_size)
        return ds, dv
    
    def update(
        self,
        inputs
    ):
        ds, dv = inputs
        ds = self.final_s(ds)
        dv = self.final_v(dv.transpose(1, 2)).transpose(1, 2)
        return ds, dv


class PaiNNUpdate(nn.Module):
    """
    PaiNN 更新层。
    利用聚合后的消息更新节点的标量和向量特征。
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # U 矩阵：用于混合向量特征
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # V 矩阵：另一个向量混合
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 标量处理网络
        self.s_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.s_mlp[0].weight)
        nn.init.xavier_uniform_(self.s_mlp[2].weight)
        
    def forward(
        self,
        s,
        v
    ):
        """
        参数:
            s: [N, hidden_dim] 标量特征
            v: [N, hidden_dim, 3] 向量特征
        返回:
            (ds, dv): 更新量
        """
        # v 的形状变换便于 Linear 层处理: [N, hidden_dim, 3] -&gt; [N, 3, hidden_dim]
        v_trans = v.transpose(1, 2)  # [N, 3, hidden_dim]
        
        Uv = self.U(v_trans).transpose(1, 2)  # [N, hidden_dim, 3]
        Vv = self.V(v_trans).transpose(1, 2)  # [N, hidden_dim, 3]
        
        # 计算向量的模长 (沿最后一个维度)
        v_norm = torch.norm(Vv, p=2, dim=-1)  # [N, hidden_dim]
        
        # 拼接标量特征和向量模长
        s_combined = torch.cat([s, v_norm], dim=-1)  # [N, 2*hidden_dim]
        
        # MLP处理
        a = self.s_mlp(s_combined)  # [N, 3*hidden_dim]
        a1, a2, a3 = torch.chunk(a, 3, dim=-1)  # 每个 [N, hidden_dim]
        
        # 计算标量更新
        ds = a1 + a2 * v_norm  # [N, hidden_dim]
        
        # 计算向量更新
        dv = a3.unsqueeze(-1) * Uv  # [N, hidden_dim, 3]
        
        return ds, dv


class PaiNNBlock(nn.Module):
    """
    完整的 PaiNN 块，包含消息传递和更新。
    增加残差缩放与数值稳定保护，防止深层爆炸。
    """
    
    def __init__(self, hidden_dim, edge_dim, residual_scale=0.1, clamp_value=100.0):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, edge_dim)
        self.update = PaiNNUpdate(hidden_dim)
        self.residual_scale = residual_scale
        self.clamp_value = clamp_value
        
    def forward(
        self,
        s,
        v,
        edge_index,
        edge_attr,
        edge_vec
    ):
        # 消息传递
        ds_msg, dv_msg = self.message(s, v, edge_index, edge_attr, edge_vec)
        
        # 残差连接（缩放）
        s = s + self.residual_scale * ds_msg
        v = v + self.residual_scale * dv_msg
        
        # 更新
        ds_update, dv_update = self.update(s, v)
        
        # 残差连接（缩放）
        s = s + self.residual_scale * ds_update
        v = v + self.residual_scale * dv_update

        # 数值稳定保护（防止进入 1e38 区间）
        s = torch.clamp(s, min=-self.clamp_value, max=self.clamp_value)
        v = torch.clamp(v, min=-self.clamp_value, max=self.clamp_value)
        
        return s, v



class PaiNNEncoder(nn.Module):
    """
    完整的 PaiNN 编码器，由多个 PaiNNBlock 堆叠而成。
    支持梯度检查点以节省显存。
    """
    
    def __init__(
        self,
        hidden_dim=128,
        num_layers=6,
        edge_dim=19,
        vocab_size=101,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 元素 Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 向量特征初始化的线性层
        self.v_init_proj = nn.Linear(3, hidden_dim, bias=False)
        
        # PaiNN 块
        self.blocks = nn.ModuleList([
            PaiNNBlock(hidden_dim, edge_dim, residual_scale=0.1, clamp_value=50.0)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.out_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        z,
        vector_features,
        edge_index,
        edge_attr,
        pos,
        initial_s=None
    ):
        """
        参数:
            z: [N] 原子序数
            vector_features: [N, 3] 初始向量特征 (手性感知)
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            pos: [N, 3] 原子坐标
            initial_s: [N, hidden_dim] 可选，用于初始化标量特征（如果提供）
        返回:
            (s_out, v_out): 最终的标量和向量特征
        """
        # 初始化标量特征
        if initial_s is None:
            s = self.embedding(z)  # [N, hidden_dim]
        else:
            s = initial_s  # 使用提供的初始特征
        
        # 初始化向量特征: [N, 3] -&gt; [N, hidden_dim, 3]
        v = self.v_init_proj(vector_features.unsqueeze(1)).transpose(1, 2)
        # 或者更简单的方式：广播（经Gemini和codex提醒不使用该方法）
        # v = vector_features.unsqueeze(1).repeat(1, self.hidden_dim, 1)  # [N, hidden_dim, 3]
        
        # 计算边向量: r_j - r_i
        row, col = edge_index
        edge_vec = pos[col] - pos[row]  # [E, 3]
        
                # ===== [DEBUG] 只打印一次 =====
        if not hasattr(self, "_debug_once_encoder"):
            self._debug_once_encoder = False

        # 输入体检
        if not self._debug_once_encoder:
            print("\n[DEBUG][PaiNNEncoder] input check")
            print("  z dtype:", z.dtype, "min/max:", int(z.min()), int(z.max()))
            print("  pos finite:", torch.isfinite(pos).all().item(), "shape:", tuple(pos.shape))
            print("  vector_features finite:", torch.isfinite(vector_features).all().item(), "shape:", tuple(vector_features.shape))
            print("  edge_attr finite:", torch.isfinite(edge_attr).all().item(), "shape:", tuple(edge_attr.shape))
            print("  edge_index shape:", tuple(edge_index.shape), "min/max:", int(edge_index.min()), int(edge_index.max()))
            print("  num_nodes:", int(pos.size(0)))
            if int(edge_index.max()) >= int(pos.size(0)) or int(edge_index.min()) < 0:
                print("  [ALERT] edge_index 越界！")
        # ===== [DEBUG] 只打印一次 =====
        if not hasattr(self, "_debug_once_encoder"):
            self._debug_once_encoder = False

        # 输入体检
        if not self._debug_once_encoder:
            print("\n[DEBUG][PaiNNEncoder] input check")
            print("  z dtype:", z.dtype, "min/max:", int(z.min()), int(z.max()))
            print("  pos finite:", torch.isfinite(pos).all().item(), "shape:", tuple(pos.shape))
            print("  vector_features finite:", torch.isfinite(vector_features).all().item(), "shape:", tuple(vector_features.shape))
            print("  edge_attr finite:", torch.isfinite(edge_attr).all().item(), "shape:", tuple(edge_attr.shape))
            print("  edge_index shape:", tuple(edge_index.shape), "min/max:", int(edge_index.min()), int(edge_index.max()))
            print("  num_nodes:", int(pos.size(0)))
            if int(edge_index.max()) >= int(pos.size(0)) or int(edge_index.min()) < 0:
                print("  [ALERT] edge_index 越界！")
        # ============================

        # 通过 PaiNN 块
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                s, v = torch.utils.checkpoint.checkpoint(
                    block, s, v, edge_index, edge_attr, edge_vec,
                    use_reentrant=False
                )
            else:
                s, v = block(s, v, edge_index, edge_attr, edge_vec)

            # 每层检查
            if not torch.isfinite(s).all() or not torch.isfinite(v).all():
                print(f"\n[DEBUG][PaiNNEncoder] NaN/Inf appears at block {i}")
                print("  s finite:", torch.isfinite(s).all().item())
                print("  v finite:", torch.isfinite(v).all().item())
                print("  s range:", float(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).min()),
                    float(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).max()))
                print("  v abs max:", float(torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).abs().max()))
                # 直接中断，避免后面污染
                raise RuntimeError(f"Non-finite detected after PaiNN block {i}")

        # 输出投影
        s_out = self.out_s(s)

        if not self._debug_once_encoder:
            print("\n[DEBUG][PaiNNEncoder] output check")
            print("  s_out finite:", torch.isfinite(s_out).all().item(), "shape:", tuple(s_out.shape))
            self._debug_once_encoder = True

        return s_out, v


