import os
import sys

# 获取当前脚本所在目录的上一级（即项目根目录），并加入系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter_add

# 导入你的模型和几何工具
from src.models.glue_cmae import GlueVAE
from src.utils.geometry import GaussianRBF

warnings.simplefilter('ignore', PDBConstructionWarning)
ELEMENT_TO_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'CL': 17}

class BinderEvaluator:
    def __init__(self, model_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        print("🧠 正在加载 CMAE 零样本物理打分器 (Binder Evaluator)...")
        self.model = GlueVAE(
            hidden_dim=config['model']['hidden_dim'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            edge_dim=config['model']['edge_dim'],
            vocab_size=config['model']['vocab_size']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(clean_state_dict)
        self.model.eval()

        self.cutoff_radius = 8.0 
        self.rbf = GaussianRBF(n_rbf=16, cutoff=self.cutoff_radius, start=0.0).to(self.device)

    def parse_complex(self, pdb_path, target_chains, binder_chains):
        """
        解析复合物，严格区分 Target (is_ligand=0) 和 Binder (is_ligand=1)
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_path)

        pos_list, z_list, is_ligand_list = [], [], []

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                if chain_id in target_chains:
                    label = 0
                elif chain_id in binder_chains:
                    label = 1
                else:
                    continue # 忽略未指定的链

                for residue in chain:
                    if residue.id[0] != ' ': 
                        continue 
                    for atom in residue:
                        pos_list.append(atom.coord)
                        element = atom.element.strip().upper()
                        z_list.append(ELEMENT_TO_Z.get(element, 6))
                        is_ligand_list.append(label)

        pos = torch.tensor(np.array(pos_list), dtype=torch.float32)
        x = torch.tensor(z_list, dtype=torch.long)
        is_ligand = torch.tensor(is_ligand_list, dtype=torch.long)
        
        # 将复合物移动到原点，防止坐标过大导致数值不稳定
        pos_center = pos.mean(dim=0, keepdim=True)
        pos = pos - pos_center

        return Data(x=x, pos=pos, is_ligand=is_ligand, pos_center=pos_center)

    def build_graph_and_features(self, data, current_pos):
        """
        根据给定的坐标 (可能是加噪后的) 重新构建图结构、拓扑边和等变向量特征。
        这里绝对保留了跨链边 (Cross-chain edges)！
        """
        device = current_pos.device
        
        # 1. 构图
        edge_index = radius_graph(current_pos, r=self.cutoff_radius, loop=False)
        row, col = edge_index
        
        diff = current_pos[row] - current_pos[col]
        dist = torch.norm(diff, p=2, dim=-1)
        
        # 2. 边类型计算 (这里跨链边 is_ligand 不同，same_chain 为 False)
        is_covalent = dist < 1.7
        same_chain = (data.is_ligand[row] == data.is_ligand[col])
        
        edge_type = torch.zeros((edge_index.size(1), 3), dtype=torch.float, device=device)
        edge_type[is_covalent, 0] = 1.0                     # 共价边
        edge_type[(~is_covalent) & same_chain, 1] = 1.0      # 链内非共价边
        edge_type[(~is_covalent) & (~same_chain), 2] = 1.0     # 🚨 跨链边 (Cross-chain)
        
        # 3. RBF
        rbf_feat = self.rbf(dist)
        edge_attr = torch.cat([edge_type, rbf_feat], dim=-1)
        
        # 4. 向量特征计算
        mask_cov = is_covalent
        row_cov = row[mask_cov]
        col_cov = col[mask_cov]
        
        N = current_pos.size(0)
        vector_features = torch.zeros(N, 3, device=device)
        
        if len(row_cov) > 0:
            vec_diff = current_pos[row_cov] - current_pos[col_cov]
            vector_features = scatter_add(vec_diff, col_cov, dim=0, dim_size=N)
            
        return edge_index, edge_attr, vector_features

    @torch.no_grad()
    def evaluate_binder(self, complex_data, interface_cutoff=8.0, noise_scale=1.0):
        """
        核心评价逻辑：加噪 -> 模型重构 -> 计算 RMSD
        """
        complex_data = complex_data.to(self.device)
        pos_true = complex_data.pos.clone()
        
        # 1. 寻找 Binder (is_ligand==1) 的界面原子
        mask_target = (complex_data.is_ligand == 0)
        mask_binder = (complex_data.is_ligand == 1)
        
        pos_target = pos_true[mask_target]
        pos_binder = pos_true[mask_binder]
        
        # 计算 Binder 到 Target 的距离矩阵
        dist_mat = torch.cdist(pos_binder, pos_target)
        min_dist_to_target, _ = dist_mat.min(dim=1)
        
        # 提取距 Target 小于 cutoff 的 Binder 原子作为界面
        binder_interface_idx_in_binder = torch.where(min_dist_to_target < interface_cutoff)[0]
        
        if len(binder_interface_idx_in_binder) == 0:
            return None, "未检测到物理接触界面 (Binder 离 Target 太远)！"
            
        # 映射回全局索引
        global_binder_indices = torch.where(mask_binder)[0]
        mask_nodes = global_binder_indices[binder_interface_idx_in_binder]
        
        # 2. 💣 挖掉坐标 (施加高斯噪声)
        pos_noisy = pos_true.clone()
        noise = torch.randn((len(mask_nodes), 3), device=self.device) * noise_scale
        pos_noisy[mask_nodes] += noise
        
        # 3. 基于“被破坏”的坐标，重新构建图结构和特征
        # 这样模型只能依靠跨链边和 Target 的未破坏表面来推断 Binder 的正确位置
        edge_index, edge_attr, vector_features = self.build_graph_and_features(complex_data, pos_noisy)
        
        # 4. 🚀 前向传播：调用编码器和解码器进行坐标重构
        
        # 🚨 极其关键的一步：严格对齐训练时的防御机制
        # 训练时你在 forward 里写了 fake_vector_features = torch.zeros_like(vector_features)
        # 这里必须照做，否则模型看到的特征分布就全乱了！
        fake_vector_features = torch.zeros_like(vector_features)

        # 第一步：编码器提取特征 (使用被破坏的坐标 pos_noisy)
        # 根据你 GlueVAE 里的 encode 签名：
        # def encode(self, z, vector_features, edge_index, edge_attr, pos)
        s, z_proj = self.model.encode(
            z=complex_data.x, 
            vector_features=fake_vector_features, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            pos=pos_noisy
        )
        
        # 第二步：解码器直接输出重构后的绝对坐标
        # 根据你 GlueVAE 里的 decode 签名：
        # def decode(self, atom_features, z_atom, fake_vector_features, edge_index, fake_edge_attr, fake_pos)
        pos_pred = self.model.decode(
            atom_features=s, 
            z_atom=complex_data.x, 
            fake_vector_features=fake_vector_features, 
            edge_index=edge_index, 
            fake_edge_attr=edge_attr, 
            fake_pos=pos_noisy
        )
        
        # 5. 📏 计算重构误差 (仅计算被挖掉界面的那一部分原子)
        pos_pred_interface = pos_pred[mask_nodes]
        pos_true_interface = pos_true[mask_nodes]
        
        # 计算 RMSD
        mse = F.mse_loss(pos_pred_interface, pos_true_interface, reduction='none').sum(dim=-1)
        rmsd = torch.sqrt(mse.mean()).item()
        
        return {
            'rmsd': rmsd,
            'num_interface_atoms': len(mask_nodes),
            'pos_pred': pos_pred,  # 可用于后续保存 PDB
            'pos_true': pos_true
        }, "Success"

# ================= 使用示例 =================
if __name__ == "__main__":
    import yaml
    
    with open('config_cmae.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    evaluator = BinderEvaluator(model_path="checkpoints/checkpoint_latest.pt", config=config)
    
    # 假设你有一个已经通过 AF3 或 HDOCK 对接好的复合物 PDB
    # Target 链是 A，Binder/环肽链是 B
    complex_pdb_path = "database/docked_binders/target_binder_1.pdb"
    
    print(f"📦 正在解析复合物: {complex_pdb_path}")
    complex_data = evaluator.parse_complex(
        pdb_path=complex_pdb_path, 
        target_chains=['A'], 
        binder_chains=['B', 'C'] # 如果你的环肽有多个片段
    )
    
    # 执行零样本物理打分
    # noise_scale 决定了你挖洞的残忍程度。1.0Å 意味着把坐标打乱 1Å
    print(f"⚖️ 开始进行物理流形重构打分...")
    results, msg = evaluator.evaluate_binder(complex_data, interface_cutoff=8.0, noise_scale=1.0)
    
    if results is None:
        print(f"❌ 评价失败: {msg}")
    else:
        rmsd = results['rmsd']
        num_atoms = results['num_interface_atoms']
        
        print("\n" + "="*50)
        print(f"🎯 Binder 评价报告")
        print("="*50)
        print(f"界面原子数量: {num_atoms} 个")
        print(f"重构 RMSD: {rmsd:.4f} Å")
        
        # 科学解释界定标准（根据你训练时的 L_rec 收敛情况来定）
        if rmsd < 1.0:
            print("🌟 结论: 完美契合！(模型认为它毫无违和感地镶嵌在口袋里)")
        elif rmsd < 2.5:
            print("✅ 结论: 合理结合。(可能需要微调侧链，但主干骨架符合物理规律)")
        else:
            print("⚠️ 结论: 存在严重排斥或极性不匹配！(模型试图把它推离当前位置)")
        print("="*50)