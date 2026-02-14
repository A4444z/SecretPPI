
import torch
import math

class GaussianRBF(torch.nn.Module):
    """高斯径向基函数 (Gaussian Radial Basis Function)，用于将距离映射到高维特征空间。"""
    def __init__(self, n_rbf: int = 16, cutoff: float = 4.5, start: float = 0.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # RBF的中心点，在 [start, cutoff] 之间均匀分布
        self.centers = torch.linspace(start, cutoff, n_rbf)
        # RBF的宽度，控制高斯函数的平滑程度
        self.widths = torch.tensor([cutoff / n_rbf] * n_rbf)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算。
        参数:
            d: (E,) 距离张量，E 为边的数量
        返回:
            (E, n_rbf) RBF特征张量
        """
        d = d.unsqueeze(-1)  # 扩展维度为 [E, 1]，以便与 centers 进行广播计算
        centers = self.centers.to(d.device)
        widths = self.widths.to(d.device)
        # 计算高斯指数：exp(-(距离 - 中心)^2 / 宽度^2)
        return torch.exp(-(d - centers)**2 / widths**2)

def get_random_rotation_matrix() -> torch.Tensor:
    """生成一个随机的 3x3 旋转矩阵 (属于 SO(3) 群)。
    
    使用随机高斯矩阵的 QR 分解方法生成均匀分布的旋转矩阵。
    """
    # 生成随机正态分布矩阵
    x = torch.randn(3, 3)
    # 进行 QR 分解，q 是正交矩阵
    q, r = torch.linalg.qr(x)
    
    # 确保行列式为 +1 (保证是旋转而不是镜像反射)
    d = torch.diag(r)
    ph = d.sign()
    q *= ph
    
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
        
    return q

def apply_rotation(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵应用到原子坐标上。
    
    参数:
        pos: [N, 3] 原始坐标张量，N 为原子数量
        rot: [3, 3] 旋转矩阵
    返回:
        [N, 3] 旋转后的坐标张量
    """
    # 坐标旋转公式：x' = xR^T
    return pos @ rot.T
