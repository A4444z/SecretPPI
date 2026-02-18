# GlueVAE：蛋白质-蛋白质相互作用生成模型

## 📢 最近更新 (2026-02-18)

### 1. CASF-2016 验证集支持
- **新增** `collect_casf_pdb_ids.py`: 收集CASF-2016数据集中的所有PDB ID
- **新增** `database/CASF-2016_pdb_ids.json`: 包含285个需排除的PDB ID
- **新增** `database/CASF-2016_USAGE_GUIDE.md`: CASF-2016使用说明
- **功能**: 在数据加载时自动过滤验证集PDB ID，确保训练集和验证集分离

### 2. 原子级VAE架构
- **改进**: 实现原子级VAE架构，保持每个原子的独特特征
- **解决**: 之前同一残基的所有原子特征完全相同的问题
- **新增**: `AtomLevelLatentEncoder` 和 `AtomLevelLatentDecoder`

### 3. 数据集加载优化
- **Dynamic Patch Sampling**: 对大界面进行随机补丁采样，避免显存溢出
- **Farthest Point Sampling (FPS)**: 系统性覆盖界面区域，确保全面性
- **CASF-2016过滤**: 在 `src/data/dataset.py` 中添加 `exclude_pdb_json` 参数

### 4. 工具脚本
- **新增** `scripts/build_lmdb_resumable.py`: 支持断点续传的LMDB数据集构建
- **新增** `test_casf_filter.py`: 测试CASF-2016过滤功能
- **新增** 多个配置文件: `config_solo.yaml`, `config_solo_test.yaml`, `config_solo_partial.yaml`

---

## 目录
- [什么是GlueVAE？](#什么是gluevae)
- [核心概念解释](#核心概念解释)
- [项目文件结构](#项目文件结构)
- [关键功能详解](#关键功能详解)
- [如何使用](#如何使用)
- [常见问题](#常见问题)

---

## 什么是GlueVAE？

### 通俗解释
想象一下，您有两个蛋白质，您想知道它们能不能像两块拼图一样拼在一起。GlueVAE就像是一个"智能拼图助手"，它通过学习大量已知的蛋白质复合物数据，来预测两个蛋白质是否可能结合在一起，甚至可以生成可能的结合方式。

### 技术定义
GlueVAE是一个基于**变分自编码器(VAE)**和**SE(3)等变神经网络(PaiNN)**的深度学习模型，专门用于：
- 蛋白质-蛋白质界面建模
- 分子胶诱导的复合物预测
- 蛋白质结构生成

---

## 核心概念解释

### 1. 什么是蛋白质？

#### 通俗理解
蛋白质就像是由氨基酸珠子串成的项链，然后这条项链会折叠成特定的3D形状。每个珠子（氨基酸）由多个原子组成。

#### 技术细节
- **残基(Residue)**：项链上的一个珠子（20种标准氨基酸）
- **原子(Atom)**：构成珠子的基本粒子（C、N、O、S等）
- **坐标(Coordinates)**：每个原子在3D空间中的位置(x, y, z)
- **链(Chain)**：一条完整的项链（一个蛋白质分子）

### 2. 什么是蛋白质-蛋白质界面？

#### 通俗理解
当两条蛋白质项链（链A和链B）靠近时，它们接触的地方就叫"界面"。就像两块积木拼在一起，接触的边缘就是界面。

#### 技术细节
- **界面原子**：距离对方链小于4.0Å（埃，长度单位，1Å=0.1纳米）的原子
- **相互作用**：界面处的氢键、疏水作用、盐桥等

### 3. 什么是SE(3)等变性？

#### 通俗理解
想象您有一个玩具，您把它旋转或移动，它的本质属性不会改变。SE(3)等变性就是说：
- 如果您旋转输入的蛋白质坐标
- 模型的输出也会相应地旋转
- 但模型学到的"知识"是一样的

#### 为什么重要？
蛋白质在空间中可以任意旋转和移动，但它们的相互作用本质不会改变。如果模型不具备SE(3)等变性，它可能会认为"旋转后的蛋白质"和"原始蛋白质"是完全不同的东西！

### 4. 什么是变分自编码器(VAE)？

#### 通俗理解
VAE就像是一个"压缩-解压"机器：
1. **编码器(Encoder)**：把复杂的蛋白质结构压缩成一个简洁的"摘要"（潜在向量）
2. **解码器(Decoder)**：从这个"摘要"重新生成蛋白质结构
3. **潜在空间(Latent Space)**：所有"摘要"组成的空间，类似一个"蛋白质目录"

#### 为什么用VAE？
- **生成新结构**：可以在潜在空间中采样，生成新的蛋白质界面
- **插值**：在两个已知结构之间平滑过渡，探索中间状态
- **降维**：把高维的原子坐标变成低维的向量，便于处理

### 5. 什么是PaiNN？

#### 通俗理解
PaiNN（Polarizable Atom Interaction Neural Network）是专门为处理3D分子数据设计的神经网络。它同时处理两种信息：
- **标量特征(Scalar Features)**：没有方向的信息，比如原子类型
- **向量特征(Vector Features)**：有方向的信息，比如原子间的相对位置

#### 为什么用PaiNN？
- 完美支持SE(3)等变性
- 能很好地捕捉分子的几何结构
- 专门为原子级精度设计

---

## 项目文件结构

```
SecretPPI/
├── src/
│   ├── data/
│   │   └── dataset.py          # 数据集加载（核心！）
│   ├── models/
│   │   ├── layers_solo.py      # PaiNN层实现
│   │   └── glue_vae_solo.py    # 主模型架构
│   └── utils/
│       ├── geometry.py          # 几何计算工具
│       ├── loss_solo.py         # 损失函数
│       └── constants_solo.py    # 常量定义
├── scripts/
│   └── build_lmdb_resumable.py # 数据集构建（支持断点续传）
├── train_solo.py                # 训练脚本
├── config_solo.yaml             # 配置文件
└── README.md                    # 就是这个文件！
```

---

## 关键功能详解

### 1. 数据集加载（dataset.py）

这是项目最复杂也最重要的文件！让我详细解释它的每个功能。

#### 功能1：从LMDB读取数据

**什么是LMDB？**
- LMDB是一个高效的键值存储数据库
- 类似于一个超大的字典，存了很多蛋白质界面数据
- 好处：读取速度快，支持多进程

**数据格式：**
每个条目包含：
- `pos`: 所有原子的3D坐标 [N, 3]
- `z`: 所有原子的类型 [N]
- `residue_index`: 每个原子属于哪个残基 [N]
- `residue_keys`: 残基的标识信息
- `meta`: PDB ID和链对等元数据

#### 功能2：Dynamic Patch Sampling（动态补丁采样）

**为什么需要这个？**
- 有些蛋白质界面非常大，可能有几万个原子
- 直接处理会导致显存溢出（GPU内存不够）
- 就像看一张大照片，一次只看一个局部区域

**它是怎么工作的？**

```
第一步：判断是否需要采样
    if 原子数 > max_atoms (默认1000):
        进行补丁采样
    else:
        直接使用全部原子

第二步：选择补丁中心
    1. 找到所有界面原子（距离对方链<4.0Å）
    2. 使用最远点采样(FPS)选择几个代表性的中心
    3. 每次调用使用不同的中心，确保覆盖整个界面

第三步：裁剪补丁
    1. 计算所有原子到中心的距离
    2. 只保留距离 < 15.0Å 的原子
    3. 确保至少保留100个原子（如果不够，放宽阈值）

第四步：更新所有数组
    - pos: 裁剪后的坐标
    - z: 裁剪后的原子类型
    - residue_index: 裁剪后的残基索引
    - is_ligand: 裁剪后的受体/配体标记
    - mask_interface: 裁剪后的界面标记
```

**最远点采样(FPS)是什么？**

想象您要在一个城市里选5个代表性地点：
1. 随机选第一个点（比如市中心）
2. 选距离第一个点最远的点（比如东郊）
3. 选距离前两个点都最远的点（比如西郊）
4. 继续直到选够5个点

这样选出来的点能最好地覆盖整个区域！

#### 功能3：图结构构建

**什么是图(Graph)？**
- **节点(Node)**：每个原子就是一个节点
- **边(Edge)**：两个原子之间的连接就是一条边

**如何构建边？**

```
第一步：初始边选择
    使用 radius_graph(r=4.5Å)
    意思是：距离小于4.5Å的原子之间连一条边

第二步：限制邻居数量
    每个原子最多有32个邻居
    如果超过32个，只保留距离最近的32个
    为什么？防止边数爆炸，节省显存

第三步：边类型分类
    Type 0: 共价键（距离 < 1.7Å）
    Type 1: 链内非共价（同一条链，距离 ≥ 1.7Å）
    Type 2: 链间相互作用（不同链，这是重点！）

第四步：边特征计算
    - 类型的One-hot编码 [3维]
    - 距离的RBF编码 [16维]
    - 拼接起来：总共19维特征
```

**什么是RBF编码？**

RBF（径向基函数）编码是把距离转换成高维向量的方法：
- 想象有16个"探测器"，每个对特定距离最敏感
- 比如第1个探测器对1.0Å最敏感，第2个对1.5Å最敏感...
- 这样距离信息就变成了16维的向量
- 好处：神经网络更容易处理

#### 功能4：向量特征计算

**为什么需要向量特征？**
- 蛋白质是3D结构，方向很重要
- 只使用标量信息（比如原子类型）不够
- 需要给模型一些方向感知

**如何计算？**

```
第一步：找到所有共价键
    边类型为Type 0的边

第二步：计算相对向量
    对于每条共价键 (i → j)
    向量 = pos[j] - pos[i]
    意思是：从j指向i的向量

第三步：累加向量
    对于每个原子i
    vector_features[i] = 所有以i为目标的共价键向量之和

第四步：处理孤立原子
    如果某个原子没有共价邻居
    给它添加一个微小的随机向量
    为什么？避免零向量导致梯度消失
```

**什么是梯度消失？**
想象您在爬山，如果路是平的（梯度为0），您就不知道往哪走了。神经网络也一样，如果某个层的输出全是0，后续层就"学不到东西"了。

#### 功能5：数据增强（随机旋转）

**什么是数据增强？**
就像您学习时，把书本旋转不同角度来看，能加深理解。数据增强就是给模型看"不同角度"的数据，让它学得更鲁棒。

**为什么在Patch采样之后？**
- 先采样Patch：计算量小（只处理Patch，不是整个蛋白）
- 再旋转Patch：保持SE(3)等变性
- 顺序很重要！

---

### 2. 模型架构（glue_vae_solo.py）

#### 整体架构

```
输入：蛋白质界面（原子坐标、类型等）
    ↓
[编码器] PaiNNEncoder
    ↓
[Pooling] 原子特征 → 残基特征
    ↓
[潜在空间] LatentEncoder → 采样z
    ↓
[解码] LatentDecoder → 残基特征
    ↓
[Unpooling] 残基特征 → 原子特征
    ↓
[解码器] ConditionalPaiNNDecoder
    ↓
输出：重构的蛋白质界面
```

#### 编码器（PaiNNEncoder）

**功能**：提取蛋白质界面的特征

**组成**：
- 嵌入层：把原子类型转换成向量
- 多层PaiNN：每一层都更新标量和向量特征
- 梯度检查点：节省显存（可选）

#### 原子→残基Pooling

**为什么需要？**
- 原子级别太细，计算量大
- 残基级别更粗，但包含足够的化学信息
- 就像从"逐像素"变成"逐块"处理

**怎么做？**
- 属于同一个残基的原子特征取平均
- 使用 `scatter_mean` 操作

#### 潜在空间（Latent Space）

**编码器**：把残基特征变成均值(mu)和对数方差(logvar)
**采样**：从高斯分布中采样 z = mu + exp(logvar/2) * ε
**为什么？** 这是VAE的核心，让潜在空间连续且有意义

#### 残基→原子Unpooling

**怎么做？**
- 每个原子直接使用它所属残基的特征
- 简单但有效

#### 解码器（ConditionalPaiNNDecoder）

**功能**：从潜在特征重构蛋白质界面
**条件生成**：可以结合受体原子的信息，生成配体
**输出**：预测的原子坐标

---

### 3. 损失函数（loss_solo.py）

#### D-RMSD Loss

**什么是RMSD？**
- RMSD（Root Mean Square Deviation）：均方根偏差
- 衡量两个结构之间的差异
- 越小越好，0表示完全相同

**什么是D-RMSD？**
- D-RMSD（Distance RMSD）：基于距离矩阵的RMSD
- 先计算所有原子对的距离，然后比较距离矩阵
- **关键优势**：SE(3)不变！不需要先对齐结构

**为什么用D-RMSD？**
- 普通RMSD需要先对齐结构（计算量大）
- D-RMSD直接比较距离，天然SE(3)不变
- 更适合训练等变模型

#### KL散度（KLLoss）

**什么是KL散度？**
- 衡量两个概率分布之间的差异
- 在VAE中，让潜在分布接近标准正态分布

**β-VAE**
- 总损失 = 重构损失 + β * KL散度
- β控制KL散度的权重
- 使用warmup策略：β从0逐渐增加到目标值

---

## 如何使用

### 1. 环境设置

```bash
# 创建conda环境
conda create -n secret python=3.10
conda activate secret

# 安装PyTorch（GPU版）
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4 -c conda-forge

# 安装PyTorch Geometric
conda install -y -c pyg -c pytorch -c nvidia pytorch-geometric

# 安装其他依赖
conda install -y biotite mdanalysis rdkit
pip install lmdb tqdm pyyaml
```

### 2. 构建数据集

```bash
# 使用支持断点续传的脚本
python scripts/build_lmdb_resumable.py \
    --pdb_dir database/3DComplex/raw \
    --lmdb_path database/3DComplex/processed_lmdb \
    --num_workers 8 \
    --progress_file processed_progress.txt
```

**参数说明**：
- `--pdb_dir`: PDB文件目录
- `--lmdb_path`: 输出LMDB路径
- `--num_workers`: 使用的CPU核数
- `--progress_file`: 进度文件（用于断点续传）

### 3. 训练模型

```bash
# 使用测试数据集快速验证
python train_solo.py --config config_solo_test.yaml

# 使用完整数据集训练
python train_solo.py --config config_solo.yaml
```

**配置文件说明（config_solo.yaml）**：

```yaml
data:
  root_dir: "database/3DComplex"
  lmdb_path: "database/3DComplex/processed_lmdb"
  batch_size: 4              # 每批样本数
  num_workers: 4             # 数据加载进程数
  train_split: 0.9           # 训练集比例

model:
  hidden_dim: 128            # 隐藏层维度
  latent_dim: 32             # 潜在空间维度
  num_encoder_layers: 6      # 编码器层数
  num_decoder_layers: 4      # 解码器层数

training:
  num_epochs: 100            # 训练轮数
  learning_rate: 1.0e-4      # 学习率
  beta:
    start: 0.0               # KL散度初始权重
    end: 0.1                 # KL散度最终权重
    warmup_steps: 10000      # warmup步数
```

### 4. 查看训练结果

```bash
# 检查点保存在
ls -la checkpoints/

# TensorBoard查看（如果使用）
tensorboard --logdir logs/
```

---

## 常见问题

### Q1: 显存溢出怎么办？

**A:** 有几个方法：

1. **减小batch_size**: 在config里把batch_size改小（比如从4改到1）
2. **减小hidden_dim**: 把hidden_dim从128改到64
3. **减小max_atoms**: 让Patch Sampling更激进（比如从1000改到500）
4. **启用梯度检查点**: 在模型初始化时设置`use_gradient_checkpointing=True`

### Q2: 训练时Loss不下降怎么办？

**A:** 检查以下几点：

1. **学习率**: 可能太大或太小，试试1e-5或1e-3
2. **beta warmup**: 确保beta不是一开始就很大
3. **数据质量**: 检查数据集是否正确构建
4. **模型初始化**: 确保参数初始化合理

### Q3: Patch Sampling会丢失信息吗？

**A:** 会有一些信息丢失，但：
1. 我们只在大界面时才采样
2. 使用FPS确保覆盖整个界面
3. 多次训练会使用不同的Patch，综合起来能学到完整信息
4. 这是"精度-显存"的权衡

### Q4: 什么是SE(3)等变性？为什么重要？

**A:** 简单说：
- 旋转输入 → 输出也旋转
- 移动输入 → 输出也移动
- 但模型的"理解"不变

重要性：
- 蛋白质在空间中可以任意朝向
- 模型不应该因为朝向不同而给出不同预测
- 等变模型能更好地泛化

### Q5: 我没有GPU能训练吗？

**A:** 理论上可以，但：
- 会非常慢（可能需要几周甚至几个月）
- 不建议
- 最好使用有GPU的服务器

---

## 下一步

- 阅读代码中的注释，理解每个细节
- 尝试用测试数据集训练
- 调整超参数，观察效果
- 阅读相关论文（PaiNN、VAE等）

---

## 参考文献

- PaiNN: https://arxiv.org/abs/2102.03150
- VAE: https://arxiv.org/abs/1312.6114
- β-VAE: https://openreview.net/forum?id=Sy2fzU9gl

---

**祝您使用愉快！如有问题，欢迎提问！** 🚀
