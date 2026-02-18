
"""
常量定义文件。
包含元素词汇表、原子半径、标准氨基酸等物理和化学常数。
"""



# 元素原子序数映射（与extract_interface.py保持一致）
ELEMENT_Z = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18,
    "K": 19, "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
    "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57,
    "W": 74, "PT": 78, "AU": 79, "HG": 80, "PB": 82,
}

# 元素符号到索引的映射（用于Embedding）
# 取前100号元素作为词汇表
MAX_ATOMIC_NUM = 100
ELEMENT_VOCAB_SIZE = MAX_ATOMIC_NUM + 1

# 原子半径（单位：埃）- 用于某些几何计算
ATOMIC_RADIUS = {
    1: 0.32,   # H
    6: 0.77,   # C
    7: 0.75,   # N
    8: 0.73,   # O
    9: 0.71,   # F
    15: 1.06,  # P
    16: 1.02,  # S
    17: 0.99,  # Cl
    19: 1.96,  # K
    20: 1.74,  # Ca
    26: 1.26,  # Fe
    30: 1.31,  # Zn
}

# 标准20种氨基酸
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

# 常见非标准残基（保留，不忽略）
COMMON_NON_STANDARD_RESIDUES = {
    "MSE",  # 硒代甲硫氨酸
    "PTR",  # 磷酸酪氨酸
    "SEP",  # 磷酸丝氨酸
    "TPO",  # 磷酸苏氨酸
    "CSO",  # S-氧化半胱氨酸
    "CSD",  # 3-磺基丙氨酸
    "CME",  # S,S-(2-羟乙基)硫代半胱氨酸
    "MLY",  # N,N-二甲基赖氨酸
    "MLZ",  # N,N,N-三甲基赖氨酸
    "HYP",  # 羟脯氨酸
}

# 模型默认超参数
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_CUTOFF = 4.5
DEFAULT_RBF_DIM = 16

