"""
使用基于接触扩展的策略提取蛋白质-蛋白质界面原子。

该脚本解析 PDB 文件，识别相互作用的链对（重原子最短距离 < 阈值），
通过迭代的链内接触扩展（而非简单的固定半径裁剪）构建界面区域，
并将全原子坐标（仅限重原子）、元素类型（原子序数）和残基索引写入 LMDB 数据库。

输出数据结构:
- pos: [N, 3] 原子坐标
- z: [N] 原子序数
- residue_index: [N] 残基索引

用法示例:
python -m src.data.extract_interface --input_dir data/raw_pdbs --output_dir data/processed_lmdb
"""

from __future__ import annotations

import os
import math
import argparse
import pickle
from typing import Dict, List, Tuple, Set, Optional

import lmdb
import numpy as np

try:
    from Bio.PDB import PDBParser, Structure, Chain, Residue, Atom, PDBIO, Select
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "需要安装 BioPython: pip install biopython"
    ) from e


# 常见元素的原子序数查找表，涵盖蛋白质和常见配体
_ELEMENT_Z: Dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18,
    "K": 19, "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
    "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57,
    "W": 74, "PT": 78, "AU": 79, "HG": 80, "PB": 82,
}


def infer_element(atom: Atom) -> str:
    """推断 Bio.PDB Atom 对象的元素符号。
    如果 Atom.element 为空，则通过解析原子名称来推断。
    """
    elem = atom.element
    if isinstance(elem, str) and elem:
        return elem.strip().upper()
    name = atom.get_name().strip()
    # 优先识别常见的双字母元素
    if len(name) >= 2 and name[:2].upper() in {"CL", "BR", "FE", "MG", "NA", "ZN", "MN", "HG", "CA"}:
        return name[:2].upper()
    # 备选：取首字母
    return name[0].upper()


def is_hydrogen(atom: Atom) -> bool:
    """判断是否为氢原子。"""
    elem = infer_element(atom)
    return elem == "H"


def is_water(residue: Residue) -> bool:
    """判断残基是否为水分。"""
    name = residue.get_resname().upper()
    if name in {"HOH", "WAT"}:
        return True
    hetflag = residue.id[0]
    return hetflag == "W"


def atom_is_heavy(atom: Atom) -> bool:
    """判断是否为重原子（非氢原子）。"""
    return not is_hydrogen(atom)


def residue_key(res: Residue, chain_id: str) -> Tuple[str, int, str, str]:
    """生成唯一的残基键值：(链ID, 残基序号, 插入码, 残基名称)。"""
    resseq: int = res.id[1]
    icode: str = res.id[2].strip() if isinstance(res.id[2], str) else ""
    return (chain_id, resseq, icode, res.get_resname().upper())


def collect_chain_atoms(structure: Structure.Structure) -> Dict[str, List[Tuple[Residue, Atom]]]:
    """按链收集重原子，排除水分子和氢原子。
    返回字典：chain_id -> [(Residue, Atom), ...]
    """
    chains: Dict[str, List[Tuple[Residue, Atom]]] = {}
    for model in structure:
        for chain in model:
            cid = chain.id
            lst: List[Tuple[Residue, Atom]] = []
            for res in chain:
                if is_water(res):
                    continue
                for atom in res.get_atoms():
                    if atom_is_heavy(atom):
                        lst.append((res, atom))
            if lst:
                chains[cid] = lst
        break  # 仅处理第一个 Model
    return chains


def min_interchain_distance(
    atoms_a: List[Tuple[Residue, Atom]], atoms_b: List[Tuple[Residue, Atom]]
) -> float:
    """计算两条链重原子之间的最短距离。"""
    if not atoms_a or not atoms_b:
        return math.inf
    posa = np.array([atom.get_coord() for _, atom in atoms_a], dtype=np.float32)  # [Na, 3]
    posb = np.array([atom.get_coord() for _, atom in atoms_b], dtype=np.float32)  # [Nb, 3]
    
    # 使用向量化方法计算成对最短距离
    aa = np.sum(posa**2, axis=1, keepdims=True)  # [Na, 1]
    bb = np.sum(posb**2, axis=1, keepdims=True)  # [Nb, 1]
    d2 = aa + bb.T - 2.0 * (posa @ posb.T)  # [Na, Nb]
    d2 = np.maximum(d2, 0.0)
    return float(np.sqrt(np.min(d2)))


def find_interacting_pairs(
    chains: Dict[str, List[Tuple[Residue, Atom]]], min_dist_threshold: float
) -> List[Tuple[str, str]]:
    """找出重原子最短距离小于阈值的链对。"""
    ids = list(chains.keys())
    pairs: List[Tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            d = min_interchain_distance(chains[a], chains[b])
            if d < min_dist_threshold:
                pairs.append((a, b))
    return pairs


def contacting_residues_between_chains(
    atoms_a: List[Tuple[Residue, Atom]],
    atoms_b: List[Tuple[Residue, Atom]],
    contact_cutoff: float,
) -> Tuple[Set[Tuple[str, int, str, str]], Set[Tuple[str, int, str, str]]]:
    """识别两条链之间发生接触的残基（原子间距离小于 contact_cutoff）。
    返回两条链各自的残基键集合。
    """
    posa = np.array([atom.get_coord() for _, atom in atoms_a], dtype=np.float32)
    posb = np.array([atom.get_coord() for _, atom in atoms_b], dtype=np.float32)
    if posa.size == 0 or posb.size == 0:
        return set(), set()
    aa = np.sum(posa**2, axis=1, keepdims=True)
    bb = np.sum(posb**2, axis=1, keepdims=True)
    d2 = aa + bb.T - 2.0 * (posa @ posb.T)
    d2 = np.maximum(d2, 0.0)
    mask = d2 <= (contact_cutoff**2)  # [Na, Nb]

    res_set_a: Set[Tuple[str, int, str, str]] = set()
    res_set_b: Set[Tuple[str, int, str, str]] = set()
    if np.any(mask):
        ia, ib = np.where(mask)
        for i in ia:
            res_a = atoms_a[i][0]
            res_set_a.add(residue_key(res_a, res_a.get_parent().id))
        for j in ib:
            res_b = atoms_b[j][0]
            res_set_b.add(residue_key(res_b, res_b.get_parent().id))
    return res_set_a, res_set_b


def build_intra_chain_residue_adjacency(
    atoms: List[Tuple[Residue, Atom]],
    intra_contact_cutoff: float,
) -> Dict[Tuple[str, int, str, str], Set[Tuple[str, int, str, str]]]:
    """构建单链内部残基的邻接关系（如果残基间重原子最短距离小于阈值则认为相邻）。"""
    # 按残基分组原子
    res_to_atoms: Dict[Tuple[str, int, str, str], List[Atom]] = {}
    for res, atom in atoms:
        key = residue_key(res, res.get_parent().id)
        res_to_atoms.setdefault(key, []).append(atom)

    keys = list(res_to_atoms.keys())
    adjacency: Dict[Tuple[str, int, str, str], Set[Tuple[str, int, str, str]]] = {k: set() for k in keys}
    # 预计算位置信息以提高效率
    pos_list: List[np.ndarray] = [
        np.array([a.get_coord() for a in res_to_atoms[k]], dtype=np.float32) for k in keys
    ]
    cutoff2 = intra_contact_cutoff**2
    for i in range(len(keys)):
        pi = pos_list[i]
        if pi.size == 0:
            continue
        aa = np.sum(pi**2, axis=1, keepdims=True)
        for j in range(i + 1, len(keys)):
            pj = pos_list[j]
            if pj.size == 0:
                continue
            bb = np.sum(pj**2, axis=1, keepdims=True)
            d2 = aa + bb.T - 2.0 * (pi @ pj.T)
            d2 = np.maximum(d2, 0.0)
            if np.any(d2 <= cutoff2):
                adjacency[keys[i]].add(keys[j])
                adjacency[keys[j]].add(keys[i])
    return adjacency


def contact_based_expansion(
    initial_residues: Set[Tuple[str, int, str, str]],
    adjacency: Dict[Tuple[str, int, str, str], Set[Tuple[str, int, str, str]]],
    max_depth: int = 2,
) -> Set[Tuple[str, int, str, str]]:
    """沿着链内邻接关系，从初始界面残基开始进行固定步数的迭代扩展。"""
    expanded: Set[Tuple[str, int, str, str]] = set(initial_residues)
    current_frontier: Set[Tuple[str, int, str, str]] = set(initial_residues)

    for _ in range(max_depth):
        next_frontier: Set[Tuple[str, int, str, str]] = set()
        for res in current_frontier:
            neighbors = adjacency.get(res, set())
            for nbr in neighbors:
                if nbr not in expanded:
                    expanded.add(nbr)
                    next_frontier.add(nbr)
        
        current_frontier = next_frontier
        if not current_frontier:
            break
            
    return expanded


def gather_atoms_for_residue_set(
    atoms: List[Tuple[Residue, Atom]],
    keep_residues: Set[Tuple[str, int, str, str]],
) -> List[Tuple[Residue, Atom]]:
    """从原子列表中过滤出属于指定残基集合的原子。"""
    out: List[Tuple[Residue, Atom]] = []
    for res, atom in atoms:
        key = residue_key(res, res.get_parent().id)
        if key in keep_residues:
            out.append((res, atom))
    return out


def atomic_number(elem: str) -> int:
    """将元素符号映射为原子序数。未知元素返回 0。"""
    if elem in _ELEMENT_Z:
        return _ELEMENT_Z[elem]
    e1 = elem.capitalize()
    if e1 in _ELEMENT_Z:
        return _ELEMENT_Z[e1]
    e2 = elem.upper()
    if e2 in _ELEMENT_Z:
        return _ELEMENT_Z[e2]
    return 0


def to_arrays(atoms: List[Tuple[Residue, Atom]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int, str, str]]]:
    """将 (Residue, Atom) 列表转换为 Numpy 数组格式。
    返回:
    - pos: [N, 3] 坐标
    - z: [N] 原子序数
    - residue_index: [N] 残基局部索引 (0..R-1)
    - residue_keys: 长度为 R 的残基键列表
    """
    res_keys: List[Tuple[str, int, str, str]] = []
    res_index_map: Dict[Tuple[str, int, str, str], int] = {}
    coords: List[np.ndarray] = []
    zs: List[int] = []
    ridxs: List[int] = []
    for res, atom in atoms:
        coord = atom.get_coord()
        if np.isnan(coord).any():
            continue
            
        elem = infer_element(atom)
        z = atomic_number(elem)
        if z == 0:
            continue
            
        key = residue_key(res, res.get_parent().id)
        if key not in res_index_map:
            res_index_map[key] = len(res_keys)
            res_keys.append(key)
        ridx = res_index_map[key]
        
        coords.append(coord)
        zs.append(z)
        ridxs.append(ridx)
        
    pos = np.array(coords, dtype=np.float32)
    z = np.array(zs, dtype=np.int64)
    residue_index = np.array(ridxs, dtype=np.int64)
    return pos, z, residue_index, res_keys


def process_pdb_file(
    pdb_path: str,
    min_chain_distance: float = 10.0,
    contact_cutoff: float = 4.5,
    intra_contact_cutoff: float = 4.5,
    max_depth: int = 1,
) -> List[
    Tuple[
        str,
        str,
        Dict[str, np.ndarray],
        List[Tuple[str, int, str, str]],
        List[Tuple[Residue, Atom]],
    ]
]:
    """处理单个 PDB 文件，返回每个发生相互作用的链对的界面数据。"""
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    except Exception as e:
        print(f"解析错误 {pdb_path}: {e}")
        return []

    chains = collect_chain_atoms(structure)
    pairs = find_interacting_pairs(chains, min_chain_distance)
    results: List[
        Tuple[
            str,
            str,
            Dict[str, np.ndarray],
            List[Tuple[str, int, str, str]],
            List[Tuple[Residue, Atom]],
        ]
    ] = []

    for a, b in pairs:
        atoms_a = chains[a]
        atoms_b = chains[b]
        # 初始界面残基
        res_a0, res_b0 = contacting_residues_between_chains(atoms_a, atoms_b, contact_cutoff)
        if not res_a0 or not res_b0:
            continue
        # 构建链内邻接关系并扩展
        adj_a = build_intra_chain_residue_adjacency(atoms_a, intra_contact_cutoff)
        adj_b = build_intra_chain_residue_adjacency(atoms_b, intra_contact_cutoff)
        keep_a = contact_based_expansion(res_a0, adj_a, max_depth=max_depth)
        keep_b = contact_based_expansion(res_b0, adj_b, max_depth=max_depth)
        # 收集选定残基的所有重原子
        kept_atoms_a = gather_atoms_for_residue_set(atoms_a, keep_a)
        kept_atoms_b = gather_atoms_for_residue_set(atoms_b, keep_b)
        merged_atoms = kept_atoms_a + kept_atoms_b
        if not merged_atoms:
            continue
        pos, z, residue_index, res_keys = to_arrays(merged_atoms)
        if len(pos) == 0:
            continue
        arrays = {"pos": pos, "z": z, "residue_index": residue_index}
        results.append((a, b, arrays, res_keys, merged_atoms))
    return results


def write_batch_to_lmdb(
    env: lmdb.Environment,
    entries: List[
        Tuple[
            str,
            str,
            Dict[str, np.ndarray],
            List[Tuple[str, int, str, str]],
            List[Tuple[Residue, Atom]],
        ]
    ],
    pdb_id: str,
) -> None:
    """将界面条目写入 LMDB 数据库环境。"""
    with env.begin(write=True) as txn:
        for a, b, arrays, res_keys, _ in entries:
            key = f"{pdb_id}|{a}-{b}".encode("utf-8")
            meta = {"pdb_id": pdb_id, "chains": (a, b)}
            payload = {**arrays, "residue_keys": res_keys, "meta": meta}
            txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


class InterfaceSelect(Select):
    """Bio.PDB 过滤器，用于在保存 PDB 文件时仅保留指定的界面残基重原子。"""
    def __init__(self, keep_residue_keys: Set[Tuple[str, int, str, str]]):
        super().__init__()
        self.keep_residue_keys = keep_residue_keys

    def accept_model(self, model) -> bool:
        return True

    def accept_chain(self, chain) -> bool:
        return True

    def accept_residue(self, residue: Residue) -> bool:
        if is_water(residue):
            return False
        chain_id = residue.get_parent().id
        key = residue_key(residue, chain_id)
        return key in self.keep_residue_keys

    def accept_atom(self, atom: Atom) -> bool:
        return atom_is_heavy(atom)


def write_interface_pdbs_from_path(
    pdb_path: str,
    entries: List[
        Tuple[
            str,
            str,
            Dict[str, np.ndarray],
            List[Tuple[str, int, str, str]],
            List[Tuple[Residue, Atom]],
        ]
    ],
    output_dir: str,
    pdb_id: str,
) -> None:
    """根据提取的界面信息，将界面部分的结构保存为新的 PDB 文件（用于验证）。"""
    os.makedirs(output_dir, exist_ok=True)
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return 
        
    io = PDBIO()
    io.set_structure(structure)
    
    for a, b, _, res_keys, _ in entries:
        keep_set = set(res_keys)
        out_path = os.path.join(output_dir, f"{pdb_id}_{a}-{b}.pdb")
        try:
            io.save(out_path, InterfaceSelect(keep_set))
        except Exception as e:
            print(f"保存 PDB 失败 {out_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于接触的界面提取（全重原子，排除水和氢）"
    )
    parser.add_argument("--input_dir", type=str, default="data/raw_pdbs", help="PDB 文件所在目录")
    parser.add_argument(
        "--output_dir", type=str, default="data/processed_lmdb", help="LMDB 输出目录"
    )
    parser.add_argument(
        "--min_chain_distance", type=float, default=10.0, help="链间最短距离阈值 (Å)"
    )
    parser.add_argument(
        "--contact_cutoff", type=float, default=4.5, help="链间接触距离阈值 (Å)"
    )
    parser.add_argument(
        "--intra_contact_cutoff", type=float, default=4.5, help="链内接触距离阈值 (Å)"
    )
    parser.add_argument(
        "--max_depth", type=int, default=2, help="接触扩展的最大步数 (Hops)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    pdb_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".pdb", ".ent"))
    ]
    if not pdb_files:
        print(f"在 {input_dir} 中未找到 PDB 文件")
        return

    os.makedirs(output_dir, exist_ok=True)
    # 打开 LMDB 环境，设置较大的 map_size (100GB)
    env = lmdb.open(output_dir, map_size=100 * 1024 * 1024 * 1024)

    try:
        for pdb_path in pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            try:
                entries = process_pdb_file(
                    pdb_path,
                    min_chain_distance=args.min_chain_distance,
                    contact_cutoff=args.contact_cutoff,
                    intra_contact_cutoff=args.intra_contact_cutoff,
                    max_depth=args.max_depth,
                )
                if entries:
                    write_batch_to_lmdb(env, entries, pdb_id)
                    # 同时保存 PDB 以供人工验证
                    write_interface_pdbs_from_path(pdb_path, entries, output_dir, pdb_id)
                    print(f"处理完成 {pdb_id}: 写入了 {len(entries)} 个链对")
                else:
                    print(f"处理完成 {pdb_id}: 无相互作用链对")
            except Exception as e:
                print(f"处理失败 {pdb_id}: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
