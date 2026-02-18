#!/usr/bin/env python
"""
收集CASF-2016数据集中的所有PDB ID
"""

import os
import json
from collections import defaultdict

def extract_pdb_id_from_filename(filename):
    """
    从文件名中提取PDB ID
    例如: 1a30_decoys.mol2 -> '1a30'
         1a30_rmsd.dat -> '1a30'
    """
    # 分割文件名，取前4个字符
    base = os.path.splitext(filename)[0]
    if '_' in base:
        pdb_id = base.split('_')[0]
    else:
        pdb_id = base
    
    # 检查是否是4字符的PDB ID
    if len(pdb_id) == 4 and pdb_id.isalnum():
        return pdb_id.lower()
    return None

def extract_pdb_ids_from_dat_file(file_path):
    """
    从 .dat 文件中提取PDB ID
    例如: CoreSet.dat, subset-*.dat
    """
    pdb_ids = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 第一列是PDB ID
                parts = line.split()
                if len(parts) >= 1:
                    pdb_id = parts[0].lower()
                    if len(pdb_id) == 4 and pdb_id.isalnum():
                        pdb_ids.add(pdb_id)
    except Exception as e:
        print(f"    警告: 无法读取文件 {file_path}: {e}")
    
    return pdb_ids

def collect_pdb_ids(casf_dir):
    """
    收集CASF-2016目录下的所有PDB ID
    """
    
    result = {
        "dataset_name": "CASF-2016",
        "description": "用于验证的基准测试集，训练时需排除这些PDB ID",
        "directories": {},
        "all_pdb_ids": [],
        "total_count": 0
    }
    
    # 定义需要扫描的子目录
    subdirs = ["coreset", "decoys_docking", "decoys_screening", 
                "power_docking", "power_ranking", "power_scoring", "power_screening"]
    
    all_pdb_set = set()
    
    for subdir in subdirs:
        subdir_path = os.path.join(casf_dir, subdir)
        
        if os.path.exists(subdir_path):
            pdb_ids = set()
            
            # 方式1: 获取该目录下的子目录名即为PDB ID
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isdir(item_path):
                    # 检查是否是4字符的PDB ID
                    if len(item) == 4 and item.isalnum():
                        pdb_id = item.lower()
                        pdb_ids.add(pdb_id)
                        all_pdb_set.add(pdb_id)
            
            # 方式2: 从文件名中提取PDB ID
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isfile(item_path):
                    pdb_id = extract_pdb_id_from_filename(item)
                    if pdb_id:
                        pdb_ids.add(pdb_id)
                        all_pdb_set.add(pdb_id)
            
            # 方式3: 从 .dat 文件中提取PDB ID
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isfile(item_path) and item.endswith('.dat'):
                    dat_pdb_ids = extract_pdb_ids_from_dat_file(item_path)
                    for pdb_id in dat_pdb_ids:
                        pdb_ids.add(pdb_id)
                        all_pdb_set.add(pdb_id)
            
            # 保存到结果
            pdb_ids_sorted = sorted(list(pdb_ids))
            result["directories"][subdir] = {
                "count": len(pdb_ids_sorted),
                "pdb_ids": pdb_ids_sorted
            }
            print(f"  {subdir}: {len(pdb_ids_sorted)} 个PDB ID")
    
    # 保存所有唯一的PDB ID
    result["all_pdb_ids"] = sorted(list(all_pdb_set))
    result["total_count"] = len(all_pdb_set)
    
    return result

def main():
    casf_dir = "/home/fit/liulei/WORK/SecretPPI/database/CASF-2016"
    output_file = "/home/fit/liulei/WORK/SecretPPI/database/CASF-2016_pdb_ids.json"
    
    print("正在收集CASF-2016的PDB ID...")
    print(f"目录: {casf_dir}")
    print()
    
    result = collect_pdb_ids(casf_dir)
    
    print()
    print(f"总计: {result['total_count']} 个唯一PDB ID")
    print()
    
    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"已保存到: {output_file}")
    
    # 打印统计摘要
    print()
    print("=== 统计摘要 ===")
    for dir_name, dir_info in result["directories"].items():
        print(f"{dir_name}: {dir_info['count']}")
    print(f"总计唯一: {result['total_count']}")

if __name__ == "__main__":
    main()
