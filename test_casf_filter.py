#!/usr/bin/env python
"""
测试CASF-2016 PDB ID过滤功能
"""

import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.dataset import GlueVAEDataset

def test_casf_filtering():
    """测试CASF-2016过滤功能"""
    
    print("=" * 60)
    print("测试CASF-2016 PDB ID过滤功能")
    print("=" * 60)
    print()
    
    # 测试1: 使用测试数据集，不使用过滤
    print("测试1: 不使用过滤")
    print("-" * 40)
    try:
        dataset_no_filter = GlueVAEDataset(
            root="test",
            lmdb_path="test/test_lmdb",
            split='train',
            exclude_pdb_json=None
        )
        print(f"数据集大小: {len(dataset_no_filter)}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    # 测试2: 使用测试数据集，使用CASF过滤
    print("测试2: 使用CASF-2016过滤")
    print("-" * 40)
    try:
        dataset_with_filter = GlueVAEDataset(
            root="test",
            lmdb_path="test/test_lmdb",
            split='train',
            exclude_pdb_json="database/CASF-2016_pdb_ids.json"
        )
        print(f"数据集大小: {len(dataset_with_filter)}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    # 测试3: 直接测试PDB ID对比
    print("测试3: PDB ID对比测试")
    print("-" * 40)
    test_ids = ["1a30", "1A30", "1bcu", "1BCU", "9zzz", "NOT_A_PDB"]
    print("测试ID列表:", test_ids)
    
    try:
        # 手动加载排除列表
        import json
        with open("database/CASF-2016_pdb_ids.json", 'r') as f:
            casf_data = json.load(f)
        exclude_set = set(pdb_id.lower() for pdb_id in casf_data['all_pdb_ids'])
        
        print(f"排除列表大小: {len(exclude_set)}")
        print()
        print("ID对比结果:")
        for pdb_id in test_ids:
            is_excluded = pdb_id.lower() in exclude_set
            status = "❌ 排除" if is_excluded else "✅ 保留"
            print(f"  {pdb_id:10s} -> {status}")
    except Exception as e:
        print(f"错误: {e}")
    print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_casf_filtering()
