#!/usr/bin/env python
"""
改进版的LMDB数据集构建脚本，支持断点续传
"""

import os
import sys
import argparse
import multiprocessing
import lmdb
import pickle
import time
from datetime import datetime
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.extract_interface import process_pdb_file

def load_processed_list(progress_file):
    """加载已处理的PDB列表"""
    processed = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            for line in f:
                pdb_id = line.strip()
                if pdb_id:
                    processed.add(pdb_id)
    return processed

def save_processed_list(progress_file, processed):
    """保存已处理的PDB列表"""
    with open(progress_file, 'w') as f:
        for pdb_id in sorted(processed):
            f.write(f"{pdb_id}\n")

def worker(args):
    """Worker function to process a single PDB file."""
    pdb_path, min_dist, contact_cutoff, intra_cutoff, max_depth = args
    try:
        # process_pdb_file returns List[Tuple[chainA, chainB, arrays, res_keys, merged_atoms]]
        entries = process_pdb_file(
            pdb_path,
            min_chain_distance=min_dist,
            contact_cutoff=contact_cutoff,
            intra_contact_cutoff=intra_cutoff,
            max_depth=max_depth
        )
        
        # Strip the last element (merged_atoms) which contains BioPython objects
        clean_entries = []
        for entry in entries:
            clean_entries.append(entry[:4])
            
        return pdb_path, clean_entries, None
    except Exception as e:
        return pdb_path, [], str(e)

def main():
    parser = argparse.ArgumentParser(description="Build LMDB dataset from 3DComplex PDBs with resume support")
    parser.add_argument("--input_dir", type=str, default="database/3DComplex", help="Input directory containing PDBs")
    parser.add_argument("--output_dir", type=str, default="database/3DComplex/processed_lmdb", help="Output LMDB directory")
    parser.add_argument("--progress_file", type=str, default="database/3DComplex/processed_progress.txt", help="File to track processed PDBs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes (default: 8)")
    parser.add_argument("--map_size", type=int, default=100 * 1024 * 1024 * 1024, help="LMDB map size (bytes)")
    
    # Extraction parameters
    parser.add_argument("--min_chain_distance", type=float, default=10.0)
    parser.add_argument("--contact_cutoff", type=float, default=4.5)
    parser.add_argument("--intra_contact_cutoff", type=float, default=4.5)
    parser.add_argument("--max_depth", type=int, default=2)
    
    args = parser.parse_args()
    
    # Collect all PDB files
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在扫描PDB文件...")
    pdb_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.pdb', '.ent')):
                pdb_files.append(os.path.join(root, f))
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 找到 {len(pdb_files)} 个PDB文件")
    if not pdb_files:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processed list
    processed = load_processed_list(args.progress_file)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已处理 {len(processed)} 个PDB文件")
    
    # Filter out already processed files
    unprocessed_files = []
    for pdb_path in pdb_files:
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        if pdb_id not in processed:
            unprocessed_files.append(pdb_path)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 剩余 {len(unprocessed_files)} 个PDB文件待处理")
    
    if not unprocessed_files:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有PDB文件已处理完毕！")
        return
    
    # Prepare arguments for workers
    worker_args = [
        (p, args.min_chain_distance, args.contact_cutoff, args.intra_contact_cutoff, args.max_depth) 
        for p in unprocessed_files
    ]
    
    # Open LMDB environment
    env = lmdb.open(args.output_dir, map_size=args.map_size)
    
    total_entries = 0
    failed_files = 0
    
    try:
        txn = env.begin(write=True)
        pdb_count = 0
        
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            iterator = pool.imap_unordered(worker, worker_args, chunksize=1)
            
            for pdb_path, entries, error in tqdm(iterator, total=len(unprocessed_files), desc="处理PDB文件"):
                pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
                
                if error:
                    print(f"\n警告: 处理 {pdb_id} 时出错: {error}")
                    failed_files += 1
                    processed.add(pdb_id)
                    save_processed_list(args.progress_file, processed)
                    continue
                
                if not entries:
                    processed.add(pdb_id)
                    save_processed_list(args.progress_file, processed)
                    continue
                
                for a, b, arrays, res_keys in entries:
                    key = f"{pdb_id}|{a}-{b}".encode("utf-8")
                    meta = {"pdb_id": pdb_id, "chains": (a, b)}
                    payload = {**arrays, "residue_keys": res_keys, "meta": meta}
                    txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                    total_entries += 1
                
                processed.add(pdb_id)
                pdb_count += 1
                
                # Commit every 100 PDBs and save progress
                if pdb_count % 100 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    save_processed_list(args.progress_file, processed)
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已处理 {pdb_count} 个PDB，共 {total_entries} 个界面")
        
        # Final commit
        txn.commit()
        save_processed_list(args.progress_file, processed)
        
    finally:
        env.close()
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理完成！")
    print(f"总条目数: {total_entries}")
    print(f"失败文件数: {failed_files}")
    print(f"已处理PDB数: {len(processed)}")

if __name__ == "__main__":
    main()