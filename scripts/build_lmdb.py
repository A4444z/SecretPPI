
import os
import sys
import argparse
import multiprocessing
import lmdb
import pickle
from tqdm import tqdm
import numpy as np

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.extract_interface import process_pdb_file

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
        # to reduce pickling overhead and avoid potential issues
        clean_entries = []
        for entry in entries:
            # entry: (chainA, chainB, arrays, res_keys, merged_atoms)
            # We keep the first 4 elements
            clean_entries.append(entry[:4])
            
        return pdb_path, clean_entries, None
    except Exception as e:
        return pdb_path, [], str(e)

def main():
    parser = argparse.ArgumentParser(description="Build LMDB dataset from 3DComplex PDBs")
    parser.add_argument("--input_dir", type=str, default="database/3DComplex", help="Input directory containing PDBs")
    parser.add_argument("--output_dir", type=str, default="data/processed_lmdb", help="Output LMDB directory")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes")
    parser.add_argument("--map_size", type=int, default=100 * 1024 * 1024 * 1024, help="LMDB map size (bytes)")
    
    # Extraction parameters
    parser.add_argument("--min_chain_distance", type=float, default=10.0)
    parser.add_argument("--contact_cutoff", type=float, default=4.5)
    parser.add_argument("--intra_contact_cutoff", type=float, default=4.5)
    parser.add_argument("--max_depth", type=int, default=2)
    
    args = parser.parse_args()
    
    # Collect all PDB files
    pdb_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.pdb', '.ent')):
                pdb_files.append(os.path.join(root, f))
    
    print(f"Found {len(pdb_files)} PDB files in {args.input_dir}")
    if not pdb_files:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare arguments for workers
    worker_args = [
        (p, args.min_chain_distance, args.contact_cutoff, args.intra_contact_cutoff, args.max_depth) 
        for p in pdb_files
    ]
    
    # Open LMDB environment
    env = lmdb.open(args.output_dir, map_size=args.map_size)
    
    total_entries = 0
    failed_files = 0
    
    try:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Use imap_unordered for better responsiveness
            iterator = pool.imap_unordered(worker, worker_args, chunksize=1)
            
            with env.begin(write=True) as txn:
                # Commit every N entries to avoid transaction growing too large? 
                # Or just one huge transaction? LMDB handles large transactions fine usually, 
                # but for safety and progress saving, maybe commit periodically.
                # However, with multiprocessing, the main loop is just writing.
                
                # Let's commit every 1000 PDBs or so.
                pdb_count = 0
                
                for pdb_path, entries, error in tqdm(iterator, total=len(pdb_files), desc="Processing PDBs"):
                    if error:
                        # print(f"Error processing {os.path.basename(pdb_path)}: {error}")
                        failed_files += 1
                        continue
                    
                    if not entries:
                        continue
                        
                    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
                    
                    for a, b, arrays, res_keys in entries:
                        key = f"{pdb_id}|{a}-{b}".encode("utf-8")
                        meta = {"pdb_id": pdb_id, "chains": (a, b)}
                        payload = {**arrays, "residue_keys": res_keys, "meta": meta}
                        txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                        total_entries += 1
                    
                    pdb_count += 1
                    if pdb_count % 1000 == 0:
                        # Commit and restart transaction
                        txn.commit()
                        txn = env.begin(write=True)
                
                # Final commit (context manager would commit if no exception, but we are inside it)
                # Actually env.begin() context manager commits on exit if write=True and no exception.
                # But since we manually committed inside loop, we need to be careful.
                # If we reassigned txn, we need to ensure the last one is committed.
                # The context manager `with env.begin() as txn` handles the txn created at start.
                # If we overwrite `txn`, the context manager might get confused or try to commit a closed txn?
                # The standard lmdb pattern with explicit commits is usually:
                # txn = env.begin(write=True)
                # ...
                # txn.commit()
                # txn = env.begin(write=True)
                # ...
                # txn.commit()
                
                # So the `with` block is tricky if we want intermediate commits.
                # Let's simplify: just one big transaction if memory allows, or manual management.
                # Given 100GB map size and potentially thousands of files, one transaction is risky for RAM?
                # No, LMDB writes to disk. But the write buffers might grow.
                # Let's use manual management.
                pass
            
            # Since I used `with env.begin(write=True) as txn`, it will try to commit at the end.
            # If I replaced `txn` inside, the original `txn` (from `with`) is gone? 
            # No, `txn` is a variable. The `__exit__` method is called on the object returned by `env.begin()`.
            # If I do `txn = env.begin()`, I lose reference to the context manager's object?
            # Actually `env.begin()` returns a Transaction object.
            # Let's DO NOT use `with` for the transaction if we want to commit periodically.
            
    except Exception as e:
        print(f"Main loop error: {e}")
        # If we used `with`, it might try to rollback/commit.
    
    finally:
        env.close()

    # Re-implementing the loop with manual transaction management for safety
    # The above `with` block is potentially buggy if I reassign `txn`.
    # Let's rewrite the main execution block logic below cleanly.

def safe_main_loop(args, pdb_files, worker_args):
    env = lmdb.open(args.output_dir, map_size=args.map_size)
    total_entries = 0
    failed_files = 0
    
    try:
        txn = env.begin(write=True)
        pdb_count = 0
        
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            iterator = pool.imap_unordered(worker, worker_args, chunksize=1)
            
            for pdb_path, entries, error in tqdm(iterator, total=len(pdb_files), desc="Processing PDBs"):
                if error:
                    failed_files += 1
                    continue
                
                if not entries:
                    continue
                    
                pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
                
                for a, b, arrays, res_keys in entries:
                    key = f"{pdb_id}|{a}-{b}".encode("utf-8")
                    meta = {"pdb_id": pdb_id, "chains": (a, b)}
                    payload = {**arrays, "residue_keys": res_keys, "meta": meta}
                    txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                    total_entries += 1
                
                pdb_count += 1
                if pdb_count % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        
        # Final commit
        txn.commit()
        
    finally:
        env.close()
        
    print(f"Done. Total entries: {total_entries}. Failed files: {failed_files}")

if __name__ == "__main__":
    # We need to parse args first to pass to safe_main_loop
    # But I put the logic in main(). Let's correct the structure.
    
    parser = argparse.ArgumentParser(description="Build LMDB dataset from 3DComplex PDBs")
    parser.add_argument("--input_dir", type=str, default="database/3DComplex", help="Input directory containing PDBs")
    parser.add_argument("--output_dir", type=str, default="data/processed_lmdb", help="Output LMDB directory")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes")
    parser.add_argument("--map_size", type=int, default=100 * 1024 * 1024 * 1024, help="LMDB map size (bytes)")
    parser.add_argument("--min_chain_distance", type=float, default=10.0)
    parser.add_argument("--contact_cutoff", type=float, default=4.5)
    parser.add_argument("--intra_contact_cutoff", type=float, default=4.5)
    parser.add_argument("--max_depth", type=int, default=2)
    
    args = parser.parse_args()
    
    pdb_files = []
    if os.path.exists(args.input_dir):
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.lower().endswith(('.pdb', '.ent')):
                    pdb_files.append(os.path.join(root, f))
    
    print(f"Found {len(pdb_files)} PDB files in {args.input_dir}")
    if not pdb_files:
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    
    worker_args = [
        (p, args.min_chain_distance, args.contact_cutoff, args.intra_contact_cutoff, args.max_depth) 
        for p in pdb_files
    ]
    
    safe_main_loop(args, pdb_files, worker_args)
