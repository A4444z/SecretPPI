# Changelog

## v2.0 (2026-04-24) — CMAE Cleanup & Bug Fixes

### Critical Bug Fixes

#### Data Split Was Broken (was leaking train into val/test)
`src/data/dataset.py` never actually partitioned data by split. Every split loaded all LMDB keys identically — the `split` argument only changed the cache filename. Fixed by deterministic MD5 hash-based partitioning:

```python
h = int(hashlib.md5(pdb_id.encode()).hexdigest(), 16) % 10000
# maps each PDB ID to train / val / test reproducibly
```

Old key caches (`keys_cache_*.pkl`) on HPC were deleted. The first run will rebuild them with correct splits. **Validation metrics will shift** since val now contains genuinely held-out complexes.

#### `eval_retrieval.py` crashed on startup
Model returns 6 values; the unpack expected 4. Fixed: `graph_z1, graph_z2, _, _, _, _ = model(...)`.

#### Device mismatch in `glue_cmae.py`
Two `torch.randint()` calls created tensors on CPU while `pos` was on GPU, causing a crash during masking. Fixed by adding `device=pos.device`.

#### Duplicate `__main__` in `extract_interface.py`
`main()` was called twice at the end of the file. Removed the duplicate.

#### `inference_screening.py` called undefined `main()`
The `__main__` block called `main()` which does not exist in this module. Replaced with an informative print directing users to `run_screening.py`.

#### Hardcoded WANDB API key in `train_cmae_submt.sh`
Removed. Now reads from the environment:
```bash
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY environment variable is not set}"
```
**Set this before submitting**: `export WANDB_API_KEY="your_key"` then `sbatch train_cmae_submt.sh`.

---

### Improvements

#### Mixed Precision Training (AMP)
`train_cmae.py` now supports `torch.cuda.amp.GradScaler`. Enabled by default via `config_cmae.yaml`:
```yaml
training:
  use_amp: true
```
Expected ~30–40% memory reduction and ~20% speedup. Scaler state is saved/restored in checkpoints for safe resume. If NaN loss occurs, set `use_amp: false`.

#### DDP `find_unused_parameters=False`
Reduced DDP overhead. All parameters in `GlueVAE` are used in every forward pass so this is safe.

#### Gradient probe now off by default
The per-batch gradient norm logging was always printing to stdout (every 50 steps). Now gated:
```yaml
logging:
  debug_gradients: true   # add this line to re-enable
```

#### Debug/health-check prints removed from `layers_solo.py`
Removed the two input-validation print blocks from `PaiNNEncoder.forward()`. The NaN/Inf safety check (raises `RuntimeError`) is kept.

#### Memory leak cap in dataset
`_sample_states` dict now evicts the oldest 50% of entries once it exceeds 10,000 keys, preventing unbounded memory growth in long training runs.

---

### Repository Cleanup (23 files deleted)

| Category | Deleted files |
|----------|--------------|
| Old VAE pipeline | `glue_vae_solo.py`, `glue_vae_atom_level.py`, `train_solo.py`, `train_solo_ddp.py`, `config_solo*.yaml`, `loss_solo.py` |
| Old scatter_mean CMAE | `glue_cmae_scatter_mean.py`, `eval_cmae_scatter_mean.py` |
| Empty placeholders | `glue_vae.py`, `layers.py`, `constants.py`, `constants_solo.py`, `parsing.py`, `train.py` |
| SLURM (old) | `train_solo_full_submit.sh`, `train_solo_ddp_submit.sh`, `train_solo_overfit_submit.sh` |
| One-time utilities | `build_lmdb.py`, `collect_casf_pdb_ids.py`, `test_casf_filter.py`, `readmeAI.md` |

Active files are unchanged. The only SLURM script is now `train_cmae_submt.sh`.

---

### Files Changed

| File | What changed |
|------|-------------|
| `src/data/dataset.py` | Hash-based split; memory leak cap; `split_ratio` param |
| `src/models/glue_cmae.py` | Removed unused import; fixed device mismatch (×2) |
| `src/models/layers_solo.py` | Removed debug prints |
| `src/data/extract_interface.py` | Removed duplicate `__main__` |
| `scripts/inference_screening.py` | Replaced undefined `main()` call |
| `train_cmae.py` | AMP support; `find_unused_parameters=False`; gradient probe gating; `split_ratio` forwarded to dataset |
| `eval_retrieval.py` | Fixed return-value unpack; `split_ratio` param |
| `eval_cmae_attn.py` | Fixed misleading description strings; `split_ratio` param |
| `train_cmae_submt.sh` | Removed hardcoded key; fixed config name in echo |
| `config_cmae.yaml` | Added `use_amp: true` |
| `.gitignore` | Added `checkpoints/`, `*.pt`, `screening_results/`, `processed_progress.txt` |
| `README.md` | Complete bilingual rewrite describing CMAE architecture |
