# Copilot / AI Agent Instructions — GlueVAE

Short, actionable guidance for code-generation agents working on GlueVAE.

- **Big picture:** GlueVAE is a generative VAE for protein–protein interfaces. It operates on ALL heavy atoms (no coarse-graining) and enforces SE(3)-equivariance / chirality-aware features.

- **Key files (read before editing):**
  - [src/data/extract_interface.py](src/data/extract_interface.py#L1-L120) — CLI to extract all-atom interfaces into LMDB. Example run below.
  - [src/data/dataset.py](src/data/dataset.py) — Graph construction and feature engineering (radius graph r<4.5Å; covalent threshold d<1.6Å; scalar `s` and vector `v` chirality features).
  - [src/models/glue_vae.py](src/models/glue_vae.py) — Encoder/Decoder and residue-level latent design (scatter_mean pooling).
  - [src/models/layers.py](src/models/layers.py) — Low-level equivariant building blocks.
  - [src/utils/parsing.py](src/utils/parsing.py), [src/utils/geometry.py](src/utils/geometry.py), [src/utils/constants.py](src/utils/constants.py) — utilities and constants used across data & model code.

- **Don't change without asking:**
  - All-atom assumption: do NOT switch to residue-only or coarse-grained representations.
  - SE(3)-equivariance and chirality injection: changes that remove vector features (`v`) or the covalent-neighbor sum must be discussed.

- **Data & LMDB format (extract_interface.py):**
  - LMDB entries store `pos` (float32 [N,3]), `z` (int64 element numbers), `residue_index` (int64 per-atom residue id), and `residue_keys`/`meta` in the pickle payload.
  - Extraction CLI (recommended):

```bash
python -m src.data.extract_interface --input_dir data/raw_pdbs --output_dir data/processed_lmdb
```

  - Defaults and conventions: inter-chain min distance 10.0Å; contact cutoff 4.5Å; intra-contact 4.5Å; expansion hops `max_depth=2`.

- **Graph & model conventions:**
  - Spatial edges: radius graph r < 4.5Å; covalent edges: d < 1.6Å and carry distinct `edge_attr`.
  - Node features: `s` = embedding of atomic number; `v` = sum of relative vectors to covalent neighbors (see `dataset.py` implementation).
  - Pooling: aggregate atom → residue with `scatter_mean` to form residue-level latents; decoder broadcasts/unpools back to atoms.

- **Dependencies & environment:**
  - Python 3.9+; PyTorch >=2.1; torch_geometric; biopython; lmdb; numpy; wandb (optional).

- **Developer conventions:**
  - Use type hints on functions and methods; document tensor shapes in comments (e.g. `# [B, N, 3]`).
  - Keep model layers, dataset, and training logic decoupled. Follow existing file patterns in `src/models` and `src/data`.

- **Typical quick debugging workflow:**
  1. Place a small set of PDBs in `test/input_pdbs` or `data/raw_pdbs`.
 2. Run the extraction CLI (above) with `--output_dir test/interface` to write LMDB and per-interface PDBs for inspection.
 3. Inspect saved interface PDBs in the output directory to verify contact-based expansion.

- **When to ask for human confirmation:**
  - Any change that removes the all-atom pipeline, reduces element scope, or alters chirality calculation.
  - Changes to the latent representation (atom vs residue level) or loss that would break SE(3)-invariance (e.g., replacing D-RMSD with naive coordinate MSE).

If anything here is unclear or you want extra examples (small runnable demo, unit test for `extract_interface.py`), tell me which area to expand.
