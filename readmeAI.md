GlueVAE: AI Development Context & Specification
Note to AI Agent (Trae/Cursor): This document defines the SINGLE SOURCE OF TRUTH for the GlueVAE project. All code generation must strictly adhere to the architecture, data structures, and physical constraints defined below. DO NOT deviate from the "All-Atom" and "SE(3)-Equivariant" principles unless explicitly instructed.

1. Project Overview
Name: GlueVAE (Chiral-aware Latent Landscape of Protein–Protein Interfaces) Goal: A generative AI model to discover "inducible" protein-protein interactions (molecular glue targets) by learning the manifold of physical protein interfaces. Core Philosophy:

Generative, not Discriminative: We use VAE to "inpaint" missing interface atoms, not to classify bind/non-bind.

All-Atom Precision: We do NOT use coarse-grained residue representations. We use all heavy atoms to capture specific chemical interactions (H-bonds, hydrophobic packing).

Chirality Aware: The model must distinguish L-proteins from D-proteins via explicit SE(3) vector features.

2. Tech Stack & Dependencies
Language: Python 3.9+

Deep Learning: PyTorch 2.1+ (Stable)

Geometric DL: torch_geometric (PyG)

Backbone Architecture: PaiNN (Polarizable Atom Interaction Neural Network) implementation via PyG or custom adapted version.

Bioinformatics: BioPython (PDB parsing)

Data Storage: lmdb (High-throughput I/O), pickle serialization.

Visualization/Logging: wandb

3. Directory Structure
Ensure all generated files follow this tree:

Plaintext
GlueVAE/
├── data/
│   ├── raw_pdbs/          # Input PDB files
│   └── processed_lmdb/    # Output LMDB database
├── src/
│   ├── data/
│   │   ├── extract_interface.py  # [Script] Raw PDB -> Interface Atoms -> LMDB
│   │   └── dataset.py            # [Class] PyG Dataset, Graph construction, Chiral features
│   ├── models/
│   │   ├── layers.py             # Basic blocks (PaiNN layers, Norms)
│   │   └── glue_vae.py           # Main VAE Architecture (Encoder, Pooling, Decoder)
│   └── utils/
│       ├── geometry.py           # Geometric utils (RMSD, pairwise distance)
│       ├── parsing.py            # PDB parsing helpers
│       └── constants.py          # Atomic radii, element vocab, etc.
├── train.py                      # Training loop
└── config.yaml                   # Hyperparameters
4. Key Implementation Details (Strict constraints)
A. Data Processing (extract_interface.py)

Input: Raw PDB files.

Logic:

Find interacting chains (min dist < 10Å).

Crop interface: Keep ALL HEAVY ATOMS (C, N, O, S, P, F, Cl...) within 20Å of the center.

CRITICAL: Do NOT filter non-standard residues (MSE, PTR, Ligands). Treat everything as a point cloud of atoms with element types.

Save: Coordinates pos [N, 3], Element z [N], Residue Index batch [N].

B. Graph Construction & Features (dataset.py)

Graph Topology:

Spatial Edges: Radius graph (r<4.5 
A
˚
 ).

Covalent Edges: Distance threshold (d<1.6 
A
˚
 ). Assign distinct edge_attr.

Node Features (Inputs to PaiNN):

s (Scalar): nn.Embedding of atomic numbers (Element type).

v (Vector): Chirality Injection.

For each atom i, calculate sum of relative vectors to covalent neighbors:  
v

  
i
​	
 =∑ 
j∈cov
​	
 ( 
r

  
j
​	
 − 
r

  
i
​	
 ).

This allows the SE(3) network to perceive local tetrahedral chirality.

C. Model Architecture (glue_vae.py)

Encoder (PaiNN):

Input: All-atom graph.

Technique: Use Gradient Checkpointing to save VRAM.

Bottleneck (Atom-to-Residue Pooling):

Use scatter_mean based on residue_index to aggregate atom features -> residue features.

Latent Space: Defined at the residue level (Coarse-grained latent, Fine-grained input).

Decoder (Conditional):

Input: Latent z (residue level) + Receptor Atoms (fine level).

Operation: Unpool/Broadcast z back to atoms or learn a super-resolution mapping.

D. Training Objectives (train.py)

Loss Function:

D-RMSD (Distance-RMSD): L=∣∣D 
pred
​	
 −D 
true
​	
 ∣∣ 
2
 . Use pairwise distance matrices instead of coordinate MSE to ensure SE(3) invariance without alignment.

KL Divergence: Standard β-VAE loss.

5. Coding Standards
Type Hinting: All functions must use Python type hints (e.g., def forward(self, x: torch.Tensor) -> torch.Tensor:).

Tensor Shapes: Document expected tensor shapes in comments (e.g., # [Batch, Num_Atoms, 3]).

Modularity: Keep model layers, data loading, and training logic decoupled.
