# 🧬 Structure-Based Feature Engineering — SARS-CoV-2 Mᵖʳᵒ (7RFW)

> Feature extraction, graph encoding, and lossless decoding pipeline for structure-based generative drug discovery.  
> Built for the **Medvolt Tech** ML Engineering Assignment.

---

## The Problem, in One Sentence

A protein structure is a 3D point cloud with chemistry attached. Generative models can't eat PDB files — this pipeline converts one into a machine-learning-ready graph and back again, with zero information loss.

---

## Structure

```
├── data/
│   └── 7rfw.pdb                  # SARS-CoV-2 Main Protease + Nirmatrelvir (PDB)
├── scripts/
│   ├── analysis.py               # PDB parsing via BioPython
│   ├── feature_extractor.py      # Physicochemical feature computation
│   ├── distance_graph.py         # Cα contact graph construction (8 Å cutoff)
│   ├── encoder.py                # Graph → NumPy tensor encoding
│   ├── decoder.py                # Encoded tensors → NetworkX graph
│   ├── validator.py              # RMSD, residue recovery, graph statistics
│   └── main.py                   # End-to-end pipeline runner
├── outputs/
│   └── encoded_graph.pkl         # Serialised encoded representation
└── requirements.txt
```

---

## Why This Protein?

**7RFW** is the crystal structure of the SARS-CoV-2 main protease (Mᵖʳᵒ / 3CLᵖʳᵒ) covalently bound to **Nirmatrelvir** — the active compound in Paxlovid. At 1.73 Å resolution, it's one of the highest-quality Mᵖʳᵒ structures available and a canonical target for structure-based drug discovery. The catalytic dyad (His41–Cys145) and the covalent inhibitor bond make it a demanding and realistic test for any structural feature pipeline.

---

## Feature Design

### Residue-Level (Why Not Atom-Level?)

Residue-level representations are chosen deliberately:

| Consideration | Atom-level | Residue-level (chosen) |
|---|---|---|
| Generative model input size | O(N_atoms) — very large | O(N_residues) — tractable |
| Noise sensitivity | High (side-chain flexibility) | Low (Cα is stable backbone anchor) |
| Transferability across proteins | Poor | Strong |
| Interpretability | Low | High |

### Per-Residue Feature Vector (26-dim)

| Feature | Dim | Source |
|---|---|---|
| Hydrophobicity | 1 | Kyte-Doolittle scale |
| Formal charge at pH 7 | 1 | Standard biochemistry |
| Polarity flag | 1 | Standard biochemistry |
| Cα coordinates (x, y, z) | 3 | PDB ATOM records |
| One-hot residue identity | 20 | 20 canonical amino acids |

**Total: 26-dimensional feature vector per residue**

### Edge Features (Contact Graph)

Edges connect residues whose Cα atoms are within **8 Å** — a well-established threshold that captures both covalent-adjacent and non-covalent spatial interactions relevant to binding pockets and allosteric networks.

| Edge Feature | Description |
|---|---|
| Euclidean distance (Å) | Spatial proximity |
| Sequence distance | \|i − j\| in primary sequence |

---

## Pipeline

```
PDB File
   │
   ▼
[analysis.py]  →  residue list (chain, residue name, Cα coords)
   │
   ▼
[feature_extractor.py]  →  26-dim feature dict per residue
   │
   ▼
[distance_graph.py]  →  NetworkX graph (nodes = residues, edges = contacts)
   │
   ▼
[encoder.py]  →  { node_features: float32 matrix, edge_index: int64, edge_features: float32 }
   │
   ├──→  saved as outputs/encoded_graph.pkl
   │
   ▼
[decoder.py]  →  NetworkX graph reconstructed from tensors
   │
   ▼
[validator.py]  →  node match, edge match, residue recovery accuracy, coordinate RMSD
```

---

## Encoding Format

The encoded output is a dictionary of NumPy arrays — directly consumable by PyTorch Geometric, DGL, or any GNN framework:

```python
{
    "node_features":  np.ndarray  # shape (N_residues, 26), dtype float32
    "edge_index":     np.ndarray  # shape (2, N_edges),     dtype int64
    "edge_features":  np.ndarray  # shape (N_edges, 2),     dtype float32
    "num_nodes":      int
    "num_edges":      int
}
```

This format maps directly to `torch_geometric.data.Data` with no transformation needed.

---

## Decoding Objective

The decoder is **lossless by construction** — no dimensionality reduction or compression is applied. The encoding is a bijective mapping from graph → tensors, so decoding inverts it exactly:

- Residue identity recovered via `argmax` over the one-hot slice → 100% accuracy
- Coordinates recovered directly from feature vector slots 3–5 → RMSD = 0.0 Å
- Graph topology recovered from edge index → exact edge match

This is intentional: the encoded representation is designed as input to a *downstream* generative model (e.g. a graph VAE or SE(3)-equivariant network), not as a compressed latent itself. The encoding is the input; the latent space lives in the model.

---

## Quickstart

```bash
pip install -r requirements.txt
cd scripts
python main.py
```

Expected output:
```
Features Extracted: 306 residues
Number of Nodes: 306
Number of Edges: ~2400+
Node Feature Matrix Shape: (306, 26)
Residue Recovery Accuracy: 1.0000
Coordinate RMSD: 0.0000 Å
PIPELINE COMPLETED SUCCESSFULLY
```

---

## Validation Results (7RFW)

| Metric | Result |
|---|---|
| Node count match | ✅ |
| Edge count match | ✅ |
| Residue identity recovery | 100% |
| Coordinate RMSD | 0.0000 Å |
| Graph density preserved | ✅ |
| Avg degree preserved | ✅ |

---

## Design Trade-offs

**Included**
- Kyte-Doolittle hydrophobicity (most validated scale for ML tasks)
- pH-7 formal charges (ASP/GLU negative, LYS/ARG/HIS positive)
- Spatial contact graph at 8 Å (captures binding pocket geometry)

**Consciously excluded**
- Secondary structure (DSSP) — adds a runtime dependency and secondary structure is derivable from coordinates by a downstream model
- B-factors — useful for flexibility modelling but adds noise for generative tasks without normalisation
- Ligand atoms (Nirmatrelvir/4WI) — out of scope for residue-level protein representation; would require a separate heterogeneous graph layer

**Scalability**
- The pipeline is O(N²) in the graph construction step. For proteome-scale datasets, replace the brute-force distance loop with a KD-tree (`scipy.spatial.KDTree`) for O(N log N) neighbour search. The encoding and decoding steps are already O(N).

---

## Dependencies

```
biopython
numpy
networkx
scipy
```

No deep learning framework required to run the pipeline. The output format is framework-agnostic.

---

## Biological Context

> The Mᵖʳᵒ active site contains the catalytic dyad **His41** and **Cys145**. Nirmatrelvir forms a covalent bond with Cys145 Sγ, visible in the PDB LINK record. The feature pipeline captures this region through the contact graph — Cys145 will appear as a high-degree node connected to the surrounding S1, S2, and S4 subsites. This makes the representation directly meaningful for structure-based generation of covalent inhibitors.

---

*Medvolt Tech ML Engineering Assignment — Feature Engineering for Structure-Based Generative Learning*
