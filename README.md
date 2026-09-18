# SEQR: Symmetry-Extended Qubit Reduction for FCC Lattice Protein Encoding

Code and data accompanying the manuscript
*A Symmetry-Extended Qubit Reduction Theorem for FCC Lattice Protein Encoding*
(Supplementary Note 19).

Archived version: [DOI to be added after Zenodo release]

## What this repository reproduces

The worked example is the six-residue amyloid-beta fragment KLVFFA on the
face-centered cubic lattice, with all ten candidate contacts scored and full
self-avoidance imposed.

| Part | Command | Reproduces | Runtime |
|---|---|---|---|
| Exact fold | `python seqr_fcc_pipeline.py --part exact` | Decoder bijection, 8,145 feasible states, ground energy -30.5700 at degeneracy 2 (canonical, 15 qubits) and 48 (unreduced, 20 qubits) | Minutes |
| Hamiltonian | `python seqr_fcc_pipeline.py --part hamiltonian` | 27,238 monomials, 32,696 Pauli terms, single-pair QUBO (Supplementary Data 1), full-fold quadratization with 367 ancillas | Up to about an hour |
| Symmetry | `python seqr_fcc_pipeline.py --part symmetry` | Lemma 2 reflection check and the Proposition 1 enumeration (5,256 classes, no complete six-qubit fixing) | Minutes |
| Variational | `python seqr_fcc_pipeline.py --part vqe` | CVaR recovery, threshold sweep and register comparison | About 7 hours |
| Hardware | `python seqr_fcc_pipeline.py --part hardware` | Ground-state probabilities, masses and attenuation from the stored `ibm_marrakesh` counts | Seconds |

Each part asserts the values reported in the manuscript and stops if one differs.

## Installation

```
pip install -r requirements.txt
```

Results were produced with Python 3.13.9, NumPy 2.3.5 and Qiskit 2.5.2 on a
CPU workstation. The greedy quadratization and the transpiler are sensitive to
library version.

## Data

| File | Contents |
|---|---|
| `optimized_parameters.json` | Optimized ansatz parameters for every reported run |
| `vqe_runs.csv` | Per-run record of every variational optimization |
| `run_manifest.json` | Environment, seeds and pinned figures |
| `hardware_jobs*.json`, `hardware_repeats.json` | IBM Quantum job identifiers and submission metadata |
| `hardware_probs*.npz`, `hardware_repeats.npz` | Measured probability distributions from `ibm_marrakesh` |
| `ground_conformation_*.xyz` | The two ground-state conformations at 3.8 A bond scaling |
| `Supplementary_Data1_QUBO_matrix.npy` | 23 x 23 QUBO matrix of the single-pair construction check |

## Notes

- Hardware results depend on the device calibration at submission and cannot
  be regenerated exactly. The `hardware` part recomputes every reported
  quantity from the stored counts. A submission function is included but
  disabled, and it requires your own IBM Quantum account.
- The canonical probability mass is taken over states below the penalty weight
  of 100, and the unreduced mass over feasible states, as stated in
  Supplementary Note 17.
- Every stochastic element is seeded.

## Licence

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the article [reference to be added] and the
archived version [DOI to be added].
