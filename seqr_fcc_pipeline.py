"""
seqr_fcc_pipeline.py
====================

Reference implementation for

    A Symmetry-Extended Qubit Reduction Theorem for FCC Lattice Protein Encoding

Regenerates every classically computed figure reported in the manuscript and
Supplementary Information for the six-residue amyloid-beta fragment KLVFFA on
the face-centered cubic lattice.

Usage
-----
    python seqr_fcc_pipeline.py --part exact        # minutes
    python seqr_fcc_pipeline.py --part symmetry     # seconds
    python seqr_fcc_pipeline.py --part hamiltonian  # about a minute
    python seqr_fcc_pipeline.py --part vqe          # about 7 hours
    python seqr_fcc_pipeline.py --part hardware     # seconds
    python seqr_fcc_pipeline.py --part all          # every part except vqe

Each part compares its output against the value reported in the manuscript and
exits non-zero if any comparison fails.

The hardware results depend on a specific device at a specific calibration and
cannot be regenerated. The hardware part recomputes every reported quantity
from the stored measurement distributions in data/.
"""

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# --------------------------------------------------------------------------
# Configuration (Supplementary Notes 12 and 14)
# --------------------------------------------------------------------------
SEQUENCE = "KLVFFA"                  # amyloid-beta residues 16 to 21
N_RES = len(SEQUENCE)                # 6
N_TURNS = N_RES - 1                  # 5 backbone transitions
Q_RAW = 4 * N_TURNS                  # 20 qubits, unreduced register
Q_CAN = Q_RAW - 5                    # 15 qubits, canonical register (Theorem 1)

D_TARGET = 2                         # squared nearest-neighbour distance
LAMBDA_PLANE = 100.0                 # plane-validity penalty
LAMBDA_SA = 100.0                    # self-avoidance penalty
T_CAN = np.array([1, 1, 0])          # canonical first direction
BOND_ANGSTROM = 3.8                  # Ca to Ca spacing
SCALE = BOND_ANGSTROM / np.sqrt(D_TARGET)

REPS = 3                             # ansatz depth
RNG_SEED = 42

# Miyazawa and Jernigan contact energies, in units of RT, for the residue
# pairs occurring among the candidate contacts of KLVFFA.
MJ = {
    frozenset("LF"): -7.28, frozenset("VF"): -6.29, frozenset("LA"): -4.91,
    frozenset("FA"): -4.81, frozenset("VA"): -4.04, frozenset("KF"): -3.36,
    frozenset("KV"): -2.49, frozenset("KA"): -1.31,
}

PAIRS = [(i, j) for i in range(1, N_RES + 1) for j in range(i + 2, N_RES + 1)]

FIX = lambda v: ("fix", v)
BIT = lambda k: ("bit", k)

# Canonical register: turn 1 fixed entirely by Lemma 1, p1 of turn 2 by Lemma 2.
SPEC_CAN = [
    (FIX(0), FIX(0), FIX(0), FIX(0)),
    (BIT(0), BIT(1), FIX(0), BIT(2)),
    (BIT(3), BIT(4), BIT(5), BIT(6)),
    (BIT(7), BIT(8), BIT(9), BIT(10)),
    (BIT(11), BIT(12), BIT(13), BIT(14)),
]
SPEC_RAW = [tuple(BIT(4 * k + i) for i in range(4)) for k in range(N_TURNS)]

# --------------------------------------------------------------------------
# Comparison bookkeeping
# --------------------------------------------------------------------------
REPORT = []


def check(name, got, expected, tol=0.0):
    g, e = float(got), float(expected)
    ok = abs(g - e) <= tol
    REPORT.append((name, g, e, ok))
    print(f"  [{'OK  ' if ok else 'DIFF'}] {name:52s} "
          f"got {g:.6g}   reported {e:.6g}")
    return ok


def summarise():
    bad = [r for r in REPORT if not r[3]]
    print(f"\n{len(REPORT)} comparisons, {len(bad)} DIFF")
    for name, g, e, _ in bad:
        print(f"  {name}: got {g:.6g}, reported {e:.6g}")
    return len(bad)


# --------------------------------------------------------------------------
# Encoding (Supplementary Notes 1, 2, 5, 6, 13)
# --------------------------------------------------------------------------
def decode_components(d2, d3, p1, p2):
    """Closed-form FCC direction decoder, Equation (S29). Scalars or arrays."""
    dx = (1 - 2 * d2) * (1 - p2)
    dz = (1 - 2 * d3) * (p1 + p2 - 2 * p1 * p2)
    dy = (1 - p1) * ((1 - p2) * (1 - 2 * d3) + p2 * (1 - 2 * d2))
    return dx, dy, dz


def dvec(t):
    return np.array(decode_components(*t), dtype=int)


VALID_CODES = [t for t in itertools.product([0, 1], repeat=4) if t[2:] != (1, 1)]
DIRS = np.array([dvec(t) for t in VALID_CODES])          # the twelve directions

# Complete decoder truth table, Supplementary Table S5.
TABLE_S5 = {
    (0, 0, 0, 0): (1, 1, 0),   (0, 0, 0, 1): (0, 1, 1),
    (0, 0, 1, 0): (1, 0, 1),   (0, 0, 1, 1): (0, 0, 0),
    (0, 1, 0, 0): (1, -1, 0),  (0, 1, 0, 1): (0, 1, -1),
    (0, 1, 1, 0): (1, 0, -1),  (0, 1, 1, 1): (0, 0, 0),
    (1, 0, 0, 0): (-1, 1, 0),  (1, 0, 0, 1): (0, -1, 1),
    (1, 0, 1, 0): (-1, 0, 1),  (1, 0, 1, 1): (0, 0, 0),
    (1, 1, 0, 0): (-1, -1, 0), (1, 1, 0, 1): (0, -1, -1),
    (1, 1, 1, 0): (-1, 0, -1), (1, 1, 1, 1): (0, 0, 0),
}

# The full octahedral group as 48 signed permutation matrices.
OH = []
for _perm in itertools.permutations(range(3)):
    for _signs in itertools.product([1, -1], repeat=3):
        _G = np.zeros((3, 3), dtype=int)
        for _i in range(3):
            _G[_i, _perm[_i]] = _signs[_i]
        OH.append(_G)


def register_bits(n_qubits):
    idx = np.arange(2 ** n_qubits, dtype=np.int64)
    return [((idx >> k) & 1).astype(np.int16) for k in range(n_qubits)]


def _resolve(entry, bits, size):
    kind, val = entry
    if kind == "bit":
        return bits[val]
    return np.full(size, val, dtype=np.int16)


def decode_all_turns(spec, bits, size):
    """Decode every turn for every register state. Shape (n_turns, 3, size)."""
    out = np.empty((len(spec), 3, size), dtype=np.int16)
    for k, s in enumerate(spec):
        d2, d3, p1, p2 = (_resolve(e, bits, size) for e in s)
        dx, dy, dz = decode_components(d2, d3, p1, p2)
        out[k, 0], out[k, 1], out[k, 2] = dx, dy, dz
    return out


def residue_coords(dirs):
    """R_1 at the origin, R_{k+1} = R_k + delta(t_k)."""
    n_turns, _, size = dirs.shape
    coords = np.zeros((n_turns + 1, 3, size), dtype=np.int16)
    np.cumsum(dirs, axis=0, out=coords[1:])
    return coords


def pairwise_sq_distances(coords, pairs):
    size = coords.shape[2]
    D = np.empty((len(pairs), size), dtype=np.int16)
    for p, (i, j) in enumerate(pairs):
        diff = coords[i - 1].astype(np.int16) - coords[j - 1]
        D[p] = (diff * diff).sum(axis=0)
    return D


def radius_of_gyration(coords_scaled):
    """Unweighted, over alpha carbons only."""
    com = coords_scaled.mean(axis=0)
    return float(np.sqrt(((coords_scaled - com) ** 2).sum(axis=1).mean()))


def build_register(spec, n_qubits):
    """Decode, place, score and classify every state of one register."""
    size = 2 ** n_qubits
    bits = register_bits(n_qubits)
    dirs = decode_all_turns(spec, bits, size)
    coords = residue_coords(dirs)
    D = pairwise_sq_distances(coords, PAIRS)

    m_pair = np.array([MJ[frozenset((SEQUENCE[i - 1], SEQUENCE[j - 1]))]
                       for i, j in PAIRS])

    bond_sq = (dirs.astype(np.int32) ** 2).sum(axis=1)
    turn_valid = bond_sq == D_TARGET
    n_invalid = (~turn_valid).sum(axis=0).astype(np.int16)
    n_coincident = (D == 0).sum(axis=0).astype(np.int16)
    feasible = (n_invalid == 0) & (n_coincident == 0)

    contact = D == D_TARGET
    e_contact = (m_pair[:, None] * contact).sum(axis=0)
    e_total = e_contact + LAMBDA_PLANE * n_invalid + LAMBDA_SA * n_coincident

    return dict(bits=bits, dirs=dirs, coords=coords, D=D, m_pair=m_pair,
                turn_valid=turn_valid, n_invalid=n_invalid, feasible=feasible,
                contact=contact, e_contact=e_contact, e_total=e_total)


def prepare(verbose=True):
    """Build both registers and run the encoding checks.

    The random generator is consumed here in a fixed order so that the
    random-parameter baseline of the variational part is reproducible.
    """
    if verbose:
        print(f"sequence {SEQUENCE}, N = {N_RES}, turns = {N_TURNS}")
        print(f"raw register {Q_RAW} qubits, canonical register {Q_CAN} qubits")

    can = build_register(SPEC_CAN, Q_CAN)
    raw = build_register(SPEC_RAW, Q_RAW)

    rng = np.random.default_rng(RNG_SEED)
    # 1. vectorised decode against the scalar decoder on 200 random states
    for state in rng.integers(0, 2 ** Q_CAN, size=200):
        b = [(int(state) >> k) & 1 for k in range(Q_CAN)]
        for k, s in enumerate(SPEC_CAN):
            vals = [b[e[1]] if e[0] == "bit" else e[1] for e in s]
            assert tuple(decode_components(*vals)) == \
                tuple(can["dirs"][k, :, state]), "vectorised decode mismatch"
    # 2. contact-indicator elimination on 500 feasible states
    grid = np.array(list(itertools.product([0, 1], repeat=len(PAIRS))))
    sample = rng.choice(np.flatnonzero(can["feasible"]), size=500, replace=False)
    for s in sample:
        active = can["contact"][:, s].astype(float)
        assert np.isclose((grid @ (can["m_pair"] * active)).min(),
                          can["e_contact"][s]), "indicator elimination fails"

    return can, raw, rng


# --------------------------------------------------------------------------
# Part: exact
# --------------------------------------------------------------------------
def part_exact(can, raw):
    print("\n=== Decoder (Supplementary Note 5) ===")
    bad = [k for k, v in TABLE_S5.items() if tuple(decode_components(*k)) != v]
    check("decoder reproduces Table S5 (16 states)", 16 - len(bad), 16)
    valid = {tuple(decode_components(*k)) for k in TABLE_S5 if k[2:] != (1, 1)}
    check("distinct valid directions", len(valid), 12)
    check("all of squared length two with one zero component",
          int(all(sum(c == 0 for c in v) == 1 and sum(c * c for c in v) == 2
                  for v in valid)), 1)
    check("turn 1 fixed to t_can", int(np.all(can["dirs"][0].T == T_CAN)), 1)

    zero = (can["dirs"].astype(np.int32) ** 2).sum(axis=1) == 0
    ok = True
    for k, s in enumerate(SPEC_CAN):
        p1 = _resolve(s[2], can["bits"], 2 ** Q_CAN).astype(bool)
        p2 = _resolve(s[3], can["bits"], 2 ** Q_CAN).astype(bool)
        ok &= np.array_equal(zero[k], p1 & p2)
    check("zero-length bonds occur exactly on (p1,p2)=(1,1)", int(ok), 1)
    check("R2 fixed to t_can", int(np.all(can["coords"][1].T == T_CAN)), 1)

    print("\n=== Exact reference, canonical register (Note 17) ===")
    e_total, feasible = can["e_total"], can["feasible"]
    e_min = e_total.min()
    ground = np.isclose(e_total, e_min)
    gidx = np.flatnonzero(ground)

    check("states enumerated", e_total.size, 32768)
    check("plane-valid states", int((can["n_invalid"] == 0).sum()), 13824)
    check("feasible states", int(feasible.sum()), 8145)
    check("feasible fraction (%)", 100 * feasible.mean(), 24.86, tol=0.005)
    check("ground-state energy", e_min, -30.5700, tol=1e-9)
    check("ground-state degeneracy", ground.sum(), 2)
    check("penalised minimum equals constrained minimum",
          int(np.isclose(e_min, can["e_contact"][feasible].min())), 1)
    check("every ground state is feasible", int(np.all(feasible[ground])), 1)

    # independent revalidation from the coordinates alone
    for s in gidx:
        c = can["coords"][:, :, int(s)].astype(int)
        bonds = np.diff(c, axis=0)
        assert np.all((bonds ** 2).sum(axis=1) == D_TARGET), "bad bond length"
        assert len({tuple(x) for x in c}) == N_RES, "residues coincide"
        made = [(i, j) for (i, j) in PAIRS
                if int(((c[i - 1] - c[j - 1]) ** 2).sum()) == D_TARGET]
        e = sum(MJ[frozenset((SEQUENCE[i - 1], SEQUENCE[j - 1]))]
                for i, j in made)
        assert np.isclose(e, e_min), "recomputed energy disagrees"
    check("ground states revalidated from coordinates", len(gidx), 2)

    g0 = can["coords"][:, :, int(gidx[0])].astype(int)
    made = [(i, j) for (i, j) in PAIRS
            if int(((g0[i - 1] - g0[j - 1]) ** 2).sum()) == D_TARGET]
    check("contacts formed", len(made), 7)
    check("contact set matches Table S12",
          int(made == [(1, 3), (1, 5), (1, 6), (2, 4), (2, 5), (3, 6), (4, 6)]), 1)
    check("radius of gyration (A)", radius_of_gyration(g0 * SCALE), 2.69, tol=0.005)

    # the two ground states are related by sigma_z
    c1 = can["coords"][:, :, int(gidx[0])].astype(int)
    c2 = can["coords"][:, :, int(gidx[1])].astype(int)
    check("ground states related by sigma_z",
          int(np.array_equal(c1 * np.array([1, 1, -1]), c2)), 1)

    print("\n=== Geometry of the fold (Note 17) ===")
    cen = g0.mean(axis=0)
    check("centroid at (1,0,0)", int(np.allclose(cen, [1, 0, 0])), 1)
    check("all residues at squared distance one from the centroid",
          int(np.allclose(((g0 - cen) ** 2).sum(axis=1), 1)), 1)
    allD = {(i, j): int(((g0[i - 1] - g0[j - 1]) ** 2).sum())
            for i in range(1, N_RES + 1) for j in range(i + 1, N_RES + 1)}
    check("pairs at contact separation", sum(v == 2 for v in allD.values()), 12)
    check("antipodal pairs are (1,4),(2,6),(3,5)",
          int(sorted(k for k, v in allD.items() if v == 4)
              == [(1, 4), (2, 6), (3, 5)]), 1)
    check("contacts at even sequence separation",
          sum((j - i) % 2 == 0 for i, j in made), 4)
    check("point-group stabilizer order of the fold",
          sum(np.array_equal(g0 @ G.T, g0) for G in OH), 1)

    n_contacts = can["contact"].sum(axis=0)
    check("maximum contacts over feasible canonical states",
          n_contacts[feasible].max(), 7)
    seven = feasible & (n_contacts == 7)
    check("feasible states forming seven contacts", seven.sum(), 12)
    patterns = np.unique(can["contact"][:, seven].T.astype(int), axis=0)
    check("distinct seven-contact patterns", len(patterns), 5)
    pat_e = np.sort(patterns @ can["m_pair"])
    check("gap to the next seven-contact pattern", pat_e[1] - pat_e[0],
          0.02, tol=1e-6)

    print("\n=== Unreduced register (Note 17) ===")
    e_raw, feas_raw = raw["e_total"], raw["feasible"]
    e_min_raw = e_raw.min()
    ground_raw = np.isclose(e_raw, e_min_raw)
    gidx_raw = np.flatnonzero(ground_raw)

    check("states enumerated", e_raw.size, 1048576)
    check("feasible states", int(feas_raw.sum()), 152532)
    check("feasible fraction (%)", 100 * feas_raw.mean(), 14.55, tol=0.005)
    check("ground energy agrees with the canonical register (Theorem 1)",
          e_min_raw, -30.5700, tol=1e-9)
    check("ground-state degeneracy", ground_raw.sum(), 48)
    check("degeneracy ratio", ground_raw.sum() / ground.sum(), 24)
    check("maximum contacts over feasible unreduced states",
          (raw["D"] == D_TARGET).sum(axis=0)[feas_raw].max(), 7)

    coord_sets = {tuple(map(tuple, raw["coords"][:, :, int(s)]))
                  for s in gidx_raw}
    check("distinct coordinate sets among them", len(coord_sets), 48)
    first = defaultdict(int)
    for s in gidx_raw:
        first[tuple(raw["dirs"][0, :, int(s)])] += 1
    check("distinct first-turn directions", len(first), 12)
    check("each direction occurs as the first turn four times",
          int(set(first.values()) == {4}), 1)

    print("\n=== Feasibility margins (Note 17) ===")
    check("lowest infeasible energy, canonical", e_total[~feasible].min(),
          58.67, tol=0.005)
    check("infeasible states at negative energy, canonical",
          np.count_nonzero((e_total < 0) & ~feasible), 0)
    check("infeasible states at negative energy, unreduced",
          np.count_nonzero((e_raw < 0) & ~feas_raw), 0)

    return dict(e_min=e_min, ground=ground, gidx=gidx,
                e_min_raw=e_min_raw, ground_raw=ground_raw)


# --------------------------------------------------------------------------
# Part: symmetry (Supplementary Note 3)
# --------------------------------------------------------------------------
def _R_bits(t):
    d2, d3, p1, p2 = t
    if (p1, p2) == (0, 0):
        d2, d3 = d3, d2
    return (d2, d3, p2, p1)


def _SZ_bits(t):
    d2, d3, p1, p2 = t
    if (p1, p2) != (0, 0):
        d3 = 1 - d3
    return (d2, d3, p1, p2)


def _R_geo(v):
    return v[..., [1, 0, 2]]


def _SZ_geo(v):
    return v * np.array([1, 1, -1])


def _chain(turns):
    d = np.array([dvec(t) for t in turns])
    return np.vstack([np.zeros(3, dtype=int), np.cumsum(d, axis=0)])


def part_symmetry():
    print("\n=== Residual stabilizer (Note 1) ===")
    check("R bit rule matches the geometric reflection",
          sum(np.array_equal(dvec(_R_bits(t)), _R_geo(dvec(t)))
              for t in VALID_CODES), 12)
    check("sigma_z bit rule matches the geometric reflection",
          sum(np.array_equal(dvec(_SZ_bits(t)), _SZ_geo(dvec(t)))
              for t in VALID_CODES), 12)

    stab = {"e": lambda v: v, "R": _R_geo, "sigma_z": _SZ_geo,
            "R sigma_z": lambda v: _SZ_geo(_R_geo(v))}
    for name, g in stab.items():
        assert np.array_equal(g(T_CAN), T_CAN), f"{name} does not fix t_can"
    for name, exp in (("e", 12), ("R", 2), ("sigma_z", 4), ("R sigma_z", 2)):
        check(f"directions fixed by {name}",
              sum(np.array_equal(stab[name](v), v) for v in DIRS), exp)
    orbits = {frozenset(tuple(g(v)) for g in stab.values()) for v in DIRS}
    check("orbits of Stab(t_can) on D, Burnside", len(orbits), 5)

    print("\n=== Lemma 2 reflection, Tables S2 and S3 (Note 3) ===")
    turns = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0),
             (0, 1, 0, 0), (0, 1, 0, 1), (0, 0, 0, 0)]
    turns_refl_exp = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                      (1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 0)]
    coords_exp = np.array([(0, 0, 0), (1, 1, 0), (1, 2, 1), (2, 2, 2),
                           (3, 1, 2), (3, 2, 1), (4, 3, 1)])
    dist_exp = [1.4142, 2.4495, 3.4641, 3.7417, 3.7417, 5.0990, 1.4142,
                2.4495, 2.8284, 2.4495, 3.7417, 1.4142, 2.4495, 2.0000,
                3.1623, 1.4142, 1.4142, 2.4495, 1.4142, 2.4495, 1.4142]

    c = _chain(turns)
    refl = [_R_bits(t) for t in turns]
    c_bit, c_geo = _chain(refl), _R_geo(_chain(turns))
    check("coordinates match Table S2", int(np.array_equal(c, coords_exp)), 1)
    check("reflected registers match Equation S16",
          int(refl == turns_refl_exp), 1)
    check("bit rule equals the geometric reflection on the full chain",
          int(np.array_equal(c_bit, c_geo)), 1)
    check("example is self-avoiding", len({tuple(x) for x in c}), 7)
    d_o = [np.linalg.norm(c[i] - c[j]) for i in range(7) for j in range(i + 1, 7)]
    d_r = [np.linalg.norm(c_bit[i] - c_bit[j])
           for i in range(7) for j in range(i + 1, 7)]
    check("21 distances match Table S3",
          sum(abs(a - b) < 1e-4 for a, b in zip(d_o, dist_exp)), 21)
    check("21 distances preserved by the reflection",
          sum(abs(a - b) < 1e-9 for a, b in zip(d_o, d_r)), 21)

    print("\n=== Proposition 1, optimality by enumeration (Note 3) ===")
    t0 = time.perf_counter()
    key = {tuple(v): i for i, v in enumerate(DIRS)}
    perm = np.array([[key[tuple(G @ v)] for v in DIRS] for G in OH])
    codebit = np.array(VALID_CODES)

    n_t = 5
    nseq = 12 ** n_t
    idx = np.arange(nseq, dtype=np.int64)
    dig = [(idx // 12 ** t) % 12 for t in range(n_t)]
    cls = np.full(nseq, np.iinfo(np.int64).max, dtype=np.int64)
    for g in range(48):
        img = np.zeros(nseq, dtype=np.int64)
        for t in range(n_t):
            img += perm[g][dig[t]] * 12 ** t
        np.minimum(cls, img, out=cls)
    _, cls_c = np.unique(cls, return_inverse=True)
    n_cls = int(cls_c.max()) + 1

    check("admissible five-transition sequences", nseq, 248832)
    check("classes under O_h", n_cls, 5256)
    fixg = np.array([(perm[g] == np.arange(12)).sum() for g in range(48)])
    check("classes by Burnside", (fixg.astype(np.int64) ** 5).sum() / 48, 5256)

    # literal L = 2 * variable + value, variable = 4 * transition + bit
    agree = [codebit[dig[t], b] == v
             for t in range(n_t) for b in range(4) for v in (0, 1)]

    def complete(lits):
        m = np.ones(nseq, dtype=bool)
        for L in lits:
            m &= agree[L]
        return np.count_nonzero(np.bincount(cls_c[m], minlength=n_cls)) == n_cls

    levels = {1: {(L,) for L in range(40) if complete((L,))}}
    k = 1
    while levels[k]:
        prev, nxt = levels[k], set()
        for F in sorted(prev):
            for L in range((F[-1] // 2 + 1) * 2, 40):
                cand = F + (L,)
                if any(cand[:i] + cand[i + 1:] not in prev for i in range(k)):
                    continue
                if complete(cand):
                    nxt.add(cand)
        k += 1
        levels[k] = nxt

    for size, exp in zip(range(1, 7), (40, 595, 4040, 10540, 7600, 0)):
        check(f"complete bit-fixings of size {size}",
              len(levels.get(size, ())), exp)
    check("the fixing of Lemmas 1 and 2 is complete",
          int((0, 2, 4, 6, 12) in levels[5]), 1)
    check("transitions touched by any size-five fixing",
          max(len({L // 2 // 4 for L in F}) for F in levels[5]), 4)

    # Remark 7: the recoded register outside the bit-fixing class
    n_t3 = 3
    idx3 = np.arange(12 ** n_t3, dtype=np.int64)
    dig3 = [(idx3 // 12 ** t) % 12 for t in range(n_t3)]
    cls3 = np.full(idx3.size, np.iinfo(np.int64).max, dtype=np.int64)
    for g in range(48):
        img = sum(perm[g][dig3[t]] * 12 ** t for t in range(n_t3))
        np.minimum(cls3, img, out=cls3)
    check("classes of three transitions", np.unique(cls3).size, 42)
    s2 = {(1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 0, 1), (0, 1, 1),
          (-1, 0, 1), (0, -1, 1)}
    s3 = {(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1),
          (-1, 0, 1), (0, 1, -1), (0, -1, -1)}
    in2 = np.array([tuple(v) in s2 for v in DIRS])
    in3 = np.array([tuple(v) in s3 for v in DIRS])
    keep = (dig3[0] == key[(1, 1, 0)]) & in2[dig3[1]] & in3[dig3[2]]
    check("Remark 7 recoded register retains every class",
          np.unique(cls3[keep]).size, 42)
    print(f"  enumeration time {time.perf_counter() - t0:.0f} s")


# --------------------------------------------------------------------------
# Multilinear and Ising expansions (Supplementary Notes 9, 10, 19)
# --------------------------------------------------------------------------
def fast_mobius(values, n):
    """Coefficients a_S with f(x) = sum_S a_S prod_{i in S} x_i, in n 2^n ops."""
    a = np.array(values, dtype=np.float64).copy()
    idx = np.arange(a.size)
    for i in range(n):
        step = 1 << i
        hi = idx[(idx & step) != 0]
        a[hi] -= a[hi ^ step]
    return a


def mobius_inverse(coeffs, n):
    f = np.array(coeffs, dtype=np.float64).copy()
    idx = np.arange(f.size)
    for i in range(n):
        step = 1 << i
        hi = idx[(idx & step) != 0]
        f[hi] += f[hi ^ step]
    return f


def walsh_hadamard(a):
    """Unnormalised fast Walsh and Hadamard transform, self-inverse up to 1/N."""
    a = np.array(a, dtype=np.float64).copy()
    n = a.size.bit_length() - 1
    for i in range(n):
        step = 1 << i
        a = a.reshape(-1, 2 * step)
        lo, hi = a[:, :step].copy(), a[:, step:].copy()
        a[:, :step], a[:, step:] = lo + hi, lo - hi
        a = a.reshape(-1)
    return a


def popcount_table(n):
    idx = np.arange(2 ** n, dtype=np.int64)
    pc = np.zeros(2 ** n, dtype=np.int16)
    for i in range(n):
        pc += ((idx >> i) & 1).astype(np.int16)
    return pc


def extract_multilinear(func, nvars):
    """Multilinear coefficients by direct subset sum.

    Used for the trimmed instance only. The iteration order of the resulting
    dictionary follows itertools.combinations, and the greedy quadratization
    below depends on that order through its tie-breaking. Replacing this with
    the fast transform changes the ancilla count.
    """
    table = {b: func(*b) for b in itertools.product([0, 1], repeat=nvars)}
    coeffs = {}
    for S in itertools.chain.from_iterable(
            itertools.combinations(range(nvars), r) for r in range(nvars + 1)):
        a = 0.0
        for T in itertools.chain.from_iterable(
                itertools.combinations(S, r) for r in range(len(S) + 1)):
            x = tuple(1 if i in set(T) else 0 for i in range(nvars))
            a += (-1) ** (len(S) - len(T)) * table[x]
        if abs(a) > 1e-9:
            coeffs[frozenset(S)] = a
    return coeffs


def quadratize(coeffs, nvars, progress=False):
    """Greedy Rosenberg quadratization (Algorithm 2, Supplementary Note 19).

    At each step the variable pair occurring most often among the remaining
    monomials of degree above two is replaced by a fresh ancilla, and the
    Rosenberg penalty is added. Ties in the pair count are resolved by first
    occurrence in the iteration order of the coefficient dictionary, which is
    what max() returns. The ancilla count is therefore sensitive to how the
    coefficients were constructed, and both construction routes used in this
    work are preserved exactly as they were run.
    """
    coeffs = {frozenset(k): float(v) for k, v in coeffs.items()}
    M = 10.0 * sum(abs(v) for v in coeffs.values()) + 100.0
    next_var, aux, t0 = nvars, [], time.perf_counter()
    while True:
        high = [S for S in coeffs if len(S) > 2]
        if not high:
            break
        pc = defaultdict(int)
        for S in high:
            Ss = sorted(S)
            for i in range(len(Ss)):
                for j in range(i + 1, len(Ss)):
                    pc[(Ss[i], Ss[j])] += 1
        a, b = max(pc, key=pc.get)
        y = next_var
        next_var += 1
        aux.append((y, a, b))
        new = defaultdict(float)
        for S, v in coeffs.items():
            Sn = frozenset((S - {a, b}) | {y}) if {a, b} <= S else S
            new[Sn] += v
        for term, w in ((frozenset({a, b}), M), (frozenset({a, y}), -2 * M),
                        (frozenset({b, y}), -2 * M), (frozenset({y}), 3 * M)):
            new[term] += w
        coeffs = {k: v for k, v in new.items() if abs(v) > 1e-9}
        if progress and len(aux) % 100 == 0:
            print(f"    {len(aux)} ancillas, {len(coeffs)} terms, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    return coeffs, next_var, aux, M


def qubo_matrix(coeffs, n):
    Q = np.zeros((n, n))
    for S, v in coeffs.items():
        S = tuple(sorted(S))
        if len(S) == 1:
            Q[S[0], S[0]] += v
        elif len(S) == 2:
            Q[S[0], S[1]] += v / 2
            Q[S[1], S[0]] += v / 2
    return Q


def qubo_to_ising(coeffs, n):
    """x = (1 - z)/2. Returns the constant, the fields and the couplings."""
    h, J, const = np.zeros(n), {}, 0.0
    for S, v in coeffs.items():
        S = tuple(sorted(S))
        if len(S) == 0:
            const += v
        elif len(S) == 1:
            const += v / 2
            h[S[0]] -= v / 2
        else:
            k, l = S
            const += v / 4
            h[k] -= v / 4
            h[l] -= v / 4
            J[(k, l)] = J.get((k, l), 0.0) + v / 4
    return const, h, {k: v for k, v in J.items() if abs(v) > 1e-9}


def qubo_minimum(coeffs, n, chunk=1 << 20, tol=1e-6):
    """Exact minimum and degeneracy of a quadratic objective by enumeration."""
    const = coeffs.get(frozenset(), 0.0)
    lin = [(next(iter(S)), v) for S, v in coeffs.items() if len(S) == 1]
    quad = [(tuple(sorted(S)), v) for S, v in coeffs.items() if len(S) == 2]
    best, deg = np.inf, 0
    for start in range(0, 2 ** n, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** n), dtype=np.int64)
        bits = [((idx >> k) & 1).astype(np.float64) for k in range(n)]
        E = np.full(idx.size, const)
        for k, v in lin:
            E += v * bits[k]
        for (k, l), v in quad:
            E += v * bits[k] * bits[l]
        m = E.min()
        if m < best - tol:
            best, deg = m, 0
        deg += int(np.count_nonzero(np.abs(E - best) < tol))
    return best, deg


# --------------------------------------------------------------------------
# Part: hamiltonian
# --------------------------------------------------------------------------
def _trimmed_objective(q5, q6, q8, q9, q10, q11, q12, c24):
    """Single-pair instance of Note 14, contact (2,4), indicator retained."""
    d2 = dvec((q5, q6, 0, q8))
    d3 = dvec((q9, q10, q11, q12))
    R2 = np.array([1, 1, 0])
    R4 = R2 + d2 + d3
    theta = 1 if int(np.dot(R2 - R4, R2 - R4)) == D_TARGET else 0
    return c24 * MJ[frozenset("LF")] * theta + LAMBDA_PLANE * q11 * q12


def part_hamiltonian(can):
    print("\n=== Multilinear objective, canonical register (Note 14) ===")
    e_total = can["e_total"]
    a_can = fast_mobius(e_total, Q_CAN)
    check("Moebius round trip",
          int(np.allclose(mobius_inverse(a_can, Q_CAN), e_total, atol=1e-8)), 1)
    nz = np.flatnonzero(np.abs(a_can) > 1e-9)
    pc15 = popcount_table(Q_CAN)
    deg = pc15[nz]
    check("monomials", nz.size, 27238)
    check("maximum degree", deg.max(), 15)
    check("monomials of degree above two", int((deg > 2).sum()), 27203)
    check("smallest coefficient magnitude", np.abs(a_can[nz]).min(), 0.33,
          tol=0.005)
    check("largest coefficient magnitude", np.abs(a_can[nz]).max(), 2.155e4,
          tol=5)

    print("\n=== Structure of D(i,j) and Theta_ij, Table S4 (Note 14) ===")
    table_s4 = {(1, 3): (2, 5, 2, 1), (2, 4): (5, 50, 5, 33),
                (3, 5): (6, 95, 6, 78), (4, 6): (6, 95, 6, 78),
                (1, 4): (5, 50, 7, 64), (2, 5): (6, 171, 9, 1207),
                (3, 6): (6, 256, 10, 2477), (1, 5): (6, 171, 11, 810),
                (2, 6): (6, 368, 13, 20797), (1, 6): (6, 368, 15, 17015)}
    for p, pair in enumerate(PAIRS):
        aD = fast_mobius(can["D"][p].astype(np.float64), Q_CAN)
        aT = fast_mobius((can["D"][p] == D_TARGET).astype(np.float64), Q_CAN)
        nzD, nzT = np.abs(aD) > 1e-9, np.abs(aT) > 1e-9
        got = (int(pc15[nzD].max()), int(nzD.sum()),
               int(pc15[nzT].max()), int(nzT.sum()))
        for lbl, g, e in zip(("degree D", "monomials D",
                              "degree Theta", "monomials Theta"),
                             got, table_s4[pair]):
            check(f"{pair} {lbl}", g, e)

    print("\n=== Compact Ising expansion, Note 10 ===")
    c_can = walsh_hadamard(e_total) / 2 ** Q_CAN
    check("Walsh and Hadamard round trip",
          int(np.allclose(walsh_hadamard(c_can), e_total, atol=1e-8)), 1)
    nzm = np.abs(c_can) > 1e-9
    w = pc15[nzm]
    check("Pauli terms", int(nzm.sum()), 32696)
    check("possible Pauli-Z strings", 2 ** Q_CAN, 32768)
    check("maximum Pauli weight", int(w.max()), 15)
    check("identity coefficient", c_can[0], 121.849, tol=0.001)
    check("smallest coefficient magnitude", np.abs(c_can[nzm]).min(), 1.77e-5,
          tol=5e-8)
    check("terms above 1e-3", int((np.abs(c_can) > 1e-3).sum()), 31518)
    check("terms above 1e-2", int((np.abs(c_can) > 1e-2).sum()), 28511)

    # the Pauli sum reproduces the tabulated energy at both ground states
    ground = np.isclose(e_total, e_total.min())
    supp = np.flatnonzero(nzm)
    for s in np.flatnonzero(ground):
        sgn = 1 - 2 * (np.array([bin(int(t) & int(s)).count("1")
                                 for t in supp]) % 2)
        assert np.isclose(float((c_can[nzm] * sgn).sum()), e_total[s], atol=1e-8)
    check("Pauli sum verified at both ground states", 2, 2)

    print("\n=== Single-pair construction check, Notes 15 and 16 ===")
    n8 = 8
    co8 = extract_multilinear(_trimmed_objective, n8)
    tab8 = np.array([_trimmed_objective(*b)
                     for b in itertools.product([0, 1], repeat=n8)])
    # itertools.product is most-significant-first, so reindex little-endian
    order = np.array([sum(((i >> k) & 1) << (n8 - 1 - k) for k in range(n8))
                      for i in range(2 ** n8)])
    tab8 = tab8[order]

    check("monomials before quadratization", len(co8), 34)
    check("maximum degree before quadratization",
          max(len(S) for S in co8), 6)
    check("total absolute coefficient mass",
          sum(abs(v) for v in co8.values()), 471.28, tol=0.005)
    check("ground energy before quadratization", tab8.min(), -7.28, tol=1e-9)
    check("degeneracy before quadratization",
          np.count_nonzero(np.abs(tab8 - tab8.min()) < 1e-9), 32)

    q8, n23, aux8, M8 = quadratize(co8, n8)
    Q23 = qubo_matrix(q8, n23)
    off = int(np.count_nonzero(np.abs(Q23[np.triu_indices(n23, 1)]) > 1e-9))
    check("penalty weight M", M8, 4812.8, tol=0.05)
    check("ancillas", len(aux8), 15)
    check("total variables", n23, 23)
    check("nonzero diagonal entries",
          np.count_nonzero(np.abs(np.diag(Q23)) > 1e-9), 15)
    check("nonzero off-diagonal pairs", off, 76)
    check("possible off-diagonal pairs", n23 * (n23 - 1) // 2, 253)
    check("matrix density (%)", 100 * off / (n23 * (n23 - 1) // 2), 30.04,
          tol=0.005)

    block_exp = np.array([
        [0, -4812.8, -3.64, 0, -4812.8],
        [-4812.8, 14438.4, 0, -4812.8, 0],
        [-3.64, 0, 14538.4, -7.28, 7.28],
        [0, -4812.8, -7.28, 14431.12, 0],
        [-4812.8, 0, 7.28, 0, 14438.4]])
    check("sub-block x7 to x11 matches Equation S37",
          int(np.allclose(Q23[7:12, 7:12], block_exp, atol=0.005)), 1)

    e23, d23 = qubo_minimum(q8, n23)
    check("ground energy after quadratization", e23, -7.28, tol=1e-6)
    check("degeneracy after quadratization", d23, 32)

    const23, h23, J23 = qubo_to_ising(q8, n23)
    nfield = int(np.count_nonzero(np.abs(h23) > 1e-9))
    mags = np.concatenate([np.abs(h23[np.abs(h23) > 1e-9]),
                           np.abs(np.array(list(J23.values())))])
    check("Ising single-qubit fields", nfield, 22)
    check("Ising two-qubit couplings", len(J23), 76)
    check("Ising total Pauli terms", 1 + nfield + len(J23), 99)
    check("smallest field or coupling magnitude", mags.min(), 1.82, tol=0.005)
    check("largest field or coupling magnitude", mags.max(), 4.81e3, tol=5)
    check("Ising additive constant", const23, 5.42e4, tol=50)

    c8 = walsh_hadamard(tab8) / 2 ** n8
    pc8 = popcount_table(n8)
    nz8 = np.abs(c8) > 1e-9
    nonid = np.abs(c8[1:][nz8[1:]])
    check("compact form Pauli terms", int(nz8.sum()), 76)
    check("compact form maximum weight", int(pc8[nz8].max()), 6)
    check("compact form identity coefficient", c8[0], 23.18, tol=0.005)
    check("compact form smallest coefficient", nonid.min(), 0.2275, tol=5e-5)
    check("compact form largest coefficient", nonid.max(), 24.55, tol=0.005)

    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "Supplementary_Data1_QUBO_matrix.npy"), Q23)
    print(f"  wrote data/Supplementary_Data1_QUBO_matrix.npy ({n23}x{n23})")

    print("\n=== Full-fold quadratization, Notes 9, 10 and 18 ===")
    # Built from the fast transform in little-endian support order. This is the
    # construction the reported figures were produced with; see quadratize().
    coA = {frozenset(k for k in range(Q_CAN) if (int(s) >> k) & 1):
           float(a_can[s]) for s in nz}
    massA = sum(abs(v) for v in coA.values())
    check("total absolute coefficient mass", massA, 1.5501e7, tol=500)

    qF, nF, auxF, MF = quadratize(coA, Q_CAN, progress=True)
    check("penalty weight M", MF, 1.5501e8, tol=5e3)
    check("ancillas A", len(auxF), 367)
    check("total variables", nF, 382)
    check("maximum degree after quadratization",
          max(len(S) for S in qF if len(S) > 0), 2)

    offF = sum(1 for S in qF if len(S) == 2)
    totF = nF * (nF - 1) // 2
    check("nonzero off-diagonal pairs", offF, 28135)
    check("possible off-diagonal pairs", totF, 72771)
    check("matrix density (%)", 100 * offF / totF, 38.66, tol=0.005)
    check("couplings contributed by the Rosenberg penalties", 3 * len(auxF), 1101)
    magsQ = np.abs(np.array([v for S, v in qF.items() if len(S) > 0]))
    check("smallest QUBO coefficient magnitude", magsQ.min(), 0.33, tol=0.005)
    check("largest QUBO coefficient magnitude", magsQ.max(), 4.65e8, tol=5e5)

    constF, hF, JF = qubo_to_ising(qF, nF)
    nfF = int(np.count_nonzero(np.abs(hF) > 1e-9))
    magsI = np.concatenate([np.abs(hF[np.abs(hF) > 1e-9]),
                            np.abs(np.array(list(JF.values())))])
    check("Ising Pauli terms", 1 + nfF + len(JF), 28518)
    check("Ising additive constant", constF, 4.27e10, tol=5e7)
    check("smallest Ising coefficient magnitude", magsI.min(), 0.0825, tol=5e-5)
    check("largest Ising coefficient magnitude", magsI.max(), 7.36e8, tol=5e5)


# --------------------------------------------------------------------------
# Part: vqe (Supplementary Note 17)
# --------------------------------------------------------------------------
def build_ansatz(n_qubits, reps):
    """Ry layers with a linear CX ladder, no entangler after the last layer."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    theta = ParameterVector("t", n_qubits * (reps + 1))
    qc = QuantumCircuit(n_qubits)
    k = 0
    for r in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(theta[k], q)
            k += 1
        if r < reps:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc, theta


def probabilities(circuit, params):
    from qiskit.quantum_info import Statevector
    bound = circuit.assign_parameters(params)
    return np.abs(Statevector.from_instruction(bound).data) ** 2


def make_cost(circuit, energies, alpha=1.0, history=None):
    """alpha = 1 gives the mean, alpha < 1 the conditional value at risk."""
    order = np.argsort(energies)
    e_sorted = energies[order]

    def cost(params):
        p = probabilities(circuit, params)[order]
        if alpha >= 1.0:
            val = float(p @ e_sorted)
        else:
            c = np.cumsum(p)
            k = min(int(np.searchsorted(c, alpha)) + 1, p.size)
            w = p[:k].copy()
            excess = w.sum() - alpha
            if excess > 0:
                w[-1] -= excess
            val = float(w @ e_sorted[:k] / alpha)
        if history is not None:
            history.append(val)
        return val
    return cost


def evaluate(circuit, params, energies, ground_mask, feasible_mask):
    p = probabilities(circuit, params)
    best = int(np.argmax(p))
    return {"mean_energy": float(p @ energies),
            "ground_overlap": float(p[ground_mask].sum()),
            "modal_state_is_ground": bool(ground_mask[best]),
            "modal_state_energy": float(energies[best]),
            "feasible_mass": float(p[feasible_mask].sum())}


def run_once(circuit, n_params, energies, ground_mask, feasible_mask,
             alpha, seed, maxiter):
    from scipy.optimize import minimize
    x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, n_params)
    hist = []
    cost = make_cost(circuit, energies, alpha=alpha, history=hist)
    t0 = time.perf_counter()
    res = minimize(cost, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-8})
    m = evaluate(circuit, res.x, energies, ground_mask, feasible_mask)
    m.update(objective_value=float(res.fun), n_evals=len(hist),
             seconds=time.perf_counter() - t0, seed=seed, alpha=alpha,
             params=res.x, history=np.array(hist))
    return m


def evals_to_solution(m, target, tol=1e-4):
    hit = np.flatnonzero(np.abs(np.asarray(m["history"]) - target) < tol)
    return int(hit[0]) + 1 if hit.size else None


def part_vqe(can, raw, rng):
    e_total, feasible = can["e_total"], can["feasible"]
    e_min = e_total.min()
    ground = np.isclose(e_total, e_min)
    e_raw, feas_raw = raw["e_total"], raw["feasible"]
    ground_raw = np.isclose(e_raw, e_raw.min())

    ansatz, theta = build_ansatz(Q_CAN, REPS)
    ops = dict(ansatz.count_ops())
    check("Ry gates equal n(r+1)", ops.get("ry", 0), Q_CAN * (REPS + 1))
    check("CX gates equal (n-1)r", ops.get("cx", 0), (Q_CAN - 1) * REPS)
    check("ansatz parameters", len(theta), 60)

    print("\n=== Null references (Note 17) ===")
    check("uniform-superposition ground overlap",
          ground.sum() / 2 ** Q_CAN, 6.104e-5, tol=5e-9)
    rand = rng.uniform(0, 2 * np.pi, len(theta))
    m_rand = evaluate(ansatz, rand, e_total, ground, feasible)
    check("random-parameter ground overlap", m_rand["ground_overlap"],
          1.806e-5, tol=5e-9)

    print("\n=== Variational recovery, five seeds (Note 17) ===")
    results = {}
    for label, alpha in (("mean", 1.0), ("CVaR alpha=0.1", 0.1)):
        runs = [run_once(ansatz, len(theta), e_total, ground, feasible,
                         alpha, s, 2000) for s in range(5)]
        results[label] = runs
        ov = np.array([m["ground_overlap"] for m in runs])
        hit = np.mean([m["modal_state_is_ground"] for m in runs])
        print(f"  {label:16s} success {hit:.0%}  "
              f"overlap {ov.mean():.4f} +/- {ov.std():.4f}  "
              f"evals {np.mean([m['n_evals'] for m in runs]):.0f}")

    cv = results["CVaR alpha=0.1"]
    check("CVaR success rate",
          np.mean([m["modal_state_is_ground"] for m in cv]), 1.0)
    check("CVaR ground overlap",
          np.mean([m["ground_overlap"] for m in cv]), 0.1168, tol=0.0005)
    check("CVaR overlap spread",
          np.std([m["ground_overlap"] for m in cv]), 0.0204, tol=0.0005)
    check("CVaR mean evaluations",
          np.mean([m["n_evals"] for m in cv]), 949, tol=0.5)
    mn = results["mean"]
    check("mean-aggregation success rate",
          np.mean([m["modal_state_is_ground"] for m in mn]), 0.2)
    check("mean-aggregation ground overlap",
          np.mean([m["ground_overlap"] for m in mn]), 0.0569, tol=0.0005)

    print("\n=== CVaR threshold sweep, Table S10 (Note 17) ===")
    sweep_exp = {0.02: (1.0, 0.6, 0.0229), 0.05: (1.0, 1.0, 0.0525),
                 0.10: (1.0, 1.0, 0.1168), 0.20: (0.8, 0.8, 0.1807),
                 0.35: (0.2, 0.2, 0.0721), 0.50: (0.0, 0.0, 0.0260)}
    sweep = {0.10: cv, 1.0: mn}
    for a in (0.02, 0.05, 0.20, 0.35, 0.50):
        sweep[a] = [run_once(ansatz, len(theta), e_total, ground, feasible,
                             a, s, 2000) for s in range(5)]
    for a, (sat_e, hit_e, ov_e) in sweep_exp.items():
        runs = sweep[a]
        sat = np.mean([np.isclose(m["objective_value"], e_min, atol=1e-4)
                       for m in runs])
        hit = np.mean([m["modal_state_is_ground"] for m in runs])
        ov = np.mean([m["ground_overlap"] for m in runs])
        check(f"alpha {a:.2f} saturated fraction", sat, sat_e)
        check(f"alpha {a:.2f} success fraction", hit, hit_e)
        check(f"alpha {a:.2f} ground overlap", ov, ov_e, tol=0.0005)

    print("\n=== Register comparison, three shared seeds (Note 17) ===")
    ansatz_raw, theta_raw = build_ansatz(Q_RAW, REPS)
    runs_raw = [run_once(ansatz_raw, len(theta_raw), e_raw, ground_raw,
                         feas_raw, 0.1, s, 1500) for s in range(3)]
    can_sub = [m for m in cv if m["seed"] in (0, 1, 2)]

    check("canonical success, three seeds",
          np.mean([m["modal_state_is_ground"] for m in can_sub]), 1.0)
    check("unreduced success, three seeds",
          np.mean([m["modal_state_is_ground"] for m in runs_raw]), 1.0)
    check("canonical ground overlap",
          np.mean([m["ground_overlap"] for m in can_sub]), 0.1265, tol=0.0005)
    check("unreduced ground overlap",
          np.mean([m["ground_overlap"] for m in runs_raw]), 0.1078, tol=0.0005)
    check("unreduced objective reaches the ground energy",
          sum(np.isclose(m["objective_value"], e_min, atol=1e-4)
              for m in runs_raw), 3)
    print(f"  canonical evaluations to first reach the ground energy: "
          f"{sorted(evals_to_solution(m, e_min) for m in can_sub)}")
    print(f"  unreduced evaluations to first reach the ground energy: "
          f"{sorted(evals_to_solution(m, e_min) for m in runs_raw)}")

    params = {}
    for label, runs in results.items():
        for m in runs:
            params[f"canonical_{label}_seed{m['seed']}"] = \
                list(map(float, m["params"]))
    for m in runs_raw:
        params[f"unreduced_CVaR_seed{m['seed']}"] = list(map(float, m["params"]))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "optimized_parameters_regenerated.json"),
              "w") as f:
        json.dump(params, f, indent=1)
    print("  wrote data/optimized_parameters_regenerated.json")


# --------------------------------------------------------------------------
# Part: hardware (Supplementary Note 17)
# --------------------------------------------------------------------------
def counts_to_probs(counts, n_qubits):
    """Qiskit keys are most-significant-qubit first, and int(bs, 2) already
    gives the integer whose bit k is qubit k. No reversal."""
    p = np.zeros(2 ** n_qubits)
    total = sum(counts.values())
    for bs, c in counts.items():
        p[int(bs, 2)] += c / total
    return p


def pick(store, label):
    """Fetch a stored distribution by error-suppression label.

    The archives were written at different points with the label either
    verbatim ("twirl only") or with spaces and the plus sign stripped
    ("twirl_only"), so match on the alphanumeric characters alone.
    """
    want = "".join(ch for ch in label.lower() if ch.isalnum())
    for k in store:
        if "".join(ch for ch in k.lower() if ch.isalnum()) == want:
            return store[k]
    raise KeyError(f"{label!r} not among {sorted(store)}")


def part_hardware(can, raw):
    e_total, feasible = can["e_total"], can["feasible"]
    ground = np.isclose(e_total, e_total.min())
    e_raw, feas_raw = raw["e_total"], raw["feasible"]
    ground_raw = np.isclose(e_raw, e_raw.min())
    phys = e_total < min(LAMBDA_PLANE, LAMBDA_SA)

    needed = ["hardware_probs.npz", "hardware_probs_20q.npz",
              "hardware_repeats.npz", "optimized_parameters.json"]
    missing = [n for n in needed if not os.path.exists(os.path.join(DATA, n))]
    if missing:
        raise SystemExit(f"missing from {DATA}: {', '.join(missing)}")

    hw = dict(np.load(os.path.join(DATA, "hardware_probs.npz")))
    hw20 = dict(np.load(os.path.join(DATA, "hardware_probs_20q.npz")))
    rep = dict(np.load(os.path.join(DATA, "hardware_repeats.npz")))
    with open(os.path.join(DATA, "optimized_parameters.json")) as f:
        params = json.load(f)

    ansatz, theta = build_ansatz(Q_CAN, REPS)
    ansatz_raw, theta_raw = build_ansatz(Q_RAW, REPS)
    p_sim = probabilities(ansatz, params["canonical_cvar_alpha0.1_seed2"])
    p_sim20 = probabilities(ansatz_raw, params["unreduced_cvar_alpha0.1_seed1"])

    print("\n=== Simulator references (Note 17) ===")
    check("canonical simulator ground probability", p_sim[ground].sum(),
          0.1568, tol=5e-5)
    check("unreduced simulator ground probability", p_sim20[ground_raw].sum(),
          0.1205, tol=5e-5)
    check("canonical simulator probability mass", p_sim[phys].sum(),
          0.8716, tol=5e-5)
    check("unreduced simulator probability mass", p_sim20[feas_raw].sum(),
          0.4345, tol=5e-5)

    print("\n=== Pooled hardware runs, twirling only (Table S13) ===")
    g15 = np.array([rep[k][ground].sum()
                    for k in sorted(rep) if k.startswith("canonical")])
    g20 = np.array([rep[k][ground_raw].sum()
                    for k in sorted(rep) if k.startswith("unreduced")])
    m15 = np.array([rep[k][phys].sum()
                    for k in sorted(rep) if k.startswith("canonical")])
    m20 = np.array([rep[k][feas_raw].sum()
                    for k in sorted(rep) if k.startswith("unreduced")])
    check("canonical runs", len(g15), 3)
    check("unreduced runs", len(g20), 3)
    check("canonical hardware ground probability", g15.mean(), 0.0919, tol=5e-5)
    check("canonical spread", g15.std(ddof=1), 0.0084, tol=5e-5)
    check("unreduced hardware ground probability", g20.mean(), 0.0593, tol=5e-5)
    check("unreduced spread", g20.std(ddof=1), 0.0015, tol=5e-5)
    check("canonical probability mass", m15.mean(), 0.8491, tol=5e-5)
    check("unreduced probability mass", m20.mean(), 0.3656, tol=5e-5)
    check("ratio of ground probabilities", g15.mean() / g20.mean(), 1.55,
          tol=0.005)
    check("ratio of probability masses", m15.mean() / m20.mean(), 2.32,
          tol=0.005)
    pooled = np.sqrt(g15.std(ddof=1) ** 2 + g20.std(ddof=1) ** 2)
    check("separation in pooled standard deviations",
          (g15.mean() - g20.mean()) / pooled, 3.82, tol=0.005)
    check("modal bitstring is a ground state, canonical",
          sum(bool(ground[int(np.argmax(rep[k]))])
              for k in sorted(rep) if k.startswith("canonical")), 3)
    check("modal bitstring is a ground state, unreduced",
          sum(bool(ground_raw[int(np.argmax(rep[k]))])
              for k in sorted(rep) if k.startswith("unreduced")), 3)

    print("\n=== Noise structure (Note 17) ===")
    top4 = np.argsort(p_sim)[::-1][:4]
    ratios = pick(hw, "twirl only")[top4] / p_sim[top4]
    check("top four keep their order on hardware",
          int(np.array_equal(np.argsort(pick(hw, "twirl only"))[::-1][:4], top4)), 1)
    check("minimum attenuation over the top four", ratios.min(), 0.53,
          tol=0.005)
    check("maximum attenuation over the top four", ratios.max(), 0.57,
          tol=0.005)
    mask = p_sim > 1e-3
    check("median attenuation above 1e-3",
          np.median(pick(hw, "twirl only")[mask] / p_sim[mask]), 0.718, tol=0.0005)
    check("decoupling ratio, canonical",
          pick(hw, "DD + twirl")[ground].sum() / pick(hw, "twirl only")[ground].sum(),
          0.933, tol=0.0005)
    check("decoupling ratio, unreduced",
          pick(hw20, "DD + twirl")[ground_raw].sum()
          / pick(hw20, "twirl only")[ground_raw].sum(), 0.892, tol=0.0005)

    print("\n=== Cumulative distributions (Note 17) ===")
    med15 = [rep[k] for k in sorted(rep) if k.startswith("canonical")][
        int(np.argsort(g15)[1])]
    med20 = [rep[k] for k in sorted(rep) if k.startswith("unreduced")][
        int(np.argsort(g20)[1])]
    check("negative-energy mass, canonical median run (%)",
          100 * med15[e_total < 0].sum(), 48.3, tol=0.05)
    check("negative-energy mass, unreduced median run (%)",
          100 * med20[e_raw < 0].sum(), 34.4, tol=0.05)
    check("tail beyond 250, canonical median run (%)",
          100 * med15[e_total > 250].sum(), 6.76, tol=0.005)
    check("tail beyond 250, unreduced median run (%)",
          100 * med20[e_raw > 250].sum(), 10.18, tol=0.005)

    for name in ("hardware_jobs.json", "hardware_jobs_20q.json",
                 "hardware_repeats.json"):
        with open(os.path.join(DATA, name)) as f:
            meta = json.load(f)
        print(f"  {name}: backend {meta.get('backend')}, "
              f"shots {meta.get('shots')}")


def submit_to_hardware(params, n_qubits, backend_name="ibm_marrakesh",
                       shots=100_000, dynamical_decoupling=False):
    """Submit one circuit to IBM Quantum. Requires your own account.

    Not called by any part. Provided so that the transpiled circuits behind
    the stored counts can be rebuilt and resubmitted. The sampled frequencies
    will differ, since they depend on the device calibration at submission.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    qc, _ = build_ansatz(n_qubits, REPS)
    bound = qc.assign_parameters(params)
    bound.measure_all()
    backend = QiskitRuntimeService().backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=RNG_SEED)
    isa = pm.run(bound)
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = dynamical_decoupling
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    return isa, sampler.run([isa]).job_id()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all",
                    choices=["exact", "symmetry", "hamiltonian", "vqe",
                             "hardware", "all"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.part == "symmetry":
        part_symmetry()
    else:
        can, raw, rng = prepare()
        if args.part in ("exact", "all"):
            part_exact(can, raw)
        if args.part == "all":
            part_symmetry()
        if args.part in ("hamiltonian", "all"):
            part_hamiltonian(can)
        if args.part == "vqe":
            part_vqe(can, raw, rng)
        if args.part in ("hardware", "all"):
            part_hardware(can, raw)

    print(f"\nwall time {time.perf_counter() - t0:.0f} s")
    sys.exit(1 if summarise() else 0)


if __name__ == "__main__":
    main()
