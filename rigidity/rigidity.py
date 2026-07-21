#!/usr/bin/env python3
"""
Task 1: Infinitesimal rigidity of Alpoge's 3D Keller counterexample.

Linearized Keller equation at F:  d/dt|_0 det J(F+tG) = tr(adj(JF) . JG).
Two versions:
  Sol0 : tr(adj(JF).JG) = 0            (det stays exactly -2 to first order)
  SolC : tr(adj(JF).JG) = const        (det stays constant-in-x to first order)

Degree convention: G = (G1,G2,G3), each component of TOTAL degree <= 7
(deg F = max deg F_i = 7). Unknowns: 3 * C(10,3) = 360 coefficients.

Trivial tangent space:
  A-side: d/dt (id+tA)oF = A o F, condition div A = 0 (Sol0) / div A = const (SolC).
    Constraint deg(AoF) <= 7 forces A affine (monomials of F have degrees 7,6,4;
    any nonlinear monomial in (F1,F2,F3) has degree >= 8).
    We still allow A of degree <= 2 and intersect with deg<=7 to catch cancellations.
  B-side: d/dt F o (id+tB) = JF . B, condition div B = 0 / const,
    intersected with {deg(JF.B) <= 7}.  B allowed up to degree 7.

All linear algebra mod p = 32003 (rechecked mod 1000003).
"""
import sympy as sp
import numpy as np
from itertools import product as iproduct
import sys

x, y, z = sp.symbols('x y z')
VARS = (x, y, z)
u = 1 + x*y
F1 = u**3*z + y**2*u*(4 + 3*x*y)
F2 = y + 3*x*u**2*z + 3*x*y**2*(4 + 3*x*y)
F3 = 2*x - 3*x**2*y - x**3*z
F = [sp.expand(F1), sp.expand(F2), sp.expand(F3)]

J = sp.Matrix(3, 3, lambda i, j: sp.diff(F[i], VARS[j]))
detJ = sp.expand(J.det())
print("deg F components:", [sp.total_degree(sp.Poly(f, x, y, z)) for f in F])
print("det JF =", detJ)
assert detJ == -2

adj = J.adjugate()  # J * adj = det(J) * I ; adj = cofactor^T
# check
chk = sp.expand(J * adj - detJ * sp.eye(3))
assert chk == sp.zeros(3, 3)

def poly_to_dict(expr):
    p = sp.Poly(sp.expand(expr), x, y, z)
    return {m: int(c) for m, c in zip(p.monoms(), p.coeffs())}

ADJ = [[poly_to_dict(adj[i, j]) for j in range(3)] for i in range(3)]

def monomials_upto(d):
    return [(a, b, c) for a in range(d+1) for b in range(d+1-a) for c in range(d+1-a-b)]

def mono_index(monos):
    return {m: k for k, m in enumerate(monos)}

def dict_mul_mono(pd, mono, coeff):
    """multiply poly-dict by coeff * x^mono"""
    out = {}
    for m, c in pd.items():
        mm = (m[0]+mono[0], m[1]+mono[1], m[2]+mono[2])
        out[mm] = out.get(mm, 0) + c*coeff
    return out

def dmono(mono, j):
    """d/dx_j of monomial -> (coeff, mono) or None"""
    e = mono[j]
    if e == 0:
        return None
    m = list(mono); m[j] -= 1
    return e, tuple(m)

# ---------- build linearized-Keller matrix ----------
DG = 7                      # deg bound on G components
G_monos = monomials_upto(DG)
nGm = len(G_monos)          # 120
NG = 3 * nGm                # 360
# tr(adj . JG) = sum_{i,j} adj[i,j] * dG_j/dx_i ; result degree <= 11 + 6 = 17
DEQ = 17
EQ_monos = monomials_upto(DEQ)
EQ_idx = mono_index(EQ_monos)
print("unknowns:", NG, "; eq rows (monomials deg<=%d):" % DEQ, len(EQ_monos))

maxadjdeg = max(max(sum(m) for m in ADJ[i][j]) if ADJ[i][j] else 0
                for i in range(3) for j in range(3))
print("max deg of adj entries:", maxadjdeg)

def build_lin_matrix(p):
    M = np.zeros((len(EQ_monos), NG), dtype=np.int64)
    for j in range(3):              # component of G
        for k, mono in enumerate(G_monos):
            col = j * nGm + k
            for i in range(3):      # derivative variable
                d = dmono(mono, i)
                if d is None:
                    continue
                cf, dm = d
                contrib = dict_mul_mono(ADJ[i][j], dm, cf)
                for mm, c in contrib.items():
                    M[EQ_idx[mm], col] = (M[EQ_idx[mm], col] + c) % p
    return M

def rref_rank(M, p, return_basis=False):
    """Gaussian elimination mod p; returns rank (and kernel basis if asked)."""
    A = M % p
    nrows, ncols = A.shape
    A = A.copy()
    pivots = []
    r = 0
    for c in range(ncols):
        # find pivot
        piv = None
        for rr in range(r, nrows):
            if A[rr, c] % p:
                piv = rr; break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p-2, p)
        A[r] = (A[r] * inv) % p
        col = A[:, c].copy()
        col[r] = 0
        A = (A - np.outer(col, A[r])) % p
        pivots.append(c)
        r += 1
        if r == nrows:
            break
    rank = r
    if not return_basis:
        return rank
    # kernel basis
    pivset = set(pivots)
    free = [c for c in range(ncols) if c not in pivset]
    K = np.zeros((len(free), ncols), dtype=np.int64)
    for idx, fc in enumerate(free):
        K[idx, fc] = 1
        for ri, pc in enumerate(pivots):
            K[idx, pc] = (-A[ri, fc]) % p
    return rank, K

# ---------- trivial space ----------
D7_monos = monomials_upto(7)
D7_idx = mono_index(D7_monos)
n7 = len(D7_monos)          # 120 ; G-vector length 3*120 = 360

Fd = [poly_to_dict(f) for f in F]

def gvec_from_polydicts(comps, p):
    v = np.zeros(3 * n7, dtype=np.int64)
    for i in range(3):
        for m, c in comps[i].items():
            if sum(m) > 7:
                raise ValueError("degree overflow in trivial vector")
            v[i * n7 + D7_idx[m]] = c % p
    return v

def build_trivial(p, const_version):
    vecs = []
    tags = []
    # ---- A-side: A of degree <= 2 in target coords (w1,w2,w3), div A = 0 (or const)
    A_monos = monomials_upto(2)   # 10 monomials per component, 30 params
    # G = A o F : component i = sum over monomials  a_{i,m} * F^m ; keep only deg<=7 images
    FdP = {}  # cache powers F^m as dicts
    def Fpow(m):
        if m in FdP:
            return FdP[m]
        out = {(0, 0, 0): 1}
        for t in range(3):
            for _ in range(m[t]):
                new = {}
                for mo, c in out.items():
                    for mo2, c2 in Fd[t].items():
                        mm = (mo[0]+mo2[0], mo[1]+mo2[1], mo[2]+mo2[2])
                        new[mm] = new.get(mm, 0) + c*c2
                out = new
        FdP[m] = out
        return out
    # unknowns a_{i,m}: build matrix of constraints:
    #   rows: deg>7 coefficients of (A o F)_i  +  div A coefficients
    # then kernel -> admissible A's -> G vectors
    nA = 3 * len(A_monos)
    rows = []
    # divergence rows: div A = sum_i dA_i/dw_i ; polynomial in w of deg <=1
    # coefficients on monomials of w: for const_version drop the constant row
    div_monos = monomials_upto(1)
    for dm_ in div_monos:
        if const_version and dm_ == (0, 0, 0):
            continue
        row = np.zeros(nA, dtype=np.int64)
        for i in range(3):
            for k, am in enumerate(A_monos):
                d = dmono(am, i)
                if d and d[1] == dm_:
                    row[i * len(A_monos) + k] = d[0] % p
        rows.append(row)
    # degree-overflow rows: (AoF)_i coefficients on monomials of degree 8..14
    # max deg of F^m for m deg<=2: 14
    hi_monos = [m for m in monomials_upto(14) if sum(m) > 7]
    hi_idx = mono_index(hi_monos)
    ov = np.zeros((3 * len(hi_monos), nA), dtype=np.int64)
    for i in range(3):
        for k, am in enumerate(A_monos):
            fp = Fpow(am)
            for mm, c in fp.items():
                if sum(mm) > 7:
                    ov[i * len(hi_monos) + hi_idx[mm], i * len(A_monos) + k] = c % p
    Mat = np.vstack([np.array(rows, dtype=np.int64), ov]) % p
    rk, K = rref_rank(Mat, p, return_basis=True)
    for kv in K:
        comps = [{}, {}, {}]
        for i in range(3):
            for k, am in enumerate(A_monos):
                c = int(kv[i * len(A_monos) + k])
                if c:
                    fp = Fpow(am)
                    for mm, cc in fp.items():
                        comps[i][mm] = (comps[i].get(mm, 0) + c*cc) % p
        vecs.append(gvec_from_polydicts(comps, p))
        tags.append('A')
    nA_side = len(K)
    # ---- B-side: B deg <= 7, div B = 0 (or const), deg(JF.B) <= 7
    B_monos = monomials_upto(7)
    nB = 3 * len(B_monos)
    dF = [[poly_to_dict(sp.diff(F[i], VARS[j])) for j in range(3)] for i in range(3)]
    rows = []
    divB_monos = monomials_upto(6)
    for dm_ in divB_monos:
        if const_version and dm_ == (0, 0, 0):
            continue
        row = np.zeros(nB, dtype=np.int64)
        for j in range(3):
            for k, bm in enumerate(B_monos):
                d = dmono(bm, j)
                if d and d[1] == dm_:
                    row[j * len(B_monos) + k] = d[0] % p
        rows.append(row)
    # overflow rows for (JF.B)_i = sum_j dF_i/dx_j * B_j, deg <= 6+7 = 13
    hi_monos = [m for m in monomials_upto(13) if sum(m) > 7]
    hi_idx = mono_index(hi_monos)
    ov = np.zeros((3 * len(hi_monos), nB), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k, bm in enumerate(B_monos):
                contrib = dict_mul_mono(dF[i][j], bm, 1)
                for mm, c in contrib.items():
                    if sum(mm) > 7:
                        ov[i * len(hi_monos) + hi_idx[mm], j * len(B_monos) + k] = \
                            (ov[i * len(hi_monos) + hi_idx[mm], j * len(B_monos) + k] + c) % p
    Mat = np.vstack([np.array(rows, dtype=np.int64), ov]) % p
    rk, K = rref_rank(Mat, p, return_basis=True)
    for kv in K:
        comps = [{}, {}, {}]
        for i in range(3):
            for j in range(3):
                for k, bm in enumerate(B_monos):
                    c = int(kv[j * len(B_monos) + k])
                    if c:
                        contrib = dict_mul_mono(dF[i][j], bm, c)
                        for mm, cc in contrib.items():
                            comps[i][mm] = (comps[i].get(mm, 0) + cc) % p
        comps = [{m: c % p for m, c in comp.items() if c % p} for comp in comps]
        # drop deg>7 (they are zero by construction)
        comps = [{m: c for m, c in comp.items() if sum(m) <= 7} for comp in comps]
        vecs.append(gvec_from_polydicts(comps, p))
        tags.append('B')
    nB_side = len(K)
    V = np.array(vecs, dtype=np.int64)
    return V, nA_side, nB_side

def run(p):
    print(f"\n===== prime p = {p} =====")
    M = build_lin_matrix(p)
    const_row = EQ_idx[(0, 0, 0)]
    rank0 = rref_rank(M, p)
    Mc = np.delete(M, const_row, axis=0)
    rankC = rref_rank(Mc, p)
    s0 = NG - rank0
    sC = NG - rankC
    print(f"dim Sol0 (tr=0)      = {s0}   (rank {rank0} of {M.shape})")
    print(f"dim SolC (tr=const)  = {sC}")
    for const_version in (False, True):
        V, na, nb = build_trivial(p, const_version)
        label = "const" if const_version else "zero"
        rt = rref_rank(V, p)
        # sanity: trivial vectors satisfy the equation
        R = (M @ V.T) % p
        if const_version:
            R = np.delete(R, const_row, axis=0)
        ok = not R.any()
        print(f"[div={label}] A-side kernel dim {na}, B-side kernel dim {nb}, "
              f"dim span(trivial) = {rt}, trivial subset of Sol: {ok}")
        sol = s0 if not const_version else sC
        print(f"[div={label}] dim Sol = {sol} vs dim Triv = {rt}  ->  "
              + ("RIGID (equal)" if sol == rt else f"NOT RIGID: {sol - rt} extra direction(s)"))
    return s0, sC

if __name__ == "__main__":
    run(32003)
    run(1000003)
