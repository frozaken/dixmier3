#!/usr/bin/env python3
"""
Phase 2: characterize the 2 extra first-order directions and test
second/third-order integrability inside the degree<=7 Keller variety.
"""
import sympy as sp
import numpy as np
import sys

p = 1000003

x, y, z = sp.symbols('x y z')
VARS = (x, y, z)
u = 1 + x*y
F = [sp.expand(u**3*z + y**2*u*(4 + 3*x*y)),
     sp.expand(y + 3*x*u**2*z + 3*x*y**2*(4 + 3*x*y)),
     sp.expand(2*x - 3*x**2*y - x**3*z)]
J = sp.Matrix(3, 3, lambda i, j: sp.diff(F[i], VARS[j]))
adj = J.adjugate()

def poly_to_dict(expr):
    q = sp.Poly(sp.expand(expr), x, y, z)
    return {m: int(c) for m, c in zip(q.monoms(), q.coeffs())}

ADJ = [[poly_to_dict(adj[i, j]) for j in range(3)] for i in range(3)]
dF = [[poly_to_dict(sp.diff(F[i], VARS[j])) for j in range(3)] for i in range(3)]

def monomials_upto(d):
    return [(a, b, c) for a in range(d+1) for b in range(d+1-a) for c in range(d+1-a-b)]

def midx(monos):
    return {m: k for k, m in enumerate(monos)}

def dmono(mono, j):
    e = mono[j]
    if e == 0:
        return None
    m = list(mono); m[j] -= 1
    return e, tuple(m)

def pmulm(pd, mono, coeff):
    out = {}
    for m, c in pd.items():
        mm = (m[0]+mono[0], m[1]+mono[1], m[2]+mono[2])
        out[mm] = (out.get(mm, 0) + c*coeff) % p
    return out

def pmul(pa, pb):
    out = {}
    for m, c in pa.items():
        for m2, c2 in pb.items():
            mm = (m[0]+m2[0], m[1]+m2[1], m[2]+m2[2])
            out[mm] = (out.get(mm, 0) + c*c2) % p
    return {m: c for m, c in out.items() if c}

def padd(*ps):
    out = {}
    for q in ps:
        for m, c in q.items():
            out[m] = (out.get(m, 0) + c) % p
    return {m: c for m, c in out.items() if c}

def pscale(q, s):
    return {m: (c*s) % p for m, c in q.items() if (c*s) % p}

G_monos = monomials_upto(7)
nGm = len(G_monos)
NG = 3 * nGm
EQ_monos = monomials_upto(17)
EQ_idx = midx(EQ_monos)

def build_lin_matrix():
    M = np.zeros((len(EQ_monos), NG), dtype=np.int64)
    for j in range(3):
        for k, mono in enumerate(G_monos):
            col = j * nGm + k
            for i in range(3):
                d = dmono(mono, i)
                if d is None:
                    continue
                cf, dm = d
                contrib = pmulm(ADJ[i][j], dm, cf)
                for mm, c in contrib.items():
                    M[EQ_idx[mm], col] = (M[EQ_idx[mm], col] + c) % p
    return M % p

def rref(Min):
    A = Min % p
    nrows, ncols = A.shape
    A = A.copy()
    pivots = []
    r = 0
    for c in range(ncols):
        piv = None
        nz = np.nonzero(A[r:, c] % p)[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p-2, p)
        A[r] = (A[r] * inv) % p
        col = A[:, c].copy(); col[r] = 0
        A = (A - np.outer(col, A[r])) % p
        pivots.append(c)
        r += 1
        if r == nrows:
            break
    return A, pivots

def kernel_basis(Min):
    A, pivots = rref(Min)
    ncols = Min.shape[1]
    pivset = set(pivots)
    free = [c for c in range(ncols) if c not in pivset]
    K = np.zeros((len(free), ncols), dtype=np.int64)
    for i, fc in enumerate(free):
        K[i, fc] = 1
        for ri, pc in enumerate(pivots):
            K[i, pc] = (-A[ri, fc]) % p
    return K

def rank_of(Min):
    return len(rref(Min)[1])

M = build_lin_matrix()
const_row = EQ_idx[(0, 0, 0)]
K0 = kernel_basis(M)                      # Sol0, dim 32
print("dim Sol0 =", K0.shape[0])

# ---------- trivial span (div=0 version), same as rigidity.py ----------
def trivial_span():
    vecs = []
    # A-side: affine traceless + translations  ->  G = M.F + v with tr M = 0
    Fd = [poly_to_dict(f) for f in F]
    basisA = []
    for i in range(3):
        for k in range(3):
            comps = [{}, {}, {}]
            comps[i] = dict(Fd[k])
            basisA.append((comps, (i, k)))
        comps = [{}, {}, {}]
        comps[i] = {(0, 0, 0): 1}
        basisA.append((comps, (i, 'v')))
    # impose tr=0: use differences M_ii - M_jj and off-diagonals and translations
    D7_idx = midx(G_monos)
    def gvec(comps):
        v = np.zeros(NG, dtype=np.int64)
        for i in range(3):
            for m, c in comps[i].items():
                v[i * nGm + D7_idx[m]] = c % p
        return v
    raw = {tag: gvec(c) for c, tag in basisA}
    for i in range(3):
        for k in range(3):
            if i != k:
                vecs.append(raw[(i, k)])
    vecs.append((raw[(0, 0)] - raw[(1, 1)]) % p)
    vecs.append((raw[(1, 1)] - raw[(2, 2)]) % p)
    for i in range(3):
        vecs.append(raw[(i, 'v')])
    # B-side: B deg <= 7, div B = 0, deg(JF.B) <= 7  (same as before)
    B_monos = monomials_upto(7)
    nB = 3 * len(B_monos)
    rows = []
    for dm_ in monomials_upto(6):
        row = np.zeros(nB, dtype=np.int64)
        for j in range(3):
            for k, bm in enumerate(B_monos):
                d = dmono(bm, j)
                if d and d[1] == dm_:
                    row[j * len(B_monos) + k] = d[0] % p
        rows.append(row)
    hi = [m for m in monomials_upto(13) if sum(m) > 7]
    hi_idx = midx(hi)
    ov = np.zeros((3 * len(hi), nB), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k, bm in enumerate(B_monos):
                contrib = pmulm(dF[i][j], bm, 1)
                for mm, c in contrib.items():
                    if sum(mm) > 7:
                        r_ = i * len(hi) + hi_idx[mm]
                        ov[r_, j * len(B_monos) + k] = (ov[r_, j * len(B_monos) + k] + c) % p
    KB = kernel_basis(np.vstack([np.array(rows), ov]) % p)
    D7_idx = midx(G_monos)
    for kv in KB:
        comps = [{}, {}, {}]
        for i in range(3):
            for j in range(3):
                for k, bm in enumerate(B_monos):
                    c = int(kv[j * len(B_monos) + k])
                    if c:
                        for mm, cc in pmulm(dF[i][j], bm, c).items():
                            comps[i][mm] = (comps[i].get(mm, 0) + cc) % p
        v = np.zeros(NG, dtype=np.int64)
        for i in range(3):
            for m, c in comps[i].items():
                if sum(m) <= 7 and c % p:
                    v[i * nGm + midx(G_monos)[m]] = c % p
        vecs.append(v)
    return np.array(vecs, dtype=np.int64)

T = trivial_span()
rT = rank_of(T)
print("dim Triv0 =", rT)

# extract 2 extra directions: kernel vectors independent modulo T
A_, piv = rref(T)
Tred = A_[:rT]  # row-reduced basis of T
def reduce_mod_T(v):
    w = v % p
    for r_i, c in enumerate(piv):
        w = (w - w[c] * Tred[r_i]) % p
    return w

extras = []
stack = Tred.copy()
for kv in K0:
    cand = np.vstack([stack, kv]) % p
    if rank_of(cand) > stack.shape[0]:
        stack = cand
        extras.append(kv % p)
    if len(extras) == 2:
        break
print("found extras:", len(extras))

# ---------- characterize extras: B = -adj.G/2, check div B, deg B ----------
inv2 = pow(2, p-2, p)
def G_from_vec(v):
    comps = []
    for i in range(3):
        d = {}
        for k, m in enumerate(G_monos):
            c = int(v[i * nGm + k])
            if c:
                d[m] = c % p
        comps.append(d)
    return comps

def B_of_G(Gc):
    # B = -(1/2) adj . G
    B = []
    for i in range(3):
        acc = {}
        for j in range(3):
            acc = padd(acc, pmul(ADJ[i][j], Gc[j]))
        B.append(pscale(acc, (-inv2) % p))
    return B

def divergence(Bc):
    out = {}
    for i in range(3):
        for m, c in Bc[i].items():
            d = dmono(m, i)
            if d:
                cf, dm = d
                out[dm] = (out.get(dm, 0) + c*cf) % p
    return {m: c for m, c in out.items() if c % p}

for t, v in enumerate(extras):
    Gc = G_from_vec(v)
    Bc = B_of_G(Gc)
    degB = [max((sum(m) for m in b), default=-1) for b in Bc]
    dv = divergence(Bc)
    degG = [max((sum(m) for m in g), default=-1) for g in Gc]
    print(f"extra {t}: deg G = {degG}, deg B = {degB}, div B == 0: {len(dv) == 0}")

# also reduced-mod-T representatives (sparser?)
for t, v in enumerate(extras):
    w = reduce_mod_T(v)
    Gc = G_from_vec(w)
    nz = sum(len(g) for g in Gc)
    degG = [max((sum(m) for m in g), default=-1) for g in Gc]
    print(f"extra {t} mod Triv: deg G = {degG}, #terms = {nz}")

np.save('/tmp/surfaceA/extras.npy', np.array(extras))
np.save('/tmp/surfaceA/M.npy', M)
np.save('/tmp/surfaceA/T.npy', T)

# ---------- second-order obstruction ----------
# F_t = F + tG + t^2 H ; t^2 coeff of det: tr(adj.JH) + detJF*e2(X) with X = JF^{-1} JG
# condition: exists H deg<=7, c in C with  L(H) = -detJF*e2(X) + c = 2*e2(X) + c
def JG_of(Gc):
    return [[{} if (d := None) else None for _ in range(3)] for _ in range(3)]  # placeholder

def jac_of(Gc):
    out = [[{} for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for m, c in Gc[i].items():
            for j in range(3):
                d = dmono(m, j)
                if d:
                    cf, dm = d
                    out[i][j][dm] = (out[i][j].get(dm, 0) + c*cf) % p
    return out

def matmul_poly(A_, B_):
    out = [[{} for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc = {}
            for k in range(3):
                acc = padd(acc, pmul(A_[i][k], B_[k][j]))
            out[i][j] = acc
    return out

def mat_trace(A_):
    return padd(A_[0][0], A_[1][1], A_[2][2])

def e2_of(X_):
    trX = mat_trace(X_)
    X2 = matmul_poly(X_, X_)
    trX2 = mat_trace(X2)
    return pscale(padd(pmul(trX, trX), pscale(trX2, p-1)), inv2)

def Xmat_of(Gc):
    # X = JF^{-1} JG = -(1/2) adj . JG
    JGm = jac_of(Gc)
    Xm = matmul_poly(ADJ, JGm)
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]

Mc = np.delete(M, const_row, axis=0)
EQ_monos_c = [m for m in EQ_monos if m != (0, 0, 0)]
EQc_idx = midx(EQ_monos_c)
rank_Mc = rank_of(Mc)
print("rank Mc =", rank_Mc)

def second_order_test(Gc, label):
    Xm = Xmat_of(Gc)
    e2 = e2_of(Xm)
    # RHS: L(H) = 2*e2(X) + c  -> r = 2*e2 restricted, need high-degree part zero
    r = pscale(e2, 2)
    hi_bad = {m: c for m, c in r.items() if sum(m) > 17}
    if hi_bad:
        print(f"[{label}] OBSTRUCTED at order 2: e2 has degree-{max(sum(m) for m in hi_bad)} terms "
              f"outside Im(L) degree range ({len(hi_bad)} monomials)")
        return None
    rv = np.zeros(len(EQ_monos_c), dtype=np.int64)
    for m, c in r.items():
        if m != (0, 0, 0):
            rv[EQc_idx[m]] = c % p
    aug = np.hstack([Mc, rv.reshape(-1, 1)])
    ra = rank_of(aug)
    if ra == rank_Mc:
        print(f"[{label}] order 2: UNOBSTRUCTED (solvable for H)")
        return rv
    else:
        print(f"[{label}] order 2: OBSTRUCTED (rank {rank_Mc} -> {ra})")
        return None

rng = np.random.default_rng(7)
e1v, e2v = extras
tests = [
    (G_from_vec(e1v), "e1"),
    (G_from_vec(e2v), "e2"),
    (G_from_vec((e1v + e2v) % p), "e1+e2"),
]
# generic kernel element
gen = np.zeros(NG, dtype=np.int64)
for kv in K0:
    gen = (gen + int(rng.integers(1, p)) * kv) % p
tests.append((G_from_vec(gen), "generic kernel elt"))
# e1 plus random trivial
tt = np.zeros(NG, dtype=np.int64)
for kv in Tred:
    tt = (tt + int(rng.integers(1, p)) * kv) % p
tests.append((G_from_vec((e1v + tt) % p), "e1 + random trivial"))
tests.append((G_from_vec(tt), "pure random trivial (control)"))

for Gc, label in tests:
    second_order_test(Gc, label)
