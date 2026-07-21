#!/usr/bin/env python3
"""
Phase 3: sanity controls + fine-grained obstruction map.
- affine-orbit directions (provably integrable: a_t o F o b_t, affine families) must pass order 2
- test each B-side trivial basis vector individually
- extend A-side search to deg <= 3 (kernel should stay 11)
"""
import sympy as sp
import numpy as np

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
    return {m: int(c) % p for m, c in zip(q.monoms(), q.coeffs())}

ADJ = [[poly_to_dict(adj[i, j]) for j in range(3)] for i in range(3)]
dF = [[poly_to_dict(sp.diff(F[i], VARS[j])) for j in range(3)] for i in range(3)]
Fd = [poly_to_dict(f) for f in F]

def monomials_upto(d):
    return [(a, b, c) for a in range(d+1) for b in range(d+1-a) for c in range(d+1-a-b)]
def midx(monos):
    return {m: k for k, m in enumerate(monos)}
def dmono(mono, j):
    e = mono[j]
    if e == 0: return None
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
nGm = len(G_monos); NG = 3*nGm
G_idx = midx(G_monos)
M = np.load('/tmp/surfaceA/M.npy')
extras = np.load('/tmp/surfaceA/extras.npy')
EQ_monos = monomials_upto(17)
EQ_idx = midx(EQ_monos)
const_row = EQ_idx[(0,0,0)]
Mc = np.delete(M, const_row, axis=0)
EQ_monos_c = [m for m in EQ_monos if m != (0,0,0)]
EQc_idx = midx(EQ_monos_c)

def rref(Min):
    A = Min % p
    nrows, ncols = A.shape
    A = A.copy(); pivots = []; r = 0
    for c in range(ncols):
        nz = np.nonzero(A[r:, c] % p)[0]
        if len(nz) == 0: continue
        piv = r + nz[0]
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p-2, p)
        A[r] = (A[r]*inv) % p
        col = A[:, c].copy(); col[r] = 0
        A = (A - np.outer(col, A[r])) % p
        pivots.append(c); r += 1
        if r == nrows: break
    return A, pivots
def rank_of(Min): return len(rref(Min)[1])
def kernel_basis(Min):
    A, pivots = rref(Min)
    ncols = Min.shape[1]; pivset = set(pivots)
    free = [c for c in range(ncols) if c not in pivset]
    K = np.zeros((len(free), ncols), dtype=np.int64)
    for i, fc in enumerate(free):
        K[i, fc] = 1
        for ri, pc in enumerate(pivots):
            K[i, pc] = (-A[ri, fc]) % p
    return K

rank_Mc = rank_of(Mc)

inv2 = pow(2, p-2, p)
def G_from_vec(v):
    comps = []
    for i in range(3):
        d = {}
        for k, m in enumerate(G_monos):
            c = int(v[i*nGm+k])
            if c: d[m] = c % p
        comps.append(d)
    return comps
def gvec(comps):
    v = np.zeros(NG, dtype=np.int64)
    for i in range(3):
        for m, c in comps[i].items():
            if c % p:
                v[i*nGm + G_idx[m]] = c % p
    return v
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
    return [[padd(*[pmul(A_[i][k], B_[k][j]) for k in range(3)]) for j in range(3)] for i in range(3)]
def mat_trace(A_): return padd(A_[0][0], A_[1][1], A_[2][2])
def e2_of(X_):
    trX = mat_trace(X_)
    trX2 = mat_trace(matmul_poly(X_, X_))
    return pscale(padd(pmul(trX, trX), pscale(trX2, p-1)), inv2)
def Xmat_of(Gc):
    Xm = matmul_poly(ADJ, jac_of(Gc))
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]

def second_order_test(Gc, label, quiet=False):
    e2 = e2_of(Xmat_of(Gc))
    r = pscale(e2, 2)
    hi_bad = {m: c for m, c in r.items() if sum(m) > 17}
    if hi_bad:
        if not quiet: print(f"[{label}] OBSTRUCTED (deg>17 residue)")
        return False
    rv = np.zeros(len(EQ_monos_c), dtype=np.int64)
    for m, c in r.items():
        if m != (0,0,0): rv[EQc_idx[m]] = c % p
    ra = rank_of(np.hstack([Mc, rv.reshape(-1,1)]))
    ok = (ra == rank_Mc)
    if not quiet: print(f"[{label}] order 2: " + ("UNOBSTRUCTED" if ok else "OBSTRUCTED"))
    return ok

# ---- controls that must pass ----
# (1) G = JF.b, b = constant field (translation): F o (id + t b) exact family
Gc = [padd(*[pmulm(dF[i][j], (0,0,0), 1) for j in [0]]) for i in range(3)]  # b = e_x
second_order_test(Gc, "control JF.e_x (translation)")
# (2) b linear traceless: b = (x, -y, 0)
Gc = [padd(pmul(dF[i][0], {(1,0,0):1}), pmul(dF[i][1], {(0,1,0):p-1})) for i in range(3)]
second_order_test(Gc, "control JF.(x,-y,0)")
# (3) A-side linear traceless: G = (F2, F1, 0)
Gc = [dict(Fd[1]), dict(Fd[0]), {}]
second_order_test(Gc, "control (F2,F1,0)")
# (4) random combo of full affine orbit: A affine tr0 + b affine tr0
rng = np.random.default_rng(11)
def rand_orbit_vec():
    v = np.zeros(NG, dtype=np.int64)
    # A-side: M.F + v, tr M = 0
    Mt = rng.integers(0, p, (3,3)); Mt[2,2] = (-int(Mt[0,0])-int(Mt[1,1])) % p
    vt = rng.integers(0, p, 3)
    comps = [{}, {}, {}]
    for i in range(3):
        for k in range(3):
            comps[i] = padd(comps[i], pscale(Fd[k], int(Mt[i,k])))
        comps[i] = padd(comps[i], {(0,0,0): int(vt[i])})
    v = (v + gvec(comps)) % p
    # b-side: b = Lx + w, tr L = 0
    L = rng.integers(0, p, (3,3)); L[2,2] = (-int(L[0,0])-int(L[1,1])) % p
    w = rng.integers(0, p, 3)
    bfield = [padd({(1,0,0): int(L[j,0])}, {(0,1,0): int(L[j,1])}, {(0,0,1): int(L[j,2])},
                   {(0,0,0): int(w[j])}) for j in range(3)]
    comps = [padd(*[pmul(dF[i][j], bfield[j]) for j in range(3)]) for i in range(3)]
    v = (v + gvec(comps)) % p
    return v
for t in range(3):
    v = rand_orbit_vec()
    second_order_test(G_from_vec(v), f"control random affine-orbit #{t}")

# ---- dim of affine-orbit tangent ----
orbvecs = []
for i in range(3):
    for k in range(3):
        comps = [{}, {}, {}]; comps[i] = dict(Fd[k]); orbvecs.append(gvec(comps))
    comps = [{}, {}, {}]; comps[i] = {(0,0,0):1}; orbvecs.append(gvec(comps))
for jj in range(3):
    for kk in range(4):  # b = x_k or 1
        bf = [{} , {}, {}]
        bf[jj] = {(1,0,0):1} if kk==0 else {(0,1,0):1} if kk==1 else {(0,0,1):1} if kk==2 else {(0,0,0):1}
        comps = [padd(*[pmul(dF[i][j], bf[j]) for j in range(3)]) for i in range(3)]
        orbvecs.append(gvec(comps))
O = np.array(orbvecs, dtype=np.int64)
print("dim affine-orbit tangent (no trace conditions, det=const version):", rank_of(O))
# with trace-0 both sides (det unchanged version): subtract 2
# ---- individual B-side trivial basis vectors ----
B_monos = monomials_upto(7)
nB = 3*len(B_monos)
rows = []
for dm_ in monomials_upto(6):
    row = np.zeros(nB, dtype=np.int64)
    for j in range(3):
        for k, bm in enumerate(B_monos):
            d = dmono(bm, j)
            if d and d[1] == dm_:
                row[j*len(B_monos)+k] = d[0] % p
    rows.append(row)
hi = [m for m in monomials_upto(13) if sum(m) > 7]
hi_idx = midx(hi)
ov = np.zeros((3*len(hi), nB), dtype=np.int64)
for i in range(3):
    for j in range(3):
        for k, bm in enumerate(B_monos):
            for mm, c in pmulm(dF[i][j], bm, 1).items():
                if sum(mm) > 7:
                    r_ = i*len(hi)+hi_idx[mm]
                    ov[r_, j*len(B_monos)+k] = (ov[r_, j*len(B_monos)+k] + c) % p
KB = kernel_basis(np.vstack([np.array(rows), ov]) % p)
print("B-side kernel dim:", KB.shape[0])
passcnt = 0
results = []
for t, kv in enumerate(KB):
    bfield = [{}, {}, {}]
    for j in range(3):
        for k, bm in enumerate(B_monos):
            c = int(kv[j*len(B_monos)+k])
            if c: bfield[j][bm] = c
    degb = [max((sum(m) for m in b), default=-1) for b in bfield]
    comps = [padd(*[pmul(dF[i][j], bfield[j]) for j in range(3)]) for i in range(3)]
    comps = [{m:c for m,c in cc.items() if sum(m) <= 7} for cc in comps]
    ok = second_order_test(comps, f"B-basis {t} degB={degb}", quiet=True)
    results.append((t, degb, ok))
    if ok: passcnt += 1
print("B-side basis order-2 pass:", passcnt, "of", KB.shape[0])
for t, degb, ok in results:
    print(f"  B-basis {t}: max deg B = {max(degb)}, pass = {ok}")

# ---- A-side extended to deg <= 3 ----
A_monos = monomials_upto(3)
FdP = {}
def Fpow(m):
    if m in FdP: return FdP[m]
    out = {(0,0,0): 1}
    for t in range(3):
        for _ in range(m[t]):
            out = pmul(out, Fd[t])
    FdP[m] = out
    return out
nA = 3*len(A_monos)
rowsA = []
for dm_ in monomials_upto(2):
    if dm_ == (0,0,0): continue  # allow div A = const (weaker; superset)
    row = np.zeros(nA, dtype=np.int64)
    for i in range(3):
        for k, am in enumerate(A_monos):
            d = dmono(am, i)
            if d and d[1] == dm_:
                row[i*len(A_monos)+k] = d[0] % p
    rowsA.append(row)
hiA = [m for m in monomials_upto(21) if sum(m) > 7]
hiA_idx = midx(hiA)
ovA = np.zeros((3*len(hiA), nA), dtype=np.int64)
for i in range(3):
    for k, am in enumerate(A_monos):
        for mm, c in Fpow(am).items():
            if sum(mm) > 7:
                ovA[i*len(hiA)+hiA_idx[mm], i*len(A_monos)+k] = c % p
KA = kernel_basis(np.vstack([np.array(rowsA), ovA]) % p)
print("A-side (deg<=3, div const) kernel dim:", KA.shape[0], "(12 = affine only)")
