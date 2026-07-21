#!/usr/bin/env python3
"""
Order-3 obstruction for the order-2 survivors (B-basis 6, 11, 14, 16) and,
as control, an affine-orbit direction. Existence question:
  exists H (deg<=7 sol of order2) and H3 (deg<=7) with
  L(H3) = 2(trX trY - tr(XY) + det X) + c,  Y = JF^{-1} JH.
H varies over particular solution + ker(L); the induced variation of RHS3 is
2(trX trY_w - tr(X Y_w)), w in ker L -- include those columns in the span.
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
def midx(monos): return {m: k for k, m in enumerate(monos)}
def dmono(mono, j):
    e = mono[j]
    if e == 0: return None
    m = list(mono); m[j] -= 1
    return e, tuple(m)
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

G_monos = monomials_upto(7); nGm = len(G_monos); NG = 3*nGm
G_idx = midx(G_monos)
M = np.load('/tmp/surfaceA/M.npy')
EQ_monos = monomials_upto(17); EQ_idx = midx(EQ_monos)
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
def solve_particular(Min, rhs):
    aug = np.hstack([Min, rhs.reshape(-1, 1)]) % p
    A, pivots = rref(aug)
    ncols = Min.shape[1]
    if ncols in pivots:
        return None
    sol = np.zeros(ncols, dtype=np.int64)
    for ri, pc in enumerate(pivots):
        sol[pc] = A[ri, -1] % p
    return sol

inv2 = pow(2, p-2, p)
def G_from_vec(v):
    return [{m: int(v[i*nGm+k]) % p for k, m in enumerate(G_monos) if int(v[i*nGm+k]) % p}
            for i in range(3)]
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
def Xmat_of(Gc):
    Xm = matmul_poly(ADJ, jac_of(Gc))
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]
def e2_of(X_):
    trX = mat_trace(X_)
    trX2 = mat_trace(matmul_poly(X_, X_))
    return pscale(padd(pmul(trX, trX), pscale(trX2, p-1)), inv2)
def det_of(X_):
    t0 = pmul(X_[0][0], padd(pmul(X_[1][1], X_[2][2]), pscale(pmul(X_[1][2], X_[2][1]), p-1)))
    t1 = pmul(X_[0][1], padd(pmul(X_[1][0], X_[2][2]), pscale(pmul(X_[1][2], X_[2][0]), p-1)))
    t2 = pmul(X_[0][2], padd(pmul(X_[1][0], X_[2][1]), pscale(pmul(X_[1][1], X_[2][0]), p-1)))
    return padd(t0, pscale(t1, p-1), t2)

BIG_monos = monomials_upto(40); BIG_idx = midx(BIG_monos)
def to_big(q):
    v = np.zeros(len(BIG_monos), dtype=np.int64)
    for m, c in q.items():
        if m != (0, 0, 0):
            v[BIG_idx[m]] = c % p
    return v
Mbig = np.zeros((len(BIG_monos), Mc.shape[1]), dtype=np.int64)
for m in EQ_monos_c:
    Mbig[BIG_idx[m]] = Mc[EQc_idx[m]]

KC = kernel_basis(Mc)   # 33-dim
# precompute Y-columns for kernel directions: variation of RHS3 via H -> H + w
def rhs3_of(X_, Y_):
    trX = mat_trace(X_); trY = mat_trace(Y_)
    XY = matmul_poly(X_, Y_)
    val = padd(pmul(trX, trY), pscale(mat_trace(XY), p-1), det_of(X_))
    return pscale(val, 2)

def order3_test(Gc, label):
    X_ = Xmat_of(Gc)
    r2 = to_big(pscale(e2_of(X_), 2))
    # order-2 solve: Mbig h = r2  (rows deg <= 17 only nonzero in Mbig)
    h = solve_particular(Mbig, r2 % p)
    if h is None:
        print(f"[{label}] obstructed already at order 2 (unexpected here)")
        return
    Hc = G_from_vec((-h) % p)   # L(H) = -r2 + c convention: careful sign
    # Condition: t^2 coeff: tr(adj JH) + detJF*e2 = const ->
    # -2 trY + -2 e2 = const -> trY = -e2 + c'. L(H) = tr(adj.JH) = -2 trY.
    # our matrix M encodes L(H)=tr(adj.JH). We need L(H) = 2 e2 + c i.e. Mbig h = r2. so H = h itself
    Hc = G_from_vec(h % p)
    Y_ = Xmat_of(Hc)
    # sanity: check order-2 identity: trY + e2 = const
    chk = padd(mat_trace(Y_), e2_of(X_))
    nonconst = {m: c for m, c in chk.items() if m != (0,0,0)}
    assert not nonconst, f"order-2 identity failed for {label}"
    r3 = to_big(rhs3_of(X_, Y_))
    # allowed span: Im(L) + C + {rhs3 variation from H -> H + w} for w in ker L
    cols = [to_big(rhs3_of(X_, Xmat_of(G_from_vec(w)))) -  to_big(det_of(Xmat_of(G_from_vec(w))))*0
            for w in KC]
    # note: rhs3_of(X, Yw) includes det X again -> subtract it (variation is bilinear part only)
    dX = to_big(pscale(det_of(X_), 2))
    cols = [ (c_ - dX) % p for c_ in cols ]
    span = np.hstack([Mbig] + [c_.reshape(-1, 1) for c_ in cols])
    rk0 = rank_of(span)
    rk1 = rank_of(np.hstack([span, (r3 % p).reshape(-1, 1)]))
    print(f"[{label}] order 3: " + ("UNOBSTRUCTED" if rk0 == rk1 else "OBSTRUCTED"))

# rebuild B-side kernel identically to before
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
            for mm0, c0 in dF[i][j].items():
                mm = (mm0[0]+bm[0], mm0[1]+bm[1], mm0[2]+bm[2])
                if sum(mm) > 7:
                    r_ = i*len(hi)+hi_idx[mm]
                    ov[r_, j*len(B_monos)+k] = (ov[r_, j*len(B_monos)+k] + c0) % p
KB = kernel_basis(np.vstack([np.array(rows), ov]) % p)

def Gc_of_Bbasis(t):
    kv = KB[t]
    Bf = [{}, {}, {}]
    for j in range(3):
        for k, bm in enumerate(B_monos):
            c = int(kv[j*len(B_monos)+k])
            if c: Bf[j][bm] = c
    comps = [padd(*[pmul(dF[i][j], Bf[j]) for j in range(3)]) for i in range(3)]
    return [{m: c for m, c in cc.items() if sum(m) <= 7} for cc in comps]

# control: affine-orbit direction (must be unobstructed at order 3)
rng = np.random.default_rng(5)
Mt = rng.integers(0, p, (3,3)); vt = rng.integers(0, p, 3)
comps0 = [{}, {}, {}]
for i in range(3):
    for k in range(3):
        comps0[i] = padd(comps0[i], pscale(Fd[k], int(Mt[i,k])))
    comps0[i] = padd(comps0[i], {(0,0,0): int(vt[i])})
order3_test(comps0, "control affine A-side")
for t in [6, 11, 14, 16]:
    order3_test(Gc_of_Bbasis(t), f"B-basis {t}")
