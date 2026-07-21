#!/usr/bin/env python3
"""
Order-4 and order-5 obstruction test for the two order-3 survivors
(B-basis 11 and 14) of Alpoge's F, deg<=7 Keller variety K_{<=7}.

Deformation calculus. F_t = F + tG + t^2 H + t^3 H3 + t^4 H4 + t^5 H5,
X,Y,Z,U,V = JF^{-1} J(G,H,H3,H4,H5).  det J F_t / det JF = det(I + tX + ...).
Conditions (each t^k coefficient constant in x,y,z):
  t^2: trY + e2(X)
  t^3: trZ + trX*trY - tr(XY) + det(X)
  t^4: trU + trX*trZ - tr(XZ) + e2(Y) + tr(adj(X) Y)
  t^5: trV + trX*trU - tr(XU) + trY*trZ - tr(YZ) + tr(adj(X) Z) + tr(adj(Y) X)
With L(W) := tr(adj(JF) JW) = -2 tr(JF^{-1} JW), each condition reads
  L(H_k) = 2 * RHS_k + const.
EXACTNESS: for FIXED H the order-4 condition is linear in (H3, H4); the
H3-freedom (h3 -> h3 + v, v in ker L) shifts RHS4 by col_v = 2(trX trZ_v
- tr(X Z_v)) -- the same columns as at order 3.  So membership of 2*RHS4 in
Im(L) + span{col_v} is an EXACT solvability test for fixed H.  H itself is
sampled from the exact solution space of the joint order-2/3 linear system.
All arithmetic mod p = 1000003 (matrix cache /tmp/surfaceA/M.npy is mod this p).
"""
import sympy as sp
import numpy as np
import sys, time, os

p = 1000003
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)

# ---------------- packed-key polynomial arithmetic ----------------
KP = 64; KP2 = KP*KP; KP3 = KP**3
def pk(m): return (m[0]*KP + m[1])*KP + m[2]
def unpk(k): a, r = divmod(k, KP2); b, c = divmod(r, KP); return (a, b, c)

def pmul(pa, pb):
    if not pa or not pb:
        return {}
    ka = np.fromiter(pa.keys(), dtype=np.int64, count=len(pa))
    ca = np.fromiter(pa.values(), dtype=np.int64, count=len(pa))
    kb = np.fromiter(pb.keys(), dtype=np.int64, count=len(pb))
    cb = np.fromiter(pb.values(), dtype=np.int64, count=len(pb))
    keys = (ka[:, None] + kb[None, :]).ravel()
    vals = ((ca[:, None] * cb[None, :]) % p).ravel()
    acc = np.bincount(keys, weights=vals.astype(np.float64), minlength=KP3)
    acc = acc.astype(np.int64) % p
    nz = np.nonzero(acc)[0]
    return {int(k): int(acc[k]) for k in nz}

def padd(*ps):
    out = {}
    for q in ps:
        for k, c in q.items():
            v = (out.get(k, 0) + c) % p
            if v: out[k] = v
            elif k in out: del out[k]
    return out

def pscale(q, s):
    s %= p
    return {k: (c*s) % p for k, c in q.items() if (c*s) % p}

def psub(pa, pb): return padd(pa, pscale(pb, p-1))

def jac_of(comps):   # comps: list of 3 packed polys -> 3x3 matrix of packed polys
    out = [[{} for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for k, c in comps[i].items():
            a, b, cc = unpk(k)
            if a: out[i][0][k-KP2] = (out[i][0].get(k-KP2, 0) + c*a) % p
            if b: out[i][1][k-KP]  = (out[i][1].get(k-KP, 0) + c*b) % p
            if cc: out[i][2][k-1]  = (out[i][2].get(k-1, 0) + c*cc) % p
    return [[{k: c for k, c in e.items() if c} for e in row] for row in out]

def matmul_poly(A_, B_):
    return [[padd(*[pmul(A_[i][kk], B_[kk][j]) for kk in range(3)])
             for j in range(3)] for i in range(3)]

def mat_trace(A_): return padd(A_[0][0], A_[1][1], A_[2][2])

def tr_prod(A_, B_):   # tr(A B) without full product
    return padd(*[pmul(A_[i][j], B_[j][i]) for i in range(3) for j in range(3)])

inv2 = pow(2, p-2, p)
def e2_of(X_):
    trX = mat_trace(X_)
    return pscale(psub(pmul(trX, trX), tr_prod(X_, X_)), inv2)

def det_of(X_):
    t0 = pmul(X_[0][0], psub(pmul(X_[1][1], X_[2][2]), pmul(X_[1][2], X_[2][1])))
    t1 = pmul(X_[0][1], psub(pmul(X_[1][0], X_[2][2]), pmul(X_[1][2], X_[2][0])))
    t2 = pmul(X_[0][2], psub(pmul(X_[1][0], X_[2][1]), pmul(X_[1][1], X_[2][0])))
    return padd(t0, pscale(t1, p-1), t2)

def adj_of(X_):   # adjugate (transpose of cofactors)
    A = [[None]*3 for _ in range(3)]
    idx = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        for j in range(3):
            r = idx[i]; c = idx[j]
            m = psub(pmul(X_[r[0]][c[0]], X_[r[1]][c[1]]),
                     pmul(X_[r[0]][c[1]], X_[r[1]][c[0]]))
            A[j][i] = m if (i+j) % 2 == 0 else pscale(m, p-1)   # transpose here
    return A

# ---------------- build F, ADJ(JF), dF ----------------
x, y, z = sp.symbols('x y z')
VARS = (x, y, z)
u = 1 + x*y
F = [sp.expand(u**3*z + y**2*u*(4 + 3*x*y)),
     sp.expand(y + 3*x*u**2*z + 3*x*y**2*(4 + 3*x*y)),
     sp.expand(2*x - 3*x**2*y - x**3*z)]
J = sp.Matrix(3, 3, lambda i, j: sp.diff(F[i], VARS[j]))
assert sp.expand(J.det()) == -2
adjJ = J.adjugate()
def poly_to_pdict(expr):
    q = sp.Poly(sp.expand(expr), x, y, z)
    return {pk(m): int(c) % p for m, c in zip(q.monoms(), q.coeffs()) if int(c) % p}
ADJ = [[poly_to_pdict(adjJ[i, j]) for j in range(3)] for i in range(3)]
dF = [[poly_to_pdict(sp.diff(F[i], VARS[j])) for j in range(3)] for i in range(3)]
log("built F, ADJ, dF")

def Xmat_of(comps):   # JF^{-1} J(comps) = -(1/2) ADJ . J(comps)
    Xm = matmul_poly(ADJ, jac_of(comps))
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]

# ---------------- monomial bookkeeping / cached matrix ----------------
def monomials_upto(d):
    return [(a, b, c) for a in range(d+1) for b in range(d+1-a) for c in range(d+1-a-b)]
G_monos = monomials_upto(7); nGm = len(G_monos); NG = 3*nGm
G_key = [pk(m) for m in G_monos]
EQ_monos = monomials_upto(17)
M = np.load('/tmp/surfaceA/M.npy')
assert M.shape == (len(EQ_monos), NG)
const_row = EQ_monos.index((0, 0, 0))
Mc = np.delete(M, const_row, axis=0) % p
EQ_keys_c = [pk(m) for m in EQ_monos if m != (0, 0, 0)]
log("loaded M.npy", M.shape)

# ---------------- linear algebra mod p ----------------
def rref(Min):
    A = Min % p
    nrows, ncols = A.shape
    A = A.copy(); pivots = []; r = 0
    for c in range(ncols):
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0: continue
        piv = r + nz[0]
        if piv != r: A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p-2, p)
        A[r] = (A[r] * inv) % p
        col = A[:, c].copy(); col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - np.outer(col[nzr], A[r])) % p
        pivots.append(c); r += 1
        if r == nrows: break
    return A, pivots

def kernel_from_rref(A, pivots, ncols):
    pivset = set(pivots)
    free = [c for c in range(ncols) if c not in pivset]
    Kb = np.zeros((len(free), ncols), dtype=np.int64)
    for i, fc in enumerate(free):
        Kb[i, fc] = 1
        for ri, pc in enumerate(pivots):
            Kb[i, pc] = (-A[ri, fc]) % p
    return Kb

def rank_of(Min): return len(rref(Min)[1])

def kernel_basis(Min):
    A, piv = rref(Min)
    return kernel_from_rref(A, piv, Min.shape[1])

def system_solve(cols_extra, rhs, want_kernel=False):
    """Solve  Mc.h + sum_j lam_j * cols_extra[j] = rhs  (as polynomials, const dropped).
    cols_extra, rhs: packed polys.  Rows = union support.  Returns (h, lam, kernel|None)
    or None if unsolvable."""
    support = set(EQ_keys_c)
    for cp in cols_extra: support |= set(cp.keys())
    support |= set(rhs.keys())
    support.discard(0)
    skeys = sorted(support); sidx = {k: i for i, k in enumerate(skeys)}
    nc = NG + len(cols_extra)
    A = np.zeros((len(skeys), nc + 1), dtype=np.int64)
    for r_, k in enumerate(EQ_keys_c):
        A[sidx[k], :NG] = Mc[r_]
    for j, cp in enumerate(cols_extra):
        for k, c in cp.items():
            if k: A[sidx[k], NG + j] = c % p
    for k, c in rhs.items():
        if k: A[sidx[k], nc] = c % p
    R, piv = rref(A)
    if nc in piv:
        return None
    sol = np.zeros(nc, dtype=np.int64)
    for ri, pc in enumerate(piv):
        sol[pc] = R[ri, nc] % p
    h, lam = sol[:NG], sol[NG:]
    ker = None
    if want_kernel:
        ker = kernel_from_rref(R[:, :nc], piv, nc)
    return h, lam, ker

def comps_from_vec(v):
    return [{G_key[k]: int(v[i*nGm+k]) % p for k in range(nGm) if int(v[i*nGm+k]) % p}
            for i in range(3)]

def assert_const(q, label):
    bad = {k: c for k, c in q.items() if k != 0}
    assert not bad, f"{label}: non-constant part, {len(bad)} monomials, e.g. {list(bad.items())[:3]}"

# ---------------- B-side trivial kernel (verbatim ordering of order3.py) ----------------
def dmono(mono, j):
    e = mono[j]
    if e == 0: return None
    m = list(mono); m[j] -= 1
    return e, tuple(m)
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
hi_idx = {m: k for k, m in enumerate(hi)}
dF_t = [[{unpk(k): c for k, c in dF[i][j].items()} for j in range(3)] for i in range(3)]
ov = np.zeros((3*len(hi), nB), dtype=np.int64)
for i in range(3):
    for j in range(3):
        for k, bm in enumerate(B_monos):
            for mm0, c0 in dF_t[i][j].items():
                mm = (mm0[0]+bm[0], mm0[1]+bm[1], mm0[2]+bm[2])
                if sum(mm) > 7:
                    r_ = i*len(hi)+hi_idx[mm]
                    ov[r_, j*len(B_monos)+k] = (ov[r_, j*len(B_monos)+k] + c0) % p
KB = kernel_basis(np.vstack([np.array(rows), ov]) % p)
log("B-side trivial kernel dim =", KB.shape[0])

def B_of_basis(t):
    kv = KB[t]
    Bf = [{}, {}, {}]
    for j in range(3):
        for k, bm in enumerate(B_monos):
            c = int(kv[j*len(B_monos)+k])
            if c: Bf[j][pk(bm)] = c
    return Bf

def Gc_of_Bbasis(t):
    Bf = B_of_basis(t)
    comps = [padd(*[pmul(dF[i][j], Bf[j]) for j in range(3)]) for i in range(3)]
    for cc in comps:
        for k in list(cc.keys()):
            a, b, c2 = unpk(k)
            assert a+b+c2 <= 7, "B-basis image not deg<=7"
    return comps

def lift(c):   # symmetric lift for printing
    c %= p
    return c - p if c > p//2 else c

def show_B(t):
    Bf = B_of_basis(t)
    names = []
    for j in range(3):
        terms = []
        for k, c in sorted(Bf[j].items()):
            a, b, cc = unpk(k)
            mono = ''.join(s*e for s, e in zip(('x', 'y', 'z'), (a, b, cc))) or '1'
            terms.append(f"{lift(c):+d}*{mono}")
        names.append(' '.join(terms) if terms else '0')
    log(f"  B[{t}] = ({names[0]} ; {names[1]} ; {names[2]})")

# ---------------- first-order kernel KC (33-dim) ----------------
KC = kernel_basis(Mc)
log("dim ker L (first-order Keller directions) =", KC.shape[0])
YW = [Xmat_of(comps_from_vec(w)) for w in KC]   # JF^{-1} J(w) for each kernel vector
log("precomputed Y_w for", len(YW), "kernel vectors")

# ---------------- the obstruction tower ----------------
def variation_cols(X_):
    """col_v = 2(trX*trW_v - tr(X W_v)) for v in KC -- exact RHS shift when the
    unknown at the PREVIOUS order moves by v (same formula at orders 3,4,5)."""
    trX = mat_trace(X_)
    cols = []
    for W_ in YW:
        cols.append(pscale(psub(pmul(trX, mat_trace(W_)), tr_prod(X_, W_)), 2))
    return cols

def tower(Gc, label, max_order=5, n_samples=3, seed=11):
    rng = np.random.default_rng(seed)
    X_ = Xmat_of(Gc)
    trX = mat_trace(X_)
    assert all(k == 0 for k in trX), f"{label}: trX not constant"
    log(f"[{label}] trX = {lift(trX.get(0,0))}")
    # ---- order 2: L(H) = 2 e2(X) + c
    r2 = pscale(e2_of(X_), 2)
    s2 = system_solve([], r2)
    if s2 is None:
        log(f"[{label}] OBSTRUCTED at order 2"); return
    h0 = s2[0]
    log(f"[{label}] order 2 solved")
    # ---- order 3 joint in (h3, s):  L(h3) - sum s_i col_i = r3(h0)
    cols = variation_cols(X_)
    detX = det_of(X_)
    def r3_of(Y_):
        return pscale(padd(psub(pmul(trX, mat_trace(Y_)), tr_prod(X_, Y_)), detX), 2)
    Y0 = Xmat_of(comps_from_vec(h0))
    # unknown s shifts Y by sum s_i YW[i], shifting r3 by sum s_i col_i -> minus sign on cols
    s3 = system_solve([pscale(c_, p-1) for c_ in cols], r3_of(Y0), want_kernel=True)
    if s3 is None:
        log(f"[{label}] OBSTRUCTED at order 3"); return
    h3_0, lam0, ker3 = s3
    log(f"[{label}] order 3 solved; joint kernel dim = {ker3.shape[0]}")
    if max_order < 4:
        return
    # ---- order 4, exact for each fixed H sample
    def try_order4(h3v, lam, tag):
        # H = h0 + sum lam_i w_i ;  H3 = h3v
        Hvec = (h0 + (lam @ KC)) % p
        Hc = comps_from_vec(Hvec)
        Y_ = [[padd(Y0[i][j], *[pscale(YW[t][i][j], int(lam[t])) for t in range(len(YW)) if lam[t]])
               for j in range(3)] for i in range(3)]
        # exact verify order-2 and order-3 for this (H, H3)
        assert_const(padd(mat_trace(Y_), e2_of(X_)), f"{label} order2 identity ({tag})")
        Z_ = Xmat_of(comps_from_vec(h3v))
        t3 = padd(mat_trace(Z_), psub(pmul(trX, mat_trace(Y_)), tr_prod(X_, Y_)), detX)
        assert_const(t3, f"{label} order3 identity ({tag})")
        # RHS4 = 2( trX trZ - tr(XZ) + e2(Y) + tr(adjX . Y) )
        adjX = adj_of(X_)
        r4 = pscale(padd(psub(pmul(trX, mat_trace(Z_)), tr_prod(X_, Z_)),
                         e2_of(Y_), tr_prod(adjX, Y_)), 2)
        s4 = system_solve([pscale(c_, p-1) for c_ in cols], r4)
        if s4 is None:
            return None
        h4, mu = s4[0], s4[1]
        # certified: verify full t^4 coefficient
        h3f = (h3v + (mu @ KC)) % p
        Zf = Xmat_of(comps_from_vec(h3f))
        t3f = padd(mat_trace(Zf), psub(pmul(trX, mat_trace(Y_)), tr_prod(X_, Y_)), detX)
        assert_const(t3f, f"{label} order3 identity after mu-shift ({tag})")
        U_ = Xmat_of(comps_from_vec(h4))
        t4 = padd(mat_trace(U_), psub(pmul(trX, mat_trace(Zf)), tr_prod(X_, Zf)),
                  e2_of(Y_), tr_prod(adjX, Y_))
        assert_const(t4, f"{label} order4 FULL identity ({tag})")
        return Hvec, Y_, h3f, Zf, h4, U_, adjX
    res = try_order4(h3_0, lam0, "particular")
    tag_used = "particular"
    if res is None:
        for it in range(n_samples):
            coef = rng.integers(1, p, ker3.shape[0])
            pert = (coef @ ker3) % p
            h3v = (h3_0 + pert[:NG]) % p
            lam = (lam0 + pert[NG:]) % p
            res = try_order4(h3v, lam, f"sample{it}")
            if res is not None:
                tag_used = f"sample{it}"
                break
    if res is None:
        log(f"[{label}] order 4: OBSTRUCTED (exact for each fixed H; particular + "
            f"{n_samples} random H-samples from joint order<=3 solution space all fail)")
        return
    log(f"[{label}] order 4: UNOBSTRUCTED (exact certificate, {tag_used}; full t^4 identity verified)")
    if max_order < 5:
        return res
    # ---- order 5, exact for fixed (H, H3): linear in (H4, H5); H4-freedom = cols again
    Hvec, Y_, h3f, Zf, h4, U_, adjX = res
    adjY = adj_of(Y_)
    r5 = pscale(padd(psub(pmul(trX, mat_trace(U_)), tr_prod(X_, U_)),
                     psub(pmul(mat_trace(Y_), mat_trace(Zf)), tr_prod(Y_, Zf)),
                     tr_prod(adjX, Zf), tr_prod(adjY, X_)), 2)
    s5 = system_solve([pscale(c_, p-1) for c_ in cols], r5)
    if s5 is None:
        log(f"[{label}] order 5: OBSTRUCTED for the found (H,H3) branch (exact in (H4,H5) "
            f"for this branch; other branches not exhausted)")
        return res
    h5, nu = s5[0], s5[1]
    h4f = (h4 + (nu @ KC)) % p
    Uf = Xmat_of(comps_from_vec(h4f))
    t4f = padd(mat_trace(Uf), psub(pmul(trX, mat_trace(Zf)), tr_prod(X_, Zf)),
               e2_of(Y_), tr_prod(adjX, Y_))
    assert_const(t4f, f"{label} order4 identity after nu-shift")
    V_ = Xmat_of(comps_from_vec(h5))
    t5 = padd(mat_trace(V_), psub(pmul(trX, mat_trace(Uf)), tr_prod(X_, Uf)),
              psub(pmul(mat_trace(Y_), mat_trace(Zf)), tr_prod(Y_, Zf)),
              tr_prod(adjX, Zf), tr_prod(adjY, X_))
    assert_const(t5, f"{label} order5 FULL identity")
    log(f"[{label}] order 5: UNOBSTRUCTED (exact certificate; full t^5 identity verified)")
    return res

# ---------------- controls ----------------
Fd = [poly_to_pdict(f) for f in F]
rng0 = np.random.default_rng(5)
Mt = rng0.integers(0, p, (3, 3)); vt = rng0.integers(0, p, 3)
ctrl = [padd(*([pscale(Fd[k], int(Mt[i, k])) for k in range(3)] + [{0: int(vt[i])}]))
        for i in range(3)]
tower(ctrl, "control affine A-side", max_order=5)

# ---------------- the survivors ----------------
for t in (11, 14):
    show_B(t)
    Gc = Gc_of_Bbasis(t)
    # sanity: direction is in ker(Mc)
    gv = np.zeros(NG, dtype=np.int64)
    kidx = {k: i for i, k in enumerate(G_key)}
    for i in range(3):
        for k, c in Gc[i].items():
            gv[i*nGm + kidx[k]] = c % p
    resid = (Mc @ gv) % p
    assert not resid.any(), f"B-basis {t} not in first-order kernel"
    tower(Gc, f"B-basis {t}", max_order=5, n_samples=3)

# re-verify order-3 verdicts of the other two order-2 survivors (regression check)
for t in (6, 16):
    tower(Gc_of_Bbasis(t), f"B-basis {t} (regression)", max_order=3)
log("DONE")
