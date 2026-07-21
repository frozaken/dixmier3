#!/usr/bin/env python3
"""
Rigorous order-4 obstruction CERTIFICATE for the two order-3 survivors,
run at a chosen prime (argv[1]), directions pinned by exact rational lifts:
  B11 = (xy/3, -2y^2/3, yz)          (div = y/3 - 4y/3 + y = 0)
  B14 = (-x^3/3, 0, 2xy + x^2 z)     (div = -x^2 + x^2 = 0)
G = JF . B  (first-order Keller direction, deg <= 7).

Certificate logic.  V = solution space of the joint order-2/3 linear system in
(H, H3) around base point (H*, H3*); its kernel is parametrized by t.  The
order-4 RHS satisfies exactly
   r4(t) - r4(0)  in  span{L_k} + span{Q_km}
where L_k are the linear variation columns and Q_km = B(W_k, W_m) the
polarized-e2 quadratic columns (B(A,C) = trA trC - tr(AC); diagonal e2(W_k) =
B(W_k,W_k)/2 lies in the same span).  Hence if
   r4(0)  NOT in  Im(L) + C + span{L_k} + span{Q_km}
then a separating functional kills every r4(t) simultaneously: NO point of V
admits H4 -- obstruction PROVED mod p (covers the full quadratic H-freedom,
no sampling).
"""
import sympy as sp
import numpy as np
import sys, time

p = int(sys.argv[1]) if len(sys.argv) > 1 else 1000003
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)
log("prime p =", p)

KP = 64; KP2 = KP*KP; KP3 = KP**3
def pk(m): return (m[0]*KP + m[1])*KP + m[2]
def unpk(k): a, r = divmod(k, KP2); b, c = divmod(r, KP); return (a, b, c)

def pmul(pa, pb):
    if not pa or not pb: return {}
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

def jac_of(comps):
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
def tr_prod(A_, B_):
    return padd(*[pmul(A_[i][j], B_[j][i]) for i in range(3) for j in range(3)])

inv2 = pow(2, p-2, p); inv3 = pow(3, p-2, p)
def e2_of(X_):
    trX = mat_trace(X_)
    return pscale(psub(pmul(trX, trX), tr_prod(X_, X_)), inv2)
def bil_e2(A_, B_):   # polarization: trA trB - tr(AB)
    return psub(pmul(mat_trace(A_), mat_trace(B_)), tr_prod(A_, B_))
def det_of(X_):
    t0 = pmul(X_[0][0], psub(pmul(X_[1][1], X_[2][2]), pmul(X_[1][2], X_[2][1])))
    t1 = pmul(X_[0][1], psub(pmul(X_[1][0], X_[2][2]), pmul(X_[1][2], X_[2][0])))
    t2 = pmul(X_[0][2], psub(pmul(X_[1][0], X_[2][1]), pmul(X_[1][1], X_[2][0])))
    return padd(t0, pscale(t1, p-1), t2)
def adj_of(X_):
    A = [[None]*3 for _ in range(3)]
    idx = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        for j in range(3):
            r = idx[i]; c = idx[j]
            m = psub(pmul(X_[r[0]][c[0]], X_[r[1]][c[1]]),
                     pmul(X_[r[0]][c[1]], X_[r[1]][c[0]]))
            A[j][i] = m if (i+j) % 2 == 0 else pscale(m, p-1)
    return A

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

def Xmat_of(comps):
    Xm = matmul_poly(ADJ, jac_of(comps))
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]

def monomials_upto(d):
    return [(a, b, c) for a in range(d+1) for b in range(d+1-a) for c in range(d+1-a-b)]
def dmono(mono, j):
    e = mono[j]
    if e == 0: return None
    m = list(mono); m[j] -= 1
    return e, tuple(m)
G_monos = monomials_upto(7); nGm = len(G_monos); NG = 3*nGm
G_key = [pk(m) for m in G_monos]
EQ_monos = monomials_upto(17)
EQ_idx_pk = {pk(m): i for i, m in enumerate(EQ_monos)}

def build_M():
    Mm = np.zeros((len(EQ_monos), NG), dtype=np.int64)
    for j in range(3):
        for k, mono in enumerate(G_monos):
            col = j*nGm + k
            for i in range(3):
                d = dmono(mono, i)
                if d is None: continue
                cf, dm = d
                dk = pk(dm)
                for kk, c in ADJ[i][j].items():
                    r = EQ_idx_pk[kk + dk]
                    Mm[r, col] = (Mm[r, col] + c*cf) % p
    return Mm

M = build_M()
const_row = EQ_monos.index((0, 0, 0))
Mc = np.delete(M, const_row, axis=0) % p
EQ_keys_c = [pk(m) for m in EQ_monos if m != (0, 0, 0)]
log("built M", M.shape)

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

def kernel_basis(Min):
    A, piv = rref(Min)
    return kernel_from_rref(A, piv, Min.shape[1])

def system_solve(cols_extra, rhs, want_kernel=False):
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
    ker = kernel_from_rref(R[:, :nc], piv, nc) if want_kernel else None
    return sol[:NG], sol[NG:], ker

def comps_from_vec(v):
    return [{G_key[k]: int(v[i*nGm+k]) % p for k in range(nGm) if int(v[i*nGm+k]) % p}
            for i in range(3)]

def assert_const(q, label):
    bad = {k: c for k, c in q.items() if k != 0}
    assert not bad, f"{label}: non-constant, e.g. {list(bad.items())[:3]}"

KC = kernel_basis(Mc)
log("dim ker L =", KC.shape[0])
YW = [Xmat_of(comps_from_vec(w)) for w in KC]
log("precomputed Y_w")

# ---- exact rational directions ----
B11 = [{pk((1, 1, 0)): inv3}, {pk((0, 2, 0)): (p - 2*inv3 % p) % p}, {pk((0, 1, 1)): 1}]
B14 = [{pk((3, 0, 0)): (p - inv3) % p}, {}, {pk((1, 1, 0)): 2, pk((2, 0, 1)): 1}]

def G_of_B(Bf, name):
    # div B check
    dv = padd(*[ {kk - (KP2, KP, 1)[j]: (c * unpk(kk)[j]) % p
                  for kk, c in Bf[j].items() if unpk(kk)[j]} for j in range(3)])
    assert_const(dv, f"div {name}")
    assert dv.get(0, 0) % p == 0, f"div {name} nonzero const"
    comps = [padd(*[pmul(dF[i][j], Bf[j]) for j in range(3)]) for i in range(3)]
    for cc in comps:
        for k in cc:
            assert sum(unpk(k)) <= 7, f"{name}: deg(JF.B) > 7"
    # in first-order kernel?
    gv = np.zeros(NG, dtype=np.int64)
    kidx = {k: i for i, k in enumerate(G_key)}
    for i in range(3):
        for k, c in comps[i].items():
            gv[i*nGm + kidx[k]] = c % p
    resid = (Mc @ gv) % p
    assert not resid.any(), f"{name} not in ker L"
    return comps

def certify(Gc, label):
    X_ = Xmat_of(Gc)
    trX = mat_trace(X_)
    assert all(k == 0 for k in trX), f"{label}: trX not constant"
    r2 = pscale(e2_of(X_), 2)
    s2 = system_solve([], r2)
    if s2 is None:
        log(f"[{label}] obstructed at order 2 (unexpected)"); return
    h0 = s2[0]
    Y0 = Xmat_of(comps_from_vec(h0))
    detX = det_of(X_)
    cols = [pscale(psub(pmul(trX, mat_trace(W_)), tr_prod(X_, W_)), 2) for W_ in YW]
    r3 = pscale(padd(psub(pmul(trX, mat_trace(Y0)), tr_prod(X_, Y0)), detX), 2)
    s3 = system_solve([pscale(c_, p-1) for c_ in cols], r3, want_kernel=True)
    if s3 is None:
        log(f"[{label}] obstructed at order 3 (unexpected)"); return
    h3_0, lam0, ker3 = s3
    m = ker3.shape[0]
    log(f"[{label}] orders 2-3 solved; joint kernel dim = {m}")
    # base point
    Ystar = [[padd(Y0[i][j], *[pscale(YW[t][i][j], int(lam0[t]))
                               for t in range(len(YW)) if lam0[t]])
              for j in range(3)] for i in range(3)]
    Zstar = Xmat_of(comps_from_vec(h3_0))
    # exact base-point identities
    assert_const(padd(mat_trace(Ystar), e2_of(X_)), f"{label} order2 base")
    assert_const(padd(mat_trace(Zstar),
                      psub(pmul(trX, mat_trace(Ystar)), tr_prod(X_, Ystar)), detX),
                 f"{label} order3 base")
    adjX = adj_of(X_)
    r4_0 = pscale(padd(psub(pmul(trX, mat_trace(Zstar)), tr_prod(X_, Zstar)),
                       e2_of(Ystar), tr_prod(adjX, Ystar)), 2)
    # adapt kernel basis: split into s-part-independent vectors and delta-s = 0 vectors
    A_s = ker3[:, NG:]                     # m x 33 (s-parts)
    Rt, piv_rows = rref(A_s.T % p)         # pivots = indices of rows with indep s-parts
    s_rows = piv_rows                      # kernel vectors carrying genuine H-motion
    null_rows = kernel_basis(A_s.T % p)    # combos with zero s-part
    log(f"[{label}] kernel split: {len(s_rows)} H-moving + {null_rows.shape[0]} H3-only")
    # H3-only vectors: delta-h3 in ker(Mc) -> L-columns are the 'cols' family; include
    # them via their exact L formula anyway (delta Z shift).
    Lcols, Wlist = [], []
    def Lcol_of(kv):
        dh3 = kv[:NG] % p
        ds = kv[NG:] % p
        dZ = Xmat_of(comps_from_vec(dh3))
        W_ = [[padd(*[pscale(YW[t][i][j], int(ds[t])) for t in range(len(YW)) if ds[t]])
               for j in range(3)] for i in range(3)]
        Lc = padd(pscale(psub(pmul(trX, mat_trace(dZ)), tr_prod(X_, dZ)), 2),
                  pscale(bil_e2(Ystar, W_), 2),
                  pscale(tr_prod(adjX, W_), 2))
        return Lc, W_
    for ri in s_rows:
        Lc, W_ = Lcol_of(ker3[ri])
        Lcols.append(Lc); Wlist.append(W_)
    for combo in null_rows:
        kv = (combo @ ker3) % p
        Lc, _ = Lcol_of(kv)
        Lcols.append(Lc)
    Qcols = []
    ns = len(Wlist)
    for a in range(ns):
        for b in range(a, ns):
            Qcols.append(bil_e2(Wlist[a], Wlist[b]))
    log(f"[{label}] built {len(Lcols)} L-columns, {len(Qcols)} Q-columns")
    s4 = system_solve([pscale(c_, p-1) for c_ in (Lcols + Qcols)], r4_0)
    if s4 is None:
        log(f"[{label}] ORDER 4 OBSTRUCTION CERTIFIED mod {p}: r4(0) not in "
            f"Im(L)+C+span(L,Q) -- no point of the order<=3 solution space "
            f"admits H4 (full quadratic H-freedom covered)")
    else:
        log(f"[{label}] certificate INCONCLUSIVE: r4(0) in extended span "
            f"(quadratic system may or may not have a zero)")

for Bf, name in ((B11, "B11 = (xy/3, -2y^2/3, yz)"),
                 (B14, "B14 = (-x^3/3, 0, 2xy+x^2z)")):
    certify(G_of_B(Bf, name), name)
log("DONE")
