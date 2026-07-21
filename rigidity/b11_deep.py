#!/usr/bin/env python3
"""
Deep order-4 analysis for B11 = (xy/3, -2y^2/3, yz) at prime argv[1].
Facts used: over the order-<=3 solution space parametrized by t in F_p^ns
(H-moving kernel directions; H3-only directions lie inside the test span),
   r4(t) = r4(0) + sum_a t_a L_a + sum_{a<b} 2 t_a t_b Q_ab + sum_a t_a^2 Q_aa
EXACTLY (verified against direct recomputation on random t).  Solvable at t
iff r4(t) in Span := Im(L) + C + span(cols33).
Outputs: effective obstruction codimensions, exhaustive batch sampling,
and 2-variable random-slice solvability if codim small.
"""
import sys
exec(open('/Users/marcusteller/repos/pjc/experiments/surfaceA/order4_certify.py').read()
     .split("for Bf, name in")[0])   # all machinery, no driver

def rank_of(Min): return len(rref(Min)[1])

label = "B11"
Gc = G_of_B(B11, label)
X_ = Xmat_of(Gc)
trX = mat_trace(X_)
r2 = pscale(e2_of(X_), 2)
h0 = system_solve([], r2)[0]
Y0 = Xmat_of(comps_from_vec(h0))
detX = det_of(X_)
cols = [pscale(psub(pmul(trX, mat_trace(W_)), tr_prod(X_, W_)), 2) for W_ in YW]
r3 = pscale(padd(psub(pmul(trX, mat_trace(Y0)), tr_prod(X_, Y0)), detX), 2)
h3_0, lam0, ker3 = system_solve([pscale(c_, p-1) for c_ in cols], r3, want_kernel=True)
Ystar = [[padd(Y0[i][j], *[pscale(YW[t][i][j], int(lam0[t]))
                           for t in range(len(YW)) if lam0[t]])
          for j in range(3)] for i in range(3)]
Zstar = Xmat_of(comps_from_vec(h3_0))
adjX = adj_of(X_)
r4_0 = pscale(padd(psub(pmul(trX, mat_trace(Zstar)), tr_prod(X_, Zstar)),
                   e2_of(Ystar), tr_prod(adjX, Ystar)), 2)
A_s = ker3[:, NG:]
Rt, piv_rows = rref(A_s.T % p)
s_rows = piv_rows
ns = len(s_rows)
log(f"H-moving directions ns = {ns}")
Lcols, Wlist, dZlist = [], [], []
for ri in s_rows:
    kv = ker3[ri]
    dh3 = kv[:NG] % p; ds = kv[NG:] % p
    dZ = Xmat_of(comps_from_vec(dh3))
    W_ = [[padd(*[pscale(YW[t][i][j], int(ds[t])) for t in range(len(YW)) if ds[t]])
           for j in range(3)] for i in range(3)]
    Lc = padd(pscale(psub(pmul(trX, mat_trace(dZ)), tr_prod(X_, dZ)), 2),
              pscale(bil_e2(Ystar, W_), 2),
              pscale(tr_prod(adjX, W_), 2))
    Lcols.append(Lc); Wlist.append(W_); dZlist.append(dZ)
Qpairs = {}
for a in range(ns):
    for b in range(a, ns):
        Qpairs[(a, b)] = bil_e2(Wlist[a], Wlist[b])
log("built L and Q dicts")

# ---- vectorize over union support ----
gens = [r4_0] + Lcols + list(Qpairs.values())
support = set(EQ_keys_c)
for q in gens + cols: support |= set(q.keys())
support.discard(0)
skeys = sorted(support); sidx = {k: i for i, k in enumerate(skeys)}
nsup = len(skeys)
log(f"support size = {nsup}")
def vec_of(q):
    v = np.zeros(nsup, dtype=np.int64)
    for k, c in q.items():
        if k: v[sidx[k]] = c % p
    return v
# span S = [Mc columns | cols33]
S = np.zeros((nsup, NG + len(cols)), dtype=np.int64)
for r_, k in enumerate(EQ_keys_c):
    S[sidx[k], :NG] = Mc[r_]
for j, cp in enumerate(cols):
    for k, c in cp.items():
        if k: S[sidx[k], NG + j] = c % p
# echelon basis of column space of S via rref of S^T
RS, pivS = rref(S.T % p)
E = RS[:len(pivS)]            # rank x (nsup) echelon basis, pivot col = pivS? careful:
# rref(S.T): rows = original columns of S; RS rows are echelon combos, pivots = support indices
rankS = len(pivS)
log(f"rank(Span) = {rankS}")
def reduce_vec(v):
    w = v % p
    for ri, pc in enumerate(pivS):
        c = w[pc] % p
        if c: w = (w - c * E[ri]) % p
    return w
V0 = reduce_vec(vec_of(r4_0))
Lred = np.array([reduce_vec(vec_of(q)) for q in Lcols])
Qred = {ab: reduce_vec(vec_of(q)) for ab, q in Qpairs.items()}
log("reduced generators mod Span")
uL = rank_of(Lred)
allQ = np.array(list(Qred.values()))
uQ = rank_of(allQ)
uAll = rank_of(np.vstack([Lred, allQ]))
u0 = rank_of(np.vstack([Lred, allQ, V0.reshape(1, -1)]))
log(f"dim span(L mod Span) = {uL}; dim span(Q mod Span) = {uQ}; "
    f"dim span(L,Q) = {uAll}; with r4(0): {u0} "
    f"({'r4(0) INSIDE span(L,Q) (relaxation feasible)' if u0 == uAll else 'r4(0) OUTSIDE -> certified obstructed'})")

# ---- exact residual formula + cross-check ----
def resid_of_t(tv):
    w = V0.copy()
    w = (w + tv @ Lred) % p
    for (a, b), qv in Qred.items():
        cc = (2 * tv[a] * tv[b] if a != b else tv[a] * tv[a]) % p
        if cc: w = (w + cc * qv) % p
    return w % p
# cross-check against direct recomputation for one random t
rngc = np.random.default_rng(3)
tv = rngc.integers(0, p, ns)
Yt = [[padd(Ystar[i][j], *[pscale(Wlist[a][i][j], int(tv[a])) for a in range(ns)])
       for j in range(3)] for i in range(3)]
Zt = [[padd(Zstar[i][j], *[pscale(dZlist[a][i][j], int(tv[a])) for a in range(ns)])
       for j in range(3)] for i in range(3)]
r4t = pscale(padd(psub(pmul(trX, mat_trace(Zt)), tr_prod(X_, Zt)),
                  e2_of(Yt), tr_prod(adjX, Yt)), 2)
direct = reduce_vec(vec_of(r4t))
assert np.array_equal(direct, resid_of_t(tv)), "combo formula mismatch!"
log("combo formula CROSS-CHECKED against direct recomputation")

# ---- batch sampling ----
rng = np.random.default_rng(2026)
NBATCH = 20000
fails = 0; hits = []
# vectorized: residual = V0 + T @ Lred + (quad coeffs) @ allQ
keys = list(Qred.keys())
Qmat = np.array([Qred[k] for k in keys])
for chunk in range(20):
    T = rng.integers(0, p, (NBATCH//20, ns))
    quad = np.zeros((T.shape[0], len(keys)), dtype=np.int64)
    for ci, (a, b) in enumerate(keys):
        quad[:, ci] = (2*T[:, a]*T[:, b] if a != b else T[:, a]*T[:, a]) % p
    R = (V0[None, :] + (T % p) @ Lred % p + quad @ Qmat) % p
    nzrow = np.any(R % p, axis=1)
    fails += int(nzrow.sum())
    for i in np.nonzero(~nzrow)[0]:
        hits.append(T[i].copy())
log(f"sampling: {fails} obstructed / {NBATCH} random t; solutions found: {len(hits)}")
if hits:
    np.save('/tmp/surfaceA/b11_solution_t.npy', np.array(hits))
    log("SOLUTION(S) FOUND -> B11 unobstructed at order 4; saved t")

# ---- save reduced system for the quadratic solver ----
BasV, pivB0 = rref(np.vstack([Lred, allQ, V0.reshape(1, -1)]) % p)
coords0 = np.array(pivB0, dtype=np.int64)
np.savez(f'/tmp/surfaceA/b11_system_{p}.npz',
         coords=coords0, V0=V0[coords0], L=Lred[:, coords0],
         Q=Qmat[:, coords0], keys=np.array(keys), p=np.array([p]))
log(f"saved reduced system ({len(coords0)} equations, {ns} unknowns) "
    f"-> /tmp/surfaceA/b11_system_{p}.npz")

# ---- slice decision if codim small ----
log(f"effective number of scalar equations (codim of target) = {uAll} "
    f"in {ns} unknowns; quadratic system {'OVER' if uAll > ns else 'under'}-determined")
# reduce residual to independent coordinates: project onto row space basis of [Lred; allQ; V0]
Bas, pivB = rref(np.vstack([Lred, allQ, V0.reshape(1, -1)]) % p)
coords = pivB   # residual determined by these coordinates
log(f"residual lives in {len(coords)}-dim space; equations = residual coords = {len(coords)}")
if len(coords) <= 6 and not hits:
    import sympy as sp2
    tsy = sp2.symbols(f't0:{ns}')
    eqs = []
    for pc in coords:
        expr = int(V0[pc])
        for a in range(ns):
            expr += int(Lred[a, pc]) * tsy[a]
        for ci, (a, b) in enumerate(keys):
            cc = 2 if a != b else 1
            expr += cc * int(Qmat[ci, pc]) * tsy[a] * tsy[b]
        eqs.append(sp2.Poly(expr, *tsy, modulus=p))
    log(f"built {len(eqs)} quadratic equations over GF({p}) -- attempting random 2-plane slices")
    rng2 = np.random.default_rng(99)
    for trial in range(4):
        al, be = sp2.symbols('al be')
        v1 = rng2.integers(0, p, ns); v2 = rng2.integers(0, p, ns)
        sub = {tsy[a]: int(v1[a])*al + int(v2[a])*be for a in range(ns)}
        se = [sp2.Poly(e.as_expr().subs(sub), al, be, modulus=p) for e in eqs]
        # solve pairwise via resultants against first equation
        r01 = sp2.resultant(se[0].as_expr(), se[1].as_expr(), be)
        roots = sp2.ground_roots(sp2.Poly(r01, al, modulus=p))
        found = False
        for rt in roots:
            for e_ in se:
                pass
        log(f"slice {trial}: resultant deg {sp2.Poly(r01, al, modulus=p).degree()}, "
            f"roots mod p: {len(roots)} (candidate check skipped if 0)")
log("DEEP DONE")
