#!/usr/bin/env python3
"""
Find an F_p-rational point on the B11 order-4 obstruction variety
(7 quadrics, 32 unknowns) via random 7-plane slices + eigenvalue splitting:
grevlex GB -> quotient basis -> multiplication matrix of one variable ->
Krylov minimal polynomial -> F_p roots -> substitute, recurse.
argv: prime  [nslices]
"""
import numpy as np
import sympy as sp
import sys, time

pfile = sys.argv[1] if len(sys.argv) > 1 else '1000003'
NSLICE = int(sys.argv[2]) if len(sys.argv) > 2 else 4
d = np.load(f'/tmp/surfaceA/b11_system_{pfile}.npz')
p = int(d['p'][0]); V0 = d['V0']; L = d['L']; Q = d['Q']; keys = d['keys']
neq = len(V0); ns = L.shape[0]
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log(f"p={p}, {neq} equations, {ns} unknowns")

ts = sp.symbols(f't0:{ns}')
eqs = []
for e in range(neq):
    expr = int(V0[e])
    for a in range(ns):
        if L[a, e]: expr += int(L[a, e]) * ts[a]
    for ci in range(Q.shape[0]):
        a, b = int(keys[ci][0]), int(keys[ci][1])
        c = int(Q[ci, e])
        if c: expr += (2*c if a != b else c) * ts[a] * ts[b]
    eqs.append(sp.expand(expr))

def eval_full(tvec):
    res = []
    for e in range(neq):
        val = int(V0[e]) + sum(int(L[a, e])*int(tvec[a]) for a in range(ns))
        for ci in range(Q.shape[0]):
            a2, b2 = int(keys[ci][0]), int(keys[ci][1])
            c = int(Q[ci, e])
            if c: val += (2 if a2 != b2 else 1)*c*int(tvec[a2])*int(tvec[b2])
        res.append(val % p)
    return res

def minpoly_roots(Mm):
    """F_p roots of the minimal polynomial of matrix Mm (Krylov)."""
    n = Mm.shape[0]
    rng_ = np.random.default_rng(17)
    v = rng_.integers(0, p, n)
    K = [v % p]
    for _ in range(n):
        K.append((Mm @ K[-1]) % p)
    A = np.array(K).T % p   # n x (n+1)
    # first linear dependence: solve smallest k with K[k] in span(K[:k])
    # do full rref with augmentation trick: find kernel vector of A
    AA = A.copy()
    nrows, ncols = AA.shape
    piv = []; r = 0
    for c in range(ncols):
        nz = np.nonzero(AA[r:, c] % p)[0]
        if len(nz) == 0:
            # column c dependent: kernel vector via back-subst
            coef = np.zeros(ncols, dtype=np.int64); coef[c] = 1
            for ri, pc in enumerate(piv):
                coef[pc] = (-AA[ri, c]) % p
            poly = sum(int(coef[k]) * sp.Symbol('lam')**k for k in range(c+1))
            P_ = sp.Poly(poly, sp.Symbol('lam'), modulus=p)
            return sp.ground_roots(P_)
        pr = r + nz[0]
        if pr != r: AA[[r, pr]] = AA[[pr, r]]
        inv = pow(int(AA[r, c]), p-2, p)
        AA[r] = (AA[r]*inv) % p
        col = AA[:, c].copy(); col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr): AA[nzr] = (AA[nzr] - np.outer(col[nzr], AA[r])) % p
        piv.append(c); r += 1
        if r == nrows: break
    return {}

def solve0(polys, gens, depth=0):
    """recursively find one F_p point of 0-dim system; None if none rational."""
    pad = "  "*depth
    consts = [sp.expand(q_) for q_ in polys]
    nonz = [q for q in consts if q != 0]
    if not gens:
        return [] if not any(q.is_number and int(q) % p != 0 for q in nonz) and \
                     all(q.is_number for q in nonz) else None
    if any(q.is_number and int(q) % p != 0 for q in nonz if q.is_number):
        return None
    nonz = [q for q in nonz if not q.is_number]
    if not nonz:
        # zero ideal: any value works; pick 0 for all remaining
        return [0]*len(gens)
    try:
        G = sp.groebner(nonz, *gens, modulus=p, order='grevlex')
    except Exception as ex:
        log(f"{pad}groebner failed: {ex}"); return None
    if any(g.is_number and g != 0 for g in G.exprs):
        return None   # unit ideal
    LTs = [sp.Poly(g, *gens, modulus=p).monoms()[0] for g in G.exprs]
    def divisible(m1, m2): return all(a >= b for a, b in zip(m1, m2))
    # quotient basis via BFS over monomials not divisible by any LT
    basis = []; frontier = [tuple([0]*len(gens))]; seen = set(frontier)
    while frontier:
        m = frontier.pop()
        basis.append(m)
        if len(basis) > 300:
            log(f"{pad}quotient too big / positive dim"); return 'BIG'
        for j in range(len(gens)):
            m2 = tuple(mm + (1 if k == j else 0) for k, mm in enumerate(m))
            if m2 in seen: continue
            if any(divisible(m2, lt) for lt in LTs):
                continue
            seen.add(m2); frontier.append(m2)
    bidx = {m: i for i, m in enumerate(basis)}
    nb = len(basis)
    # multiplication matrix of last generator
    v = gens[-1]
    Mm = np.zeros((nb, nb), dtype=np.int64)
    for i, m in enumerate(basis):
        mono = sp.prod([g**e for g, e in zip(gens, m)]) * v
        red = G.reduce(mono)[1]
        pr = sp.Poly(red, *gens, modulus=p)
        for mo, co in zip(pr.monoms(), pr.coeffs()):
            if tuple(mo) not in bidx:
                return None   # shouldn't happen
            Mm[bidx[tuple(mo)], i] = int(co) % p
    roots = minpoly_roots(Mm)
    if not roots:
        log(f"{pad}dim {nb}: no F_p eigenvalue for {v} (all points irrational here)")
        return None
    for r_ in sorted(roots, key=lambda q: int(sp.Integer(q) % p)):
        rv = int(sp.Integer(r_) % p)
        sub = [sp.expand(q.subs(v, rv)) for q in polys]
        rec = solve0([q for q in sub if q != 0], gens[:-1], depth+1)
        if rec == 'BIG': rec = None
        if rec is not None:
            return rec + [rv]
    return None

rng = np.random.default_rng(int(pfile) % 1009 + 7)
s = sp.symbols('s0:7')
for trial in range(NSLICE):
    A = rng.integers(0, p, (ns, 7)); b = rng.integers(0, p, ns)
    sub = {ts[a]: sum(int(A[a, j])*s[j] for j in range(7)) + int(b[a]) for a in range(ns)}
    se = [sp.expand(e.subs(sub)) for e in eqs]
    log(f"slice {trial}: solving ...")
    sol = solve0(se, list(s))
    ncuts = 0
    while sol == 'BIG' and ncuts < 7:
        cut = sum(int(rng.integers(1, p))*s[j] for j in range(7)) + int(rng.integers(0, p))
        se = se + [sp.expand(cut)]
        ncuts += 1
        log(f"slice {trial}: adding random hyperplane cut #{ncuts}")
        sol = solve0(se, list(s))
    if sol == 'BIG': sol = None
    if sol is None:
        log(f"slice {trial}: no rational point (empty slice or irrational points)")
        continue
    sval = [int(x) % p for x in sol]
    tvec = [(sum(int(A[a, j])*sval[j] for j in range(7)) + int(b[a])) % p
            for a in range(ns)]
    res = eval_full(tvec)
    if not any(res):
        log(f"WITNESS FOUND at slice {trial}: t = {tvec}")
        np.save(f'/tmp/surfaceA/b11_witness_{pfile}.npy', np.array(tvec, dtype=np.int64))
        sys.exit(0)
    else:
        log(f"slice {trial}: candidate failed full verification ({res[:3]}...)")
log("NO RATIONAL WITNESS FOUND in tested slices")
