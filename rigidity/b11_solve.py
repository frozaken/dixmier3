#!/usr/bin/env python3
"""
Decide the B11 order-4 obstruction: the reduced system is 7 quadrics in 32
unknowns t over F_p (saved by b11_deep.py).  Slice with random affine 7-planes
t = A s + b -> square 7x7 quadratic system; Groebner mod p decides
empty/nonempty over the algebraic closure; lex triangularization attempts an
explicit rational point.  Nonempty slice => the obstruction variety is
nonempty over F_p-bar (=> order-4 UNobstructed over C, heuristically);
unit ideal on many random slices + both primes => strong evidence obstructed.
"""
import numpy as np
import sympy as sp
import sys, time

pfile = sys.argv[1] if len(sys.argv) > 1 else '1000003'
NSLICE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
d = np.load(f'/tmp/surfaceA/b11_system_{pfile}.npz')
p = int(d['p'][0]); V0 = d['V0']; L = d['L']; Q = d['Q']; keys = d['keys']
neq = len(V0); ns = L.shape[0]
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log(f"p={p}, {neq} equations, {ns} unknowns, {Q.shape[0]} quadratic gens")

ts = sp.symbols(f't0:{ns}')
eqs = []
for e in range(neq):
    expr = int(V0[e])
    for a in range(ns):
        if L[a, e]: expr += int(L[a, e]) * ts[a]
    for ci in range(Q.shape[0]):
        a, b = int(keys[ci][0]), int(keys[ci][1])
        c = int(Q[ci, e])
        if c:
            expr += (2*c if a != b else c) * ts[a] * ts[b]
    eqs.append(sp.expand(expr))
log("built symbolic system")

rng = np.random.default_rng(int(pfile) % 1009)
s = sp.symbols('s0:7')
for trial in range(NSLICE):
    A = rng.integers(0, p, (ns, 7)); b = rng.integers(0, p, ns)
    sub = {ts[a]: sum(int(A[a, j])*s[j] for j in range(7)) + int(b[a]) for a in range(ns)}
    se = [sp.Poly(sp.expand(e.subs(sub)), *s, modulus=p) for e in eqs]
    log(f"slice {trial}: substituted; computing groebner (grevlex) ...")
    G = sp.groebner([e_.as_expr() for e_ in se], *s, modulus=p, order='grevlex')
    if G.exprs == [sp.Integer(1)] or (len(G.exprs) == 1 and G.exprs[0].is_number):
        log(f"slice {trial}: UNIT IDEAL -> slice variety EMPTY over F_p-bar")
        continue
    log(f"slice {trial}: NONEMPTY over F_p-bar ({len(G.exprs)} groebner gens) "
        f"-> obstruction variety has points; attempting rational point via lex ...")
    try:
        Gl = sp.groebner([e_.as_expr() for e_ in se], *s, modulus=p, order='lex')
        # try univariate last generator
        last = sp.Poly(Gl.exprs[-1], s[6], modulus=p)
        roots = sp.ground_roots(last)
        log(f"  lex done; last gen deg {last.degree()} in s6; F_p roots: "
            f"{list(roots.keys())[:5]}")
        for r6 in roots:
            # back-substitute greedily
            vals = {s[6]: r6}
            ok = True
            for gexpr in reversed(Gl.exprs[:-1]):
                g_ = sp.Poly(sp.expand(gexpr.subs(vals)), *s, modulus=p)
                free = [v for v in g_.gens if g_.degree(v) > 0]
                if not free:
                    if g_.as_expr() != 0: ok = False; break
                    continue
                v = free[-1]
                uni = sp.Poly(g_.as_expr(), v, modulus=p)
                rr = sp.ground_roots(uni)
                if not rr: ok = False; break
                vals[v] = list(rr.keys())[0]
            if ok and len(vals) == 7:
                sol = [int(sp.Integer(vals[s[j]])) % p for j in range(7)]
                tvec = [(sum(int(A[a, j])*sol[j] for j in range(7)) + int(b[a])) % p
                        for a in range(ns)]
                # verify on the reduced system
                res = []
                for e in range(neq):
                    val = int(V0[e]) + sum(int(L[a, e])*tvec[a] for a in range(ns))
                    for ci in range(Q.shape[0]):
                        a2, b2 = int(keys[ci][0]), int(keys[ci][1])
                        c = int(Q[ci, e])
                        if c:
                            val += (2 if a2 != b2 else 1)*c*tvec[a2]*tvec[b2]
                    res.append(val % p)
                if not any(res):
                    log(f"  RATIONAL POINT FOUND AND VERIFIED: t = {tvec}")
                    np.save(f'/tmp/surfaceA/b11_witness_{pfile}.npy',
                            np.array(tvec, dtype=np.int64))
                    sys.exit(0)
                else:
                    log(f"  candidate failed verification (residual {res})")
    except Exception as ex:
        log(f"  lex/point extraction failed: {type(ex).__name__}: {ex}")
log("ALL SLICES PROCESSED")
