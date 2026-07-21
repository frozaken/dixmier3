#!/usr/bin/env python3
"""
Task 3: 2D probe. K_4 = {(P,Q) : deg <= 4, jac(P,Q) = 1} in C^30
(coefficients of P and Q on monomials x^a y^b, a+b <= 4; 15 each).
Equations: coefficients of jac(P,Q) - 1 on monomials deg <= 6: 28 equations.

Strata (by Jung-van der Kulk + Moh deg<=100: every K_4 point IS an automorphism):
  S4  : a o E_p o b,  a,b affine, p = p2 x^2 + p3 x^3 + p4 x^4       (length 1)
  S22 : a0 o E_q o a1 o E_p o a2, p = p2 x^2, q = q2 y^2             (length 2, multidegree (2,2))
Compute (mod two primes):
  D1, D2   = dims of strata images (rank of parametrization Jacobian), jac=1 slices: D1-1, D2-1
  t1, t2   = dim of scheme tangent space T K_4 at generic points of each stratum (jac normalized to 1)
Verdicts: smoothness of components, reducibility of K_4 (via t2 < D1-1 argument or explicit).
"""
import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

def rref_rank(Min, p):
    A = Min % p
    nrows, ncols = A.shape
    A = A.copy(); r = 0; pivots = []
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
    return r

MON4 = [(a, b) for a in range(5) for b in range(5-a)]           # 15
MON6 = [(a, b) for a in range(7) for b in range(7-a)]           # 28
M4_idx = {m: k for k, m in enumerate(MON4)}
M6_idx = {m: k for k, m in enumerate(MON6)}

def coeffs_of(expr, monos, idx):
    q = sp.Poly(sp.expand(expr), x, y)
    v = [0]*len(monos)
    for m, c in zip(q.monoms(), q.coeffs()):
        v[idx[m]] = c
    return v

def compose2(outer, inner):
    """outer, inner: pairs of exprs in (x,y); returns outer o inner"""
    X, Y = inner
    return (sp.expand(outer[0].subs({x: X, y: Y}, simultaneous=True)),
            sp.expand(outer[1].subs({x: X, y: Y}, simultaneous=True)))

def jac2(P, Q):
    return sp.expand(sp.diff(P, x)*sp.diff(Q, y) - sp.diff(P, y)*sp.diff(Q, x))

# ---------------- parametrizations with symbolic params ----------------
def affine_sym(tag):
    ps = sp.symbols(f'{tag}0:6')
    return (ps[0]*x + ps[1]*y + ps[2], ps[3]*x + ps[4]*y + ps[5]), list(ps)

def S4_param():
    a, pa = affine_sym('a')
    b, pb = affine_sym('b')
    p2, p3, p4 = sp.symbols('p2 p3 p4')
    E = (x, y + p2*x**2 + p3*x**3 + p4*x**4)
    comp = compose2(a, compose2(E, b))
    return comp, pa + pb + [p2, p3, p4]

def S22_param():
    a0, pa0 = affine_sym('c')
    a1, pa1 = affine_sym('d')
    a2, pa2 = affine_sym('e')
    p2, q2 = sp.symbols('pp qq')
    Ep = (x, y + p2*x**2)
    Eq = (x + q2*y**2, y)
    comp = compose2(a0, compose2(Eq, compose2(a1, compose2(Ep, a2))))
    return comp, pa0 + pa1 + pa2 + [p2, q2]

def stratum_dim(param_fn, p, rng):
    comp, params = param_fn()
    vals = {s: int(rng.integers(2, p if p < 10**6 else 1000)) for s in params}
    cols = []
    for s in params:
        dP = sp.expand(sp.diff(comp[0], s).subs(vals))
        dQ = sp.expand(sp.diff(comp[1], s).subs(vals))
        cols.append(coeffs_of(dP, MON4, M4_idx) + coeffs_of(dQ, MON4, M4_idx))
    Mat = np.array(cols, dtype=object).T
    Mat = np.array([[int(e) % p for e in row] for row in Mat], dtype=np.int64)
    return rref_rank(Mat, p), (comp, params, vals)

def point_from(comp, vals, p):
    P = sp.expand(comp[0].subs(vals))
    Q = sp.expand(comp[1].subs(vals))
    jj = sp.expand(jac2(P, Q))
    assert jj.is_number, "jac not constant?!"
    jj = sp.Rational(jj)
    # normalize: scale first component of the OUTER map: P -> P/jj (post-compose diag(1/jj,1))
    P = sp.expand(P/jj)
    assert sp.expand(jac2(P, Q)) == 1
    return P, Q

def tangent_dim(P, Q, p):
    """dim ker of d(jac)(dP,dQ) = jac(dP,Q)+jac(P,dQ) as 28 x 30 matrix mod p"""
    cols = []
    for m in MON4:
        mono = x**m[0]*y**m[1]
        cols.append(coeffs_of(jac2(mono, Q), MON6, M6_idx))
    for m in MON4:
        mono = x**m[0]*y**m[1]
        cols.append(coeffs_of(jac2(P, mono), MON6, M6_idx))
    Mat = np.array(cols, dtype=object).T
    num = np.array([[sp.Rational(e) for e in row] for row in Mat], dtype=object)
    # clear denominators per column not needed: reduce rationals mod p
    def redr(r_):
        r_ = sp.Rational(r_)
        return (int(r_.p) * pow(int(r_.q) % p, p-2, p)) % p
    Mi = np.array([[redr(e) for e in row] for row in num], dtype=np.int64)
    rank = rref_rank(Mi, p)
    return 30 - rank, rank

for p in (32003, 1000003):
    rng = np.random.default_rng(3)
    print(f"===== p = {p} =====")
    D1, (comp1, par1, val1) = stratum_dim(S4_param, p, rng)
    D2, (comp2, par2, val2) = stratum_dim(S22_param, p, rng)
    print(f"dim S4  (no jac=1) = {D1}  -> jac=1 slice: {D1-1}")
    print(f"dim S22 (no jac=1) = {D2}  -> jac=1 slice: {D2-1}")
    P1, Q1 = point_from(comp1, val1, p)
    P2, Q2 = point_from(comp2, val2, p)
    t1, r1 = tangent_dim(P1, Q1, p)
    t2, r2 = tangent_dim(P2, Q2, p)
    print(f"tangent dim T K_4 at S4 point : {t1} (rank {r1} of 28x30)")
    print(f"tangent dim T K_4 at S22 point: {t2} (rank {r2})")
    print(f"smooth on S4 stratum: {t1 == D1-1} ; smooth on S22 stratum: {t2 == D2-1}")
