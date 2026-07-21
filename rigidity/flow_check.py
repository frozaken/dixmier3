#!/usr/bin/env python3
"""Check whether the non-affine div-free B directions integrate via the flow:
F o phi_t has Taylor terms (1/k!)(B.grad)^k F ; if all have deg <= 7, the family
F o exp(tB) stays in K_{<=7} to that order (and is composition-trivial)."""
import sympy as sp
import numpy as np

p = 1000003
x, y, z = sp.symbols('x y z')
VARS = (x, y, z)
u = 1 + x*y
F = [sp.expand(u**3*z + y**2*u*(4 + 3*x*y)),
     sp.expand(y + 3*x*u**2*z + 3*x*y**2*(4 + 3*x*y)),
     sp.expand(2*x - 3*x**2*y - x**3*z)]
dFsym = [[sp.diff(F[i], VARS[j]) for j in range(3)] for i in range(3)]
def poly_to_dict(expr):
    q = sp.Poly(sp.expand(expr), x, y, z)
    return {m: int(c) % p for m, c in zip(q.monoms(), q.coeffs())}
dF = [[poly_to_dict(dFsym[i][j]) for j in range(3)] for i in range(3)]

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
def pderiv(q, j):
    out = {}
    for m, c in q.items():
        d = dmono(m, j)
        if d:
            cf, dm = d
            out[dm] = (out.get(dm, 0) + c*cf) % p
    return {m: c for m, c in out.items() if c}
def pdeg(q): return max((sum(m) for m in q), default=-1)

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

# rebuild B-side kernel (same construction/ordering as before)
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

Fd = [poly_to_dict(f) for f in F]
for t in [6, 11, 13, 14, 15, 16, 17, 18]:
    kv = KB[t]
    Bf = [{}, {}, {}]
    for j in range(3):
        for k, bm in enumerate(B_monos):
            c = int(kv[j*len(B_monos)+k])
            if c: Bf[j][bm] = c
    degs = []
    cur = [dict(f) for f in Fd]
    for k in range(1, 9):
        cur = [padd(*[pmul(Bf[j], pderiv(ci, j)) for j in range(3)]) for ci in cur]
        degs.append(max(pdeg(ci) for ci in cur))
        if degs[-1] < 0:
            break
    print(f"B-basis {t}: degB={[pdeg(b) for b in Bf]}, deg (B.grad)^k F, k=1..: {degs}")
