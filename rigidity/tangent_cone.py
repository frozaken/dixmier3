#!/usr/bin/env python3
"""
Phase 4: kernel of the linearized obstruction at a generic affine-orbit point.
Work in det=const convention (SolC, dim 33; orbit tangent dim 23).
omega(G) = [2 e2(X_G)] in coker(L)+C ; d omega_{v0}(w) = [2(trX0 trXw - tr(X0 Xw))].
Compute dim ker(d omega_{v0}|_{SolC}).  If == 23 -> unobstructed cone = orbit locally.
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

inv2 = pow(2, p-2, p)
def G_from_vec(v):
    return [{m: int(v[i*nGm+k]) % p for k, m in enumerate(G_monos) if int(v[i*nGm+k]) % p}
            for i in range(3)]
def gvec(comps):
    v = np.zeros(NG, dtype=np.int64)
    for i in range(3):
        for m, c in comps[i].items():
            if c % p: v[i*nGm+G_idx[m]] = c % p
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
def Xmat_of(Gc):
    Xm = matmul_poly(ADJ, jac_of(Gc))
    return [[pscale(Xm[i][j], (-inv2) % p) for j in range(3)] for i in range(3)]

# SolC kernel (det = const version): dim 33
KC = kernel_basis(Mc)
print("dim SolC =", KC.shape[0])

# generic orbit point v0 (det-const affine orbit: a o F o b, a,b affine)
rng = np.random.default_rng(23)
Mt = rng.integers(0, p, (3, 3)); vt = rng.integers(0, p, 3)
comps0 = [{}, {}, {}]
for i in range(3):
    for k in range(3):
        comps0[i] = padd(comps0[i], pscale(Fd[k], int(Mt[i, k])))
    comps0[i] = padd(comps0[i], {(0, 0, 0): int(vt[i])})
L_ = rng.integers(0, p, (3, 3)); w_ = rng.integers(0, p, 3)
bfield = [padd({(1,0,0): int(L_[j,0])}, {(0,1,0): int(L_[j,1])}, {(0,0,1): int(L_[j,2])},
               {(0,0,0): int(w_[j])}) for j in range(3)]
compsB = [padd(*[pmul(dF[i][j], bfield[j]) for j in range(3)]) for i in range(3)]
v0 = (gvec(comps0) + gvec(compsB)) % p
X0 = Xmat_of(G_from_vec(v0))
trX0 = mat_trace(X0)

# bilinear obstruction rows for each w in KC basis
# r_w = 2( trX0*trXw - tr(X0 Xw) )  in poly space, degrees up to 34
BIG_monos = monomials_upto(34); BIG_idx = midx(BIG_monos)
Rcols = []
for kv in KC:
    Xw = Xmat_of(G_from_vec(kv))
    trXw = mat_trace(Xw)
    pol = padd(pmul(trX0, trXw), pscale(mat_trace(matmul_poly(X0, Xw)), p-1))
    r = pscale(pol, 2)
    col = np.zeros(len(BIG_monos), dtype=np.int64)
    for m, c in r.items():
        if m != (0,0,0):     # constants absorbed
            col[BIG_idx[m]] = c
    Rcols.append(col)
R = np.array(Rcols, dtype=np.int64).T   # rows = monomials deg<=34, cols = 33

# extend Mc to the big row space (rows deg 18..34 are zero for L)
Mbig = np.zeros((len(BIG_monos), Mc.shape[1]), dtype=np.int64)
for m in EQ_monos_c:
    Mbig[BIG_idx[m]] = Mc[EQc_idx[m]]
rk_M = rank_of(Mbig)
rk_MR = rank_of(np.hstack([Mbig, R]))
img_dim = rk_MR - rk_M
ker_dim = KC.shape[0] - img_dim
print("rank of d_omega|_SolC modulo Im(L)+C :", img_dim)
print("dim ker d_omega at generic orbit point:", ker_dim, " (orbit tangent = 23)")
