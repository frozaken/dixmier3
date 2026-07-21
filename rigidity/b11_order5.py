#!/usr/bin/env python3
"""
ORDER-5 obstruction decision for the B11 = y*(x/3,-2y/3,z) branch of Alpoge's F.

The task asked to work over F_{p^2}, on the premise that the surviving order-4
solution branch (7 quadrics, 32 unknowns, /tmp/surfaceA/b11_system_{p}.npz) has
points only over F_{p^2} (Galois-conjugate pairs), not F_p.  THAT PREMISE IS
FALSE: it is an artefact of a break-bug in b11_point.py's `minpoly_roots`
(`if r_==nrows: break`), which returns an EMPTY root set for a nonderogatory
2x2 Krylov matrix and was misread as "no F_p eigenvalue -> points irrational".
With the bug fixed, a correct lex-Groebner triangular solve produces GENUINE
F_p points on the branch, verified to kill all 7 quadrics.  The order-4 scheme
is generically NON-REDUCED (multiplicity 2) with F_p-rational reduced points
(the elimination poly of a generic slice is (s-c)^2, c in F_p).

Because F_p points exist on the branch, the order-5 test is carried out AT those
genuine branch points.  The full F_{p^2}=F_p[w]/(w^2-r) layer is retained and
exercised (any w-parts stay 0 for an F_p point); order-5 solvability is an
F_p-linear condition (L-operator Mc and the cols freedom are over F_p), so its
verdict is identical over F_p and F_{p^2}.  We evaluate at SEVERAL distinct
branch points and confirm the verdict is stable.

Method = order4.py's exact t^5 machinery: build the base tower for B11, move to
the branch point t*, solve order 4 for H4, then test whether r5 in Im(L)+span(cols)
(H5 with the H4-kernel freedom).  Full t^4 and t^5 identities are re-verified
exactly.  argv: prime.
"""
import sympy as sp
import numpy as np
import sys, time

p = int(sys.argv[1]) if len(sys.argv) > 1 else 1000003
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)
log("ORDER-5 / B11 / prime p =", p)

# ---- inherit ALL F_p machinery from order4_certify.py (no driver) ----
_src = open('/Users/marcusteller/repos/pjc/experiments/surfaceA/order4_certify.py').read()
_src = _src.split("for Bf, name in")[0]
_src = _src.replace("p = int(sys.argv[1]) if len(sys.argv) > 1 else 1000003", f"p = {p}")
exec(_src)
log("inherited F_p machinery; NG =", NG, " dim ker L =", KC.shape[0])
inv2mod = pow(2, p-2, p)

# =====================================================================
# F_{p^2} = F_p[w]/(w^2 - r)     (both primes are 3 mod 4)
# =====================================================================
def nonresidue(pp):
    for rr in range(2, 10000):
        if pow(rr, (pp-1)//2, pp) == pp-1:
            return rr
    raise RuntimeError("no nonresidue")
r = nonresidue(p)
assert p % 4 == 3
def sqrt_fp(a):
    a %= p; s = pow(a, (p+1)//4, p)
    assert (s*s) % p == a; return s
inv_r = pow(r, p-2, p)
log(f"F_p^2 = F_p[w]/(w^2 - {r})")

def s_add(u, v): return ((u[0]+v[0]) % p, (u[1]+v[1]) % p)
def s_sub(u, v): return ((u[0]-v[0]) % p, (u[1]-v[1]) % p)
def s_mul(u, v):
    a, b = u; c, dd = v
    return ((a*c + r*b*dd) % p, (a*dd + b*c) % p)
def s_scal(u, c): c %= p; return ((u[0]*c) % p, (u[1]*c) % p)
def s_inv(u):
    a, b = u; den = (a*a - r*b*b) % p; di = pow(den, p-2, p)
    return ((a*di) % p, ((-b) % p * di) % p)
def s_is0(u): return u[0] % p == 0 and u[1] % p == 0

# =====================================================================
# reconstruct b11_deep's order<=3 base for B11  (verbatim construction)
# =====================================================================
Gc = G_of_B(B11, "B11 = (xy/3,-2y^2/3,yz)")
X_ = Xmat_of(Gc); trX = mat_trace(X_)
assert all(k == 0 for k in trX)
r2 = pscale(e2_of(X_), 2)
h0 = system_solve([], r2)[0]
Y0 = Xmat_of(comps_from_vec(h0)); detX = det_of(X_)
cols = [pscale(psub(pmul(trX, mat_trace(W_)), tr_prod(X_, W_)), 2) for W_ in YW]   # 33
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
s_rows = piv_rows; ns = len(s_rows)
Lcols, Wlist, dZlist = [], [], []
for ri in s_rows:
    kv = ker3[ri]; dh3 = kv[:NG] % p; ds = kv[NG:] % p
    dZ = Xmat_of(comps_from_vec(dh3))
    W_ = [[padd(*[pscale(YW[t][i][j], int(ds[t])) for t in range(len(YW)) if ds[t]])
           for j in range(3)] for i in range(3)]
    Lc = padd(pscale(psub(pmul(trX, mat_trace(dZ)), tr_prod(X_, dZ)), 2),
              pscale(bil_e2(Ystar, W_), 2), pscale(tr_prod(adjX, W_), 2))
    Lcols.append(Lc); Wlist.append(W_); dZlist.append(dZ)
Qpairs = {}
for a in range(ns):
    for b in range(a, ns):
        Qpairs[(a, b)] = bil_e2(Wlist[a], Wlist[b])
log(f"reconstructed order<=3 base; ns (H-moving dirs) = {ns}")

# ---- reduce mod Span, cross-check vs saved npz ----
gens = [r4_0] + Lcols + list(Qpairs.values())
support = set(EQ_keys_c)
for q in gens + cols: support |= set(q.keys())
support.discard(0)
skeys = sorted(support); sidx = {k: i for i, k in enumerate(skeys)}; nsup = len(skeys)
def vec_of(q):
    v = np.zeros(nsup, dtype=np.int64)
    for k, c in q.items():
        if k: v[sidx[k]] = c % p
    return v
S = np.zeros((nsup, NG + len(cols)), dtype=np.int64)
for r_, k in enumerate(EQ_keys_c): S[sidx[k], :NG] = Mc[r_]
for j, cp in enumerate(cols):
    for k, c in cp.items():
        if k: S[sidx[k], NG + j] = c % p
RS, pivS = rref(S.T % p); E = RS[:len(pivS)]
def reduce_vec(v):
    w = v % p
    for ri, pc in enumerate(pivS):
        c = w[pc] % p
        if c: w = (w - c * E[ri]) % p
    return w
V0r = reduce_vec(vec_of(r4_0))
Lred = np.array([reduce_vec(vec_of(q)) for q in Lcols])
keys = list(Qpairs.keys())
Qmat = np.array([reduce_vec(vec_of(Qpairs[k])) for k in keys])
d = np.load(f'/tmp/surfaceA/b11_system_{p}.npz')
coords = d['coords']
assert np.array_equal(V0r[coords] % p, d['V0'] % p)
assert np.array_equal(Lred[:, coords] % p, d['L'] % p)
assert np.array_equal(Qmat[:, coords] % p, d['Q'] % p)
assert np.array_equal(np.array(keys), d['keys'])
neq = len(coords)
V0e, Le, Qe, keye = d['V0'], d['L'], d['Q'], d['keys']
log(f"reduced system cross-checked vs saved npz: {neq} equations, {ns} unknowns")

# =====================================================================
# point finder: genuine points on the order-4 variety via lex-GB
# triangular solve.  Handles F_p roots (default here) and, if a level's
# univariate is an irreducible quadratic, an F_{p^2} root (my sqrt).
# =====================================================================
ts = sp.symbols(f't0:{ns}')
eqs = []
for e in range(neq):
    expr = int(V0e[e])
    for a in range(ns):
        if Le[a, e]: expr += int(Le[a, e]) * ts[a]
    for ci in range(Qe.shape[0]):
        a, b = int(keye[ci][0]), int(keye[ci][1]); c = int(Qe[ci, e])
        if c: expr += (2*c if a != b else c) * ts[a] * ts[b]
    eqs.append(sp.expand(expr))

def reduced_residual_fp2(tA, tB):
    """residual of the saved reduced system at t=(tA+tB*w); returns max nonzero flag."""
    for e in range(neq):
        va, vb = int(V0e[e]) % p, 0
        for a in range(ns):
            if Le[a, e]:
                va = (va + Le[a, e]*tA[a]) % p; vb = (vb + Le[a, e]*tB[a]) % p
        for ci in range(Qe.shape[0]):
            a2, b2 = int(keye[ci][0]), int(keye[ci][1]); c = int(Qe[ci, e])
            if c:
                coef = (2*c if a2 != b2 else c) % p
                pr = s_mul((int(tA[a2]), int(tB[a2])), (int(tA[b2]), int(tB[b2])))
                va = (va + coef*pr[0]) % p; vb = (vb + coef*pr[1]) % p
        if va % p or vb % p:
            return False
    return True

def solve_slice_point(se, svars, rng):
    """cut se to 0-dim, lex-GB, triangular solve over F_{p^2}.  Returns list of
    F_{p^2} (a,b) values for svars, or None."""
    cur = list(se); ncut = 0
    while True:
        G = sp.groebner(cur, *svars, modulus=p, order='grevlex')
        if any(g.is_number and g != 0 for g in G.exprs):
            return None
        LTs = [sp.Poly(g, *svars, modulus=p).monoms()[0] for g in G.exprs]
        def divisible(m1, m2): return all(x >= y for x, y in zip(m1, m2))
        basis = []; frontier = [tuple([0]*len(svars))]; seen = set(frontier); big = False
        while frontier:
            m = frontier.pop(); basis.append(m)
            if len(basis) > 220: big = True; break
            for j in range(len(svars)):
                m2 = tuple(mm + (1 if k == j else 0) for k, mm in enumerate(m))
                if m2 in seen or any(divisible(m2, lt) for lt in LTs): continue
                seen.add(m2); frontier.append(m2)
        if big:
            if ncut >= 10: return None
            cut = sum(int(rng.integers(1, p))*svars[j] for j in range(len(svars))) \
                  + int(rng.integers(0, p))
            cur = list(cur) + [sp.expand(cut)]; ncut += 1; continue
        break
    Gl = sp.groebner(cur, *svars, modulus=p, order='lex')
    # triangular solve over F_{p^2}
    vals = {}   # svar -> (a,b)
    for g in reversed(Gl.exprs):
        P = sp.Poly(g, *svars, modulus=p)
        # substitute already-known F_{p^2} values, collect univariate in remaining top var
        rem = [v for v in svars if P.degree(v) > 0 and v not in vals]
        if not rem:
            continue
        if len(rem) > 1:
            continue   # not triangular in a single unknown yet; a later gen resolves it
        v = rem[-1]
        # build F_{p^2} coefficients of powers of v
        coeffP = {}   # power -> (a,b)
        for mono, co in zip(P.monoms(), P.coeffs()):
            cval = (int(co) % p, 0)
            pw = 0
            for idx, vv in enumerate(svars):
                e_ = mono[idx]
                if e_ == 0: continue
                if vv is v: pw = e_
                else:
                    cval = s_mul(cval, pow_fp2(vals[vv], e_))
            coeffP[pw] = s_add(coeffP.get(pw, (0, 0)), cval)
        deg = max(coeffP)
        root = solve_univar_fp2(coeffP, deg)
        if root is None:
            return None
        vals[v] = root
    for vv in svars:
        vals.setdefault(vv, (0, 0))
    return [vals[vv] for vv in svars]

def pow_fp2(u, e):
    res = (1, 0)
    for _ in range(e): res = s_mul(res, u)
    return res

def solve_univar_fp2(coeffP, deg):
    """one root in F_{p^2} of sum coeffP[k]*v^k.  Handles deg 1 and (irreducible
    or split) deg 2.  Returns (a,b) or None."""
    if deg == 1:
        c1 = coeffP.get(1, (0, 0)); c0 = coeffP.get(0, (0, 0))
        if s_is0(c1): return None
        return s_mul(s_sub((0, 0), c0), s_inv(c1))
    if deg == 2:
        a2 = coeffP.get(2, (0, 0)); a1 = coeffP.get(1, (0, 0)); a0 = coeffP.get(0, (0, 0))
        if s_is0(a2):
            return solve_univar_fp2(coeffP, 1)
        ia = s_inv(a2)
        B_ = s_mul(a1, ia); C_ = s_mul(a0, ia)
        disc = s_sub(s_mul(B_, B_), s_scal(C_, 4))
        sq = s_sqrt_fp2(disc)
        if sq is None: return None
        inv2 = pow(2, p-2, p)
        return s_scal(s_sub(sq, B_), inv2)
    return None

def s_sqrt_fp2(D):
    """a square root of D in F_{p^2}, or None if none (should not happen in
    an algebraically-closed-enough sense: F_{p^2} closes all degree<=2)."""
    a, b = D[0] % p, D[1] % p
    if b == 0:
        if a == 0: return (0, 0)
        if pow(a, (p-1)//2, p) == 1:      # a is F_p square
            return (sqrt_fp(a), 0)
        # a nonresidue: sqrt(a) = sqrt_p(a/r)*w
        return (0, sqrt_fp(a * inv_r % p))
    # general a+bw: solve (x+yw)^2 = a+bw -> x^2+r y^2 = a, 2xy = b
    # norm N = x^2 - r y^2? here w^2=r so (x+yw)^2 = x^2 + r y^2 + 2xy w
    # => x^2 + r y^2 = a, 2 x y = b.  Let s=x^2: s + r*(b/(2x))^2? solve quartic.
    # Use: x^2 - a + r*(b^2/(4x^2)) = 0 -> 4 x^4 - 4a x^2 + r b^2 = 0
    # t=x^2: 4 t^2 - 4a t + r b^2 = 0 -> t = (4a +- sqrt(16a^2 -16 r b^2))/8
    disc = (16*a*a - 16*r*b*b) % p
    if pow(disc, (p-1)//2, p) not in (0, 1):
        return None
    sd = sqrt_fp(disc % p) if disc % p else 0
    inv8 = pow(8, p-2, p)
    for sgn in (1, p-1):
        t = ((4*a + sgn*sd) % p * inv8) % p
        if pow(t, (p-1)//2, p) == 1 or t == 0:
            x = sqrt_fp(t) if t else 0
            if x == 0: continue
            y = (b * pow(2*x, p-2, p)) % p
            if (x*x + r*y*y) % p == a and (2*x*y) % p == b:
                return (x, y)
    return None

# ---- collect several distinct genuine branch points ----
rng = np.random.default_rng(p % 1009 + 2024)
points = []            # list of (tA, tB, genuine_fp2, slice_idx)
seen_pts = set()
for trial in range(30):
    if len(points) >= 3: break
    A = rng.integers(0, p, (ns, 7)); bvec = rng.integers(0, p, ns)
    s = sp.symbols('s0:7')
    sub = {ts[a]: sum(int(A[a, j])*s[j] for j in range(7)) + int(bvec[a]) for a in range(ns)}
    se = [sp.expand(e.subs(sub)) for e in eqs]
    sval = solve_slice_point(se, list(s), rng)
    if sval is None:
        continue
    tA = np.zeros(ns, dtype=np.int64); tB = np.zeros(ns, dtype=np.int64)
    for a in range(ns):
        acc = (int(bvec[a]) % p, 0)
        for j in range(7):
            acc = s_add(acc, s_scal(sval[j], int(A[a, j])))
        tA[a], tB[a] = acc[0] % p, acc[1] % p
    if not reduced_residual_fp2(tA, tB):
        continue
    key = tuple(tA.tolist()) + tuple(tB.tolist())
    if key in seen_pts:
        continue
    seen_pts.add(key)
    genuine = bool(np.any(tB % p))
    points.append((tA, tB, genuine, trial))
    log(f"branch point #{len(points)} (slice {trial}) verified; "
        f"genuinely F_(p^2) (w-part!=0): {genuine}")
if not points:
    log("FATAL: no branch point found"); sys.exit(1)
np.savez(f'/tmp/surfaceA/b11_points_{p}.npz',
         **{f'tA{i}': pt[0] for i, pt in enumerate(points)},
         **{f'tB{i}': pt[1] for i, pt in enumerate(points)}, r=np.array([r]))

# =====================================================================
# F_{p^2} POLYNOMIAL layer (P0 + w*P1) on top of the F_p ops
# =====================================================================
def FADD(*Ps): return (padd(*[P[0] for P in Ps]), padd(*[P[1] for P in Ps]))
def FSUB(A, B): return (psub(A[0], B[0]), psub(A[1], B[1]))
def FMUL(A, B):
    lo = padd(pmul(A[0], B[0]), pscale(pmul(A[1], B[1]), r))
    hi = padd(pmul(A[0], B[1]), pmul(A[1], B[0]))
    return (lo, hi)
def Fscal_fp(A, c): c %= p; return (pscale(A[0], c), pscale(A[1], c))
def Fscal_s(A, sc):
    s0, s1 = sc
    lo = padd(pscale(A[0], s0), pscale(A[1], (r*s1) % p))
    hi = padd(pscale(A[0], s1), pscale(A[1], s0))
    return (lo, hi)
def Fmat_of(M): return [[(M[i][j], {}) for j in range(3)] for i in range(3)]
def Fmat_trace(A): return FADD(A[0][0], A[1][1], A[2][2])
def Ftr_prod(A, B):
    return FADD(*[FMUL(A[i][j], B[j][i]) for i in range(3) for j in range(3)])
def Fe2(X):
    t_ = Fmat_trace(X)
    return Fscal_fp(FSUB(FMUL(t_, t_), Ftr_prod(X, X)), inv2mod)
def Fadj(X):
    Am = [[None]*3 for _ in range(3)]; idx = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        for j in range(3):
            rr = idx[i]; cc = idx[j]
            m = FSUB(FMUL(X[rr[0]][cc[0]], X[rr[1]][cc[1]]),
                     FMUL(X[rr[0]][cc[1]], X[rr[1]][cc[0]]))
            Am[j][i] = m if (i+j) % 2 == 0 else Fscal_fp(m, p-1)
    return Am
def Fjac_of(comps):
    P0 = jac_of([c[0] for c in comps]); P1 = jac_of([c[1] for c in comps])
    return [[(P0[i][j], P1[i][j]) for j in range(3)] for i in range(3)]
_ADJf = [[(ADJ[i][j], {}) for j in range(3)] for i in range(3)]
def FXmat_of(comps):
    Jm = Fjac_of(comps)
    Xm = [[FADD(*[FMUL(_ADJf[i][k], Jm[k][j]) for k in range(3)])
           for j in range(3)] for i in range(3)]
    return [[Fscal_fp(Xm[i][j], (-inv2mod) % p) for j in range(3)] for i in range(3)]
def comps_from_vec_fp2(vpair):
    v0, v1 = vpair; c0 = comps_from_vec(v0); c1 = comps_from_vec(v1)
    return [(c0[i], c1[i]) for i in range(3)]
def Fconst_only(A, tag):
    for part, nm in ((A[0], "1-part"), (A[1], "w-part")):
        bad = {k: c for k, c in part.items() if k != 0}
        assert not bad, f"{tag}: {nm} non-constant, e.g. {list(bad.items())[:3]}"

Xf = Fmat_of(X_); adjXf = Fmat_of(adjX); trXf = (trX, {})
Ystar_f = Fmat_of(Ystar); Zstar_f = Fmat_of(Zstar)
Wlist_f = [Fmat_of(W_) for W_ in Wlist]; dZlist_f = [Fmat_of(dZ) for dZ in dZlist]
negcols = [pscale(c_, p-1) for c_ in cols]

def rank_report(rhs):
    support = set(EQ_keys_c)
    for cp in negcols: support |= set(cp.keys())
    support |= set(rhs.keys()); support.discard(0)
    sk = sorted(support); si = {k: i for i, k in enumerate(sk)}
    ncx = NG + len(negcols)
    Amat = np.zeros((len(sk), ncx + 1), dtype=np.int64)
    for r_, k in enumerate(EQ_keys_c): Amat[si[k], :NG] = Mc[r_]
    for j, cp in enumerate(negcols):
        for k, c in cp.items():
            if k: Amat[si[k], NG + j] = c % p
    for k, c in rhs.items():
        if k: Amat[si[k], ncx] = c % p
    return len(rref(Amat[:, :ncx])[1]), len(rref(Amat)[1])

def solve_split(rhs_pair):
    s0 = system_solve(negcols, rhs_pair[0]); s1 = system_solve(negcols, rhs_pair[1])
    if s0 is None or s1 is None:
        return None, (s0 is not None, s1 is not None)
    return ((s0[0], s1[0]), (s0[1], s1[1])), (True, True)

def run_order5(tA, tB, tag):
    tstar_sc = [(int(tA[a]), int(tB[a])) for a in range(ns)]
    def combine(base_f, list_f):
        return [[FADD(base_f[i][j],
                      *[Fscal_s(list_f[a][i][j], tstar_sc[a]) for a in range(ns)])
                 for j in range(3)] for i in range(3)]
    Yt = combine(Ystar_f, Wlist_f); Zt = combine(Zstar_f, dZlist_f)
    Fconst_only(FADD(Fmat_trace(Yt), Fe2(Xf)), f"{tag} order2")
    Fconst_only(FADD(Fmat_trace(Zt),
                     FSUB(FMUL(trXf, Fmat_trace(Yt)), Ftr_prod(Xf, Yt)),
                     (detX, {})), f"{tag} order3")
    # order 4
    r4t = Fscal_fp(FADD(FSUB(FMUL(trXf, Fmat_trace(Zt)), Ftr_prod(Xf, Zt)),
                        Fe2(Yt), Ftr_prod(adjXf, Yt)), 2)
    o4, o4stat = solve_split(r4t)
    assert o4 is not None, f"{tag}: order-4 unexpectedly obstructed (parts {o4stat})"
    h4, mu = o4
    U = FXmat_of(comps_from_vec_fp2(h4))
    mu0, mu1 = mu
    Zf = [[FADD(Zt[i][j], *[Fscal_s((YW[t][i][j], {}), (int(mu0[t]), int(mu1[t])))
                            for t in range(len(YW))])
           for j in range(3)] for i in range(3)]
    Fconst_only(FADD(Fmat_trace(U), FSUB(FMUL(trXf, Fmat_trace(Zf)), Ftr_prod(Xf, Zf)),
                     Fe2(Yt), Ftr_prod(adjXf, Yt)), f"{tag} order4 FULL")
    # order 5
    adjYt = Fadj(Yt)
    r5 = Fscal_fp(FADD(FSUB(FMUL(trXf, Fmat_trace(U)), Ftr_prod(Xf, U)),
                       FSUB(FMUL(Fmat_trace(Yt), Fmat_trace(Zf)), Ftr_prod(Yt, Zf)),
                       Ftr_prod(adjXf, Zf), Ftr_prod(adjYt, Xf)), 2)
    rc0, ra0 = rank_report(r5[0]); rc1, ra1 = rank_report(r5[1])
    o5, o5stat = solve_split(r5)
    exists = o5 is not None
    if exists:
        h5, nu = o5; V = FXmat_of(comps_from_vec_fp2(h5)); nu0, nu1 = nu
        Uf = [[FADD(U[i][j], *[Fscal_s((YW[t][i][j], {}), (int(nu0[t]), int(nu1[t])))
                               for t in range(len(YW))])
               for j in range(3)] for i in range(3)]
        Fconst_only(FADD(Fmat_trace(Uf), FSUB(FMUL(trXf, Fmat_trace(Zf)), Ftr_prod(Xf, Zf)),
                         Fe2(Yt), Ftr_prod(adjXf, Yt)), f"{tag} order4 after nu-shift")
        Fconst_only(FADD(Fmat_trace(V),
                         FSUB(FMUL(trXf, Fmat_trace(Uf)), Ftr_prod(Xf, Uf)),
                         FSUB(FMUL(Fmat_trace(Yt), Fmat_trace(Zf)), Ftr_prod(Yt, Zf)),
                         Ftr_prod(adjXf, Zf), Ftr_prod(adjYt, Xf)), f"{tag} order5 FULL")
    return exists, (rc0, ra0, ra1), o5stat

verdicts = []
for i, (tA, tB, genuine, sl) in enumerate(points):
    tag = f"pt{i}"
    exists, ranks, o5stat = run_order5(tA, tB, tag)
    rc0, ra0, ra1 = ranks
    log(f"[{tag}] genuine_fp2={genuine}; order-5 [Mc|-cols] rank={rc0}, "
        f"aug(1-part)={ra0}, aug(w-part)={ra1}; "
        f"parts solvable={o5stat}; ORDER-5 {'EXISTS' if exists else 'OBSTRUCTED'}"
        + ("  [full t^5 identity verified]" if exists else ""))
    verdicts.append(exists)

allsame = len(set(verdicts)) == 1
final = verdicts[0]
log(f"VERDICT p={p}: ORDER-5 {'EXISTS' if final else 'OBSTRUCTED'} on the B11 branch "
    f"(evaluated at {len(verdicts)} distinct branch points; "
    f"{'ALL AGREE' if allsame else '*** POINTS DISAGREE ***'})")
log("DONE")
