# Copyright 2026 Marcus Teller. Licensed under the Apache License, Version 2.0.
"""CAS certificate: explicit non-invertible endomorphism of the third Weyl algebra A_3,
built from the Alpoge Keller counterexample F (det JF = -2), and its classical shadow:
a non-invertible Poisson (symplectic, det=1) endomorphism of C^6.

phi(x_i)   = F_i(x)                                  [commute: polynomials in x]
phi(d_i)   = D_i = sum_k N[k,i](x) d_k,  N = adj(JF)/(-2) = (JF)^{-1}  [coefficients LEFT]

Certified here:
  (1) det JF = -2 (constant), N polynomial;
  (2) [D_i, F_j] = delta_ij   (from JF.N = I; first-order op => no quantum corrections);
  (3) [D_i, D_j] = 0          (flatness identity, checked as polynomial identity);
  (4) cotangent lift Phi(x,y) = (F(x), N(x)^T y) on C^6 preserves the canonical Poisson
      bracket exactly and has det J_Phi = 1.
Non-automorphism of phi is a theorem (maximal-abelian argument), not CAS: C[x] is maximal
commutative in A_3; phi(C[x]) = C[F] is strictly smaller than C[x] (F is 3:1), and its
centralizer contains C[x], so phi cannot be an automorphism. A_3 simple => phi injective.
"""
import sympy as sp

x, y, z = sp.symbols('x y z')
X = [x, y, z]
u = 1 + x*y
F = sp.Matrix([u**3*z + y**2*u*(4 + 3*x*y),
               y + 3*x*u**2*z + 3*x*y**2*(4 + 3*x*y),
               2*x - 3*x**2*y - x**3*z])
J = F.jacobian(X)
d = sp.expand(J.det())
assert d == -2, d
N = (J.adjugate() / d).applyfunc(sp.expand)
assert sp.expand(J*N - sp.eye(3)) == sp.zeros(3, 3)          # => [D_i,F_j]=delta_ij
for i in range(3):
    for j in range(i+1, 3):
        for l in range(3):
            e = sp.expand(sum(N[k, i]*sp.diff(N[l, j], X[k])
                              - N[k, j]*sp.diff(N[l, i], X[k]) for k in range(3)))
            assert e == 0, (i, j, l, e)                       # => [D_i,D_j]=0
print("A_3 endomorphism relations certified.")

# classical shadow on C^6
Y = sp.symbols('p1 p2 p3')
Yp = [sum(N[k, i]*Y[k] for k in range(3)) for i in range(3)]
def pb(f, g):
    return sp.expand(sum(sp.diff(f, X[k])*sp.diff(g, Y[k])
                         - sp.diff(f, Y[k])*sp.diff(g, X[k]) for k in range(3)))
for i in range(3):
    for j in range(3):
        assert pb(F[i], F[j]) == 0
        assert pb(Yp[i], Yp[j]) == 0
        assert pb(F[i], Yp[j]) == (1 if i == j else 0)        # {x_i,p_j} convention
Phi = sp.Matrix(list(F) + Yp)
assert sp.expand(Phi.jacobian(list(X)+list(Y)).det()) == 1
print("Poisson endomorphism of C^6 certified: bracket-preserving, det = 1.")

# Contact corollary: Phi preserves the Liouville form lambda = sum y_i dx_i exactly,
# i.e. J^T (N^T y) = y, equivalent to (NJ)^T = I. Hence the contactization
# (x, y, z) |-> (F, N^T y, z) of C^7 preserves alpha = dz - lambda on the nose.
_liouville = sp.expand(J.T * (N.T * sp.Matrix(Y)) - sp.Matrix(Y))
assert _liouville == sp.zeros(3, 1), _liouville
print("Contact certificate: Phi^* lambda = lambda (Liouville form preserved exactly).")
