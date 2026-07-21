-- Copyright 2026 Marcus Teller. Licensed under the Apache License, Version 2.0.
import Mathlib

/-!
Machine-checked certificates for Alpöge's counterexample to the Jacobian
conjecture in dimension 3 (announced 2026-07-19).

With u = 1 + x*y the map F : ℂ³ → ℂ³ is
  F₁ = u³z + y²u(4+3xy),  F₂ = y + 3xu²z + 3xy²(4+3xy),  F₃ = 2x − 3x²y − x³z.

Certified here:
  1. `alpoge_not_injective` — three distinct rational points share one image.
  2. `alpoge_keller`        — det of the Jacobian matrix is the constant −2,
                              with the entries themselves certified as the
                              formal partial derivatives (`MvPolynomial.pderiv`)
                              of the components, viewed as polynomials in
                              `MvPolynomial (Fin 3) ℤ` (x = X 0, y = X 1,
                              z = X 2); the nine entries are computed
                              explicitly in the lemmas `J₁₁` … `J₃₃`.
  3. `alpoge_identity₁/₂`   — the structural identities behind the rational
                              factorization mechanism.
-/

open MvPolynomial

namespace Alpoge

/-! ### The map, as functions (for point evaluations) -/

def f₁ (x y z : ℚ) : ℚ := (1+x*y)^3 * z + y^2*(1+x*y)*(4+3*x*y)
def f₂ (x y z : ℚ) : ℚ := y + 3*x*(1+x*y)^2 * z + 3*x*y^2*(4+3*x*y)
def f₃ (x y z : ℚ) : ℚ := 2*x - 3*x^2*y - x^3*z

/-- **Non-injectivity**: (0,0,−¼), (1,−3/2,13/2), (−1,3/2,13/2) all map to (−¼,0,0). -/
theorem alpoge_not_injective :
    (f₁ 0 0 (-1/4), f₂ 0 0 (-1/4), f₃ 0 0 (-1/4)) = (-1/4, 0, 0) ∧
    (f₁ 1 (-3/2) (13/2), f₂ 1 (-3/2) (13/2), f₃ 1 (-3/2) (13/2)) = (-1/4, 0, 0) ∧
    (f₁ (-1) (3/2) (13/2), f₂ (-1) (3/2) (13/2), f₃ (-1) (3/2) (13/2)) = (-1/4, 0, 0) := by
  refine ⟨?_, ?_, ?_⟩ <;> simp only [f₁, f₂, f₃, Prod.mk.injEq] <;> norm_num

/-! ### The map, as multivariate polynomials over ℤ (for formal derivatives) -/

noncomputable section

abbrev R := MvPolynomial (Fin 3) ℤ

def x : R := X 0
def y : R := X 1
def z : R := X 2

def F₁ : R := (1+x*y)^3 * z + y^2*(1+x*y)*(4+3*x*y)
def F₂ : R := y + 3*x*(1+x*y)^2 * z + 3*x*y^2*(4+3*x*y)
def F₃ : R := 2*x - 3*x^2*y - x^3*z

/-- The Jacobian matrix, entries defined as formal partial derivatives. -/
def J : Matrix (Fin 3) (Fin 3) R :=
  Matrix.of fun i j => pderiv j (![F₁, F₂, F₃] i)

/-- A formal partial derivative kills numeral constants. -/
lemma pderiv_ofNat (i : Fin 3) (n : ℕ) [n.AtLeastTwo] :
    pderiv i (ofNat(n) : R) = 0 := by
  rw [← map_ofNat (C : ℤ →+* R) n]
  exact pderiv_C

/-! The nine entries of `J`, certified one by one.  The right-hand sides were
computed independently (CAS); each proof unfolds the formal derivative with
the Leibniz rules and closes with `ring`. -/

lemma J₁₁ : pderiv (0 : Fin 3) F₁ =
    3*x^2*y^3*z + 6*x*y^4 + 6*x*y^2*z + 7*y^3 + 3*y*z := by
  simp only [F₁, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 0 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 0 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₁₂ : pderiv (1 : Fin 3) F₁ =
    3*x^3*y^2*z + 12*x^2*y^3 + 6*x^2*y*z + 21*x*y^2 + 3*x*z + 8*y := by
  simp only [F₁, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 1 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 1 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₁₃ : pderiv (2 : Fin 3) F₁ =
    x^3*y^3 + 3*x^2*y^2 + 3*x*y + 1 := by
  simp only [F₁, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 2 by decide),
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 2 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₂₁ : pderiv (0 : Fin 3) F₂ =
    9*x^2*y^2*z + 18*x*y^3 + 12*x*y*z + 12*y^2 + 3*z := by
  simp only [F₂, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 0 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 0 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₂₂ : pderiv (1 : Fin 3) F₂ =
    6*x^3*y*z + 27*x^2*y^2 + 6*x^2*z + 24*x*y + 1 := by
  simp only [F₂, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 1 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 1 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₂₃ : pderiv (2 : Fin 3) F₂ =
    3*x^3*y^2 + 6*x^2*y + 3*x := by
  simp only [F₂, x, y, z, map_add, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 2 by decide),
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 2 by decide), pderiv_one, pderiv_ofNat]
  ring

lemma J₃₁ : pderiv (0 : Fin 3) F₃ =
    -(3*x^2*z) - 6*x*y + 2 := by
  simp only [F₃, x, y, z, map_sub, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 0 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 0 by decide), pderiv_ofNat]
  ring

lemma J₃₂ : pderiv (1 : Fin 3) F₃ = -(3*x^2) := by
  simp only [F₃, x, y, z, map_sub, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 1 by decide),
    pderiv_X_of_ne (show (2 : Fin 3) ≠ 1 by decide), pderiv_ofNat]
  ring

lemma J₃₃ : pderiv (2 : Fin 3) F₃ = -(x^3) := by
  simp only [F₃, x, y, z, map_sub, pderiv_mul, pderiv_pow, pderiv_X_self,
    pderiv_X_of_ne (show (0 : Fin 3) ≠ 2 by decide),
    pderiv_X_of_ne (show (1 : Fin 3) ≠ 2 by decide), pderiv_ofNat]
  ring

set_option maxHeartbeats 1000000 in
-- the 3×3 determinant expansion is a large polynomial identity for `ring`
/-- **Keller property**: det J = −2, a nonzero constant. -/
theorem alpoge_keller : J.det = (-2 : R) := by
  have h : ∀ i j, J i j = pderiv j (![F₁, F₂, F₃] i) := fun _ _ => rfl
  rw [Matrix.det_fin_three]
  simp only [h, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons, J₁₁, J₁₂, J₁₃, J₂₁, J₂₂, J₂₃, J₃₁, J₃₂, J₃₃]
  ring

/-- Structural identity 1: 3x·F₁ − u·F₂ = −y·u. -/
theorem alpoge_identity₁ : 3*x*F₁ - (1+x*y)*F₂ = -(y*(1+x*y)) := by
  simp only [F₁, F₂]; ring

/-- Structural identity 2: x³·F₁ + u³·F₃ = x·u·(u+1). -/
theorem alpoge_identity₂ : x^3*F₁ + (1+x*y)^3*F₃ = x*(1+x*y)*((1+x*y)+1) := by
  simp only [F₁, F₃]; ring

end

end Alpoge
