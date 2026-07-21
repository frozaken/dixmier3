-- Copyright 2026 Marcus Teller. Licensed under the Apache License, Version 2.0.
import Pjc.Alpoge

/-!
Machine-checked certificates for the polynomial-identity core of the
constructive disproof of the Dixmier conjecture DC₃ (third Weyl algebra),
built from Alpöge's Jacobian counterexample `F = (F₁, F₂, F₃)`.

The endomorphism φ of A₃ sends xᵢ ↦ Fᵢ and ∂ᵢ ↦ Dᵢ = Σₖ N k i · ∂ₖ where
N = (JF)⁻¹ = adjugate(JF) / (−2) is polynomial because det JF = −2
(`Alpoge.alpoge_keller`).  Division by −2 is not available over
`R = MvPolynomial (Fin 3) ℤ`, so every identity below is certified in
scalar-cleared form with `M := (Alpoge.J).adjugate` (so `M = −2·N`):

1. `dixmier_inverse` / `dixmier_inverse'` — J·M = (−2)•1 = M·J.
   Cleared form of JF·N = 1 = N·JF, which gives [Dᵢ, Fⱼ] = δᵢⱼ (the Dᵢ are
   first-order operators, so there are no quantum corrections).

2. `dixmier_flatness` — for all i j l : Fin 3,
   Σₖ (M k i · ∂ₖ(M l j) − M k j · ∂ₖ(M l i)) = 0.
   Cleared form ((−2)² = 4 clears both terms uniformly) of the flatness
   identity Σₖ (N k i · ∂ₖ(N l j) − N k j · ∂ₖ(N l i)) = 0, which gives
   [Dᵢ, Dⱼ] = 0.  The nine nontrivial instances (i < j) are the lemmas
   `flat₀₁₀` … `flat₁₂₂`; the full statement follows by antisymmetry.

3. `poisson_det_scaled` — the classical shadow: the cotangent lift
   Φ(x, y) = (F(x), Nᵀy) on ℂ⁶ has block lower-triangular Jacobian
   [[JF, 0], [∗, Nᵀ]], so det JΦ = det JF · det Nᵀ = (−2)·(−1/2) = 1.
   Scalar-cleared: for EVERY lower-left block C,
   det (fromBlocks J 0 C Mᵀ) = −8 = (−2)³ · 1, using det M = 4
   (`det_adjugate_J`).  Quantifying over C certifies that the
   x-derivative block of Nᵀy is irrelevant to the determinant.

Every polynomial identity was CAS-verified (sympy) in exactly the stated
scalar-cleared form before formalization; see
`experiments/dixmier_lean_gen.py` and
`experiments/dixmier3_endomorphism_check.py`.

Non-invertibility of φ itself is a theorem (maximal-abelian argument), not
a polynomial identity, and is not formalized here.
-/

open MvPolynomial

namespace Dixmier

open Alpoge

noncomputable section

/-! ### The adjugate of the Jacobian, entry by entry

The nine polynomials below are the entries of `(Alpoge.J).adjugate`
(row i, column j for `mᵢⱼ`), computed independently by CAS. -/

def m₀₀ : R := -6*x^6*y*z - 18*x^5*y^2 - 6*x^5*z - 6*x^4*y + 8*x^3
def m₀₁ : R := 3*x^6*y^2*z + 9*x^5*y^3 + 6*x^5*y*z + 12*x^4*y^2 + 3*x^4*z - x^3*y
  - 3*x^2
def m₀₂ : R := 3*x^6*y^4*z + 9*x^5*y^5 + 12*x^5*y^3*z + 30*x^4*y^4 + 18*x^4*y^2*z
  + 32*x^3*y^3 + 12*x^3*y*z + 9*x^2*y^2 + 3*x^2*z - 3*x*y - 1
def m₁₀ : R := -6*x^4*y*z - 18*x^3*y^2 - 6*x^3*z - 6*x^2*y + 6*x
def m₁₁ : R := 3*x^4*y^2*z + 9*x^3*y^3 + 6*x^3*y*z + 12*x^2*y^2 + 3*x^2*z - 2
def m₁₂ : R := 3*x^4*y^4*z + 9*x^3*y^5 + 12*x^3*y^3*z + 30*x^2*y^4 + 18*x^2*y^2*z
  + 33*x*y^3 + 12*x*y*z + 12*y^2 + 3*z
def m₂₀ : R := 18*x^5*y*z^2 + 90*x^4*y^2*z + 18*x^4*z^2 + 108*x^3*y^3 + 60*x^3*y*z
  + 54*x^2*y^2 - 18*x^2*z - 42*x*y - 2
def m₂₁ : R := -9*x^5*y^2*z^2 - 45*x^4*y^3*z - 18*x^4*y*z^2 - 54*x^3*y^4
  - 75*x^3*y^2*z - 9*x^3*z^2 - 81*x^2*y^3 - 21*x^2*y*z - 6*x*y^2 + 6*x*z + 16*y
def m₂₂ : R := -9*x^5*y^4*z^2 - 45*x^4*y^5*z - 36*x^4*y^3*z^2 - 54*x^3*y^6
  - 165*x^3*y^4*z - 54*x^3*y^2*z^2 - 189*x^2*y^5 - 216*x^2*y^3*z - 36*x^2*y*z^2
  - 222*x*y^4 - 117*x*y^2*z - 9*x*z^2 - 89*y^3 - 21*y*z

/-! ### Certificate 1: scalar-cleared inverse -/

/-- Scalar-cleared inverse: J · adjugate J = (−2) • 1.  Together with
`alpoge_keller` this is the identity behind [Dᵢ, Fⱼ] = δᵢⱼ. -/
theorem dixmier_inverse :
    J * J.adjugate = (-2 : R) • (1 : Matrix (Fin 3) (Fin 3) R) := by
  rw [Matrix.mul_adjugate, alpoge_keller]

/-- Transposed version: adjugate J · J = (−2) • 1. -/
theorem dixmier_inverse' :
    J.adjugate * J = (-2 : R) • (1 : Matrix (Fin 3) (Fin 3) R) := by
  rw [Matrix.adjugate_mul, alpoge_keller]

/-! ### The adjugate entries, certified one by one -/

/-- All nine entries of `J.adjugate` at once (`adjugate_fin_three` cofactors
of the certified Jacobian entries `J₁₁ … J₃₃`, expanded by `ring`). -/
private lemma adj_entries :
    J.adjugate 0 0 = m₀₀ ∧ J.adjugate 0 1 = m₀₁ ∧ J.adjugate 0 2 = m₀₂ ∧
    J.adjugate 1 0 = m₁₀ ∧ J.adjugate 1 1 = m₁₁ ∧ J.adjugate 1 2 = m₁₂ ∧
    J.adjugate 2 0 = m₂₀ ∧ J.adjugate 2 1 = m₂₁ ∧ J.adjugate 2 2 = m₂₂ := by
  have h : ∀ i j, J i j = pderiv j (![F₁, F₂, F₃] i) := fun _ _ => rfl
  rw [Matrix.adjugate_fin_three]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    · simp only [Matrix.cons_val', Matrix.cons_val_zero, Matrix.empty_val',
        Matrix.cons_val_fin_one, Matrix.cons_val_one, Matrix.head_cons,
        Matrix.head_fin_const, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.of_apply, h, J₁₁, J₁₂, J₁₃, J₂₁, J₂₂, J₂₃, J₃₁, J₃₂, J₃₃,
        m₀₀, m₀₁, m₀₂, m₁₀, m₁₁, m₁₂, m₂₀, m₂₁, m₂₂]
      ring

lemma adj₀₀ : J.adjugate 0 0 = m₀₀ := adj_entries.1
lemma adj₀₁ : J.adjugate 0 1 = m₀₁ := adj_entries.2.1
lemma adj₀₂ : J.adjugate 0 2 = m₀₂ := adj_entries.2.2.1
lemma adj₁₀ : J.adjugate 1 0 = m₁₀ := adj_entries.2.2.2.1
lemma adj₁₁ : J.adjugate 1 1 = m₁₁ := adj_entries.2.2.2.2.1
lemma adj₁₂ : J.adjugate 1 2 = m₁₂ := adj_entries.2.2.2.2.2.1
lemma adj₂₀ : J.adjugate 2 0 = m₂₀ := adj_entries.2.2.2.2.2.2.1
lemma adj₂₁ : J.adjugate 2 1 = m₂₁ := adj_entries.2.2.2.2.2.2.2.1
lemma adj₂₂ : J.adjugate 2 2 = m₂₂ := adj_entries.2.2.2.2.2.2.2.2

/-! ### Certificate 2: the nine flatness identities

For each 0 ≤ i < j ≤ 2 and each l, the identity
`Σₖ (M k i · ∂ₖ(M l j) − M k j · ∂ₖ(M l i)) = 0` with `M = J.adjugate`,
written out with `Fin.sum_univ_three` shape.  Each proof unfolds the formal
partial derivatives with the Leibniz rules and closes with `ring`. -/

local macro "pderiv_simp" : tactic =>
  `(tactic| simp only [m₀₀, m₀₁, m₀₂, m₁₀, m₁₁, m₁₂, m₂₀, m₂₁, m₂₂, x, y, z,
      map_add, map_sub, map_neg, pderiv_mul, pderiv_pow, pderiv_X_self,
      pderiv_one, pderiv_ofNat,
      pderiv_X_of_ne (show (1 : Fin 3) ≠ 0 by decide),
      pderiv_X_of_ne (show (2 : Fin 3) ≠ 0 by decide),
      pderiv_X_of_ne (show (0 : Fin 3) ≠ 1 by decide),
      pderiv_X_of_ne (show (2 : Fin 3) ≠ 1 by decide),
      pderiv_X_of_ne (show (0 : Fin 3) ≠ 2 by decide),
      pderiv_X_of_ne (show (1 : Fin 3) ≠ 2 by decide)])

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₁₀ :
    (m₀₀ * pderiv (0 : Fin 3) m₀₁ - m₀₁ * pderiv (0 : Fin 3) m₀₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₀₁ - m₁₁ * pderiv (1 : Fin 3) m₀₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₀₁ - m₂₁ * pderiv (2 : Fin 3) m₀₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₁₁ :
    (m₀₀ * pderiv (0 : Fin 3) m₁₁ - m₀₁ * pderiv (0 : Fin 3) m₁₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₁₁ - m₁₁ * pderiv (1 : Fin 3) m₁₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₁₁ - m₂₁ * pderiv (2 : Fin 3) m₁₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₁₂ :
    (m₀₀ * pderiv (0 : Fin 3) m₂₁ - m₀₁ * pderiv (0 : Fin 3) m₂₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₂₁ - m₁₁ * pderiv (1 : Fin 3) m₂₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₂₁ - m₂₁ * pderiv (2 : Fin 3) m₂₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₂₀ :
    (m₀₀ * pderiv (0 : Fin 3) m₀₂ - m₀₂ * pderiv (0 : Fin 3) m₀₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₀₂ - m₁₂ * pderiv (1 : Fin 3) m₀₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₀₂ - m₂₂ * pderiv (2 : Fin 3) m₀₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₂₁ :
    (m₀₀ * pderiv (0 : Fin 3) m₁₂ - m₀₂ * pderiv (0 : Fin 3) m₁₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₁₂ - m₁₂ * pderiv (1 : Fin 3) m₁₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₁₂ - m₂₂ * pderiv (2 : Fin 3) m₁₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₀₂₂ :
    (m₀₀ * pderiv (0 : Fin 3) m₂₂ - m₀₂ * pderiv (0 : Fin 3) m₂₀)
  + (m₁₀ * pderiv (1 : Fin 3) m₂₂ - m₁₂ * pderiv (1 : Fin 3) m₂₀)
  + (m₂₀ * pderiv (2 : Fin 3) m₂₂ - m₂₂ * pderiv (2 : Fin 3) m₂₀) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₁₂₀ :
    (m₀₁ * pderiv (0 : Fin 3) m₀₂ - m₀₂ * pderiv (0 : Fin 3) m₀₁)
  + (m₁₁ * pderiv (1 : Fin 3) m₀₂ - m₁₂ * pderiv (1 : Fin 3) m₀₁)
  + (m₂₁ * pderiv (2 : Fin 3) m₀₂ - m₂₂ * pderiv (2 : Fin 3) m₀₁) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₁₂₁ :
    (m₀₁ * pderiv (0 : Fin 3) m₁₂ - m₀₂ * pderiv (0 : Fin 3) m₁₁)
  + (m₁₁ * pderiv (1 : Fin 3) m₁₂ - m₁₂ * pderiv (1 : Fin 3) m₁₁)
  + (m₂₁ * pderiv (2 : Fin 3) m₁₂ - m₂₂ * pderiv (2 : Fin 3) m₁₁) = 0 := by
  pderiv_simp; ring

set_option maxHeartbeats 8000000 in
-- expanded flatness identity: `ring` on ~10^3-monomial products
lemma flat₁₂₂ :
    (m₀₁ * pderiv (0 : Fin 3) m₂₂ - m₀₂ * pderiv (0 : Fin 3) m₂₁)
  + (m₁₁ * pderiv (1 : Fin 3) m₂₂ - m₁₂ * pderiv (1 : Fin 3) m₂₁)
  + (m₂₁ * pderiv (2 : Fin 3) m₂₂ - m₂₂ * pderiv (2 : Fin 3) m₂₁) = 0 := by
  pderiv_simp; ring

set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 1000000 in
-- 27-way case dispatch over (i, j, l)
/-- **Flatness** (scalar-cleared): with M = adjugate J, for all i j l,
`Σₖ (M k i · ∂ₖ(M l j) − M k j · ∂ₖ(M l i)) = 0`.  This is the identity
behind `[Dᵢ, Dⱼ] = 0` for `Dᵢ = Σₖ N k i ∂ₖ`, `N = M/(−2)` (the clearing
factor (−2)² = 4 multiplies both terms uniformly). -/
theorem dixmier_flatness (i j l : Fin 3) :
    ∑ k, (J.adjugate k i * pderiv k (J.adjugate l j)
        - J.adjugate k j * pderiv k (J.adjugate l i)) = 0 := by
  fin_cases i <;> fin_cases j <;> fin_cases l <;>
    simp only [Fin.sum_univ_three, Fin.isValue, Fin.zero_eta, Fin.mk_one,
      Fin.reduceFinMk, adj₀₀, adj₀₁, adj₀₂, adj₁₀, adj₁₁, adj₁₂, adj₂₀, adj₂₁,
      adj₂₂] <;>
    first
      | simp only [sub_self, add_zero]
      | linear_combination flat₀₁₀
      | linear_combination flat₀₁₁
      | linear_combination flat₀₁₂
      | linear_combination flat₀₂₀
      | linear_combination flat₀₂₁
      | linear_combination flat₀₂₂
      | linear_combination flat₁₂₀
      | linear_combination flat₁₂₁
      | linear_combination flat₁₂₂
      | linear_combination -flat₀₁₀
      | linear_combination -flat₀₁₁
      | linear_combination -flat₀₁₂
      | linear_combination -flat₀₂₀
      | linear_combination -flat₀₂₁
      | linear_combination -flat₀₂₂
      | linear_combination -flat₁₂₀
      | linear_combination -flat₁₂₁
      | linear_combination -flat₁₂₂

/-! ### Certificate 3: the Poisson determinant, scalar-cleared -/

/-- det (adjugate J) = (det J)² = 4. -/
theorem det_adjugate_J : (J.adjugate).det = (4 : R) := by
  rw [Matrix.det_adjugate, alpoge_keller]
  norm_num [Fintype.card_fin]

/-- **Poisson determinant** (scalar-cleared): the cotangent lift
Φ(x, y) = (F(x), Nᵀy) has Jacobian `fromBlocks J 0 C Nᵀ` (∂F/∂y = 0; C is
the x-derivative block of Nᵀy).  With N replaced by M = adjugate J = −2·N,
the determinant is (−2)·4 = −8 = (−2)³ — i.e. det JΦ = 1 after removing
the clearing factor.  Quantifying over C certifies that the lower-left
block is irrelevant. -/
theorem poisson_det_scaled (C : Matrix (Fin 3) (Fin 3) R) :
    (Matrix.fromBlocks J 0 C (J.adjugate).transpose).det = (-8 : R) := by
  rw [Matrix.det_fromBlocks_zero₁₂, Matrix.det_transpose, det_adjugate_J,
    alpoge_keller]
  norm_num

end

end Dixmier
