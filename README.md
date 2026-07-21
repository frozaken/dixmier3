# An explicit counterexample to the Dixmier conjecture for the third Weyl algebra

Marcus Teller, July 2026. Live page: https://frozaken.github.io/dixmier3/

Alpöge's July 2026 counterexample to the Jacobian conjecture in dimension three
implies, through the classical implication DC₃ ⇒ JC₃, that the Dixmier
conjecture fails for the third Weyl algebra A₃. This note writes down the
explicit witness: a concrete injective, non-surjective ℂ-algebra endomorphism
of A₃, the cotangent lift of Alpöge's map. Byproducts: an explicit
Poisson-conjecture PC₃ counterexample and a symplectic Keller counterexample on
ℂ⁶ with Jacobian determinant identically 1.

## Contents

- `index.html` — the note (open the [live page](https://frozaken.github.io/dixmier3/))
- `main.pdf` — the full write-up with proofs
- `ancillary_check.py` — sympy certificate (`python3 ancillary_check.py`)
- `lean/` — Lean 4 / mathlib proofs of the identity core (no `sorry`,
  axiom-clean); toolchain pinned to `lean4:v4.32.0` + mathlib `v4.32.0`

## Certification

Every computational claim is verified twice: by the sympy script and by the
Lean development. The Lean theorems (`alpoge_keller`, `alpoge_not_injective`,
`dixmier_inverse`, `dixmier_flatness`, `poisson_det_scaled`) depend only on the
three standard axioms `propext`, `Classical.choice`, `Quot.sound`.

## License

Code (`ancillary_check.py`, `lean/`) is Apache-2.0 (see `LICENSE`). The exposition
(`index.html`, `main.pdf`) is © 2026 Marcus Teller, all rights reserved pending
arXiv posting.

Prepared with substantial assistance from AI systems (Claude Fable 5, Anthropic;
cross-checked with GPT-5.6-sol, OpenAI). All computational claims are
machine-certified. Corrections welcome: marcumail (at) gmail (dot) com
