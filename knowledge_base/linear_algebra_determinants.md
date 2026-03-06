# Linear Algebra: Determinants

## 2×2 Determinant
|a b|
|c d| = ad - bc

## 3×3 Determinant (Expansion along first row)
|a₁ b₁ c₁|
|a₂ b₂ c₂| = a₁(b₂c₃-b₃c₂) - b₁(a₂c₃-a₃c₂) + c₁(a₂b₃-a₃b₂)
|a₃ b₃ c₃|

## Properties of Determinants
- det(Aᵀ) = det(A)
- det(AB) = det(A) × det(B)
- det(kA) = k^n × det(A) for n×n matrix
- Swapping two rows/columns changes sign
- Two identical rows/columns → det = 0
- Row/column of zeros → det = 0
- det(A⁻¹) = 1/det(A)

## Cramer's Rule (2 variables)
For a₁x + b₁y = c₁ and a₂x + b₂y = c₂:
x = |c₁ b₁| / |a₁ b₁|
    |c₂ b₂|   |a₂ b₂|

## Inverse of 2×2 Matrix
A⁻¹ = (1/det(A)) × | d  -b|
                      |-c   a|

## Conditions for System of Linear Equations
- Unique solution: det(A) ≠ 0
- No solution or infinite solutions: det(A) = 0
  - Inconsistent (no solution): at least one equation contradicts
  - Dependent (infinite solutions): equations are multiples

## Common Mistakes
- Sign errors in cofactor expansion
- Forgetting determinant changes sign with row swap
- Not checking det ≠ 0 before finding inverse
