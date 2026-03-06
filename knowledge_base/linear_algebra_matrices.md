# Linear Algebra: Matrices

## Matrix Types
- Square matrix: rows = columns
- Identity matrix (I): diagonal = 1, rest = 0
- Zero matrix: all elements = 0
- Diagonal matrix: non-zero only on main diagonal
- Symmetric: A = Aᵀ
- Skew-symmetric: A = -Aᵀ (diagonal elements = 0)
- Orthogonal: AAᵀ = AᵀA = I

## Matrix Operations
- Addition: (A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ (same dimensions required)
- Scalar multiplication: (cA)ᵢⱼ = c × Aᵢⱼ
- Matrix multiplication: (AB)ᵢⱼ = Σ Aᵢₖ × Bₖⱼ
  - A(m×n) × B(n×p) = C(m×p)
  - AB ≠ BA in general (not commutative)

## Transpose Properties
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ
- (kA)ᵀ = kAᵀ

## Common Mistakes
- Assuming matrix multiplication is commutative
- Wrong dimensions in multiplication
- Forgetting to reverse order in transpose of product
