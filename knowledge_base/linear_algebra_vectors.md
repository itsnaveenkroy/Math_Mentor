# Linear Algebra: Vectors

## Vector Basics
- Magnitude: |a| = √(a₁² + a₂² + a₃²)
- Unit vector: â = a/|a|
- Position vector of point P(x,y,z): OP = xi + yj + zk

## Dot Product (Scalar Product)
a · b = |a||b|cos θ = a₁b₁ + a₂b₂ + a₃b₃
- Perpendicular if a · b = 0
- Parallel if a × b = 0

## Cross Product (Vector Product)
a × b = |a||b|sin θ × n̂
|i  j  k |
|a₁ a₂ a₃| = i(a₂b₃-a₃b₂) - j(a₁b₃-a₃b₁) + k(a₁b₂-a₂b₁)
|b₁ b₂ b₃|
- |a × b| = area of parallelogram formed by a and b
- a × b = -b × a (anti-commutative)

## Scalar Triple Product
[a b c] = a · (b × c) = volume of parallelepiped
- Coplanar if [a b c] = 0

## Projection
Projection of a on b = (a · b)/|b|
Vector projection = ((a · b)/|b|²) × b

## Section Formula
Point dividing AB in ratio m:n:
P = (mB + nA)/(m + n)  [internal]
P = (mB - nA)/(m - n)  [external]

## Common Mistakes
- Confusing dot product (scalar) with cross product (vector)
- Sign errors in cross product computation
- Forgetting that cross product is not commutative
