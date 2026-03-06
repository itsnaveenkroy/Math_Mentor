# Probability: Permutations and Combinations

## Factorial
n! = n × (n-1) × (n-2) × ... × 1
0! = 1

## Permutations (Order matters)
- nPr = n! / (n-r)!
- Arrangements of n distinct objects = n!
- Arrangements with repetition: n₁ identical of type 1, n₂ of type 2, ...
  = n! / (n₁! × n₂! × ...)

## Circular Permutations
- n objects in a circle = (n-1)!
- If clockwise = anticlockwise: (n-1)!/2

## Combinations (Order doesn't matter)
- nCr = n! / (r! × (n-r)!)
- nCr = nC(n-r)
- nC0 = nCn = 1
- nC1 = n
- nCr + nC(r+1) = (n+1)C(r+1)  [Pascal's Rule]

## Binomial Theorem
(a + b)^n = Σ nCr × a^(n-r) × b^r, for r = 0 to n

## Important Results
- Total subsets of n elements = 2^n
- nC0 + nC1 + ... + nCn = 2^n
- nC0 - nC1 + nC2 - ... = 0

## JEE-Style Tips
- Stars and bars: distributing n identical objects into r groups = (n+r-1)C(r-1)
- Derangements: D_n = n! × Σ(-1)^k/k! for k = 0 to n
- Inclusion-exclusion principle for counting

## Common Mistakes
- Using permutations when combinations are needed
- Forgetting to handle identical objects
- Double counting in distribution problems
