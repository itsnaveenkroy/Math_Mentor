# Probability: Bayes' Theorem and Distributions

## Bayes' Theorem
P(A|B) = P(B|A) × P(A) / P(B)

## Total Probability Theorem
If B₁, B₂, ..., Bₙ partition the sample space:
P(A) = Σ P(A|Bᵢ) × P(Bᵢ)

## Binomial Distribution
X ~ B(n, p)
- P(X = k) = nCk × p^k × (1-p)^(n-k)
- Mean: E(X) = np
- Variance: Var(X) = np(1-p)

## Poisson Distribution
X ~ Poisson(λ)
- P(X = k) = e^(-λ) × λ^k / k!
- Mean = Variance = λ

## Geometric Distribution
P(X = k) = (1-p)^(k-1) × p
- Mean: 1/p
- Represents: number of trials until first success

## Expected Value
E(X) = Σ xᵢ × P(X = xᵢ)
E(aX + b) = aE(X) + b

## Variance
Var(X) = E(X²) - [E(X)]²
Var(aX + b) = a²Var(X)

## JEE-Style Applications
- Coin toss problems → Binomial
- Conditional probability with disease testing → Bayes
- "At least one" problems → Use complement: P(≥1) = 1 - P(0)

## Common Mistakes
- Not identifying the correct distribution
- Confusing P(A|B) with P(B|A) (prosecutor's fallacy)
- Forgetting to use complement for "at least" problems
