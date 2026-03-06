# Calculus: Integration

## Basic Integrals
- ∫x^n dx = x^(n+1)/(n+1) + C, n ≠ -1
- ∫1/x dx = ln|x| + C
- ∫e^x dx = e^x + C
- ∫a^x dx = a^x/ln(a) + C

## Trigonometric Integrals
- ∫sin x dx = -cos x + C
- ∫cos x dx = sin x + C
- ∫sec²x dx = tan x + C
- ∫csc²x dx = -cot x + C
- ∫sec x tan x dx = sec x + C
- ∫csc x cot x dx = -csc x + C
- ∫tan x dx = -ln|cos x| + C = ln|sec x| + C
- ∫cot x dx = ln|sin x| + C

## Important Integrals (JEE)
- ∫1/(x²+a²) dx = (1/a)tan⁻¹(x/a) + C
- ∫1/√(a²-x²) dx = sin⁻¹(x/a) + C
- ∫1/(x²-a²) dx = (1/2a)ln|(x-a)/(x+a)| + C

## Integration Techniques
1. **Substitution**: Replace u = g(x), du = g'(x)dx
2. **Integration by Parts**: ∫u dv = uv - ∫v du (ILATE rule for choosing u)
3. **Partial Fractions**: Decompose rational functions
4. **Trigonometric Substitution**: For √(a²-x²), √(a²+x²), √(x²-a²)

## ILATE Rule (Priority for choosing u in IBP)
I - Inverse trig
L - Logarithmic
A - Algebraic
T - Trigonometric
E - Exponential

## Definite Integrals Properties
- ∫[a,b] f(x)dx = -∫[b,a] f(x)dx
- ∫[a,b] f(x)dx = ∫[a,c] f(x)dx + ∫[c,b] f(x)dx
- King's rule: ∫[0,a] f(x)dx = ∫[0,a] f(a-x)dx

## Common Mistakes
- Forgetting the constant of integration (+C) in indefinite integrals
- Wrong substitution limits in definite integrals
- Applying ILATE incorrectly
