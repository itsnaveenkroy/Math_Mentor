# Calculus: Limits

## Definition
lim(x→a) f(x) = L means f(x) approaches L as x approaches a.

## Basic Limit Properties
- lim(f ± g) = lim(f) ± lim(g)
- lim(f × g) = lim(f) × lim(g)
- lim(f/g) = lim(f)/lim(g), provided lim(g) ≠ 0
- lim(cf) = c × lim(f)

## Standard Limits (JEE Important)
- lim(x→0) sin(x)/x = 1
- lim(x→0) tan(x)/x = 1
- lim(x→0) (1 - cos(x))/x² = 1/2
- lim(x→0) (e^x - 1)/x = 1
- lim(x→0) ln(1 + x)/x = 1
- lim(x→0) (a^x - 1)/x = ln(a)
- lim(x→0) (1 + x)^(1/x) = e
- lim(x→∞) (1 + 1/x)^x = e
- lim(x→0) sin⁻¹(x)/x = 1
- lim(x→0) tan⁻¹(x)/x = 1

## L'Hôpital's Rule
If lim f(x)/g(x) gives 0/0 or ∞/∞:
lim f(x)/g(x) = lim f'(x)/g'(x)

## Squeeze Theorem (Sandwich Theorem)
If g(x) ≤ f(x) ≤ h(x) near a, and lim g = lim h = L, then lim f = L.

## Indeterminate Forms
0/0, ∞/∞, 0×∞, ∞-∞, 0^0, ∞^0, 1^∞
For 1^∞ form: lim f(x)^g(x) = e^(lim g(x)×(f(x)-1))

## Common Mistakes
- Applying L'Hôpital when not in indeterminate form
- Forgetting to check left and right limits
- Not simplifying before applying limits
