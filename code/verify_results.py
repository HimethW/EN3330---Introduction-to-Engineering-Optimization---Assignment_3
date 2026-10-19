# EN3330 Assignment 03 - numerical check of the analytical results.

import numpy as np

f = lambda x: x**2
h = lambda x: 1 - x
L = lambda mu, x: f(x) + mu * h(x)
d = lambda mu: mu - mu**2 / 4

x_star, mu_star, p_star = 1.0, 2.0, 1.0
grid = np.linspace(-3, 4, 700001)


print("Q1(a)  minimiser of L is x*(mu) = mu/2")
for mu in (0, 1, 2, 3):
    brute = grid[np.argmin(L(mu, grid))]
    assert abs(mu / 2 - brute) < 1e-4
    print(f"   mu={mu}   x*={mu/2:.1f}   L={L(mu, mu/2):.2f}")


print("\nQ1(b)  dual function d(mu) = mu - mu^2/4")
for mu in (0, 1, 2, 3):
    assert abs(d(mu) - L(mu, grid).min()) < 1e-5
    print(f"   d({mu}) = {d(mu):.2f}")

m = np.linspace(0, 6, 60001)
assert abs(m[np.argmax(d(m))] - mu_star) < 1e-3
assert abs(d(mu_star) - p_star) < 1e-12
print(f"   mu* = {mu_star:.0f},  d* = {d(mu_star):.0f},  p* = {p_star:.0f}, "
      f"gap = {p_star - d(mu_star):.0f}  (strong duality)")


print("\nQ1(c)  g(x) = max over mu of L")
big = np.linspace(0, 1e6, 100001)
for x in (0.0, 0.5, 1.0, 2.0, 3.0):
    sup = L(big, x).max()
    if x >= 1:
        assert abs(sup - f(x)) < 1e-9
        print(f"   x={x:.1f}  g = x^2 = {sup:.2f}")
    else:
        assert sup > 1e5
        print(f"   x={x:.1f}  g -> infinity  (reached {sup:.1e})")


print("\nQ2(a)  saddle point at (1, 2)")
assert L(mu_star, x_star) == 1
print(f"   (i)   L(mu*,x*) = {L(mu_star, x_star):.0f}")

assert all(L(mu, x_star) == 1 for mu in (0, 1, 2, 3, 10, 1000))
print("   (ii)  L(mu,x*) = 1 for every mu  ->  left inequality is an equality")

xs = np.linspace(-3, 4, 70001)
assert np.allclose(L(mu_star, xs), (xs - 1) ** 2 + 1)
assert L(mu_star, xs).min() >= 1 - 1e-12
print("   (iii) L(mu*,x) = (x-1)^2 + 1 >= 1  ->  right inequality holds")


print("\nQ2(b)  optimality chain, for feasible x")
for x in (1.5, 2.0, 3.0, 10.0):
    a, b = f(x_star), L(mu_star, x_star)
    c, e = L(mu_star, x), f(x) + mu_star * h(x)
    assert a == b <= c and abs(c - e) < 1e-12 and e <= f(x)
    print(f"   x={x:>4.1f}:  {a:.0f} = {b:.0f} <= {c:.2f} = {e:.2f} <= {f(x):.2f}")


print("\nQ2(c)  complementary slackness")
assert mu_star * h(x_star) == 0
print(f"   mu* h(x*) = {mu_star:.0f} * {h(x_star):.0f} = 0   (constraint active)")

h2 = lambda x: -1 - x                       # the constraint x >= -1
d2 = lambda mu: -mu - mu**2 / 4
assert h2(0) < 0                            # unconstrained minimiser is feasible
assert d2(0) >= max(d2(m))                  # dual is maximised at the boundary
print(f"   x >= -1:  h2(0) = {h2(0):.0f} < 0 so inactive, and mu* = 0")


print("\nQ3(a)  weak duality: d(mu) <= p*")
for mu in (0, 1, 2, 3):
    assert d(mu) <= p_star + 1e-12
    print(f"   d({mu}) = {d(mu):.2f} <= {p_star:.0f}   gap = {p_star - d(mu):.2f}")


print("\nQ3(b)  lower envelope of the three lines")
mg = np.linspace(0, 6, 60001)
env = np.min([L(mg, x) for x in (0, 1, 2)], axis=0)
assert np.diff(env, 2).max() < 1e-12       # concave: second difference <= 0
assert np.all(env >= d(mg) - 1e-9)         # min over 3 points overestimates
print("   slopes +1, 0, -1 with kinks at mu = 1 and 3  ->  concave")
print("   envelope lies above the true d(mu), as expected")

print("\nall checks passed")
