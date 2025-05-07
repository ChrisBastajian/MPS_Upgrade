from scipy.integrate import tplquad
import numpy as np

# Vector field dot dl = 2*phi*dr + r*cos(phi)*dz
# We define this as a function f(z, r, phi) because tplquad uses (z, r, phi) order

def integrand(z, r, phi):
    return 2*phi + r*np.cos(phi)

# Integration limits
phi_lower, phi_upper = 0, 2*np.pi
r_lower, r_upper = 0, 5
z_lower, z_upper = 0, 2

# Perform the triple integral
result, error = tplquad(integrand,
                        phi_lower, phi_upper,
                        lambda phi: r_lower, lambda phi: r_upper,
                        lambda phi, r: z_lower, lambda phi, r: z_upper)

print(f"Triple Integral Result = {result:.5f}")
print(f"Estimated Error = {error:.2e}")