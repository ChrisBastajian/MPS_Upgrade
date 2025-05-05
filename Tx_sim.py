import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sympy as sp

R= 15 * 1e-3 #15 mm
muo = 4 * np.pi * 1e-7
I = 5 #A
n_turns = 206
height = 73 * 1e-3 #73mm
dtps = 10000 #number of data points

phi = np.linspace(0, 2 * np.pi * n_turns, dtps) #the azimuthal length (x-y plane)

#Current path (dl) in cylindrical coordinates:
lx, ly, lz = np.array([R*np.cos(phi), R*np.sin(phi), height* (phi/np.max(phi))])

fig, ax = plt.subplots()
ax.plot(lx, ly)
plt.show()

#sympy numerical computations:
phi, x, y, z = sp.symbols('phi,x,y,z')
l = sp.Matrix([R*sp.cos(phi), R*sp.sin(phi), height])

r= sp.Matrix([x,y,z]) #this is an observation point

#distance between point and current line path:
d = r-l

integrand = ((muo*I)/(4*np.pi)) * sp.diff(l, phi).cross(d)
