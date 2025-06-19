import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad, dblquad
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
import functions as analyze

R= 17.5 * 1e-3 #17.5 mm
muo = 4 * np.pi * 1e-7
I = 5 #A
n_turns = 65 *3
height = 65 * 1e-3 #65mm
dtps = 100000
#number of data points

phi = np.linspace(0, 2 * np.pi * n_turns, dtps) #the azimuthal length (x-y plane)

#Current path (dl) in cylindrical coordinates:
lx, ly, lz = np.array([R*np.cos(phi), R*np.sin(phi), height* (phi/np.max(phi))])

fig, ax = plt.subplots()
ax.plot(lx, ly)
plt.show()

#sympy numerical computations:
phi, x, y, z = sp.symbols('phi,x,y,z')
l = R * sp.Matrix([sp.cos(phi), sp.sin(phi), (height/(R * n_turns * 2 * np.pi))*phi])

r= sp.Matrix([x,y,z]) #this is an observation point

#distance between point and current line path:
d = r-l # this is like R' (the distance vector from the observation point to the current path/wire

integrand = ((muo*I)/(4*np.pi)) * ((sp.diff(l, phi).cross(d))/(d.norm()**3))

#lambdifying the integrand (giving numerical values):
dBx = sp.lambdify([phi, x, y, z], integrand[0])
dBy = sp.lambdify([phi, x, y, z], integrand[1])
dBz = sp.lambdify([phi, x, y, z], integrand[2])

#Function to integrate:
def B(x, y, z):
    return(np.array([
        quad(dBx, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBy, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBz, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
    ]))

#computing integral over NMR Tube diameter:
D_nmr = 10 * 1e-3 #approx. 10 mm
R_nmr = D_nmr/2
lnmr = 10 * 1e-3 # 10mm height variance for the sample

num_points = 125#num_points = 64 #4*4*4=64
i = np.linspace(-2*R_nmr, 2*R_nmr, int(num_points**(1/3)))
j = np.linspace(-2*R_nmr, 2*R_nmr, int(num_points**(1/3)))
kprime= np.ones(int(num_points**(1/3))) * (height/2) #in the middle of the coil
knmr = np.linspace(-lnmr/2, lnmr/2, int(num_points**(1/3)) )
k= kprime + knmr

xi, yi, zi = np.meshgrid(i, j, k)

B_center = B(0, 0, height/2)
print("Magnetic field at the center:", B_center)

#get the whole field by vectorizing the result of calling B with the meshgrid elements:
B_field = np.vectorize(B, signature='(),(),()->(n)')(xi,yi,zi) #the signature in this
                                    # case is about joining all three inputs into one vector source (n)
Bx = B_field[:,:,:,0]
By = B_field[:,:,:,1]
Bz = B_field[:,:,:,2]

#Finding out how uniform the field is:
Bz_mean = np.mean(Bz)
add=0
diff_squared = []

dim = 3 #there are 3 dimensions (x,y,z)
n = int(num_points**(1/dim)) #per dimension

for a in range(n):
    for b in range(n):
        for c in range(n):
            diff_squared.append((B_field[a,b,c,2] - Bz_mean)**2)
RMSE = np.sqrt(np.mean(diff_squared))
print(f"RMSE= {RMSE}")
PPM = (RMSE / abs(Bz_mean)) * 1e6
print(f"PPM = {PPM:.3f}")

B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Mean and RMSE
B_mag_mean = np.mean(B_mag)
RMSE_mag = np.sqrt(np.mean((B_mag - B_mag_mean)**2))

# PPM for total magnitude
PPM_mag = (RMSE_mag / B_mag_mean) * 1e6

print(f"RMSE (|B|) = {RMSE_mag:.10f}")
print(f"PPM (|B|) = {PPM_mag:.3f}")

#plot everything
data = go.Cone(x=xi.ravel(), y = yi.ravel(), z=zi.ravel(), u =Bx.ravel(),
               v=By.ravel(), w = Bz.ravel())
fig = go.Figure(data= data)
fig.add_scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color= 'black'))

fig.show()

############# Finding the Inductance ##############
# Compute flux through one turn (central cross-section of coil)
k = np.linspace(height/2 - 1e-3, height/2 + 1e-3, int(num_points**(1/3)))  # was constant before
xi, yi, zi = np.meshgrid(i, j, k, indexing='ij')  # 'ij' indexing for correct shape

grid_points = [i, j, k]  # x, y, z axes
Bz_interp = RegularGridInterpolator((i, j, k), Bz)  # 3D interpolator

def Bz_center(r, phi):
    xj = r * np.cos(phi)
    yj = r * np.sin(phi)
    zj = height / 2
    try:
        Bz_val = Bz_interp((xj, yj, zj))
    except ValueError:
        Bz_val = 0  # out of bounds
    return Bz_val * r

flux, error = dblquad(Bz_center,
            0, 2 * np.pi,
            lambda phi: 0, lambda phi: R)

total_flux = flux * n_turns

L = total_flux/I

print(f"Inductance: {L:.3e} H")

#Comparing to Standard Solenoid inductance formula:
L = (muo * (n_turns**2)*np.pi * (R**2))/height

print(f"Standard Solenoid Inductance: {L:.3e} H")

#Computing using energy:
i = np.linspace(-100*R, 100*R, n)
j = np.linspace(-100*R, 100*R, n)
k = np.linspace(0, height, n)

dx = i[1] - i[0]
dy = j[1] - j[0]
dz = k[1] - k[0]

xi, yi, zi = np.meshgrid(i, j, k)

#get the whole field by vectorizing the result of calling B with the meshgrid elements:
B_field = np.vectorize(B, signature='(),(),()->(n)')(xi,yi,zi) #the signature in this
                                    # case is about joining all three inputs into one vector source (n)
Bx = B_field[:,:,:,0]
By = B_field[:,:,:,1]
Bz = B_field[:,:,:,2]

dV = dx * dy * dz

B_squared = Bx**2 + By**2 + Bz**2
U = (1 / (2 * muo)) * np.sum(B_squared * dV)
L = 2 * U / I**2

print(f"Energy Based Inductance: {L:.3e} H")

#Neumann inductance formula (in functions):
wire_radius = 2e-3  # Example: 0.5 mm copper wire
Y = 0.0  # For DC, set Y = 0

L_corr, L_uncorr = analyze.neumann_self_inductance(lx, ly, lz, wire_radius, Y)
print(f"Neumann Inductance (corrected): {L_corr:.6e} H")
print(f"Neumann Inductance (uncorrected): {L_uncorr:.6e} H")