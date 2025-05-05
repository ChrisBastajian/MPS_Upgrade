import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.vector import cross
from scipy.integrate import quad
import plotly.graph_objects as go

#curve path of current:
R = 0.042 # the radius of the curve itself (not the distance for each vector) in m
mu0 = 4 * np.pi * 1e-7
I = 5 #current intensity in A
n_turns = 636
max_height = 0.133

#finding thickness of litz wire:
total_thickness = 0.008 #mm
data_points_coil = 100000
num_coils = 4
coil_thickness = total_thickness/num_coils

def coil_arrays(num_coils, coil_thickness):
    phi = {}

    for n in range(num_coils):
        n_string = str(n)
        phi[n_string] = np.linspace(0, n_turns *2 * np.pi, round(data_points_coil/num_coils))

    return phi

a = coil_arrays(num_coils, coil_thickness)
#print(a['3']/a['1']) #to check that all arrays are the same
#print(len(a['1'])) #and that they have the length corresponding to dividing the datapoints wanted by the number of coils
#height = np.linspace(0, max_height, 100000)
#phi = np.linspace(0, n_turns * 2* np.pi, 100000)

def l(phi, R, coil_thickness, coil_number):
    R = R + coil_thickness * coil_number
    lengths = R * np.array([np.cos(phi), np.sin(phi), (max_height / (R* (n_turns * 2 * np.pi))) * phi]) #to create a circle with increments (loops)
    return lengths

#store dictionaries of lx, ly, and lz:
lx, ly, lz = {}, {}, {}

#all 3 dimensions for the curves:
for m in range(num_coils):
    m_string = str(m)
    lx[m_string], ly[m_string], lz[m_string] = l(a[m_string], R, coil_thickness, m + 1)

#Plot to check that the curves are all what we want:
plt.plot(lx['0'],ly['0'])
plt.plot(lx['1'],ly['1'])
plt.plot(lx['2'],ly['2'])
plt.plot(lx['3'], ly['3'])
plt.show()

#solve for integrals using sympy:
phi, x, y, z = sp.symbols('phi,x,y,z')

#recreate function of current path but in sympy to integrate:
l = R * sp.Matrix([sp.cos(phi), sp.sin(phi), (max_height/(R * n_turns * 2 * np.pi))*phi])

#distance away from path:
r = sp.Matrix([x, y, z])

#B(r) = ((mu0 * I)/(4piR)) * integral_along_curve\loop([(dl/dt x (r-l))/norm(r-l)^3)]dt
#so we can simplify by defining r-l (vector of separation):
difference = r-l
#print(difference)

#function to integrate (commented above):
integrand = (mu0 * I / (4 * np.pi)) * sp.diff(l, phi).cross(difference)/ difference.norm()**3 #this will hold dB/dt for all 3 dimensions

print(integrand)

#need to lambdify all 3 functions into actual array representations of them:
dBx = sp.lambdify([phi,x, y, z], integrand[0])
dBy = sp.lambdify([phi,x, y, z], integrand[1])
dBz = sp.lambdify([phi,x, y, z], integrand[2])

print(dBz(2*np.pi, 1, 1,1 )) #prints the z component at the given location

#function to get B from the integrand (integrate):
def B(x, y, z):
    return(np.array([
        quad(dBx, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBy, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBz, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
    ]))

#calculate B at the center (0, 0, 0)
B_center = B(0, 0, max_height/2)
print("Magnetic field at the center:", B_center)
B_SPIO = B(0,0,max_height - 0.045) #the sample is 45 mm from the top
print(B_SPIO)

#meshgrid to display:
i = np.linspace(-2*R, 2*R, 10)
j = np.linspace(0, 2*max_height, 10)
xi,yi,zi = np.meshgrid(i,i,j)

#get the whole field by vectorizing the result of calling B with the meshgrid elements:
B_field = np.vectorize(B, signature='(),(),()->(n)')(xi,yi,zi) #the signature in this
                                    # case is about joining all three inputs into one vector source (n)
Bx = B_field[:,:,:,0]
By = B_field[:,:,:,1]
Bz = B_field[:,:,:,2]

#plot everything
data = go.Cone(x=xi.ravel(), y = yi.ravel(), z=zi.ravel(), u =Bx.ravel(),
               v=By.ravel(), w = Bz.ravel())
fig = go.Figure(data= data)
fig.add_scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color= 'black'))

fig.show()




# Plotting using matplotlib in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the current path
ax.plot(lx, ly, lz, label='Current Path', color='black')

# Plot the magnetic field vectors
ax.quiver(xi, yi, zi, Bx, By, Bz, length=0.01, normalize=True, color='blue')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Magnetic Field Vectors and Current Path')

plt.legend()
plt.show()