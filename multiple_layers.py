import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.vector import cross
from scipy.integrate import quad
import plotly.graph_objects as go

#curve path of current:
r = 0.0175 # the radius of the curve itself (not the distance for each vector) in m
mu0 = 4 * np.pi * 1e-7
I = 1 #current intensity in A
n_turns = 48*4
max_height = 0.065

#finding thickness of litz wire:
#total_thickness = 0.008 #in m (we use #18 AWG = 1.32 mm total for litz wire)
data_points_coil = 100000
num_coils = 4 #there are four layers
n_turns = n_turns/num_coils #number of turns per coil
#coil_thickness = total_thickness/num_coils
coil_thickness = 2 * 1e-3

def coil_arrays(num_coils): #gives the trajectory along which the current flows in the "phi" dimension and splits data points for each coil
    phi = {}

    for n in range(num_coils):
        n_string = str(n)
        phi[n_string] = np.linspace(0, n_turns *2 * np.pi, round(data_points_coil/num_coils))

    return phi

a = coil_arrays(num_coils)
#print(a['3']/a['1']) #to check that all arrays are the same
#print(len(a['1'])) #and that they have the length corresponding to dividing the datapoints wanted by the number of coils
#height = np.linspace(0, max_height, 100000)
#phi = np.linspace(0, n_turns * 2* np.pi, 100000)

def l(phi, r, coil_thickness, coil_number): #extends each point of phi along a z_offset in l
    r = r + coil_thickness * coil_number
    lengths = r * np.array([np.cos(phi), np.sin(phi), (max_height / (r* (n_turns * 2 * np.pi))) * phi]) #to create a circle with increments (loops)
    return lengths

#store dictionaries of lx, ly, and lz:
lx, ly, lz = {}, {}, {}

#all 3 dimensions for the curves:
for m in range(num_coils):
    m_string = str(m)
    lx[m_string], ly[m_string], lz[m_string] = l(a[m_string], r, coil_thickness, m + 1)

#Plot to check that the curves are all what we want:

try:
    fig1 = plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(lx['0'], ly['0'])
    plt.plot(lx['1'], ly['1'])
    plt.plot(lx['2'], ly['2'])
    plt.plot(lx['3'], ly['3'])
    plt.title("Current Path X_Y")
    plt.xlabel("X")
    plt.ylabel("Y")

    ax = fig1.add_subplot(122, projection='3d')

    ax.plot(lx['0'], ly['0'], lz['0'], label='Coil 1', color='red')
    ax.plot(lx['1'], ly['1'], lz['1'], label='Coil 2', color='black')
    ax.plot(lx['2'], ly['2'], lz['2'], label='Coil 3', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

except:
    print('not enough coils for that')



##########################################################################################################################

#solve for integrals using sympy:
phi, x, y, z = sp.symbols('phi,x,y,z')

#set libraries:
l, R, difference, integrands = {}, {}, {}, {}
dBx, dBy, dBz = {}, {}, {}

#distance from path:
d = sp.Matrix([x, y, z])

#recreate function of current path but in sympy to integrate:
for k in range(num_coils):
    k_string = str(k)
    R[k_string] = r + (k+1) * coil_thickness
    l[k_string] = R[k_string] * sp.Matrix([sp.cos(phi), sp.sin(phi), (max_height/(R[k_string] * n_turns * 2 * np.pi))*phi])
    difference[k_string] = d - l[k_string]
    integrands[k_string] = (mu0 * I / (4 * np.pi)) * sp.diff(l[k_string], phi).cross(
        difference[k_string]) / difference[k_string].norm() ** 3  # this will hold dB/dt for all 3 dimensions
    # need to lambdify all 3 functions into actual array representations of them:
    dBx[k_string] = sp.lambdify([phi, x, y, z], integrands[k_string][0])
    dBy[k_string] = sp.lambdify([phi, x, y, z], integrands[k_string][1])
    dBz[k_string] = sp.lambdify([phi, x, y, z], integrands[k_string][2])


#function to get B from the integrand (integrate):
def B(x, y, z):
    Bx, By, Bz = 0, 0, 0
    for j in range(num_coils):
        j_string = str(j)
        Bx += quad(dBx[j_string], 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0]
        By += quad(dBy[j_string], 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0]
        Bz += quad(dBz[j_string], 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0]

    return(np.array([Bx, By, Bz]))

#calculate B at the center (0, 0, 0)
B_center = B(0, 0, max_height/2)
print("Magnetic field at the center:", B_center)
B_SPIO = B(0,0, max_height - 0.045) #the sample is 45 mm from the top
print(B_SPIO)

#meshgrid to display:
i = np.linspace(-2*r, 2*r, 10)
j = np.linspace(0, 2 * max_height, 10)
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
fig.add_scatter3d(x=lx['0'], y=ly['0'], z=lz['0'], mode='lines', line=dict(color= 'black'))
fig.add_scatter3d(x=lx['1'], y=ly['1'], z=lz['1'], mode='lines', line=dict(color= 'black'))
fig.add_scatter3d(x=lx['2'], y=ly['2'], z=lz['2'], mode='lines', line=dict(color= 'black'))

fig.show()




# Plotting using matplotlib in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the current path
ax.plot(lx['0'], ly['0'], lz['0'], label='Coil 1', color='red')
ax.plot(lx['1'], ly['1'], lz['1'], label='Coil 2', color='black')
ax.plot(lx['2'], ly['2'], lz['2'], label='Coil 3', color='blue')

# Plot the magnetic field vectors
ax.quiver(xi, yi, zi, Bx, By, Bz, length=0.01, normalize=True, color='blue')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Magnetic Field Vectors and Current Path')

plt.legend()
plt.show()