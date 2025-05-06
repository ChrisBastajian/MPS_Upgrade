import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
import plotly.graph_objects as go

matplotlib.use('TkAgg')

R= 15 * 1e-3 #15 mm
muo = 4 * np.pi * 1e-7
I = 5 #A
n_turns = 206
height = 20 * 1e-3 #73mm
dtps = 10000 #number of data points

#since we have two coils here:
distance = 15 * 1e-3 #this is the distance between coils
n_turns_coil = int(n_turns/2)
height_coil = (height-distance)/2

#For hemholtz:
#R= 150 * 1e-3
#height = R + 73 * 1e-3
#distance = R-(73/4 * 1e-3)  #centers are at a distance of R from each other for helmholtz
#n_turns_coil = int(n_turns/2)
#height_coil = (height-distance)/2
print(height_coil)

phi = np.linspace(0, 2 * np.pi * n_turns_coil, dtps) #the azimuthal length (x-y plane)

lx1, ly1, lz1 = np.array([R*np.cos(phi), R*np.sin(phi), height_coil* (phi/np.max(phi))])
lx2, ly2, lz2 = np.array([R*np.cos(phi), R*np.sin(phi), height_coil + distance + (height_coil)* (phi/np.max(phi))])
fig, ax = plt.subplots()
ax.plot(lx1, lz1)
ax.plot(lx2, lz2)
plt.show()

#sympy version:
phi, x, y, z = sp.symbols('phi,x,y,z')
l1 = R * sp.Matrix([sp.cos(phi), sp.sin(phi), (height_coil/(R * n_turns * 2 * np.pi))*phi])
l2 = R * sp.Matrix([sp.cos(phi), sp.sin(phi),#x and y
                    ((height_coil + distance)/R)+ (height_coil/(R * n_turns * 2 * np.pi))*phi]) #z

r= sp.Matrix([x,y,z]) #this is an observation point

#distance between point and current line path:
d1 = r-l1 # this is like R' (the distance vector from the observation point to the current path/wire
d2 = r-l2

integrand1 = ((muo*I)/(4*sp.pi)) * ((sp.diff(l1, phi).cross(d1))/(d1.norm()**3))
integrand2 = ((muo*I)/(4*sp.pi)) * ((sp.diff(l2, phi).cross(d2))/(d2.norm()**3))


#lambdifying the integrand (giving numerical values):
dBx1 = sp.lambdify([phi, x, y, z], integrand1[0])
dBy1 = sp.lambdify([phi, x, y, z], integrand1[1])
dBz1 = sp.lambdify([phi, x, y, z], integrand1[2])

dBx2 = sp.lambdify([phi, x, y, z], integrand2[0])
dBy2 = sp.lambdify([phi, x, y, z], integrand2[1])
dBz2 = sp.lambdify([phi, x, y, z], integrand2[2])


#Function to integrate:
def B(x, y, z, dBx, dBy, dBz):
    return(np.array([
        quad(dBx, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBy, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
        quad(dBz, 0, n_turns * 2 * np.pi, args=(x, y, z), limit=10000)[0],
    ]))

#computing integral over NMR Tube diameter:
D_nmr = 5 * 1e-3 #approx. 5 mm
R_nmr = D_nmr/2

num_points = 125#num_points = 64 #4*4*4=64
i = np.linspace(-2*R_nmr, 2*R_nmr, int(num_points**(1/3)))
j = np.linspace(-2*R_nmr, 2*R_nmr, int(num_points**(1/3)))
k= np.ones(int(num_points**(1/3))) * (height_coil + distance/2) #in the middle between the two coils

xi, yi, zi = np.meshgrid(i, j, k)

#get the whole field by vectorizing the result of calling B with the meshgrid elements:
B_field1 = np.vectorize(B, signature='(),(),(),(),(),()->(n)')(xi,yi,zi, dBx1, dBy1, dBz1) #the signature in this
                                    # case is about joining all three inputs into one vector source (n)
B_field2 = np.vectorize(B, signature='(),(),(),(),(),()->(n)')(xi,yi,zi, dBx2, dBy2, dBz2)
B_field = B_field1 + B_field2

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

#plot everything
data = go.Cone(x=xi.ravel(), y = yi.ravel(), z=zi.ravel(), u =Bx.ravel(),
               v=By.ravel(), w = Bz.ravel())
fig = go.Figure(data= data)
fig.add_scatter3d(x=lx1, y=ly1, z=lz1, mode='lines', line=dict(color= 'black'))
fig.add_scatter3d(x=lx2, y=ly2, z=lz2, mode='lines', line=dict(color= 'black'))

fig.show()