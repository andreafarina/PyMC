"""
This module implements anaytical functions under the Diffusion Approximation
for light transport in biological tissues.
ref Martelli, Binzoni

Created on Wed Oct  12 2022

@author: andreafarina
"""
import numpy as np

def REBC_TR(time,rho,mua,mus,d,c,n):
    n_ext = 1.00
    n_rel = n / n_ext
    if (n_rel<=1):
        A = 3.084635-6.531194*n_rel+8.357854*n_rel**2-5.082751*n_rel**3+1.171382*n_rel**4
    else:
        A = 504.332889-2641.00214*n_rel+5923.699064*n_rel**2-7376.355814*n_rel**3+5507.53041*n_rel**4-2463.357945*n_rel**5+610.956547*n_rel**6-64.8047*n_rel**7

    v = c/n #cm/ps velocit� della luce
    dip = 24; #numero di dipoli
    z0 = 1/mus
    D = 1/(3*mus)
    ze = 2*A*D
    mu = 1/(4*D*v*time)

    Rifle = np.zeros(np.size(time))
    for  k in range(-dip,dip):

        z_3 = -2*k*d - 4*k*ze - z0        #contributo positivo
        z_4 = -2*k*d - (4*k-2)*ze + z0    #contributo negativo
        Rifle = Rifle + (z_3*np.exp(-mu*z_3**2) - z_4*np.exp(-mu*z_4**2))
    Rifle = -np.exp(-mua*v*time-mu*rho**2)/(2*(4*np.pi*D*v)**(1.5)*(time**(2.5)))*Rifle
    Rifle[np.isnan(Rifle)]=0
    return Rifle