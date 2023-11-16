"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
"""
Main code for Monte Carlo simulation of light propagation in biological tissues

Created on Tue Oct  4 16:01:54 2022

@author: andreafarina
"""
import matplotlib.pyplot as plt
import MC
import numpy as np
import DIFF
import time

# ================ UNITS ===========================
ulength = "cm"  # length unit
utime = "ps"    # time unit
c = 0.03        # speed of light in the unit ulength/utime

# ================ INPUT PARAMETERS ================
mua = 0.1  
musp = 10    
g = 0.8
n = 1.4
d = 1000      

# define medium
medium = MC.medium(mua,musp,g,zmin=0,zmax = d)

# define detector
det = MC.detector(RT=1,rmin=1.4,rmax=1.6)

# the simulation
N = 1000
Wth = 1e-4
Tmax = 10e3  # maximum propagation time

# binning parameters
Nbins = 100

# ------------ EXERCISE 4: the MC workflow -----------------
# some initalizations
launched = 0
detected = 0
Lmax = Tmax*c/n
Path = np.zeros((N,1))
Weight = np.zeros((N,1))

#fig, ax = MC.initTrajectoriesPlot(medium,det)
t = time.time()
#np.random.seed(3)
print("launched\t detected\t elapsed time")
while launched < N:#          # external loop
    
    photon = MC.photon()  # initialize photon packet
    photon.medium = medium
    launched+=1
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    
    while photon.weight > 0:
        s = MC.stepsize(photon.medium.mut)
        hitB = photon.HitBoundary(s)
        if hitB > 0:
            photon.HopToBoundary()
            if MC.Reflect(photon.mu,medium) == 1:
                photon.mu[2]*=-1    # reflect the photon
            else:
                hitD = photon.HitDetector(hitB, det)
                if hitD == 1:       #photon detected
                    
                    ## save detected photon data
                    Path[detected] = photon.path
                    Weight[detected] = photon.weight
                    detected+=1
                    photon.weight = 0
                    #MC.plotTrajectory(fig,ax,photon.ph)
                else:
                    photon.weight = 0
                    #MC.plotTrajectory(fig,ax,photon.ph)
        else:
            photon.Hop(s)       # move the photon packet
            #ax.scatter(photon.p[0], photon.p[1])
            #plt.show()
            photon.Spin(MC.costheta(g),MC.psi())
        
        if photon.path > Lmax:
            photon.weight = 0

        # if photon.weight < Wth: # roulette
        #     if MC.PhotonSurvive(m):
        #         photon.weight = m * photon.weight
        #     else:
        #         photon.weight = 0
    if launched % (N // 10) == 0:
        print(launched,"\t\t",detected,"\t\t","{0:1.2f} s".format((time.time()-t)))
        
print("elapsed time: {0:1.2f} s".format((time.time()-t)))
# ------------ EXERCISE 5: bin and plot output data --------------
print("launched photons: ",launched)
print("detected photons: ",detected)

# delete unused memory
Path = Path[:detected]
Weight = Weight[:detected]

# Extract simulation
y,t,s = MC.Fluence(Path,Weight,Nbins,det.area,c/n,launched)

# plot fluence
fig, ax = plt.subplots() 
ax.plot(t,y,"C0o",fillstyle='none',label='MC')
# ax.set_yscale('log')
# ax.set_xlabel("time [" + utime + "]")
# ax.set_ylabel("Fluence [" + ulength +"$^{-1}$" + utime + "$^{-1}$]")

# compare with DE
yDIFF = DIFF.REBC_TR(t,np.mean(det.r),mua,musp,d,c,n)
ax.plot(t,yDIFF,"red",label='DIFF')
#ax.errorbar(t,y,s)
props = {
    'title':'MC vs DIFFUSION',
    'yscale':'log',
    'xlabel':"time [" + utime + "]",
    'ylabel':"Fluence [" + ulength +"$^{-1}$" + utime + "$^{-1}$]"
}
ax.set(**props)
ax.legend(loc='upper right')

plt.show()         
fig.savefig("pippo.pdf",dpi=300)            
# ============== PART2: COMPARISON WITH STATISTICAL OPTICAL PROPERTIES =============   
