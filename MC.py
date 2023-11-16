"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
"""
This module implements functions used for Monte Carlo methods for light transpor
in biological tissues.
ref Lihong Wang, S. Jacques

@author: andreafarina
"""
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def stepsize(mut):
    """ Samples a new step-size

    Args:
        mut: interaction coefficient >0.

    Returns:
        a number representing the stepsize in the length units of mut
    """
    return -np.log(np.random.random())/mut

def costheta(g):
    """ Samples a new zenithal direction using HG phase function

    Args:
        g: anisotropy factor.

    Returns:
        a number representing the cosine of the zenithal angle
    """
    if g == 0:
        return 2*np.random.random() - 1
    else:
        return 1/(2*g)*(1+g**2-((1-g**2)/(1-g+2*g*np.random.random()))**2)

def psi():
    """ Samples uniformly a new azimuthal direction 

    Returns:
        the new azimuthal angle
    """
    return 2*np.pi*np.random.random()

def PhotonSurvive():
    """ Roulette 
    Args: 
        an integer >1
        
    Returns:
        a boolean the photon survive or not
    """
    m = 0
    surv = 0
    if np.random.random() < 1/m:
        surv = 1
    return surv

def Reflect(mu0,medium):
    """Fresnel refraction: decide if the photon is reflected or not
    Args:
    
    Returns: 
        a boolean for reflection or not
    """
    muz = np.abs(mu0[2])
    th_in = np.arccos(muz)
    if th_in >= medium.th_lim:
        return 1
    
    if muz>0.99999:
        R = (1-medium.n)**2/(1+medium.n)**2
    else:
        th_t = np.arcsin(medium.n*np.sin(th_in))
        Rs = (np.sin(th_in-th_t)/np.sin(th_in+th_t))**2
        Rp = (np.tan(th_in-th_t)/np.tan(th_in+th_t))**2
        R = 0.5*(Rs + Rp)
    if np.random.random()<R:
        return 1
    else:
        return 0

def Fluence(Path,Weight,Nbins,area,v,Ntot):
    """ Extract the time-resolved fluence from the output of the simulation
        Args:
            Path, Weight
            Nbins: number of temporal bins
            area:  detector area
            v:     speed of light in the medium c/n
            Ntot:  number of total photon launched
    """
    hist, edges = np.histogram(Path,bins = Nbins,weights=Weight)
    histN, edges = np.histogram(Path,bins = Nbins)
    l = edges[:-1]
    l = l + np.diff(edges)[0]/2
    time = l/v
    dt = np.diff(time)[0]
    y = hist/Ntot*1/area*1/dt
    a = np.sqrt(histN)
    #sigma = y/np.sqrt(histN)
    sigma = np.zeros(np.size(y))
    valid = np.where(histN>0)
    sigma[valid] = np.divide(y[valid],np.sqrt(histN[valid]))
    sigma[np.isnan(sigma)] = 0
    return y,time,sigma

def initTrajectoriesPlot(medium,detector):
    #fig  = plt.figure()
    ls = 1/medium.musp  # reduced scattering length
    fig,ax = plt.subplots(2,2)
    ax[0,0].set(visible=False)
    ax[0,0] = fig.add_subplot(2,2,1,projection = '3d')
    latlim = 2 * np.average(detector.r)
    ax[0,0].set(xlim=(-latlim,latlim),xlabel='X')
    ax[0,0].set(ylim=(-latlim,latlim),ylabel='Y')

    if detector.RT == 1:
        dlimit = np.diff(medium.z) if np.diff(medium.z)<=10*ls else ls*10
    else:
        dlimit = np.diff(medium.z)
    ax[0,0].set_box_aspect(aspect = (2*latlim,2*latlim,dlimit))
    ax[0,0].set(zlim=(medium.z[0],medium.z[0] + dlimit ),zlabel='Z')
    #ax.invert_zaxis()
    
    # plot detector
    theta = np.linspace(0,2*np.pi,1000)
    xdet1 = detector.r[0]*np.cos(theta)
    ydet1 = detector.r[0]*np.sin(theta)
    xdet2 = detector.r[1]*np.cos(theta)
    ydet2 = detector.r[1]*np.sin(theta)
    if detector.RT == 1:
        zdet = medium.z[0]
    else:
        zdet = medium.z[1]
    ax[0,0].plot(xdet1,ydet1,"blue",zs = zdet,zdir = 'z')
    ax[0,0].plot(xdet2,ydet2,"blue",zs = zdet,zdir = 'z')
    
    ax[0,1].plot(xdet1,ydet1,"blue")
    ax[0,1].plot(xdet2,ydet2,"blue")
    
    # plot source
    xs = 0.1*np.cos(theta)
    ys = 0.1*np.sin(theta)
    ax[0,0].plot(xs,ys,"red",zs = 0,zdir = 'z')
    
    # add second subplot
    #ax[] = fig.add_subplot(2,2,2)
    ax[0,1].set(xlim=(-latlim,latlim),xlabel='X')
    ax[0,1].set(ylim=(-latlim,latlim),ylabel='Y')
    ax[1,0].set(xlim=(-latlim,latlim),xlabel='X')
    ax[1,0].set(ylim=(medium.z[0],medium.z[0] + dlimit ),ylabel='Z')
    ax[1,0].invert_yaxis()
    ax[1,1].set(xlim=(-latlim,latlim),xlabel='Y')
    ax[1,1].set(ylim=(medium.z[0],medium.z[0] + dlimit ),ylabel='Z')
    ax[1,1].invert_yaxis()
   
    return fig, ax

def plotTrajectory(fig,ax,pos):
    # def animate(i):
    #     ax.plot(pos[0,0:i].T,pos[1,0:i].T,pos[2,0:i].T,"red")
    # ani = animation.FuncAnimation(fig,animate,pos.shape[1],interval = 1)  
    # plt.show()
    ax[0,0].plot(pos[0,:].T,pos[1,:].T,pos[2,:].T)
    ax[0,1].plot(pos[0,:].T,pos[1,:])
    ax[1,0].plot(pos[0,:].T,pos[2,:].T)
    ax[1,1].plot(pos[1,:].T,pos[2,:].T)
    
    
    
# ===================== DIFFUSIVE MEDIUM ===========================
class medium(object):
    """ Diffusive medium class
        Attributes:
            mua,mus,g,n,zmin,zmax,mut,k,th_lim

        Properties:
    """
    def __init__(self,mua=0.1,musp=10,g=0.8,n=1.4,zmin=0,zmax=5):
        mus = musp/(1-g)
        medium.mua = mua
        medium.mus = mus
        medium.musp = musp
        medium.g = g
        medium.n = n
        assert zmin<zmax,print("zmin > zmax !!")
        medium.z = (zmin,zmax)
        # useful for propagation
        medium.mut = mua + mus
        medium.k = mua/(mua + mus) # for weight decrement
        medium.th_lim = np.arcsin(1/n)
# =================== DETECTOR RING =========================
class detector(object):
    """ Detector class
        Attributes:
            RT: 1 for reflectance, 2 for transmittance
            r[rnim,rmax]: detector min/max radius
        Properties:
            Area: 
        """
    def __init__(self,RT=1,rmin=0.5,rmax=1):
        # 1: Reflectance, 2: Transmittance
        detector.RT = RT
        assert rmin<rmax, print("rmin > rmax !!")
        detector.r = (rmin,rmax)
    @property
    def area(self):
        return np.pi*(self.r[1]**2 - self.r[0]**2)
    

# =================== PHOTON PACKET ===========================
class photon(object):
    """ Photon packet class

    Attributes:
        p: actual position
        mu: actual direction
        path: pathlength accumulated
        weight: photon weight
    Properties:
        medium: current diffusive medium
                  
    """
    def __init__(self):
        self.p = np.zeros((3,1))
        self.mu = np.array([[0,0,1]]).T #collimated photon injected
        # self.mu = np.array([[0,0.5,0.1]]).T 
        # norm = np.sqrt(np.sum(self.mu**2))
        # self.mu = self.mu/norm
        self.weight = 1
        self.path = 0
        #self.ph = np.array(self.p) #position history

    @property
    def medium(self):
        return self._medium

    @medium.setter
    def medium(self,value):
        self._medium = value

    def Spin(self,cost,psi):
        """ Calculate the new photon direction
        Args:
        mu: vector of incidence direction.
        cost: zenithal cosine
        psi: azimuthal angle

    Returns:
        a new direction for the photon
    """
        mu = self.mu
        mup = np.zeros((3,1))
        if np.abs(mu[2])>0.99999:
            mup[0] = np.sqrt(1-cost**2)*np.cos(psi)
            mup[1] = np.sqrt(1-cost**2)*np.sin(psi)
            mup[2] = np.sign(mu[2])*cost
        else:
            A = np.sqrt(1-cost**2)/np.sqrt(1-mu[2]**2)
            mup[0] = A*(mu[0]*mu[2]*np.cos(psi)-mu[1]*np.sin(psi)) + mu[0]*cost
            mup[1] = A*(mu[1]*mu[2]*np.cos(psi)+mu[0]*np.sin(psi)) + mu[1]*cost
            mup[2] = -np.sqrt(1-cost**2)*np.cos(psi)*np.sqrt(1-mu[2]**2)+mu[2]*cost
        self.mu = mup

    def Hop(self,s):
        """ Calculate the new photon position

        Args:
            #p0: vector of the actual position of the photon.
            #mu: vector of the propagation direction
            s: stepsize
            k: factor of weght decrement:w = w * (1-k)

        Returns:
            the photon with updated position path and weight
        """
        #self.p = self.p + self.mu * s
        self.p += self.mu * s
        self.path += s
        self.weight *= (1 - self.medium.k)
        #self.ph = np.hstack((self.ph,self.p))
    
    # def HopPlot(self,s,fig,ax):
    #     """ Calculate the new photon position

    #     Args:
    #         #p0: vector of the actual position of the photon.
    #         #mu: vector of the propagation direction
    #         s: stepsize
    #         k: factor of weght decrement:w = w * (1-k)

    #     Returns:
    #         the photon with updated position path and weight
    #     """
    #     p0 = self.p
    #     self.p = self.p + self.mu * s
    #     p1 = self.p
    #     self.path += s
    #     self.weight *= (1 - self.medium.k)
    #     x = [p0[0,0],p1[0,0]]
    #     y = [p0[1,0],p1[1,0]]
    #     z = [p0[2,0],p1[2,0]]
        
    #     ax.plot(x,y,z)

    def HopToBoundary(self):
        """ Propagate the photon to the boundary

        Returns:
            the photon with updated position, path and weight
        """
        if self.mu[2] > 0:
            s = (self.medium.z[1] - self.p[2])/self.mu[2]
        else:
            s = (self.medium.z[0] - self.p[2])/self.mu[2]
       
        self.Hop(s)
        
    def HitBoundary(self,s):
        """ Check if the photon hits the boundary and which boundary
        Args:
        
        Returns:
            0 if the photon is still inside
            1 if the photon hit the plane at z = 0
            2 if the photon hit the plane at z = d
        """
        hitB = 0
        # test hop z coordinate
        pz = self.p[2] + self.mu[2] * s
        if pz < self.medium.z[0]:
            hitB = 1    # up
        elif pz > self.medium.z[1]:
            hitB = 2    # down
        return hitB
    
    def HitDetector(self,hitB,detector):
        """ Check if the photon hit the detector
        Args:
        
        Return:
            a boolean for hitting or not the detector
        """
        hitDet = 0
        if hitB == detector.RT:
            l = np.sqrt(self.p[0]**2 + self.p[1]**2)
            if (l<=detector.r[1]) and (l>=detector.r[0]):
                hitDet = 1
        return hitDet
    
    # =================== PROPAGATION KERNEL FOR USE IN COMBINATION WITH PARFOR =====
def propagate(medium,detector,Lmax,seed):
    """ propagate a single photon
        Args:
            medium:     medium object
            detector:   detector object
            Lmax:       max pathlength in the medium
            seed:       seed for the random number generator
        
        Returns:
            Path:   pathlength spent in the medium
            Weight: weight of the photon
            if zeros the photon is not detected
        """
    np.random.seed(seed) #this is to avoid the process has the same sequence
    p = photon()  # initialize photon packet
    p.medium = medium
    Path = 0
    Weight = 0  
    while p.weight > 0:
        s = stepsize(p.medium.mut)
        hitB = p.HitBoundary(s)
        if hitB > 0:
            p.HopToBoundary()
            if Reflect(p.mu,medium) == 1:
                p.mu[2]*=-1    # reflect the photon
            else:
                hitD = p.HitDetector(hitB, detector)
                if hitD == 1:       #photon detected
                    ## save detected photon data
                    Path = p.path
                    Weight = p.weight
                    p.weight = 0
                else:
                    p.weight = 0
        else:
            p.Hop(s)       # move the photon packet
            p.Spin(costheta(medium.g),psi())
        
        if p.path > Lmax:
            p.weight = 0

        # if photon.weight < Wth: # roulette
        #     if MC.PhotonSurvive(m):
        #         photon.weight = m * photon.weight
        #     else:
        #         photon.weight = 0
    return Path,Weight
    
        

    








    