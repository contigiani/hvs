from galpy.potential import  HernquistPotential, MiyamotoNagaiPotential, KeplerPotential
from galpy.potential import NFWPotential, TriaxialNFWPotential
from astropy import units as u
import numpy as np



def MWPotential(Ms=0.76, rs=24.8, c=1., T=True):
    '''
        Milky Way potential from Marchetti 2017b -- see galpy for the definitions of the potential components

        Parameters
        ----------
            Ms : float
                NFW profile scale mass in units of e12 Msun
            rs : float
                Radial profile in units of kpc
            c : float
                Axis ratio
            T : bool
                If True, use triaxialNFWPotential 
    '''

    # NFW profile
    Ms = Ms*1e12*u.Msun
    rs = rs*u.kpc

    #Disk
    Md = 1e11 * u.Msun
    ad = 6.5 * u.kpc
    bd = 260. * u.pc

    #Bulge
    Mb = 3.4*1e10*u.Msun
    Rb = 0.7*u.kpc

    #BH mass in 1e6 Msun
    Mbh = 4e6*u.Msun
    if(T):
        halop = TriaxialNFWPotential(amp=Ms, a=rs, c=c, normalize=False)
    else:
        halop = NFWPotential(amp=Ms, a=rs, normalize=False)
    diskp = MiyamotoNagaiPotential(amp=Md, a=ad, b=bd)
    bulgep = HernquistPotential(amp=2*Mb, a=Rb) #Factor 2 because of the galpy definition
    bh = KeplerPotential(amp=Mbh)

    return [halop, diskp, bulgep, bh]
