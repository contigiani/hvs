import numpy as np
from scipy import interpolate
from astropy import units as u
import hurley_stellar_evolution as hse
from astropy import constants as const
import os

interp_data = os.path.join(os.path.dirname(__file__), 'interp_data.txt')
spectrum_data = os.path.join(os.path.dirname(__file__), 'spectrum_data.txt')

Id, A_v, GMag_0, VMag_0, IcMag_0 = np.loadtxt(interp_data, unpack = True)
rbf_2_G =  interpolate.Rbf(Id, A_v, GMag_0, function = 'linear')
rbf_2_V =  interpolate.Rbf(Id, A_v, VMag_0, function = 'linear')
rbf_2_Ic = interpolate.Rbf(Id, A_v, IcMag_0, function = 'linear')

def closest_spectrum(Teff,Logg):
    '''
        Finds the spectrum from the BaSel library which matches the given
        Teff, Logg
    '''
    Met = 0. # Assumption: Considering only Solar Metallicities!
    Vturb = 2.00 # Atmospheric micro-turbulence velocity [km/s]
    XH = 0.00 # Mixing length

    files, Id, T, logg, met, Vt, Xh = np.loadtxt(spectrum_data, dtype = 'str', unpack=True)
    Id = np.array(Id,dtype='float')
    T = np.array(T,dtype='float')
    logg = np.array(logg,dtype='float')
    met = np.array(met,dtype='float')
    Vt = np.array(Vt, dtype = 'float')
    Xh = np.array(Xh, dtype='float')

    ds = np.sqrt( (T - Teff)**2. + (logg - Logg)**2. + (Met - met)**2. + (Vturb - Vt)**2. + (Xh - XH)**2. )
    indexm = np.where(ds == np.min(ds)) # Chi-square minimization

    identification = Id[indexm]

    return identification

def G_to_GRVS( G, V_I ):
    # From Gaia G band magnitude to Gaia G_RVS magnitude
    # Jordi+ 2010 , Table 3, second row:

    a = -0.0138
    b = 1.1168
    c = -0.1811
    d = 0.0085

    f = a + b * V_I + c * V_I**2. + d * V_I**3.

    return G - f # G_RVS magnitude

def get_errors(r, l, b, M, age, dust):
    '''
        Computes Gaia Grvs magnitudes and errorbars given the input.
        Written by TM (see author list)

        Parameters
        ----------
            r : Quantity
                distance form the Earth
            l : Quantity
                Galactic latitude
            b : Quantity
                Galactic longitude
            age : Quantity
                Stellar age
            dust : DustMap
                DustMap to be used

        Returns
        -------
            e_par, e_pmra, e_pmdec : Quantity
                errors in parallax, pmra* and pmdec.
    '''
    r, l, b, M, age = r*u.kpc, l*u.deg, b*u.deg, M*u.Msun, age*u.Myr

    T, R = hse.get_TempRad( M.to(u.solMass).value, 0, age.to(u.Myr).value) # Temperature [K], radius [solRad]

    T = T * u.K                   # Temperature of the star at t = tage [K]
    R = (R * u.solRad).to(u.m)    # Radius of the star at t = tage [m]

    logg = np.log10((const.G * M / R**2.).to(u.cm / u.s**2).value) # Log of surface gravity in cgs

    Id = closest_spectrum(T.value, logg) # ID of the best-matching spectrum (chi-squared minimization)
    Id = Id.squeeze() # Removes single-dimensional axes, essential for interpolating magnitudes

    mu = 5.*np.log10(r.to(u.pc).value) - 5. # Distance modulus
    Av = dust.query_dust(l.to(u.deg).value, b.to(u.deg).value, mu) * 2.682


    #Interpolation: from Id, Av to magnitudes (not corrected for the distance!)

    GMag0 = rbf_2_G(Id, Av) # Gaia G magnitude, [mag]
    VMag0 = rbf_2_V(Id, Av) # Johnson-Cousins V magnitude, [mag]
    IcMag0 = rbf_2_Ic(Id, Av) # Johnson-Cousins Ic magnitude, [mag]

    dist_correction_Mag = (- 2.5 * np.log10(((R/r)**2.).to(1))).value # Distance correction for computing the unreddened flux at Earth, [mag]

    #Magnitudes corrected for distance:
    GMag = GMag0 + dist_correction_Mag # Gaia G magnitude, [mag]
    VMag = VMag0 + dist_correction_Mag # Johnson-Cousins V magnitude, [mag]
    IcMag = IcMag0 + dist_correction_Mag # Johnson-Cousins Ic magnitude, [mag]

    V_I = VMag - IcMag # V - Ic colour, [mag]

    '''
    # ============== Errors! ================== #
    from pygaia.errors.astrometric import properMotionError
    from pygaia.errors.astrometric import parallaxError

    e_par = parallaxError(GMag, V_I, beta) * u.uas # Parallax error (PyGaia) [uas]
    e_pmra, e_pmdec = properMotionError(GMag, V_I, beta) * u.uas / u.yr # ICRS proper motions error (PyGaia) [uas/yr]
    '''

    GRVS = G_to_GRVS( GMag, V_I )

    return GRVS

get_GRVS = np.vectorize(get_errors)
