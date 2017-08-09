import numpy
from scipy import interpolate
from astropy import units as u
np = numpy



'''
    To be reimplemented using Marchetti+ 2017b
'''


#PyGaia
from pygaia.errors.astrometric import  properMotionError, parallaxError, positionErrorSkyAvg, positionError
from pygaia.errors.spectroscopic import vradErrorSkyAvg
from pygaia.photometry.transformations import vminGrvsFromVmini

#Interpolation points 
#http://www.cosmos.esa.int/web/gaia/science-performance
#

_GUMSFITM = [0.5, 0.7, 0.8, 0.95, 1.10, 1.7, 2.1, 3.25, 6.5, 9, 18.]
_GUMSFITG = [8.8, 7., 6., 4.6, 4.1, 2.2, 1.5, -0.03, -1.9, -2.7, -3.5]
_GUMSFITV = [8.62, 7.21, 5.58, 4.78, 4.24, 2.19, 1.43, 0.0, -1.71, -2, -3.5]
_GUMSFITVI = [1.71, 1.23, 0.87, 0.74, 0.67, 0.38, 0.15, 0.01, -0.15, -0.2, -0.35]
_GUMSFITVRADA, _GUMSFITVRADB, _VCF, _VZM = [1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.0, 0.9,0.9, 0.9], [0.29, 0.29, 0.5, 0.6, 0.7, 1.5, 4.0, 5.5, 26.0, 32., 50.0], 0.5, 12.7


#Interpolation functions
_GfromM = interpolate.InterpolatedUnivariateSpline(_GUMSFITM, _GUMSFITG, k=1)
_VfromM = interpolate.InterpolatedUnivariateSpline(_GUMSFITM, _GUMSFITV, k=1)
_VIfromM = interpolate.InterpolatedUnivariateSpline(_GUMSFITM, _GUMSFITVI, k=1)
_AfromM = interpolate.InterpolatedUnivariateSpline(_GUMSFITM, _GUMSFITVRADA, k=1)
_BfromM = interpolate.InterpolatedUnivariateSpline(_GUMSFITM, _GUMSFITVRADB, k=1)


#Conversion factors from A(B-V) to absorption in different passbands and colors
_FACG = 2.425
_FACV = 2.742
_FACVI = 1.237
_FACGRVS = 1.322

def GRVS(M, Abv, mu):
    '''
        Magnitude in GRVS band.
        
        Parameters
        ----------
        M : float
            Stellar mass. [Msun]
    '''
    M = M.to('Msun').value
    return -vminGrvsFromVmini(_VIfromM(M)) + _VfromM(M) + _FACGRVS*Abv + mu


def vradError(M, Av, mu):
    '''
        Error in radial velocity.
        
        Parameters
        ----------
        M : float
            Stellar mass. [Msun]
    '''
    return (_VCF + _BfromM(M)*np.exp(_AfromM(M)*(_VfromM(M) + _FACV*Abv + mu -_VZM))) * u.km/u.s

def pmError(M, Abv, mu, beta=None):
    '''
        Error in propermotion in the two directions (RA*, DEC).
        
        Parameters
        ----------
        M : float
            Stellar mass. [Msun]
        beta : float
            Ecliptic latitude (rad)
    '''
    if(beta is None):
        return properMotionErrorSkyAvg(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv) *u.uas / u.year
    return properMotionError(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv, beta) *u.uas / u.year

def posError(M, Abv, mu, beta=None):
    '''
        Error in position in the two directions (RA*, DEC).
        
        Parameters
        ----------
        M : float
            Stellar mass.
        beta : float
            Ecliptic latitude (rad)
    '''
    if(beta is None):
        return positionErrorSkyAvg(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv)*u.uas
    return positionError(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv, beta)*u.uas

def piError(M, Abv, mu, beta=None):
    '''
        Error in parallax.
        
        Parameters
        ----------
        M : float
            Stellar mass.
        beta : float
            Ecliptic latitude (rad)
    '''
    if(beta is None):
        return parallaxError(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv)*u.uas
    return parallaxError(_GfromM(M) + mu + _FACG*Abv, _VIfromM(M) + _FACVI*Abv, beta)*u.uas
