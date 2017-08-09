from astropy import units as u
import numpy  as np
from ejection import EjectionModel
import time

class HVSsample:
    '''
        HVS sample class. Main features:
        
        - Generate a sample of HVS at ejection according to a specified ejection model
        - Propagate the ejection sample in the Galaxy
        - #TODO Perform the Gaia selection cut and computes the expected errorbars in the phase space observables
        - #TODO Computes the sample likelihood of a given ejection model and galactic potential
        - Save/Load resulting catalog as FITS file (astropy.table)
        
        See README for usage examples.

        Attributes
        ---------
            self.size : int
                Size of the sample
            self.name : str
                Catalog name, 'Unknown'  by default
            self.ejmodel_name : str
                String identifier of the ejection model used to generate the sample, 'Unknown' by default
            self.cattype : int 
                0 if ejection sample, 1 is galactic sample, 2 if Gaia sample
            self.dt : Quantity
                Timestep used for orbit integration, 0.01 Myr by default
                
            self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0
                Initial phase space coordinates at ejection in cylindrical coordinates
            self.ra, self.dec, self.dist, self.pmra, self.pmdec, self.vlos : Quantity
                Equatorial coordinates of the stars. Right ascension and declination (ra, dec), heliocentric 
                distance (dist), proper motion in ra and dec directions (pmra must be declination-corrected), 
                line of sight velocity (vlos)
            self.e_ra, self.e_dec, self.e_dist, self.e_pmra, self.e_pmdec, self.e_vlos : Quantity
                Errors on the coordinates
            
            self.m
                Stellar masses of the sample            
            self.GRVS : ndarray 
                Magnitudes in the GRVS band
            self.tage, self.tflight : Quantity
                Age and flight time of the stars
                
        Methods
        -------
            __init__():
                Initializes the class, loads catalog if one is provided, otherwise creates one based on a given
                ejection model
                
            propagate(): 
                Propagates the sample in the Galaxy, changes cattype from 0 to 1
            gaia():
                Performs the Gaia selection cut, changes cattype from 1 to 2
            likelihood()
                Checks the likelihood of the sample for a given potential&ejection model combination
            
            save():
                Saves the sample in a FITS file
    '''

    solarmotion = [-14*u.km/u.s, 12.24*u.km/u.s, 7.25*u.km/u.s] 


    def __init__(self, inputdata=None, name=None, **kwargs):
        '''
        Parameters
        ----------
            inputdata : EjectionModel or str
                Instance of an ejection model or string to the catalog path 
            name : str
                Name of the catalog
            **kwargs : dict
                Arguments to be passed to the ejection model sampler if inputdata is an EjectionModel instance
        '''
        if(inputdata is None):
            raise ValueError('Initialize the class by either providing an \
                                ejection model or the path of an HVS catalog.')
        
        
        if isinstance(inputdata, EjectionModel):
            self._eject(inputdata, **kwargs)
        
        if isinstance(inputdata, basestring):
            self._load(inputdata)
 
        if(name is None):
            self.name = 'HVS catalog '+str(time.time()) 
    
    
    def _eject(self, ejmodel, **kwargs):
        '''
            Initializes the sample as an ejection sample
        '''
        self.ejmodel_name = ejmodel._name
        
        self.cattype = 0 
        
        self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
            self.m, self.tage, self.tflight, self.size = ejmodel.sampler(**kwargs)
        
    
    def propagate(self, potential, dt=0.01*u.Myr):
        '''
            Propagates the sample in the Galaxy, changes cattype from 0 to 1.
            
            Parameters
            ----------
                potential : galpy potential instance
                    Potential instance of the galpy library used to integrate the orbits
                dt : Quantity
                    Integration timestep
        '''
        from galpy.orbit import Orbit
        from galpy.util.bovy_coords import pmllpmbb_to_pmrapmdec, lb_to_radec, vrpmllpmbb_to_vxvyvz, lbd_to_XYZ
        
        if(self.cattype > 0):
            raise RuntimeError('This sample is already propagated!')   
        
        # Integration time step 
        self.dt = dt
        nsteps = numpy.ceil((self.tflight/self.dt).to('1').value)
        nsteps[nsteps<100] = 100
        
        # Initialize position in cylindrical coords
        rho = self.r0 * np.sin(self.theta0)
        z = self.r0 * np.cos(self.theta0)
        phi = self.phi0
        
        #... and velocity
        vR = self.v0 * np.sin(self.thetav0) * np.cos(self.phi0)
        vT = self.v0 * np.sin(self.thetav0) * np.sin(self.phi0)
        vz = self.v0 * np.cos(self.thetav0)    

        # Initialize empty arrays to save orbit data and integration steps
        self.pmll, self.pmbb, self.ll, self.bb, self.vlos, self.dist = (numpy.zeros(self.size) for i in xrange(6))
        self.orbits = [None] * self.size
    
    
        #Integration loop for the self.size orbits
        for i in xrange(self.size):
            #print(i)
            
            ts = np.linspace(0, 1, nsteps[i])*self.tflight[i]
            
            self.orbits[i] = Orbit(vxvv = [rho[i], vR[i], vT[i], z[i], vz[i], phi[i]], solarmotion=self.solarmotion)
            self.orbits[i].integrate(ts, potential, method='dopr54_c')
            
            # Export the final position           
            self.dist[i], self.ll[i], self.bb[i], self.pmll[i], self.pmbb[i], self.vlos[i] = \
                                                self.orbits[i].dist(self.tflight[i], use_physical=True), \
                                                self.orbits[i].ll(self.tflight[i], use_physical=True), \
                                                self.orbits[i].bb(self.tflight[i], use_physical=True), \
                                                self.orbits[i].pmll(self.tflight[i], use_physical=True) , \
                                                self.orbits[i].pmbb(self.tflight[i], use_physical=True)  , \
                                                self.orbits[i].vlos(self.tflight[i], use_physical=True)  


        # Radial velocity and distance + distance modulus
        self.vlos, self.dist = self.vlos * u.km/u.s, self.dist * u.kpc

        # Sky coordinates and proper motion
        data = pmllpmbb_to_pmrapmdec(self.pmll, self.pmbb, self.ll, self.bb, degree=True)*u.mas / u.year
        self.pmra, self.pmdec = data[:, 0], data[:, 1]
        data = lb_to_radec(self.ll, self.bb, degree=True)* u.deg
        self.ra, self.dec = data[:, 0], data[:, 1]
        
        # Done propagating
        self.cattype = 1    
    
    
    def gaia(self):
        #TODO
        return True
        
        
    def likelihood(self, potential, ejmodel):
        #TODO
        return True
        
        
    def save(self, path):
        '''
            Saves the sample in a FITS file. 
            
            Parameters
            ----------
                path : str
                    Path to the ourput fits file
        '''
        from astropy.table import Table
        
        meta_var = {'name' : self.name, 'ejmodel' : self.ejmodel_name, 'cattype' : self.cattype, 'dt' : self.dt}
        
        if(cattype == 0):
            # Ejection catalog
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight']
            
            
            
        if(cattype == 1):
            # Propagated catalog
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight, self.ra, self.dec, self.pmra, self.pmdec, \
                        self.dist, self.vlos]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', \
                        'dec', 'pmra', 'pmdec', 'dist', 'vlos']
            
        if(self.cattype == 2):
            # Gaia catalog
            #TODO
            return True

        data_table = Table(datalist, namelist, meta=meta_var)
        data_table.write(path, overwrite=True)
    
    
    def _load(self, path):
        '''
            Loads an HVS sample from a fits table
        '''
        from astropy.table import Table
        
        namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight', 'ra', 'dec', 'pmra', 
                    'pmdec', 'dist', 'vlos', 'e_ra', 'e_dec', 'e_pmra', 'e_pmdec', 'e_dist', 'e_vlos', 'GRVS']
        
        data_table = Table.read(path)
        
        #METADATA    
        self.name = 'Unkown'
        self.ejmodel_name = 'Unknown'
        self.dt = 0*u.Myr
        
        if('name' in data_table.meta):
            self.name = data_table.meta['name']
        
        if('ejmodel' in data_table.meta):
            self.ejmodel_name = data_table.meta['ejmodel']

        if('dt' in data_table.meta):
            self.dt = data_table.meta['ejmodel']
        
        if('cattype' not in data_table.meta):
            raise ValueError('Loaded fits table must contain the cattype metavariable!')
            return False
        self.cattype = data_table.meta['cattype']
        self.size = len(data_table)
        
        #DATA 
        i=0
        for colname in data.colnames:
            try:
                i = namelist.index(colname)
                setattr(self, colname, data[i].quantity)
                i+=1
            except ValueError:
                print('Column not recognized: ' + str(colname))
                i+=1
                continue

