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
            self.T_MW : Quantity
                Milky Way maximum lifetime
            
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
            
            solarmotion : Quantity
                Solar motion  
            
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

    # U, V, W in km/s in galactocentric coordinates. Galpy notation requires U to have a minus sign. 
    solarmotion = [-14., 12.24, 7.25]  
    dt = 0.01*u.Myr
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015
    
    
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
                                ejection model or an input HVS catalog.')
        
        if(name is None):
            self.name = 'HVS catalog '+str(time.time())
        else:
            self.name = name
        
        if isinstance(inputdata, EjectionModel):
            self._eject(inputdata, **kwargs)
        
        if isinstance(inputdata, basestring):
            self._load(inputdata)
    
    
    def _eject(self, ejmodel, **kwargs):
        '''
            Initializes the sample as an ejection sample
        '''
        self.ejmodel_name = ejmodel._name
        
        self.cattype = 0 
        
        self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
            self.m, self.tage, self.tflight, self.size = ejmodel.sampler(**kwargs)
        
    
    def propagate(self, potential, dt=0.01*u.Myr, check=False, threshold=0.01):
        '''
            Propagates the sample in the Galaxy, changes cattype from 0 to 1.
            
            Parameters
            ----------
                potential : galpy potential instance
                    Potential instance of the galpy library used to integrate the orbits
                dt : Quantity
                    Integration timestep. Defaults to 0.01 Myr
                check : bool
                    Flag to check the energy of the orbits remains constant over time, also saves a E_data.fits
                    file in the current directory containing the percentage of integration steps that are defined as
                    energy outliars. Defaults to False
                threshold : float
                    Maximum relative energy difference between the initial energy and the energy at any point needed
                    to consider an integration step an energy outliar. Defaults to 0.01, meaning that any excess or 
                    deficit of 1% (or more) of the initial energy is enough to be registered as outliar
                
        '''
        from galpy.orbit import Orbit
        from galpy.util.bovy_coords import pmllpmbb_to_pmrapmdec, lb_to_radec, vrpmllpmbb_to_vxvyvz, lbd_to_XYZ
        
        if(self.cattype > 0):
            raise RuntimeError('This sample is already propagated!')   
        
        # Integration time step 
        self.dt = dt
        nsteps = np.ceil((self.tflight/self.dt).to('1').value)
        nsteps[nsteps<100] = 100
        
        # Initialize position in cylindrical coords
        rho = self.r0 * np.sin(self.theta0)
        z = self.r0 * np.cos(self.theta0)
        phi = self.phi0
        
        #... and velocity
        vR = self.v0 * np.sin(self.thetav0) * np.cos(self.phiv0)
        vT = self.v0 * np.sin(self.thetav0) * np.sin(self.phiv0)
        vz = self.v0 * np.cos(self.thetav0)    

        # Initialize empty arrays to save orbit data and integration steps
        self.pmll, self.pmbb, self.ll, self.bb, self.vlos, self.dist, self.energy_var = \
                                                                            (np.zeros(self.size) for i in xrange(7))
        self.orbits = [None] * self.size
    
        #Integration loop for the self.size orbits
        for i in xrange(self.size):
            print(i)
            
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

            # Energy check
            if(check):
                energy_array = self.orbits[i].E(ts)
                idx_energy = np.absolute(energy_array/energy_array[0] - 1) >  threshold
                self.energy_var[i] = float(idx_energy.sum())/nsteps[i] # percentage of outliars

        # Radial velocity and distance + distance modulus
        self.vlos, self.dist = self.vlos * u.km/u.s, self.dist * u.kpc

        # Sky coordinates and proper motion
        data = pmllpmbb_to_pmrapmdec(self.pmll, self.pmbb, self.ll, self.bb, degree=True)*u.mas / u.year
        self.pmra, self.pmdec = data[:, 0], data[:, 1]
        data = lb_to_radec(self.ll, self.bb, degree=True)* u.deg
        self.ra, self.dec = data[:, 0], data[:, 1]
        
        # Done propagating
        self.cattype = 1    
    
        # Save the energy check information
        if(check):
            from astropy.table import Table
            e_data = Table([self.m, self.tflight, self.energy_var], names=['m', 'tflight', 'pol'])
            e_data.write('E_data.fits', overwrite=True)
        
    def gaia(self):
        #TODO
        return True
        
        
    def likelihood(self, potential, ejmodel, dt=0.01*u.Myr, xi = 0, individual=False, n_samples=1, cov = None, \
                    weights=None):
        '''
        Computes the non-normalized log-likelihood of a given potential and ejection model for a given potential.
        When comparing different ejection models, make sure you renormalize the likelihood accordingly. 
        
        Can return the log-likelihoods of individual stars if individual is set to True. 
        
        Parameters
        ----------
        potential : galpy potential
            Potential to integrate the orbits with.
        ejmodel : EjectionModel object
            Ejectionmodel to be tested.
        individual : bool
            If set to True, returns . The default value is False.
        n_samples : int
            Number of point to sample for the errorbars
        cov : iterable
            Covariance matrix for errors in the observed quantities. If not speficied, the errors in the sample are
            used instead. 
        weights : iterable
            List or array containing the weights for the log-likelihoods of the different stars.
        xi : float or array
            Assumed metallicity for stellar lifetime
        
        
        Returns
        -------
        
        log likelihood values : numpy.array or float
            Returns the log-likelihood of the entire sample or the log-likelihood for every single star if individual
            is True. 
        
        '''
        
        n_samples = np.array(n_samples)
        weights = np.array(weights)
        cov = np.array(cov)
        
        
        if(self.cattype == 0):
            raise ValueError("The likelihood can be computed only for a propagated sample.")
            
        if(self.size > 1e3):
            print("You are computing the likelihood of a large sample. This might take a while.")

        if((weights is not None) and (weights.size != self.size)):
            raise ValueError('The length of weights must be equal to the number of HVS in the sample.')
        
        if(cov is not None):
            if((cov.shape is not (7,7)) or (cov.shape is not (6,6))):
                raise ValueError('Covariance matrix must have shape 6x6 or 7x7 (if error on mass is considered)')
                
        
        #TODO sampling errorspace
        
        
        self.backwards_orbits = [None] * self.size
        self.back_dt = dt
        self.lnlike = np.ones(self.size) * (-np.inf)
        
        from hvs.utils import t_ms
        lifetime = t_MS(self.m, xi)
        lifetime[lifetime>self.T_MW] = self.T_MW
        nsteps = np.ceil((lifetime/self.back_dt).to('1').value)
        nsteps[nsteps<100] = 100
        
        
        import astropy.coordinates as coord
        from gala.coordinates import vhel_to_gal
        
        
        vSun = [-self.solarmotion[0], self.solarmotion[1], self.solarmotion[2]] * u.km / u.s # (U, V, W) 
        vrot = [0., 220., 0.] * u.km / u.s

        RSun = 8. * u.kpc
        zSun = 0 * u.pc 

        v_sun = coord.CartesianDifferential(vSun + vrot)
        gc = coord.Galactocentric(galcen_distance=RSun, z_sun=zSun, galcen_v_sun=v_sun)
        
        
        for i in xrange(self.size):
            ts = np.linspace(0, 1, nsteps[i])*self.lifetime[i]
            self.backwards_orbits[i] = Orbit(vxvv = [self.ra[i], self.dec[i], self.dist[i], \
                                    self.pmra[i], self.pmdec[i], self.vlos[i]], \
                                    solarmotion=self.solarmotion, radec=True).flip()
            self.backwards_orbits[i].integrate(ts, potential, method='dopr54_c')
                
            dist, ll, bb, pmll, pmbb, vlos = self.backwards_orbits[i].dist(ts, use_physical=True) * u.kpc, \
                                                self.backwards_orbits[i].ll(ts, use_physical=True) * u.deg, \
                                                self.backwards_orbits[i].bb(ts, use_physical=True) * u.deg, \
                                                self.backwards_orbits[i].pmll(ts, use_physical=True) *u.mas/u.year, \
                                                self.backwards_orbits[i].pmbb(ts, use_physical=True) * u.mas/u.year, \
                                                self.backwards_orbits[i].vlos(ts, use_physical=True) * u.km/u.s 


            galactic = coord.Galactic(l=ll, b=bb, distance=dist, pm_l_cosb=pmll, pm_b=pmbb, radial_velocity=vlos)
            gal = galactic.transform_to(gc)
            vtot = np.sqrt(gal.v_x**2. + gal.v_y**2. + gal.v_z**2.).to(u.km/u.s)            
            r = np.sqrt(gal.x**2. + gal.y**2. + gal.z**2.).to(u.kpc) 
            
            self.lnlike[i] = np.log((ejmodel.R(self.m, vtot, rtot) * ejmodel.g(np.linspace(0, 1, nsteps[i]))).sum())
            
        if(individual):
            return self.lnlike 
        return self.lnlikes.sum()
        
    def save(self, path):
        '''
            Saves the sample in a FITS file. 
            
            Parameters
            ----------
                path : str
                    Path to the ourput fits file
        '''
        from astropy.table import Table
        
        meta_var = {'name' : self.name, 'ejmodel' : self.ejmodel_name, 'cattype' : self.cattype, \
                    'dt' : self.dt.to('Myr').value}
        
        if(self.cattype == 0):
            # Ejection catalog
            datalist = [self.r0, self.phi0, self.theta0, self.v0, self.phiv0, self.thetav0, \
                        self.m, self.tage, self.tflight]
            namelist = ['r0', 'phi0', 'theta0', 'v0', 'phiv0', 'thetav0', 'm', 'tage', 'tflight']
            
            
            
        if(self.cattype == 1):
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

        data_table = Table(data=datalist, names=namelist, meta=meta_var)
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
        data_table.meta =  {k.lower(): v for k, v in data_table.meta.items()}
        self.name = 'Unkown'
        self.ejmodel_name = 'Unknown'
        self.dt = 0*u.Myr
        
        if ('name' in data_table.meta):
            self.name = data_table.meta['name']
        
        if('ejmodel' in data_table.meta):
            self.ejmodel_name = data_table.meta['ejmodel']

        if('dt' in data_table.meta):
            self.dt = data_table.meta['dt']*u.Myr
        
        
        if('cattype' not in data_table.meta):
            raise ValueError('Loaded fits table must contain the cattype metavariable!')
            return False
        self.cattype = data_table.meta['cattype']
        self.size = len(data_table)
        
        #DATA 
        i=0
        for colname in data_table.colnames:
            try:
                i = namelist.index(colname)
                setattr(self, colname, data_table[colname].quantity)
                i+=1
            except ValueError:
                print('Column not recognized: ' + str(colname))
                i+=1
                continue

