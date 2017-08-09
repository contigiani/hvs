import numpy as np
from astropy import units as u



class EjectionModel:
    '''
        Ejection model class
    '''
    
    _name = 'Unknown'
    
    def __init__(self, name):
        self.name = name
        
    def g(self, r):
        '''
            Survival function of HVS as a function of flight time expressed as lifetime fraction (r)
        '''
        raise NotImplementedError
    
    def R(self, m, v, r):
        '''
            Ejection density distribution, function of mass (m), total velocity (v), distance from galactic center (r)
        '''
        raise NotImplementedError
    
    def sampler(self):
        '''
            Sampler of the ejection distribution
        '''
        raise NotImplementedError



class Rossi2017(EjectionModel):
    '''
        HVS ejection model from Rossi 2017. Isotropic ejection from the GC, smooth Gaussian in 
        the radial direction and with powerlaw mass/velocity distribution. Can generate an ejection sample using a 
        montecarlo approach (see Rossi 2014).
        
        Attributes
        ---------
        _name : "Rossi 2017"
            Name of the Ejection method
        v_range : Quantity
            Allowed range of HVS initial velocities
        m_range : Quantity
            Allowed range of HVS masses
            
        Methods
        -------
        g(self, r) :
            Hypervelocity star survival function as a function of the flight time expressed as a lifetime fraction
        R(self, m, v, r) :
            Ejection density distribution, function of mass, total velocity, distance from GC
    ''' 
    v_range = [450, 5000]*u.km/u.s
    m_range = [0.1, 10]*u.Msun
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015
    
    
    def __init__(self, name_modifier = None, vm_params = [1530*u.km/u.s, -0.65, -1.7, -6.3, -1], \
                    r_params = [3*u.pc, 100*u.pc, 4]):
        '''
        Parameters
        ----------
        name_modifier : str
            Add a modifier to the class name
        vm_params : list
            Parameters of the velocity/mass ejection distribution component. These are the parameters a, b, c, d, e
            in Marchetti 2017b
        r_params : list
            Parameters of the radial ejection distribution component. In order: mean value, standard deviation, max 
            number of sigmas, see Marchetti 2017b
        '''
        if(name_modifier is not None):
            self._name = 'Rossi 2017 - ' + name_modifier
        else:
            self._name = 'Rossi 2017'
    
    
        self.a, self.b, self.c, self.d, self.e = vm_params
        self.centralr, self.sigmar, self.Nsigma = r_params
    
    
    def g(self, r):
        '''
            Hypervelocity star survival function as a function of the flight time expressed as lifetime fraction, used
            for likelihood computation
            
            Parameters
            ----------
                r : ndarray of float
                    Lifetime fractions
        '''
        
        result = np.full(r.shape, np.nan)
        idx = (r < 1.) & (r>0)
        result[idx] = 1. - r[idx] + (r[idx])*np.log(r[idx])
        result[r==0.]  = 1
        result[r==1.] = 0
        
        return result
        
        
    def R(self, m, v, r):
        '''
            Ejection rate distribution
            
            Parameters
            ----------
                m : Quantity
                    HVS stellar mass
                v : Quantity
                    HVS velocity at ejection point
                r : Quantity
                    Distance of the ejection point from the GC
        '''
        
        size = r.size
        r = ((r - self.centralr)/self.sigmar).to(1).value # normalized r
        
        
        if((m.shape != v.shape) or (v.shape != r.shape)):
            raise ValueError('The input Quantities must have the same shape.')
        
        #Boundaries of the space:
        idx = (v > self.v_range[0]) & (v < self.v_range[1]) & (m > self.m_range[0]) \
                & (m < self.m_range[1]) & (r < self.Nsigma) & (r>=0)
        
        result = np.full(r.shape, np.nan)
        result[~idx] = 0
        
        # Mass-velocity component
        v0 = np.full(r.shape, -np.inf)*u.km/u.s
        v0[idx] = np.power((m[idx]/u.Msun).to('1').value, self.b)*self.a
        
        idx1 = idx & (v > v0)
        idx2 = idx & (v < v0)
        
        result[idx1] = np.power(m[idx1], self.c) * np.power(v[idx1]/v0[idx1], self.d) 
        result[idx2] = np.power(m[idx2], self.c) * np.power(v[idx2]/v0[idx2], self.e)
        

        # Radial component
        result[idx] *= np.exp(-np.power(r[idx], 2.)/2.)
        
        return result
    
    
    def _lnprobmv(self, data):
        '''
            Log probability in mass - ejection velocity space. Simply invokes self.R() on a constant r
            
            Parameters
            ----------
                m : Quantity
                    HVS stellar mass
                v : Quantity
                    HVS velocity at ejection point
        '''
        
        result = self.R(np.atleast_1d(data[0])*u.Msun, \
                             np.atleast_1d(data[1])*u.km/u.s, \
                             self.centralr*np.atleast_1d(np.ones_like(data[0])))
        
        result[result>0] = np.log(result[result>0])
        result[result==0] = -np.inf
        
        
        return result
        

    def _lnprob_q_a_mp(self, data):   
        '''
        
        '''
        
        q, a, mp = data[0]. data[1]. data[2]
        result = np.full(q.shape, np.nan)
        
        #boundary
        idxboundary = (q >= 0.1/mp) & (q <= 1) & (mp >= 0.1) & (mp <= 100) & (a>=2.5*mp) & (a<2000)
        
        result[~idxboundary] = -np.inf
        result[idxboundary] = self._lnprobq(q[idxboundary]) + self.lnproba(a[idxboundary]) + \
                                self._lnprob(mp[idxboundary])
    
        return result
    
    
    def _lnprobq(self, q):
        # Auxilary function for _lnprob_q_a_mp - mass ratio distribution
        
        return -3.5*np.log(q)
    
    
    def _lnproba(self, a):
        # Auxilary function for _lnprob_q_a_mp - semi-major axis distribution
        
        return -np.log(a)
    
    
    def _lnprobmp(self, mp):
        # Auxilary function for _lnprob_q_a_mp - IMF for primary mass
        
        result = np.full(mp.shape, np.nan)
        idx = m > 0.5 #Cutoff of the Kroupa IMF
        
        result[idx] = -2.3*np.log(mp) + np.log(0.5)
        result[~idx] = -1.3*np.log(mp)
        
        return result    
    
    
    def sampler(self, n, xi = 0, pl=False, verbose=False):
        '''
            Samples from the ejection distribution to generate an ejection sample. 
            If pl is True, the distribution in mass and velocity space is sampled from the power-law fit from
            Marchetti 2017b. If pl is False, the distribution is generated using a Montecarlo approach (Rossi 2014).
            
            In this second case, the functions _lnprobq(), _lnproba(), _lnprobmp() dictate the parameters of the 
            progenitor binary population. They are respectively the distributions of the mass ratio, semi-major axis 
            and primary mass. The following boundaries are imposed by default:
            
            ::    0.1/mp<q<1, Rsun*(mp/Msun)<a<2000*Rsun, 0.1<mp<1
            
            Parameters
            ----------
                n : int
                    Expected Size of the sample. It is always rounded to the nearest multiple of 100. The output sample
                    might have a different size if 
                xi : float
                    Assumed stellar metallicity. See utils.mainsequence.t_MS()
                    
                pl : bool
                    Power-law flag, see description
                verbose : bool
                    Verbose flag, used to monitor the MCMC sampling
            
            Returns
            -------
                r0, phi0, theta0, v, phiv0, thetav0 : Quantity
                    Initial phase space position in spherical coordinates, centered on the GC
                m, tage, tflight
                    Stellar mass of the HVS, age at observation and tflight between ejection and observation
                    
                n : int
                    Size of the output ejection sample
            
        '''
        from utils.mainsequence import t_MS
        from math import ceil
        from astropy import constants as const
        import emcee
        PI = np.pi
        
        nwalkers = 100
        n = int(ceil(n/nwalkers)*nwalkers)
        
        # Mass and velocity magnitude
        if(pl):
            ndim = 2
            p0 = [np.random.rand(2)*np.array([1, 100])+np.array([3, 1000]) for i in xrange(nwalkers)]
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprobmv)
        
            if(verbose):
                print('burn in...')
            pos, prob, state = sampler.run_mcmc(p0, 100)
            if(verbose):
                print('burn in done')

            sampler.reset()
            sampler.run_mcmc(pos, int(n/nwalkers), rstate0=state)
            
            if(verbose):
                try:
                    print("Mean acceptance fraction:")
                    print(np.mean(sampler.acceptance_fraction))
                    print("Autocorrelation time:")
                    print(sampler.get_autocorr_time())
                except Exception:
                    pass
        
            m, v = sampler.flatchain[:,0]*u.Msun, sampler.flatchain[:,1]*u.km/u.s
        
        else:
            # q, a, mp
            ndim = 3
            nwalkers = 100
            p0 = [np.random.rand(3)*np.array([0.1,1.,1.])+np.array([0.5, 10, 3]) for i in xrange(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob_q_a_mp)
            
            if(verbose):
                print('burn in...')
            pos, prob, state = sampler.run_mcmc(p0, 100)
            if(verbose):
                print('burn in done')

            sampler.reset()
            sampler.run_mcmc(pos, int(n/nwalkers), rstate0=state)
            
            if(verbose):
                try:
                    print("Mean acceptance fraction:")
                    print(np.mean(sampler.acceptance_fraction))
                    print("Autocorrelation time:")
                    print(sampler.get_autocorr_time())
                except Exception:
                    pass
        
            q, a, mp = sampler.flatchain[:,0], sampler.flatchain[:,1]*u.Rsun, sampler.flatchain[:,2]*u.Msun
            
            ur = np.random.uniform(0,1,n)
            idx = ur>=0.5
            
            M_HVS, M_C = np.zeros(n)*u.Msun, np.zeros(n)*u.Msun
            M_HVS[idx] = Mp[idx]
            M_HVS[~idx] = Mp[~idx]*q[~idx]
            M_C[idx] = Mp[idx]*q[idx]
            M_C[~idx] = Mp[~idx]
            
            v = (np.sqrt( 2.*const.G.cgs*M_C / a )  * ( MBH/(M_C+M_HVS) )**(1./6.))
            
            
            idx = (M_HVS > self.m_range[0]) & (M_HVS < self.m_range[1]) & (v > self.v_range[0]) & (v < self.v_range[1])
            n = idx.sum()
            
            m, v = M_HVS[idx], v[idx]
        
        # Distance from GC normally distributed
        r0 = np.abs(np.random.normal(0, 1, n))*self.sigmar+self.centralr
        
        # Isotropic position unit vector in spherical coordinates
        phi0 = np.random.uniform(0,2*PI, n)*u.rad
        theta0 = np.arccos( np.random.uniform(-1,1, n))*u.rad

        # Isotropic velocity direction in spherical coordinates, only outwards
        phiv0 = np.random.uniform(-PI/2,PI/2, n)*u.rad
        thetav0 = np.arccos( np.random.uniform(-1,1, n))*u.rad 

        
        # Age and flight time
        T_max = t_MS(m, xi)
        T_max[T_max>self.T_MW] = self.T_MW
        
        e1, e2 = np.random.random((2, n))
        tage, tflight = T_max * (1-e1+e1*e2), T_max * e1 * e2
        
        return r0, phi0, theta0, v, phiv0, thetav0, m, tage, tflight, n
    
