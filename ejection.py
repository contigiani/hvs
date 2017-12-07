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
        HVS ejection model from Rossi+ 2017. Isotropic ejection from the GC, smooth Gaussian in
        the radial direction and with powerlaw mass/velocity distribution. Can generate an ejection sample using a
        Monte Carlo approach based on inverse transform sampling.

        Cannot be used to fit data, because it lacks an analytic form for the ejection distribution rate.

        Attributes
        ---------
        _name : string
            Name of the Ejection method
        v_range : Quantity
            Allowed range of HVS initial velocities
        m_range : Quantity
            Allowed range of HVS masses
        T_MW : Quantity
            Milky Way lifetime
        M_BH : Quantity
            Mass of the BH at the GC
        alpha : float
            Exponent of the power-law for the distribution of the semi-major axis in binaries
        gamma : float
            Exponent of the power-law for the distribution of the mass ratio in binaries


        Methods
        -------
        g :
            Hypervelocity star survival function as a function of the flight time expressed as a lifetime fraction
    '''
    v_range = [450, 5000]*u.km/u.s
    m_range = [0.5, 9]*u.Msun
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015
    M_BH = 4e6*u.Msun # Black hole mass

    alpha = -1.
    gamma = -3.5

    def __init__(self, name_modifier = None, r_params = [3*u.pc, 100*u.pc, 4]):
        '''
        Parameters
        ----------
        name_modifier : str
            Add a modifier to the class name
        r_params : list
            Parameters of the radial ejection distribution component. In order: mean value, standard deviation, max
            number of sigmas, see Marchetti+ 2018
        '''
        if(name_modifier is not None):
            self._name = 'Rossi 2017 - ' + name_modifier
        else:
            self._name = 'Rossi 2017'

        self.centralr, self.sigmar, self.Nsigma = r_params

    def g(self, r):
        '''
            Hypervelocity star survival function as a function of the flight time expressed as lifetime fraction.
            See Marchetti+ 2017 and Contigiani+ 2018.

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

    def _inverse_cumulative_mp(self, x):
        '''
            Inverse of the cumulative function of the Kroupa IMF f(mp) as a function of x.

            F(mp) = int_[0.1, 100] f(mp)

            returns mp such that

            F(mp) = x

        '''
        x = np.atleast_1d(x)

        F_cut = 0.729162 # F(x) at the breaking point
        total_int = 6.98626 # total integral of the broken power law between 0.1 and 100


        idx_cut = x<F_cut
        result = np.zeros(x.shape)

        result[idx_cut] = ((0.1)**(-0.3)-x[idx_cut]*0.3*0.5*total_int)**(1./(-0.3))
        result[~idx_cut] = (-total_int * (x[~idx_cut] - F_cut)*(1.3) + (0.5)**(-1.3))**(-1./1.3)

        return result

    def _inverse_cumulative_a(self, x, mp):
        amin = 2.5*mp
        amax = 2000.

        if self.alpha==-1:
            return amin*(amax/amin)**x
        else:
            return (  (amax**(1.+self.alpha) - amin**(1.+self.alpha))*x + amin**(1.+self.alpha) )**(1./(1.+self.alpha))


    def _inverse_cumulative_q(self, x, mp):
        qmin = 0.1/mp
        qmax = 1.

        if self.gamma==-1:
            return qmin*(qmax/qmin)**x
        else:
            return (  (qmax**(1.+self.gamma) - qmin**(1.+self.gamma))*x + qmin**(1.+self.gamma) )**(1./(1.+self.gamma))

    def sampler(self, n, xi = 0, verbose=False):
        '''
            Samples from the ejection distribution to generate an ejection sample.
            The distribution mass and velocity is generated using a Montecarlo approach (Rossi 2014). The functions
            _inverse_cumulative_mp, _inverse_cumulative_a, _inverse_cumulative_q dictate the parameters of the
            progenitor binary population. They are the inverse cumulative distributions
            of the mass ratio, semi-major axis and primary mass respectively.

            The velocity vector is assumed radial.

            The distance from the origin (Galactic Center or GC) is Gaussian, with parameters r_params

            The following boundaries are imposed by default on the binary parameters:

            ::    0.1/mp<q<1, Rsun*(mp/Msun)<a<2000*Rsun, 0.1<mp<100

            Parameters
            ----------
                n : int
                    Expected Size of the sample. It is always rounded to the nearest multiple of 100. The output sample
                    might have a different size depending on the allowed ranges of HVS velocity and mass.
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

        # Sample the binary properties q, a, mp using inverse sampling
        from scipy.optimize import fsolve

        n = int(n)

        # Inverse sampling
        uniform_for_mp, uniform_for_q, uniform_for_a = np.random.uniform(0, 1, (3, n))
        mp = self._inverse_cumulative_mp(uniform_for_mp)
        a, q = self._inverse_cumulative_a(uniform_for_a, mp), self._inverse_cumulative_q(uniform_for_q, mp)
        mp, a = mp*u.Msun, a*u.Rsun

        if(verbose):
            from matplotlib import pyplot as plt

            plt.figure()
            plt.hist(mp, bins=np.logspace(-1, 2, 30))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('mp distribution')
            plt.show()

            plt.figure()
            plt.hist(q, bins=np.logspace(-3, 1, 30))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('q distribution')
            plt.show()

            plt.figure()
            plt.hist(a, bins=np.logspace(np.log10(0.25), np.log10(2000), 30))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('a distribution')
            plt.show()



        ur = np.random.uniform(0,1,n)
        idx = ur>=0.5

        M_HVS, M_C = np.zeros(n)*u.Msun, np.zeros(n)*u.Msun
        M_HVS[idx] = mp[idx]
        M_HVS[~idx] = mp[~idx]*q[~idx]
        M_C[idx] = mp[idx]*q[idx]
        M_C[~idx] = mp[~idx]

        v = (np.sqrt( 2.*const.G.cgs*M_C / a )  * ( self.M_BH/(M_C+M_HVS) )**(1./6.)).to('km/s')


        idx = (M_HVS > self.m_range[0]) & (M_HVS < self.m_range[1]) & (v > self.v_range[0]) & (v < self.v_range[1])
        n = idx.sum()

        m, v = M_HVS[idx], v[idx]


        # Distance from GC normally distributed
        r0 = np.abs(np.random.normal(0, 1, n))*self.sigmar+self.centralr

        # Isotropic position unit vector in spherical coordinates
        phi0 = np.random.uniform(0,2*PI, n)*u.rad
        theta0 = np.arccos( np.random.uniform(-1,1, n))*u.rad

        # The velocity vector points radially.
        phiv0 = np.zeros(n)*u.rad#np.random.uniform(-PI/2,PI/2, n)*u.rad
        thetav0 = theta0 #np.arccos( np.random.uniform(-1,1, n))*u.rad


        # Age and flight time
        T_max = t_MS(m, xi)
        T_max[T_max>self.T_MW] = self.T_MW

        e1, e2 = np.random.random((2, n))
        tage, tflight = T_max * (1-e1+e1*e2), T_max * e1 * e2

        return r0, phi0, theta0, v, phiv0, thetav0, m, tage, tflight, n



class Contigiani2018(EjectionModel):
    '''
        HVS ejection model from Contigiani+ 2018. Isotropic ejection from the GC, smooth Gaussian in
        the radial direction and with powerlaw for mass&velocity distribution.

        Attributes
        ---------
        _name : string
            Name of the Ejection method
        v_range : Quantity
            Allowed range of HVS initial velocities
        m_range : Quantity
            Allowed range of HVS masses
        T_MW : Quantity
            Milky Way lifetime
        Lmax : Quantity
            Maximum specific angular momentum allowed (i.e. L < Lmax implies L \sim 0)


        Methods
        -------
        g :
            Hypervelocity star survival function as a function of the flight time expressed as a lifetime fraction
        R :
            Ejection density distribution, function of mass, velocity, distance w.r.t. GC
    '''
    v_range = [450, 5000]*u.km/u.s
    m_range = [0.5, 9]*u.Msun
    T_MW = 13.8*u.Gyr # MW maximum lifetime from Planck2015
    Lmax = 100*u.pc*u.km/u.s

    def __init__(self, name_modifier = None, vm_params = [1530*u.km/u.s, -0.65, -1.7, -6.3, -1], \
                    r_params = [3*u.pc, 100*u.pc, 4]):
        '''
        Parameters
        ----------
        name_modifier : str
            Add a modifier to the class name
        vm_params : list
            Parameters of the velocity/mass ejection distribution component. These are the parameters a, b, c, d, e
            in Contigiani+ 2018
        r_params : list
            Parameters of the radial ejection distribution component. In order: mean value, standard deviation, max
            number of sigmas, see Contigiani+ 2018
        '''
        if(name_modifier is not None):
            self._name = 'Contigiani 2018 - ' + name_modifier
        else:
            self._name = 'Contigiani 2018'


        self.a, self.b, self.c, self.d, self.e = vm_params
        self.centralr, self.sigmar, self.Nsigma = r_params

    def g(self, r):
        '''
            Hypervelocity star survival function as a function of the flight time expressed as lifetime fraction, used
            for likelihood computation. See Marchetti+ 2017.

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


    def R(self, m, vx, vy, vz, x, y, z):
        '''
            Ejection rate distribution for likelihood

            Parameters
            ----------
                m : Quantity
                    HVS stellar mass
                vx, vy, vz : Quantity
                    HVS velocity at ejection point in Galactocentric Cartesian coords
                x, y, z : Quantity
                    HVS position in Galactocentric Cartesian coords
        '''

        r = np.sqrt(x**2. + y**2. + z**2.)
        v = np.sqrt(vx**2. + vy**2. + vz**2.)
        L = np.sqrt((y*vz - z*vy)**2. + (x*vz-z*vx)**2. + (x*vy-y*vx)**2.)

        r = ((r - self.centralr)/self.sigmar).to(1).value # normalized r


        if(m.size == 1):
            m =  m*np.ones(vx.shape)

        if(vx.shape != x.shape):
            raise ValueError('The input Quantities must have the same shape.')

        #Boundaries:
        idx = (v >= self.v_range[0]) & (v <= self.v_range[1]) & (m >= self.m_range[0]) \
                & (m <= self.m_range[1]) & (r < self.Nsigma) & (r>=0) & (L <self.Lmax)

        result = np.full(r.shape, np.nan)
        result[~idx] = 0

        # Mass-velocity component
        v0 = np.full(r.shape, 0)*u.km/u.s
        v0[idx] = np.power((m[idx]/u.Msun).to('1').value, self.b)*self.a

        idx1 = idx & (v >= v0)
        idx2 = idx & (v < v0)

        result[idx1] = np.power(m[idx1], self.c) * np.power(v[idx1]/v0[idx1], self.d) * np.exp(-np.power(r[idx1], 2.)/2.)
        result[idx2] = np.power(m[idx2], self.c) * np.power(v[idx2]/v0[idx2], self.e) * np.exp(-np.power(r[idx2], 2.)/2.)

        return result


    def _lnprobmv(self, data):
        '''
            Log probability in mass - ejection velocity space

            Parameters
            ----------
                m : Quantity
                    HVS stellar mass
                v : Quantity
                    HVS velocity at ejection point
        '''

        m = data[0]
        v = data[1]

        result = np.full(m.shape, np.nan)

        idx = (v*u.km/u.s >= self.v_range[0]) & (v*u.km/u.s <= self.v_range[1]) & (m*u.Msun >= self.m_range[0]) \
                & (m*u.Msun <= self.m_range[1])

        result[~idx] = -np.inf

        v0 = np.full(m.shape, -np.inf)
        v0[idx] = np.power(m[idx], self.b)*(self.a.to('km/s').value)

        idx1 = idx & (v >= v0)
        idx2 = idx & (v < v0)

        result[idx1] = self.c*np.log(m[idx1]) + self.d*np.log(v[idx1]/v0[idx1])
        result[idx2] = self.c*np.log(m[idx2]) + self.e*np.log(v[idx2]/v0[idx2])

        return result


    def sampler(self, n, xi = 0, verbose=False):
        '''
            Samples from the ejection distribution to generate an ejection sample.
            The distribution in mass and velocity space is sampled from the power-law fit from
            Contigiani+ 2018.

            The velocity vector is assumed radial.

            Parameters
            ----------
                n : int
                    Expected Size of the sample. It is always rounded to the nearest multiple of 100.
                xi : float
                    Assumed stellar metallicity. See utils.mainsequence.t_MS()

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

        # Sample stellar mass and velocity magnitude
        ndim = 2
        p0 = [np.random.rand(2)*np.array([1, 100])+np.array([3, 1000]) for i in xrange(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprobmv)

        if(verbose):
            print('burn in...')
        pos, prob, state = sampler.run_mcmc(p0, 100)
        if(verbose):
            print('burn in done')

            try:
                print("Mean acceptance fraction at burn in:")
                print(np.mean(sampler.acceptance_fraction))
                print("Autocorrelation time at burn in:")
                print(sampler.get_autocorr_time())
            except Exception:
                print("----")
                pass

        sampler.reset()
        sampler.run_mcmc(pos, int(n/nwalkers), rstate0=state)

        if(verbose):
            try:
                print("Mean acceptance fraction when sampling:")
                print(np.mean(sampler.acceptance_fraction))
                print("Autocorrelation time when sampling:")
                print(sampler.get_autocorr_time())
            except Exception:
                print("----")
                pass

        m, v = sampler.flatchain[:,0]*u.Msun, sampler.flatchain[:,1]*u.km/u.s

        # Distance from GC normally distributed
        r0 = np.abs(np.random.normal(0, 1, n))*self.sigmar+self.centralr

        # Isotropic position unit vector in spherical coordinates
        phi0 = np.random.uniform(0,2*PI, n)*u.rad
        theta0 = np.arccos( np.random.uniform(-1,1, n))*u.rad

        # The velocity vector points radially.
        phiv0 = np.zeros(n)*u.rad#np.random.uniform(-PI/2,PI/2, n)*u.rad
        thetav0 = theta0 #np.arccos( np.random.uniform(-1,1, n))*u.rad


        # Age and flight time
        T_max = t_MS(m, xi)
        T_max[T_max>self.T_MW] = self.T_MW

        e1, e2 = np.random.random((2, n))
        tage, tflight = T_max * (1-e1+e1*e2), T_max * e1 * e2

        return r0, phi0, theta0, v, phiv0, thetav0, m, tage, tflight, n
