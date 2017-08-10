'''
    Compares the two sampling methods, one using an analytic form, one using a Montecarlo method. 
'''
import numpy as np
from hvs import HVSsample, Rossi2017


ejectionmodel = Rossi2017()

# generate...

fit_sample = HVSsample(ejectionmodel, name='FIT', n=1e5, pl=True, verbose=True)
fit_sample.save('ejcatalog_fit.fits')
mc_sample = HVSsample(ejectionmodel, name='MC', n=1e5, verbose=True) # pl=False is the default value
mc_sample.save('ejcatalog_mc.fits')


# ...or load
'''
mc_sample = HVSsample('ejcatalog_mc.fits')
fit_sample = HVSsample('ejcatalog_fit.fits')
'''

# Plot the mass distributions, for comparison
from matplotlib import pyplot as plt

plt.hist(mc_sample.m, bins=np.logspace(-1, 1, 20), histtype='step', label='MC', normed=1)
plt.hist(fit_sample.m, bins=np.logspace(-1, 1, 20), histtype='step', label='FIT', normed=1)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.ylim([1e-5, 20])
plt.legend()
plt.show()
