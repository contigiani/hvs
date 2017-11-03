import numpy as np
from hvs import HVSsample, Rossi2017

ejectionmodel = Rossi2017() # The name will appear in the final catalog
mysample = HVSsample(ejectionmodel, n=3e2)

from hvs.utils.mwpotential import MWPotential
from astropy import units as u

default_potential = MWPotential()  # This potential can be personalized, check the documentation using help()
mysample.propagate(potential = default_potential, dt=0.01*u.Myr) # See documentation

vtheta, vr = mysample.precision_check(default_potential)

vtheta_a, vr_a = np.array(vtheta.value), np.array(vr.value)
np.save('vtheta', vtheta_a)
np.save('vr', vr_a)

vt = np.load('vtheta.npy')
vr = np.load('vr.npy')
from matplotlib import pyplot as plt
print vt/vr, vt
plt.hist(vt/vr)
plt.show()
