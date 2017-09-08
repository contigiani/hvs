'''
    Simple catalog
'''
import numpy as np
from hvs import HVSsample, Rossi2017


'''
    Create ejection catalog
'''

# Initialize an ejection model, in this case the default Rossi2017 with some minor customizations
ejectionmodel = Rossi2017(name_modifier=' Customized') # The name will appear in the final catalog
print ejectionmodel._name

# Print the allowed ranges of HVS mass and initial velocity for this model -- can be changed
print ejectionmodel.v_range
print ejectionmodel.m_range

# Create a sample of n HVSs by sampling the fit to the ejection distribution rate
mysample = HVSsample(ejectionmodel, name='My small sample', n=1e2, pl=False, verbose=True)

# Save it for later!
mysample.save('myfirstcatalog.fits')
'''
    Propagate it through the Galaxy
'''


# Take the default MW galactic potential

from hvs.utils.mwpotential import MWPotential
from astropy import units as u

default_potential = MWPotential()  # This potential can be personalized, check the documentation using help()
mysample.propagate(potential = default_potential, dt=1*u.Myr, check=False, threshold = 1e-7) # See documentation

mysample.save('myfirstcatalog_propagated.fits')
