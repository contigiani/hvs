'''
    Simple catalog
'''
import numpy as np
from hvs import HVSsample, Rossi2017


'''
    Create ejection catalog
'''

# Initialize an ejection model, in this case the default Rossi2017 with some minor customizations
ejectionmodel = Rossi2017(name_modifier=' Customized')
print ejectionmodel._name

# Print the allowed ranges of HVS mass and initial velocity for this model -- cannot be changed as of now
print ejectionmodel.v_range
print ejectionmodel.m_range

# Create a sample of 1000 HVS by sampling the fit to the ejection distribution rate
mysample = HVSsample(ejectionmodel, name='My small sample', n=1e3, pl=True, verbose=True)

# Save it for later!
mysample.save('myfirstcatalog.fits')


'''
    Propagate it through the Galaxy
'''


# Take the default MW galactic potential

from hvs.utils.mwpotential import MWPotential
from astropy import units as u

default_potential = MWPotential()  # This potential can be personalized, check the documentation using help()
mysample.propagate(potential = default_potential, dt=1*u.Myr, check=True, threshold = 0.01) # See documentation

mysample.save('myfirstcatalog_propagated.fits')
