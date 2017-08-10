'''
    Simple ejection catalog
'''
import numpy as np
from hvs import HVSsample, Rossi2017


# Initialize an ejection model, in this case the default Rossi2017 with some minor customizations
ejectionmodel = Rossi2017(name_modifier=' Customized')
print ejectionmodel._name

# Print the allowed ranges of HVS mass and initial velocity for this model -- can be changed
print ejectionmodel.v_range
print ejectionmodel.m_range

# Create a sample of 1000 HVS by sampling the fit to the ejection distribution rate
mysample = HVSsample(ejectionmodel, name='My Sample', n=1e5, pl=True, verbose=True)

# Save it for later!
mysample.save('myfirstcatalog.fits')
