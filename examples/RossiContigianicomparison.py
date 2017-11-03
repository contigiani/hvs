'''
    Compares the two sampling methods, one using an analytic form, one using a Montecarlo method. 
'''
import numpy as np
from hvs import HVSsample, Rossi2017, Contigiani2018


ejectionmodel1 = Rossi2017()
ejectionmodel2 = Contigiani2018()


# generate...

sample1 = HVSsample(ejectionmodel1, name='Rossi MC binaries', n=1e7, verbose=True)
sample1.save('ejcatalog_mc.fits')

sample2 = HVSsample(ejectionmodel2, name='Contigiani powerlaw fit', n=1e5, verbose=True)
sample2.save('ejcatalog_fit.fits')

# ...or load

sample1 = HVSsample('ejcatalog_mc.fits')
sample2 = HVSsample('ejcatalog_fit.fits')


# Plot the mass distributions, for comparison
from matplotlib import pyplot as plt


plt.hist(sample1.m, bins=np.logspace(np.log(0.5), np.log(9), 20), histtype='step', label='Rossi+ 2017', normed=1)
plt.hist(sample2.m, bins=np.logspace(np.log(0.5), np.log(9), 20), histtype='step', label='Contigiani+ 2018', normed=1)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xlabel('M / Msun')
plt.legend()
plt.show()

'''
from fastkde import fastKDE

myPDF, axes = fastKDE.pdf(sample1.m.to('Msun').value, sample1.v0.value)

print sample1.m.to('Msun').value

import pylab as PP


#Extract the axes from the axis list
v1,v2 = axes

#Plot contours of the PDF should be a set of concentric ellipsoids centered on
#(0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
#should be large
PP.contour(v1,v2,np.log(myPDF))
PP.gca().loglog()
PP.show()
'''
