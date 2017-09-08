from hvs.utils import split_fits, MWPotential, concatenate_fits
from hvs import HVSsample
import shutil
import os

# Split the catalog
split_fits("myfirstcatalog.fits", "myfirstdir", 10) # Loads myfirstcatalog.fits

default_potential = MWPotential()

def propagate_function(i):
    mysample = HVSsample('myfirstdir/'+str(i)+'.fits')
    mysample.propagate(default_potential)
    mysample.save('myfirstdir/'+str(i)+'.fits')

# Propagate every catalog individually (this can be parallelized easily, just run this for loop on multiple 
# machines/python instances)
for i in xrange(10):
    # The files
    if(os.path.isfile('myfirstdir/.'+str(i))):
        continue  
    open('myfirstdir/.'+str(i), 'wa').close()
    propagate_function(i)
    
# Concatenate the catalog
concatenate_fits("myfirstcatalog_propagated.fits", "myfirstdir") # Rewrites everything in myfirstcatalog.fits
shutil.rmtree("myfirstdir/")
