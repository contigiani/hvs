# hvs

Generate mock catalogs of hypervelocity stars in the Milky Way.  


## Setup

Despite the fact that is a python module, the `setup.py` file does not exists (yet). To use the module, make sure that `hvs/` is in a directory included in your **PYTHONPATH** environment variable.

## Documentation

You can access every method's docstring by using the help() function in python.

## Workflow

1. Define an ejection method/mechanism as a subclass of EjectionModel

2. Create an instantaneous ejection sample. Because the ejection rate (# HVS per unit time) is assumed constant, 
the expected flight time and age at observation can also be sampled at this time. 

3. Propagate the sample in the Galaxy using the propagate() method. 



### Ejection models
The class `Rossi2017` inside `ejection.py` is an example of a pre-modelled ejection method.
    
The class `EjectionModel` is the basic structure every ejection model class should be based on. Please make sure that custom ejection models are based on that.

### Examples and more
See the `examples/` directory for some basic scripts.
    
- `myfirstcatalog.py` shows the basic workings of the class HVSsample() and Rossi2017()

- `split.py` shows how the orbit propagation and be parallelized over multiple machines or processes.

- `Rossi2017comparison.py` is a comparison between the two sampling methods for this ejection mechanism. 
    
    

## References

* Contigiani+ (in prep.)

* Marchetti+ (in prep.)

* Rossi E. M., Kobayashi S., Sari R., 2014, ApJ ([arxiv](https://arxiv.org/abs/1307.1134))

* Rossi E. M., Marchetti T., Cacciato M., Kuiack M., Sari R., 2017, MNRAS ([arxiv](https://arxiv.org/abs/1608.02000))

* Hills J.G., 1988, Nature, 331, 687
