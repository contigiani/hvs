# hvs

Generate mock catalogs of hypervelocity stars in the Galaxy. 


## Setup

__...__

## Documentation

You can access every method's docstring by using the help() function in python.

## Workflow and usage examples

1. Define an ejection method/mechanism as a subclass of EjectionModel

2. Create an instantaneous ejection sample. Because the ejection rate (# HVS per unit time) is assumed constant, 
the expected flight time and age at observation can also be sampled at this time. 

3. Propagate the sample in the Galaxy using the propagate() method. 


See the `example/` directory for some basic scripts. The class `Rossi2017` inside `ejection.py` is an example 
of a pre-modelled ejection method.


## References

* Contigiani+ (in prep.)

* Marchetti+ (in prep.)

* Rossi E. M., Kobayashi S., Sari R., 2014, ApJ ([arxiv](https://arxiv.org/abs/1307.1134))

* Rossi E. M., Marchetti T., Cacciato M., Kuiack M., Sari R., 2017, MNRAS ([arxiv](https://arxiv.org/abs/1608.02000))
