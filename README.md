# hvs

Generate mock catalogs of hypervelocity stars in the Milky Way and compute the likelihood of an observed sample for a given Galactic potential/ejection mechanism combination. Based on [scipy](https://www.scipy.org/), [galpy](http://galpy.readthedocs.io/), and many others.


## Setup

This is a python module, but there is no `setup.py` file. To use the module, just make sure that `hvs/` sits in a directory included in your **PYTHONPATH** environment variable.

## Documentation

You can access every method's docstring by using the help() function in python.

## Workflow

1. Define an ejection method/mechanism as a subclass of EjectionModel

2. Create an instantaneous ejection sample. Because the ejection rate (# HVSs per unit time) is assumed constant, the expected flight time and age at observation can also be sampled at this time.

3. Propagate the sample in the Galaxy using the propagate() method.

4. Compute the GRVS photometry using the photometry() method (requires a Dustmap)

5. Calculate the likelihood of a well-formatted sample using the likelihood() method.

6. Have fun with the HVS sample!

### Ejection models
The classes `Rossi2017` and `Contigiani2018` inside `ejection.py` are two examples of pre-modelled ejection methods.

The class `EjectionModel` is the basic structure every ejection model class should be based on. Custom ejection models should be subclasses and follow the same structure.

### Examples and more
See the `examples/` directory for some basic scripts.

- `myfirstcatalog.py` shows the basic workings of the class HVSsample() and Rossi2017()

- `split.py` shows how the orbit propagation and be parallelized over multiple machines or processes.

- `RossiContigianicomparison.py` is a comparison between the two sampling methods for these ejection mechanisms.



## References

* O. Contigiani, Rossi E. M., Marchetti T., 2018, MNRAS (submitted)

* Marchetti T., O. Contigiani, Rossi E. M., Albert J. G., Brown A. G. A., Sesana A., 2018, MNRAS ([arxiv](https://arxiv.org/abs/1711.11397))

* Rossi E. M., Kobayashi S., Sari R., 2014, ApJ ([arxiv](https://arxiv.org/abs/1307.1134))

* Rossi E. M., Marchetti T., Cacciato M., Kuiack M., Sari R., 2017, MNRAS ([arxiv](https://arxiv.org/abs/1608.02000))

* Hills J.G., 1988, Nature, 331, 687
