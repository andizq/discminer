Running the MWC 480 12CO MCMC fit
=================================

From inside the ``fit/`` directory, start a new backend with:

.. code-block:: bash

   python -W ignore fit_mc_mwc480.py -b 0

The flag ``-b 0`` creates a new ``emcee`` backend. Subsequent runs can reuse
the existing backend with:

.. code-block:: bash

   python -W ignore fit_mc_mwc480.py -b 1

Model prescriptions
-------------------

The script defines the prescriptions used in the example:

* Keplerian rotation with vertical dependence;
* upper and lower exponentially tapered emitting surfaces;
* bell-shaped line profiles;
* a radial intensity profile with an outer cutoff;
* free parameters for orientation, velocity, intensity, linewidth, line slope,
  and both emitting surfaces.

The fitting step can be computationally expensive (the precomputed
230walker, 10000-step run took ~6 hours on 48 cores and used
about 600 MB of memory per core; performance will vary by system). For this
reason, the parent example folder contains the precomputed
``log_pars_mwc480_12co_0p2_maps_cube_230walkers_10000steps.json`` file so that
the analysis workflow can be reproduced without rerunning the MCMC fit while
retaining the fit metadata.
