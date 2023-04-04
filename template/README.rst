
Quick Example
-------------

The following example is based on 12CO J=2-1 line data from the disc of MWC 480 as observed by the ALMA large program MAPS. The parameters provided in the accompanying '.txt' file in this folder were obtained from previous modelling of this object with ``discminer``. Since the ``discminer`` model assumes a smooth and Keplerian disc, any differences that appear from the comparison between the data and the model are mainly tracing deviations from Keplerian rotation and intensity perturbations driven by temperature, turbulence and density variations in the gas disc.

To start with, let's download and prepare (clip and downsample) the datacube that will be used throughout this guide:

.. code-block:: bash

   ./download-MAPS.sh #Download MWC480 12CO data from the MAPS website
   python prepare_data.py #Clip and downsample cube

Here is a quick view of selected channel maps from this disc and tracer,

.. image:: ../images/channel_maps_data.png
   :width: 500
   
Mining scripts
==============

Now, in the ``../_mining`` folder you will find several scripts that will guide you through the analysis of the structure and dynamics of the disc. Those scripts are adapted to read metadata for the disc of interest from a parameter file generated automatically from command line as follows,

.. code-block:: bash

   python ../_mining/make_parfile.py

Next, two additional *make* scripts must be run in order to produce the model channel and moment maps necessary for the rest of the analysis,

.. code-block:: bash

   python ../_mining/make_channels.py
   python ../_mining/make_single_moments.py -k gaussian

The former command saves and displays data and best-fit model channel maps, as well as residuals resulting from the subtraction of model channel intensities to those of the data,



The latter command produces three different types of moment maps: (a) **peak intensities** (b) **line widths** and (c) **centroid velocities**, which are simply the attributes of (in this case) Gaussian kernels fitted along the velocity axis of the input data and model cubes. You can visualise the output moment maps and residuals with,

.. code-block:: bash

   python ../_mining/plot_moment+offset.py -m peakintensity 

   python ../_mining/plot_moment+residuals.py -m linewidth
   python ../_mining/plot_moment+residuals.py -m velocity
   

.. note::
   The majority of the *mining* scripts support multiple arguments that allow you do different things directly from command line. A list of those arguments can be printed using the ``-h`` flag as in ``python ../_mining/plot_moment+offset.py -h``, which produces the following output,

.. code-block:: bash

   Plot moment map [velocity, linewidth, [peakintensity, peakint]?

   optional arguments:
		-h, --help            show this help message and exit
		-m {velocity,linewidth,lineslope,peakint,peakintensity}, --moment {velocity,linewidth,lineslope,peakint,peakintensity}
		velocity, linewidth or peakintensity
		-k {gauss,gaussian,bell,dgauss,doublegaussian,dbell,doublebell}, --kind {gauss,gaussian,bell,dgauss,doublegaussian,dbell,doublebell}
		gauss(or gaussian), dbell(or doublebell)
		-s {up,upper,low,lower}, --surface {up,upper,low,lower}
                upper or lower surface moment map		

Carrying on with the tutorial, you can have a quick look at the radial dependence of the main model attributes retrieved for both upper and lower emitting surfaces of the disc via,

.. code-block:: bash

   python ../_mining/plot_attributes_model.py


Now, it is also possible to display residual maps in Cartesian or polar coordinates in the disc reference frame. Internally, this requires knowledge of the disc vertical structure and orientation in order to translate celestial into disc coordinates; the ``discminer`` best-fit model provides this information.

.. code-block:: bash

   python ../_mining/plot_residuals+all.py -c disc
   
   python ../_mining/plot_residuals+deproj.py -m peakint
   python ../_mining/plot_residuals+deproj.py -m velocity
   python ../_mining/plot_residuals+deproj.py -m velocity -p polar

Finally, the following script attempts to reveal asymmetric and localised signatures in the disc by analysing the distribution of peak residuals,

.. code-block:: bash

   python ../_mining/plot_peak_residuals.py -m velocity -i 2



.. note::   
   You can easily access the different attributes and methods associated with a given variable by running your scripts on an ``IPython`` terminal,

   .. code-block:: bash

      ipython
      run ../_mining/plot_attributes_model.py
      model.skygrid #print dictionary with sky grid information
   
