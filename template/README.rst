
Quick Example
-------------

The following example takes 12CO J=2-1 line emission from the disc of MWC 480 as observed by the ALMA large program MAPS. The parameters provided in the accompanying '.txt' file in this folder were obtained from previous modelling of this object with ``discminer``.

To start with, let's download and prepare (clip and downsample) the datacube that we will be using throughout this guide:

.. code-block:: bash

   ./download-MAPS.sh #Download MWC480 12CO data from the MAPS website
   python prepare_data.py #Clip and downsample cube

Mining scripts
==============

Now, in the ``../_mining`` folder you will find various scripts to analyse the structure and dynamics of the disc and produce useful outputs. Those scripts are adapted to read metadata and model properties from a parameter file generated as follows,

.. code-block:: bash

   python ../_mining/make_parfile.py

Next, two additional *make* scripts will produce the model channel and moment maps necessary for the rest of the analysis,

.. code-block:: bash

   python ../_mining/make_channels.py
   python ../_mining/make_single_moments.py -k gaussian

The former command displays data and best-fit model channel maps, and the latter produces three different types of moment maps: (a) **peak intensities** (b) **line widths** and (c) **centroid velocities** for both data and model, using Gaussian fits along the velocity axis of the cubes. You can visualise the output moment maps and residuals with,

.. code-block:: bash

   python ../_mining/plot_moment+offset.py -m peakintensity 

   python ../_mining/plot_moment+residuals.py -m linewidth
   python ../_mining/plot_moment+residuals.py -m velocity

   
- Tip: The majority of the *mining* scripts support multiple arguments that allow you do different things directly from command line. A list of those arguments can be printed using the ``-h`` flag as in ``python ../_mining/plot_moment+offset.py -h``, which produces the following output,

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

Carrying on with the tutorial, it is also possible to display residual maps in Cartesian or polar coordinates in the disc reference frame. Internally, this requires knowledge of the disc vertical structure and orientation in order to translate celestial into disc coordinates; the ``discminer`` best-fit model provides this information.

.. code-block:: bash

   python ../_mining/plot_residuals+all.py -c disc
   
   python ../_mining/plot_residuals+deproj.py -m peakint
   python ../_mining/plot_residuals+deproj.py -m velocity
   python ../_mining/plot_residuals+deproj.py -m velocity -p polar

Finally, the following script attempts to reveal asymmetric and localised signatures in the disc by analysing the distribution of peak residuals,

.. code-block:: bash

   python ../_mining/plot_peak_residuals.py -m velocity -i 2

   
- Tip: You can easily access the different attributes and methods associated with a given variable by running your scripts on an ``IPython`` terminal,

.. code-block:: bash

   ipython
   run ../_mining/plot_attributes_model.py
   model.skygrid #print dictionary with sky grid information
   
