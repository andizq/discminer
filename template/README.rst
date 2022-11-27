
How to use
-----------

The prototype scripts used here to test the latest version of ``discminer`` are based on a model of the 12CO emission from the disc of MWC 480. The parameters provided in the accompanying '.txt' file were obtained from previous modelling of this object with ``discminer``. If you wish to play with these scripts, you must first run the following lines to download and prepare (clip and downsample) the data cube:

.. code-block:: bash

   ./download-MAPS.sh #Download MWC480 12CO data from MAPS website
   python prepare_data.py 

Next, you can simply run the prototype scripts from a terminal:

.. code-block:: bash

   python make_channels_moments.py #show model cube, save channels and moment maps
   python make_velocity_maps.py #load and show velocity maps

Running the prototype scripts on an ``IPython`` terminal (or a Jupyter notebook) is helpful to inspect the contents of the variables and methods defined,

.. code-block:: bash

   ipython
   run make_model_attributes.py
   model.skygrid #print dictionary with sky grid information
   
