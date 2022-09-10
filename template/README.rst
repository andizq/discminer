
To run once  
-----------

The prototype script used here as a test template for the latest version of discminer is based on the disc of MWC 480. The parameters provided in the accompanying '.txt' file were obtained from previous modelling of this object with discminer. If you wish to use the script as is, you must first run the following lines to download and prepare (clip and downsample) the data cube:

.. code-block:: bash

   ./download-MAPS.sh #Download MWC480 12CO data from MAPS website
   python prepare_data.py 

Next, the prototype script can be run simply as,

.. code-block:: bash

   python prototype_mwc480_12co.py

If needed, it can also be executed in an IPython terminal to use, or simply inspect, the contents of the variables and methods defined in the script,

.. code-block:: bash

   ipython
   run prototype_mwc480_12co.py
