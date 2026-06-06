Discminer examples
==================

This directory collects examples that illustrate different ways of using
``discminer`` on observational and synthetic datacubes. Each example is kept in
its own folder and includes dedicated instructions with the data preparation
steps, required inputs, and suggested analysis commands.

The examples are intended to be copied, modified, and adapted to new sources or
molecular tracers.

Available examples
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Folder
     - Description
   * - ``mwc480_12co``
     - Basic analysis workflow using the MAPS 12CO J=2-1 datacube of MWC 480.

Discminer commands
------------------

The full list of available command-line tools can be shown with:

.. code-block:: bash

   discminer -h

At the time this example set was written, the main commands include:

.. code-block:: text

    parfile             Make JSON parameter file based on input log_pars.txt and prepare_data.py
    channels            Make model channel maps and compare to data
    moments1d           Make (gaussian, bell, or quadratic) moment maps and save output into .fits files
    moments2d           Make (double Gauss or double Bell) moment maps and save output into .fits files
    prepare             Generate prepare_data.py + preview of suggested spatial and spectral clipping (Beta)
    parcube             Show cube reconstructed from fit parameters vs data cube
    channels+peakint    Show Data vs Model channel maps, peak intensities, and residuals
    attributes          Show model attributes (z, v, I, Lw) as a function of radius
    radprof             Extract and show radial profiles from moment maps AND residuals
    radprof+wedge       Extract and show radial profiles from moment residuals within specific wedges
    azimprof            Extract and show azimuthal profiles from moment maps OR residuals
    spectra             Extract and show line profiles along a specific annulus, every 30 deg
    mirrorspectra       Extract and show line profiles around a pixel and from the mirror location on the other side of the disc
    moment+residuals    Show Data vs Model moment map and residuals
    moment+offset       Show moment map and a zoom-in illustrating offset from the centre
    residuals+deproj    Show residuals from a moment map, deprojected onto the disc reference frame
    residuals+all       Show ALL moment map residuals, deprojected onto the sky OR disc reference frame
    pick                Use Pick tools. Fold residual maps and identify peak and clustered residuals
    gradient            Show peak, radial AND/OR azimuthal gradient from residual maps
    isovelocities       Show Data vs Model isovelocity contours
    pv                  Show PV diagram extracted along a specific axis
    skewkurt            Make skewness and kurtosis maps, and save output into .fits files
    intensdistrib       Extract and display the intensity distribution of pixels within selected radial regions
    stack               Azimuthally stack line profiles across the disc radial extent
    stackcube           Make stacked cube with lines shifted to their centroid velocity
    destackcube         Undo stacked cube with lines shifted to their original centroid velocity

Each example explores a subset of these commands in a specific science
case. For command-specific options, run e.g.:

.. code-block:: bash

   discminer channels -h
   discminer radprof -h
