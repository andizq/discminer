# Mining scripts

Attempt to automatically produce analysis plots for all discs based on log_pars.txt and prepare_data.py files.

## How to use

```bash
cd discminer_analysis 
cd mwc480_12co # --> Get in the folder of the disc you wish to analyse
python prepare_data.py # --> Make sure you have the reduced datacube in the folder

#MANDATORY
python ../_mining/make_parfile.py # --> Make parfile.json for the current disc 
python ../_mining/make_channels.py # --> Make model cube and plot channels
python ../_mining/make_single_moments.py -k gaussian # --> Make gaussian or bell moment maps

#OR
python ../_mining/make_double_moments.py -k doublebell # --> Make doublegaussian or doublebell moment maps
python ../_mining/show_parcube.py -k doublebell # --> Check goodness of channels reconstructed with doublebell profiles

#PRETTIER CHANNELS
python ../_mining/plot_channel+residuals.py # --> Data vs Model channels + residuals
python ../_mining/plot_channel+peakint.py # --> As above but first column shows peakintensity map for reference

#LINE PROFILES
python ../_mining/plot_lineprofiles.py -m velocity -r 200 # --> Plot spectra, every 30 deg, at R=200 au and show velocity map in the central panel

#OPTIONAL DIAGNOSTIC
python ../_mining/plot_attributes_model.py # --> Plot model vertical structure, velocity and intensity attributes 

#1D MAPS
python ../_mining/plot_azimuthal_profiles.py -t model -m peakint # --> Plot azimuthal profiles for [model, data or residuals] from moments

python ../_mining/plot_radial_profiles.py -m velocity -w 1 # --> Plot velocity components and rotation curve, and write profiles into .txt files
python ../_mining/plot_radial_profiles.py -m peakint # --> Plot radial profile of [peakint or linewidth]

#2D MAPS
python ../_mining/plot_moment+offset.py -m velocity # --> Plot moment map+offset [velocity, linewidth, peakint]
python ../_mining/plot_moment+residuals.py -m velocity # --> Plot moment map+residuals [velocity, linewidth, peakint]

python ../_mining/plot_residuals+all.py -c disc # --> Summary of three types of residuals in 'disc' or 'sky' coordinates
python ../_mining/plot_residuals+deproj.py -p cartesian -m velocity # --> 'cartesian' or 'polar' deprojection of [velocity, linewidth or peakint] residuals

python ../_mining/plot_velocity_components.py # --> Visualise 3D velocity structure in a pie map + radial gradient of vphi (pressure modulations)

python ../_mining/plot_gradient.py -g r -m velocity # --> Compute radial, azimuthal or peak [r, phi, peak] gradient of [velocity, linewidth or peakint] residuals

python ../_mining/plot_residuals+filaments.py -m peakint # --> Run FilFinder and fit spirals to the retrieved [velocity, linewidth, peakint] substructures

python ../_mining/plot_peak_residuals.py -m velocity # --> Extract peak [velocity, linewidth, peakint] residuals and identify localised perturbations using clustering algorithm

#3D MAPS
python ../_mining/plot_moment+3d.py -t residuals -m peakint # --> Visualise [velocity, linewidth, peakint] map of residuals/data/model shown on disc surface in a 3D canvas 
```

## Help

To control various aspects of the analysis most `_mining` scripts support input of multiple command line arguments. Use the `-h` flag to view argument summaries for each script, short descriptions, and default values. For example,


```bash
python ../_mining/plot_radial_profiles.py -h
```

produces the following output,

```
usage: plot radial profiles [-h] [-m {velocity,linewidth,lineslope,peakint,peakintensity}] [-b MASK_MINOR] [-a MASK_MAJOR]
                            [-k {gauss,gaussian,bell,dgauss,doublegaussian,dbell,doublebell}] [-ki {mask,sum}] [-s {up,upper,low,lower}] [-w {0,1}] [-i RINNER]
                            [-o ROUTER] [-si SIGMA]

Compute radial profiles from moment maps and residuals.

optional arguments:
  -h, --help            show this help message and exit
  -m {velocity,linewidth,lineslope,peakint,peakintensity}, --moment {velocity,linewidth,lineslope,peakint,peakintensity}
                        Type of moment map to be analysed. DEFAULTS to 'velocity'
  -b MASK_MINOR, --mask_minor MASK_MINOR
                        +- azimuthal mask around disc minor axis for computation of vphi and vz velocity components. DEFAULTS to 30.0 deg
  -a MASK_MAJOR, --mask_major MASK_MAJOR
                        +- azimuthal mask around disc major axis for computation of vR velocity component. DEFAULTS to 30.0 deg
  -k {gauss,gaussian,bell,dgauss,doublegaussian,dbell,doublebell}, --kernel {gauss,gaussian,bell,dgauss,doublegaussian,dbell,doublebell}
                        Kernel utilised for line profile fit and computation of moment maps. DEFAULTS to 'gaussian'
  -ki {mask,sum}, --kind {mask,sum}
                        How the upper and lower surface kernel profiles must be merged. DEFAULTS to 'mask'
  -s {up,upper,low,lower}, --surface {up,upper,low,lower}
                        Use upper or lower surface moment map. DEFAULTS to 'upper'
  -w {0,1}, --writetxt {0,1}
                        write output into txt file(s). DEFAULTS to 1
  -i RINNER, --Rinner RINNER
                        Number of beams to mask out from disc inner region. DEFAULTS to 1.00
  -o ROUTER, --Router ROUTER
                        Fraction of Rout to consider as the disc outer radius for the analysis. DEFAULTS to 0.98
  -si SIGMA, --sigma SIGMA
                        Mask out pixels with values below sigma threshold. DEFAULTS to 5.0
```



