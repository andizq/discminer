from discminer.core import Data
import astropy.units as u

file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor'
dpc = 162.0*u.pc

#**********************
#DATACUBE FOR PROTOTYPE
#**********************
datacube = Data(file_data+'.fits', dpc)

datacube.clip(npix=250,  overwrite=True) # can also clip along the velocity axis using e.g. channels={"interval": [15, 115]})
datacube.downsample(2, tag='_2pix') # Downsample cube and add tag to filename

#**********************
#DATACUBE FOR MCMC FIT
#**********************
datacube = Data(file_data+'_clipped.fits', dpc)
datacube.downsample(10, tag='_10pix')
