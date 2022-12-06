from discminer.core import Data

file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor'

#**********************
#DATACUBE FOR PROTOTYPE
#**********************
datacube = Data(file_data+'.fits')
datacube.clip(npix=250,  overwrite=True) #, channels={"interval": [15, 115]})
datacube.downsample(2, tag='_2pix') # Downsample cube and add tag at the end of filename

#**********************
#DATACUBE FOR MCMC FIT
#**********************
datacube = Data(file_data+'_clipped.fits')
datacube.downsample(10, tag='_10pix')


