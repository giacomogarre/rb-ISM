import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import pyISM as cPs
import pandas as pd
import os


# optical settings
# exwavelength = 635 # nm as focus ISM par
# emwavelength = 655 # nm as focus ISM par

exwavelength = 640 # nm as focus ISM par
emwavelength = 680

NA = 1.4
M = 500# overall magnification of the system (e.g. 100x obj + 5x extra magnification)

n = 1.5

AU = 1.22*emwavelength/NA
DOF = 2*emwavelength*n/NA**2

# SPAD detector settings
pxpitch = 75e3 # nm - spad array pixel pitch (real space)
pxdim = 50e3 # nm - spad pixel size (real space) 57.3e-3 for cooled spad
N = 5 # number of pixels in the detector in each dimension (5x5 typically)

# grid settings
Nx = 101
pxsizex = 50 # nm
Nz =2
pxsizez =DOF/2#nm





parameters={'exwavelength' : exwavelength ,'emwavelength' : emwavelength , 'NA' : NA, 'M' : M ,'n' : n, 'AU' : AU,
            'DOF' : DOF, 'pxpitch' : pxpitch ,'pxdim' : pxdim,'N' :N,'Nx' :Nx,
            'pxsizex' : pxsizex , 'Nz' : Nz, 'pxsizez' :pxsizez}






set_param=pd.DataFrame(data=parameters,index=[0])
os.makedirs('/Users/ggarre/Documents/GitHub/volumetric_ism',exist_ok=True)

set_param.to_csv('/Users/ggarre/Documents/GitHub/volumetric_ism/parameters.csv')



exOps = cPs.simSettings()
exOps.na = NA   # numerical aperture
exOps.wl = exwavelength   # wavelength [nm]
exOps.gamma = 45 # parameter describing the light polarization
exOps.beta = 90  # parameter describing the light polarization
exOps.n = n
exOps.abe_index=1
exOps.abe_ampli=1
exOps.mask='Zernike'
exOps.mask_sampl=100


emOps = cPs.simSettings()
emOps.na = NA   # numerical aperture
emOps.wl = emwavelength   # wavelength [nm]
emOps.gamma = 45 # parameter describing the light polarization
emOps.beta = 90  # parameter describing the light polarization
emOps.n = n
emOps.abe_index=1
emOps.abe_ampli=1
emOps.mask='Zernike'
emOps.mask_sampl=100


[Psf3, detPSF3, exPSF3]=cPs.SPAD_PSF_3D(N, Nx, pxpitch, pxdim, pxsizex, M, exOps, emOps,Nz,pxsizez)




np.save('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum2',Psf3)
np.save('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum3',detPSF3)
np.save('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum4',exPSF3)
