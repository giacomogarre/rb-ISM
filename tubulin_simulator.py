import numpy as np
import matplotlib.pyplot as plt
import scipy
import SimulatorTubulin as st
import pyISM as cPs
import ISM_processing.APR_lib as apr
import ISM_processing.FRC_lib as frc
import pandas as pd
import os
import napari




param=pd.read_csv('/Users/ggarre/Documents/GitHub/volumetric_ism/parameters.csv')


exwavelength = float(param['exwavelength'])
emwavelength = float( param['emwavelength'])
NA =  float(param['NA'])
M = float( param['M'])
n =  float(param['n'])
AU = float( param['AU'])
DOF = float( param['DOF'])
pxpitch = float( param['pxpitch'])
pxdim = float( param['pxdim'])
N = int( param['N'])
Nx = int( param['Nx'])
pxsizex =float(  param['pxsizex'])
Nz = int( param['Nz'])
pxsizez =float( param['pxsizez'])




#simulazione di tubulina con filamenti intra-piano assiale
# tubulin = st.tubSettings()
# tubulin.xy_pixel_size = pxsizex
# tubulin.xy_dimension = Nx
# tubulin.xz_dimension = Nz
# tubulin.z_pixel = 1 # pxsizez     
# tubulin.n_filament = 8
# tubulin.radius_filament = 20
# tubulin.intensity_filament = [1, 1]  
# phTub = st.functionPhTub(tubulin)
# phTub = np.swapaxes(phTub, 2, 0)



#simulazione di tubulina con un solo filamento per piano
tubulin_planar = st.tubSettings()
tubulin_planar.xy_pixel_size = pxsizex
tubulin_planar.xy_dimension = Nx
tubulin_planar.xz_dimension = 1
tubulin_planar.z_pixel =  1   
tubulin_planar.n_filament = 3
tubulin_planar.radius_filament = 35
tubulin_planar.intensity_filament = [1, 1]
 

phTub=np.zeros([Nz,Nx,Nx])

for i in range(Nz):
        phTub_planar= st.functionPhTub(tubulin_planar)
        phTub_planar = np.swapaxes(phTub_planar, 2, 0)
        phTub[i,:,:] = phTub_planar*(np.power(3,np.abs(i-2)))
        


 
plt.figure()
plt.imshow( np.sum(phTub,axis=0) ) 


viewer = napari.view_image(phTub)
napari.run()



np.save('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum1',phTub)