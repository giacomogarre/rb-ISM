import numpy as np
import matplotlib.pyplot as plt
import scipy as s
from scipy import signal
import pyISM as cPs
import ISM_processing.APR_lib as apr
import ISM_processing.FRC_lib as frc
import ISM_processing.FocusISM_lib as fISM
import pandas as pd
import math
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rbISM_lib as vISM


plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

# %%

param = pd.read_csv(
    '/Users/ggarre/Documents/GitHub/volumetric_ism/parameters.csv')


exwavelength = float(param['exwavelength'])
emwavelength = float(param['emwavelength'])
NA = float(param['NA'])
M = float(param['M'])
n = float(param['n'])
AU = float(param['AU'])
DOF = float(param['DOF'])
pxpitch = float(param['pxpitch'])
pxdim = float(param['pxdim'])
N = int(param['N'])
Nx = int(param['Nx'])
pxsizex = float(param['pxsizex'])
Nz = int(param['Nz'])
pxsizez = float(param['pxsizez'])


# %% importing saved PSFs and tubuline phantom

Psf3 = np.load('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum2.npy')
detPSF3 = np.load(
    '/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum3.npy')
exPSF3 = np.load(
    '/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum4.npy')


phTub = np.load('/Users/ggarre/Documents/GitHub/volumetric_ism/Alg_volum1.npy')




Signal = 900
SNR = np.sqrt(Signal)



# %%commputong the ISM data through 2d conv and then summing up on level to see if the results is the same as computing ISM
# dataset through 3d convolution
# In this case complete ISM dataset is the whole fin
ground_truth_noise = np.zeros(Psf3.shape)
data_ISM_noise = np.empty([Nx, Nx, N**2])

ground_truth = np.zeros(Psf3.shape)
data_ISM = np.zeros([Nx, Nx, N**2])

for i in range(N**2):
    for j in range(Nz):
        ground_truth[j, :, :, i] = signal.convolve(
            Psf3[j, :, :, i], phTub[j, :, :], mode='same', method='fft')
        data_ISM[:, :, i] += ground_truth[j, :, :, i]


ground_truth_noise = np.random.poisson(lam=np.abs(
    (Signal)*ground_truth/np.max(ground_truth)), size=None)
data_ISM_noise = np.sum(ground_truth_noise, axis=0)



# %%
# shift PSF3 and collect shift vector calculated on PSFs
usf = 20
ref = N**2//2

Psf3_s = np.zeros(Psf3.shape)
shift_vectors_psf = np.empty([N**2, 2, Nz])
shift_x_psf = np.empty([N**2, Nz])
shift_y_psf = np.empty([N**2, Nz])

for i in range(Nz):
    shift_vectors_psf[:, :, i], Psf3_s[i, :, :, :] = apr.APR(
        Psf3[i, :, :, :], usf, ref, pxsize=pxsizex, cutoff=None, apodize=None, degree=None)
    shift_x_psf[:, i], shift_y_psf[:, i] = - \
        shift_vectors_psf[:, 1, i], -shift_vectors_psf[:, 0, i]

# %% calcuating fingerprints and normalizing


fingerprints_matrix = np.zeros([N*N, Nz])
fingerprint = cPs.Fingerprint(Psf3, volumetric=True)


for i in range(Nz):
    fingerprint[i, :, :] = fingerprint[i, :, :]/np.sum(fingerprint[i, :, :])

for i in range(Nz):
    fingerprints_matrix[:, i] = np.reshape(fingerprint[i, :, :], N**2)

for i in range(Nz):
    fingerprints_matrix[:, i] = fingerprints_matrix[:,i] / np.sum(fingerprints_matrix[:, i])

# %%
# APR on ground truth images

g_t = np.zeros(ground_truth.shape)
g_t_noise = np.zeros(ground_truth.shape)

shift_vectors_imm = np.empty([N**2, 2, Nz])

shift_vectors_psf = np.empty([N**2, 2, Nz])
shift_x_imm = np.empty([N**2, Nz])
shift_y_imm = np.empty([N**2, Nz])

for i in range(Nz):
    shift_vectors_imm[:, :, i], tempo = apr.APR(
        ground_truth[i, :, :, :], usf, ref, pxsize=pxsizex, cutoff=None, apodize=None, degree=None)
    tempo_noise = apr.APR(ground_truth_noise[i, :, :, :], usf, ref,
                          pxsize=pxsizex, cutoff=None, apodize=None, degree=None)[1]
    g_t[i, :, :, :] = Signal*tempo/np.max(tempo)
    shift_x_imm[:, i], shift_y_imm[:, i] = - \
        shift_vectors_imm[:, 1, i], -shift_vectors_imm[:, 0, i]
    g_t_noise[i, :, :, :] = Signal*tempo_noise/np.max(tempo_noise)


# %% algorithm and associated plots
b = np.finfo(float).eps
itera = 60


data_ISM_noise_shift = apr.APR(
    data_ISM_noise, usf, ref, pxsize=pxsizex, cutoff=None, apodize=None, degree=None)[1]

I = data_ISM_noise_shift
I[I < 0] = 0


stop,pr,ricos= vISM.rb_ISM_pr_regu(I, Nz, Nx, N, itera, fingerprints_matrix,0,regularization='off')




# %% plot dataset and results


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
datass = ax.imshow(np.sum(data_ISM_noise_shift, axis=-1), cmap='inferno')
plt.axis('off')
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(datass, cax=cax)
ax.text(1.45, 0.1, r'Intensity [photons] ', rotation=90, transform=ax.transAxes)
scalebar = ScaleBar(
    pxsizex, "nm",  # default, extent is calibrated in meters
    box_alpha=0,
    color='w',
    length_fraction=0.25)
ax.add_artist(scalebar)
plt.savefig("data_fin_noabe_5x5_adjusted_fin.pdf", format="pdf", bbox_inches="tight")

# fingerprints plot
# ground truth plot

fig = plt.figure(figsize=(15, 15))
plt.title('ground truth and reconstructions')
plt.axis('off')
for i in range(Nz):
    ax = fig.add_subplot(3, Nz, i+1)
    datas = ax.imshow(np.sum(g_t_noise[i, :, :, :]/3, axis=2), cmap='inferno')
    plt.axis('off')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(datas, cax=cax)
    ax.text(1.45, 0.1, r'Intensity [photons] ',
            rotation=90, transform=ax.transAxes)
    scalebar = ScaleBar(
        pxsizex, "nm",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)
    ax.add_artist(scalebar)

    ax = fig.add_subplot(3, Nz, i+Nz+1)
    datass = ax.imshow(ricos[i, :, :], cmap='inferno')
    plt.axis('off')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(datass, cax=cax)
    ax.text(1.45, 0.1, r'Intensity [photons] ',rotation=90, transform=ax.transAxes)
    scalebar = ScaleBar(
        pxsizex, "nm",  # default, extent is calibrated in meters
        box_alpha=0,
        color='w',
        length_fraction=0.25)
    ax.add_artist(scalebar)

    ax = fig.add_subplot(3, Nz, i+1+Nz+Nz)
    datasss = ax.imshow(fingerprint[i, :, :], cmap='inferno')
    plt.axis('off')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(datasss, cax=cax)
    ax.text(1.45, 0.1, r'Intensity [photons] ',rotation=90, transform=ax.transAxes)


plt.savefig("Cells_images_fin_noabe_5x5_regularizeed_adjusted_fin.pdf", format="pdf", bbox_inches="tight")

