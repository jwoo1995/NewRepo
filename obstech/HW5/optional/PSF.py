import numpy as np
import matplotlib.pyplot as plt
# ================================= 01. filter data ===============================
filter_dat = np.loadtxt('ACS_WFC_606.dat')
filter_wv = filter_dat[:,0]
filter_transmit = filter_dat[:,1]

# ================================= 02. sed data ===============================
Vega_dat = np.loadtxt('Vega.sed',skiprows=8)
Vega_wv = Vega_dat[:,0]
Vega_flux = Vega_dat[:,1]

ind = np.where((np.min(Vega_wv) <= filter_wv) & (np.max(Vega_wv) >= filter_wv))
filter_wv = filter_wv[ind]
filter_transmit = filter_transmit[ind]

# ================================= 03. interpolation (Vega sed) ===============================
from scipy.interpolate import interp1d  # linear interpolation
f = interp1d(Vega_wv, Vega_flux)
x_new = filter_wv
f_new = f(filter_wv)

# ================================= 04. weight ===============================
multiplication = f_new * filter_transmit
weight = multiplication/np.sum(multiplication)

# =========================== 05. pupil data loading ===============================
from astropy.io import fits
hdulist = fits.open('hst_pupil.fits')
image = hdulist[0].data

# ============================ 06. FFT -> PSF ====================================
U = np.fft.ifftshift(np.fft.ifft2(image))
I = abs(U)**2
ind = np.where(image > 0)
print('ind',ind)
x = np.linspace(0,999,1000,endpoint=True)
center = (x[0] + x[-1])/2

xv, yv = np.meshgrid(x,x)
xv = xv - center
yv = yv - center

xv = xv[ind]
yv = yv[ind]

diameter = np.max(np.sqrt(xv **2 + yv**2) * 2)
print('diameter',diameter)
# ===================================== 03. slicing ========================================
J_y = I[500:501,:]
J_y = np.ravel(J_y)
J_x = np.linspace(1,1000,1000,endpoint=True)

# ===================================== 04. interpolation ========================================
distance = 1.9444

# ===================================== 05. pixel scale ========================================
F_ratio = 25
wavelength = x_new * 10 ** -10
diameter_of_primary = 2.4
n = 1000 / diameter
print('n',n)


# distance = 1

ps = F_ratio * wavelength / n
pl = np.sum(distance * ps * weight) # physical length
print('pl',pl)
focal_length = F_ratio * diameter_of_primary
theta_rad = pl / focal_length
print('rad',theta_rad)
theta_arcsec = theta_rad * 180 / np.pi * 3600
theta_FWHM = theta_arcsec

final_FWHM = theta_FWHM
print('final',final_FWHM)

