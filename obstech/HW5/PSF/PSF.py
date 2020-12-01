from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u
import pandas as pd
# ===================================== 01. load file ========================================
hdulist = fits.open('hst_pupil.fits')
image = hdulist[0].data

# ===================================== 02. PSF ========================================
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
J_x = np.linspace(0,999,1000,endpoint=True)

# ========================= 04. interpolation ===========================
Max = np.max(J_y)

from scipy.interpolate import CubicSpline
cs = CubicSpline(J_x,J_y)
xs = np.linspace(0, 999, 90000,endpoint=True)

from scipy.interpolate import interp1d
cubic = cs(xs)
ind = np.where(abs(Max/2 - cubic) < 10 ** -4)
print(ind)
distance = np.mean(abs(len(xs)/2 - ind[0])) / 90

# ======================= 05. pixel scale ==============================
F_ratio = 25
wavelength = 600 * 10**-9
diameter_of_primary = 2.4
n = 1000 / diameter
print('n',n)


# distance = 1

ps = F_ratio * wavelength / n
print('ps',ps)
pl = distance * ps  # physical length
print('pl', 2* pl)
focal_length = F_ratio * diameter_of_primary
theta_rad = pl / focal_length
print('rad', 2 * theta_rad)
theta_arcsec = theta_rad * 180 / np.pi * 3600
FWHM = 2*theta_arcsec
print(FWHM)