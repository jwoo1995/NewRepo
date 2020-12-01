from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
import time
"============================== 01. DATA 가져 오기 ============================="
hdulist = fits.open('Moon.fits')
image = hdulist[0].data

x = np.linspace(1,image.shape[1],image.shape[1])
y = np.linspace(1,image.shape[0],image.shape[0])

"==================== 02. 달에 반지름에 해당하는 pixel 수 구하기 ==================="
x_median = int(np.round(np.median(x)))
y_median = int(np.round(np.median(y)))
slice = image[:,x_median-1:x_median]
ind = slice > 2
new_slice = slice[ind]
diameter_pixel = new_slice.shape[0] # the number of pixels = diameter of moon

"===================== 03. X,Y PAIR coordinate AND PROB FUNC ====================="
Prob_xy = image/np.sum(image)
x = np.linspace(1,image.shape[1],image.shape[1])
y = np.linspace(1,image.shape[0],image.shape[0])
xv, yv = np.meshgrid(x,y)

xv = xv.ravel()  # x coordinate
yv = yv.ravel()  # y coordinate
Prob_xy = Prob_xy.ravel()  # Prob function
moon_ind = np.where(Prob_xy > 2/np.sum(image))
xv = xv[moon_ind]  # x coordinate
yv = yv[moon_ind]  # y coordinate


noise = np.random.normal(0,1,len(xv))
xv = xv + noise
yv = yv + noise
Prob_xy = Prob_xy[moon_ind]  # Prob function

"======================== 04. the number of photons from V=-13================"
# V -> pinhole에 분당 도달하는 photon 수만큼 Prob_xy vs z -> dx',dy' 중에 random하게
# 한 개 선택 -> x', y' 결정
V_filter = np.loadtxt('Transmission of V filter.txt') # V filter data
wavelength = V_filter[:,0] * 10**-10 * u.m
Transmission = V_filter[:,1]

f_zp = 3.64 * 10 ** -23 * u.W * u.meter ** -2 * u.Hz ** -1  # zero point frequency spectral flux density W m^-2 s^-1 Hz^-1
V = -13  # apparent magnitude
f_nu = 10 ** (0.4 * -V) * f_zp # spectral density of Moon
exposure_time =0.001 * u.second
h = 6.63 * 10**-34 * u.meter **2 * u.kg * u.second**-1   # planck constant
c = 3 * 10**8 * u.meter/u.second  # velocity of light
f_lambda = c * f_nu / wavelength**2  # wavelength spectral flux density W m^-2 s^-1 Hz^-1

def number_of_phtons(d):  # d:diameter of pinhole
  radius = d/2 * 10**-3 * u.meter
  y_value = f_lambda * wavelength * Transmission
  N_ph = (np.trapz(y_value, x=wavelength)/ (h*c)).to(u.m ** -2 * u.s **-1)  # number of photons per second per square meter
  return int(np.round(N_ph * exposure_time * np.pi * radius**2,0))  # number of accepted photons during exposure time

"============================== 05. generate random number z and choose xy pair ============================="
d = 1 # 1mm, 2mm, 10mm
num_pho = number_of_phtons(d)
x_list = np.zeros(0)
y_list = np.zeros(0)

while (True):
  z= np.random.uniform(low=0.0,high=np.max(Prob_xy),size=len(xv))
  ind_rejection_method = np.where(z<Prob_xy)
  ind_rejection_method = np.random.choice(ind_rejection_method[0],int(len(ind_rejection_method[0])/5),replace=False)
  x_new = xv[ind_rejection_method]
  y_new = yv[ind_rejection_method]
  x_list = np.append(x_list,x_new)
  y_list = np.append(y_list,y_new)
  if len(x_list) > num_pho:
    break
x_list = x_list[0:num_pho]
y_list = y_list[0:num_pho]

"============================== 06. x',y' on CCD ============================="
x_center = np.median(x)
y_center = np.median(y)

x_list = x_list[0:num_pho] - x_center # XY coordinate
y_list = y_list[0:num_pho] - y_center

distance_per_one_pixel = 3474/diameter_pixel * u.km
x_list = (x_list * distance_per_one_pixel).to(u.m)
y_list = (y_list * distance_per_one_pixel).to(u.m)

f = 0.5 # 0.5m, 0.8m, 1m
D1 = (384400 * u.km).to(u.m) # from moon to the hole
D2 = f * u.m # from the hole to the CCD


x_Prime = - (x_list * D2/D1) # physical distance
y_Prime = - (y_list * D2/D1) # physical distance

length_of_side = 10**-5 * u.m
center_CCD = (0+999)/2

x_CCD = -x_Prime/length_of_side+center_CCD
y_CCD = -y_Prime/length_of_side+center_CCD
print('x_CCD',x_CCD[0:100])
print(len(x_CCD),len(y_CCD))

xedges = np.arange(1000)
yedges = np.arange(1000)

H, xedges, yedges = np.histogram2d(x_CCD,y_CCD,bins=(xedges,yedges))
H = H.T
H = H

plt.imshow(H,origin='low',cmap='gray',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],vmin=0,vmax=np.max(H))
cmap = plt.colorbar()
cmap.set_label('the number of photons in each pixel', labelpad= 10 ,rotation=270)
plt.tight_layout()
plt.show()
