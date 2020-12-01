print(3474/384400*0.5/10**-2)

import numpy as np

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
# import time
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
Prob_xy = Prob_xy[moon_ind]  # Prob function

"======================== 04. the number of photons from V=-13================"
V_filter = np.loadtxt('Transmission of V filter.txt') # V filter data
wavelength = V_filter[:,0] * 10**-10 * u.m
Transmission = V_filter[:,1]

f_zp = 3.64 * 10 ** -23 * u.W * u.meter ** -2 * u.Hz ** -1  # zero point frequency spectral flux density W m^-2 s^-1 Hz^-1
V = -13  # apparent magnitude
f_nu = 10 ** (0.4 * -V) * f_zp # spectral density of Moon
exposure_time =60 * u.second
h = 6.63 * 10**-34 * u.meter **2 * u.kg * u.second**-1   # planck constant
c = 3 * 10**8 * u.meter/u.second  # velocity of light
f_lambda = c * f_nu / wavelength**2  # wavelength spectral flux density W m^-2 s^-1 Hz^-1

def number_of_phtons(d):  # d:diameter of pinhole
  radius = d/2 * 10**-3 * u.meter
  y_value = f_lambda * wavelength * Transmission
  N_ph = (np.trapz(y_value, x=wavelength)/ (h*c)).to(u.m ** -2 * u.s **-1)  # number of photons per second per square meter
  return int(np.round(N_ph * exposure_time * np.pi * radius**2,0))  # number of accepted photons during exposure time

"============================== 05. generate random number z and choose xy pair ============================="
d = 0.1
num_pho = number_of_phtons(d)
x_list = np.zeros(0)
y_list = np.zeros(0)

z= np.random.uniform(low=0.0,high=np.max(Prob_xy),size=num_pho)
print('ok')
print(np.append(np.zeros(0),2))


print(np.round(1.3),np.round(1.7),np.round(-1.3),np.round(-1.7))
# print(np.max(dx_list))


# print(num_pho)
# print('x_list',len(x_list))
# print('y_list',len(y_list))
# print('dx_list',len(dx_list))
# print('dy_list',len(dy_list))
# check = dx_list**2 + dy_list**2
# print(len(np.where(check<(d/2)**2)[0]))

a = np.array([1,3,2])
a = a * 2
print('repeat',a)
b = a.argsort()

print(b)
print(a[b])


a = np.array([[0,0,0],[0,0,0]])
cccc = np.array([[1,1],[1,2]])
cccc = cccc.tolist()
print(a)

a[[1,1],[1,2]]=np.array([1,10])
da=np.array([1,10])
print(a)
print(da.shape())
c = np.where(a == np.array([[1],[2]]))
print(np.array([[1],[2]]))
print(c)
print(a[c])

d = np.array([1,2,2,2,3,4])
u, indices = np.unique(d, return_index=True)
print('unique',u,indices)

x = np.array([1,1,1,2,2,2,2,2])
y = np.array([1,2,2,1,1,1,2,2])

x_ind = np.where(x==2)
y_ind = np.where(y==2)
print('ind',x_ind,y_ind)

x_ind[0]
# np.where()

