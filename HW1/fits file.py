from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
import time
"============================== 01. DATA 가져 오기 ============================="
hdulist = fits.open('Moon.fits')

image = hdulist[0].data
# print(image.shape)

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
# print('sum of Prob xy',np.sum(Prob_xy))

x = np.linspace(1,image.shape[1],image.shape[1])
y = np.linspace(1,image.shape[0],image.shape[0])
xv, yv = np.meshgrid(x,y)

xv = xv.ravel()  # x coordinate
yv = yv.ravel()  # y coordinate
Prob_xy = Prob_xy.ravel()  # Prob function
# print('제거전',len(Prob_xy))
moon_ind = np.where(Prob_xy > 2/np.sum(image))
xv = xv[moon_ind]  # x coordinate
yv = yv[moon_ind]  # y coordinate
Prob_xy = Prob_xy[moon_ind]  # Prob function

# print('제거후',len(Prob_xy))
# print('Prob_max',np.max(Prob_xy))

"======================== 04. the number of photons from V=-13================"
# V -> pinhole에 분당 도달하는 photon 수만큼 Prob_xy vs z -> dx',dy' 중에 random하게
# 한 개 선택 -> x', y' 결정
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
d = 0.01
num_pho = number_of_phtons(d)
x_list = np.zeros(0)
y_list = np.zeros(0)

while (True):
  z= np.random.uniform(low=0.0,high=np.max(Prob_xy),size=len(xv))
  ind_rejection_method = np.where(z<Prob_xy)
  print(ind_rejection_method)
  print(ind_rejection_method[0])
  ind_rejection_method = np.random.choice(ind_rejection_method[0],int(len(ind_rejection_method[0])/2),replace=True)
  print(ind_rejection_method)
  # print(len(ind_rejection_method[0]))
  x_new = xv[ind_rejection_method]
  y_new = yv[ind_rejection_method]
  x_list = np.append(x_list,x_new)
  y_list = np.append(y_list,y_new)
  print(len(x_list)/num_pho*100)
  if len(x_list) > num_pho:
    break
x_list = x_list[0:num_pho]
y_list = y_list[0:num_pho]


dx_list = np.zeros(0)
dy_list = np.zeros(0)
while (True):
  dx_new = np.random.uniform(-d/2,d/2,size=len(x_list))
  dy_new = np.random.uniform(-d/2,d/2,size=len(x_list))
  d_new = dx_new**2 + dy_new**2
  d_inx = np.where(d_new<(d/2)**2)
  dx_new = dx_new[d_inx]
  dy_new = dy_new[d_inx]
  dx_list = np.append(dx_list,dx_new)
  dy_list = np.append(dy_list,dy_new)
  if len(dx_list)>num_pho:
    break
dx_list = dx_list[0:num_pho]
dy_list = dy_list[0:num_pho]

"============================== 06. x',y' on CCD ============================="
x_center = np.median(x)
y_center = np.median(y)

x_list = x_list[0:num_pho] - x_center # XY coordinate
y_list = y_list[0:num_pho] - y_center

distance_per_one_pixel = 3474/diameter_pixel * u.km
x_list = (x_list * distance_per_one_pixel).to(u.m)
y_list = (y_list * distance_per_one_pixel).to(u.m)
# print(np.max(x_list))
# print(np.min(x_list))

dx_list = dx_list * 10**-3 * u.m
dy_list = dy_list * 10**-3 * u.m

D1 = (384400 * u.km).to(u.m) # from moon to the hole
D2 = 0.5 * u.m # from the hole to the CCD

x_Prime = (1+D2/D1) * dx_list - x_list * D2/D1 # physical distance
y_Prime = (1+D2/D1) * dy_list - y_list * D2/D1 # physical distance

length_of_side = 10**-5 * u.m
center_CCD = (0+999)/2

x_CCD = np.round(-x_Prime/length_of_side+center_CCD)
y_CCD = np.round(-y_Prime/length_of_side+center_CCD)

my_dict = {"x_CCD": x_CCD, "y_CCD": y_CCD}
df = pd.DataFrame(my_dict, columns=['x_CCD','y_CCD'])
new_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'value'})
#  print(new_df)

"============================== 07. make image of moon ============================="
x_df = new_df[['x_CCD']]
y_df = new_df[['y_CCD']]
value_df = new_df[['value']]

z1 = np.zeros((1000,1000))
new_x_df = x_df.to_numpy(dtype=np.int32)
new_x_df = new_x_df.tolist()
new_y_df = y_df.to_numpy(dtype=np.int32)
new_y_df = new_y_df.tolist()

#print(new_xy_df)
#print('index shape',new_xy_df.shape)
new_value_df = value_df.to_numpy()
new_value_df = np.ravel(new_value_df)
new_value_df = new_value_df.tolist()

print(np.max(new_value_df))
for i in range(0,len(value_df)):
  x_coordinate = new_x_df[i]
  y_coordinate = new_y_df[i]
  z1_value = new_value_df[i]
  z1[[tuple(x_coordinate),tuple(y_coordinate)]] = z1_value

print(np.max(z1))
plt.imshow(z1.T,cmap='gray',vmin=0,vmax=np.max(z1),origin='lower')

plt.show()

