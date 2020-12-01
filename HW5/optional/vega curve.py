import numpy as np
import matplotlib.pyplot as plt
# ================================= 01. filter data ===============================
filter_dat = np.loadtxt('ACS_WFC_606.dat')
filter_wv = filter_dat[:,0]
filter_transmit = filter_dat[:,1]
filter_weight = filter_transmit/np.sum(filter_transmit)

# ================================= 02. sed data ===============================
Vega_dat = np.loadtxt('Vega.sed',skiprows=8)
Vega_wv = Vega_dat[:,0]
Vega_flux = Vega_dat[:,1]

ind = np.where((np.min(filter_wv) <= Vega_wv) & (np.max(filter_wv) >= Vega_wv))
Vega_wv =  Vega_wv[ind]
Vega_flux = Vega_flux[ind]

plt.plot(Vega_wv, Vega_flux* np.max(filter_transmit)/np.max(Vega_flux),c='r')
# plt.plot(filter_wv, filter_transmit,c='b')
plt.title('Vega sed')
plt.show()