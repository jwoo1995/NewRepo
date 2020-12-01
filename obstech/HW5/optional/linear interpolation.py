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

plt.scatter(x_new,f_new,c='b',label= 'linear interpolation', s= 0.5, alpha=0.5)
plt.scatter(Vega_wv,Vega_flux,c='r', label = 'Vega sed', s= 1)
plt.xlim(np.min(x_new), np.max(x_new))
plt.tight_layout()
plt.legend()
plt.show()




# print(len(Vega_wv), np.min(Vega_wv), np.max(Vega_wv))
# print(len(filter_wv), np.min(filter_wv), np.max(filter_wv))


# multiplication = Vega_flux * filter_transmit
# weight = multiplication/np.sum(multiplication)
# plt.plot(Vega_wv,weight,c='g')
# plt.title('weight')
# plt.show()