import numpy as np
import matplotlib.pyplot as plt
filter_dat = np.loadtxt('ACS_WFC_606.dat')
filter_wv = filter_dat[:,0]
filter_transmit = filter_dat[:,1]
plt.plot(filter_wv,filter_transmit)
plt.title('ACS_WFC_606')
plt.show()