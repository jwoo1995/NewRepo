import numpy as np
import matplotlib.pyplot as plt

# ========================================== 01. load data ======================================================
PS_crit = np.loadtxt('PS_critical_mVector_PLANCK-SMT.txt', skiprows=12, usecols=(0, 7))
PS_crit_x = PS_crit[:, 0]
PS_crit_y = PS_crit[:, 1]

PS_mean = np.loadtxt("PS_mean_mVector_PLANCK-SMT.txt", skiprows=12, usecols=(0, 7))
PS_mean_x = PS_mean[:, 0]
PS_mean_y = PS_mean[:, 1]

SMT_crit = np.loadtxt('SMT_critical_mVector_PLANCK-SMT.txt', skiprows=12, usecols=(0, 7))
SMT_crit_x = SMT_crit[:, 0]
SMT_crit_y = SMT_crit[:, 1]



SMT_mean = np.loadtxt('SMT_mean_mVector_PLANCK-SMT.txt', skiprows=12, usecols=(0, 7))
SMT_mean_x = SMT_mean[:, 0]
SMT_mean_y = SMT_mean[:, 1]

print(SMT_mean_y)
# ========================================== 02. plotting ======================================================
size = 2
plt.figure(1)
plt.scatter(PS_crit_x, PS_crit_y, label ='PS critical density',s=size)
plt.scatter(PS_mean_x, PS_mean_y, label ='PS mean density',s=size)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-6, 1)

plt.figure(2)
plt.scatter(SMT_crit_x, SMT_crit_y, label ='SMT critical density',s=size, alpha=0.5)
plt.scatter(SMT_mean_x, SMT_mean_y, label ='SMT critical density',s=size, alpha=0.5)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-6, 1)

plt.figure(3)
plt.scatter(PS_crit_x, PS_crit_y, label ='SMT critical density',s=size, alpha=0.5)
plt.scatter(SMT_crit_x, SMT_crit_y, label ='SMT critical density',s=size, alpha=0.5)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-6, 1)

plt.figure(4)
plt.scatter(PS_mean_x, PS_mean_y, label ='SMT critical density',s=size, alpha=0.5)
plt.scatter(SMT_mean_x, SMT_mean_y, label ='SMT critical density',s=size, alpha=0.5)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-6, 1)
plt.show()

