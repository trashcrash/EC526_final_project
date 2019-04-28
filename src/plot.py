import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.stats import sem
NAMES1 = ["v_256_0lev_10stride.dat", "v_256_1lev_10stride.dat",
             "v_256_2lev_10stride.dat", "v_256_3lev_10stride.dat", 
             "v_256_4lev_10stride.dat", "v_256_5lev_10stride.dat", 
             "v_256_6lev_10stride.dat", "v_256_7lev_10stride.dat"]
NAMES2 = ["w_256_0lev_10stride.dat", "w_256_1lev_10stride.dat",
             "w_256_2lev_10stride.dat", "w_256_3lev_10stride.dat", 
             "w_256_4lev_10stride.dat", "w_256_5lev_10stride.dat", 
             "w_256_6lev_10stride.dat", "w_256_7lev_10stride.dat"]
NAMES3 = ["v_256_0lev_10stride_gauss.dat", "v_256_1lev_10stride_gauss.dat",
             "v_256_2lev_10stride_gauss.dat", "v_256_3lev_10stride_gauss.dat", 
             "v_256_4lev_10stride_gauss.dat", "v_256_5lev_10stride_gauss.dat", 
             "v_256_6lev_10stride_gauss.dat", "v_256_7lev_10stride_gauss.dat"]
NAMES4 = ["new_w_256_0lev_10stride_gauss.dat", "new_w_256_1lev_10stride_gauss.dat",
             "new_w_256_2lev_10stride_gauss.dat", "new_w_256_3lev_10stride_gauss.dat", 
             "new_w_256_4lev_10stride_gauss.dat", "new_w_256_5lev_10stride_gauss.dat", 
             "new_w_256_6lev_10stride_gauss.dat", "new_w_256_7lev_10stride_gauss.dat"]

def readfiles(filename):
    a = zeros(10)
    f = open(filename, 'r')
    for i in range(len(a)):
        a[i] = f.readline()
    return mean(a), sem(a)

xaxis = [0, 1, 2, 3, 4, 5, 6, 7]
mean1 = []
se1 = []
mean2 = []
se2 = []
mean3 = []
se3 = []
mean4 = []
se4 = []
for i in range(len(xaxis)):
    mean1.append(readfiles(NAMES1[i])[0])
    se1.append(readfiles(NAMES1[i])[1])
    mean2.append(readfiles(NAMES2[i])[0])
    se2.append(readfiles(NAMES2[i])[1])
    mean3.append(readfiles(NAMES3[i])[0])
    se3.append(readfiles(NAMES3[i])[1])
    mean4.append(readfiles(NAMES4[i])[0])
    se4.append(readfiles(NAMES4[i])[1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Levels', fontsize = 16)
ax.set_ylabel('Time Spent (s)', fontsize = 16)

ax.axis([0, 7, 5, 20])
linestyle = {"linestyle":"-", "linewidth":1, "markeredgewidth":1, "elinewidth":1, "capsize":2}
ax.errorbar(xaxis, mean1, yerr = se1, **linestyle, label='V-cycle Jacobi')
ax.errorbar(xaxis, mean2, yerr = se2, **linestyle, label='W-cycle Jacobi')
ax.errorbar(xaxis, mean3, yerr = se3, **linestyle, label='V-cycle Gauss-Seidel')
ax.errorbar(xaxis, mean4, yerr = se4, **linestyle, label='W-cycle Gauss-Seidel')
plt.title("256 by 256 grid", fontsize = 16)
ax.legend()
plt.show()
