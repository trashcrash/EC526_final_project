import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy.stats import sem
NAMES1 = ["v_512_0lev_10stride.dat", "v_512_1lev_10stride.dat",
             "v_512_2lev_10stride.dat", "v_512_3lev_10stride.dat", 
             "v_512_4lev_10stride.dat", "v_512_5lev_10stride.dat", 
             "v_512_6lev_10stride.dat", "v_512_7lev_10stride.dat"]
NAMES2 = ["w_512_0lev_10stride.dat", "w_512_1lev_10stride.dat",
             "w_512_2lev_10stride.dat", "w_512_3lev_10stride.dat", 
             "w_512_4lev_10stride.dat", "w_512_5lev_10stride.dat", 
             "w_512_6lev_10stride.dat", "w_512_7lev_10stride.dat"]
NAMES3 = ["mpi_v_512_0lev_10stride.dat", "mpi_v_512_1lev_10stride.dat",
             "mpi_v_512_2lev_10stride.dat", "mpi_v_512_3lev_10stride.dat",
             "mpi_v_512_4lev_10stride.dat", "mpi_v_512_5lev_10stride.dat",
             "mpi_v_512_6lev_10stride.dat", "mpi_v_512_7lev_10stride.dat"]
NAMES4 = ["mpi_w_512_0lev_10stride.dat", "mpi_w_512_1lev_10stride.dat",
             "mpi_w_512_2lev_10stride.dat", "mpi_w_512_3lev_10stride.dat",
             "mpi_w_512_4lev_10stride.dat", "mpi_w_512_5lev_10stride.dat",
             "mpi_w_512_6lev_10stride.dat", "mpi_w_512_7lev_10stride.dat"]
NAMES5 = ["acc_v_512_0lev.dat", "acc_v_512_1lev.dat",
             "acc_v_512_2lev.dat", "acc_v_512_3lev.dat", 
             "acc_v_512_4lev.dat", "acc_v_512_5lev.dat", 
             "acc_v_512_6lev.dat", "acc_v_512_7lev.dat"]
NAMES6 = ["acc_w_512_0lev.dat", "acc_w_512_1lev.dat",
             "acc_w_512_2lev.dat", "acc_w_512_3lev.dat", 
             "acc_w_512_4lev.dat", "acc_w_512_5lev.dat", 
             "acc_w_512_6lev.dat", "acc_w_512_7lev.dat"]

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
mean5 = []
se5 = []
mean6 = []
se6 = []
for i in range(len(xaxis)):
    mean1.append(readfiles(NAMES1[i])[0])
    se1.append(readfiles(NAMES1[i])[1])
    mean2.append(readfiles(NAMES2[i])[0])
    se2.append(readfiles(NAMES2[i])[1])
    mean3.append(readfiles(NAMES3[i])[0])
    se3.append(readfiles(NAMES3[i])[1])
    mean4.append(readfiles(NAMES4[i])[0])
    se4.append(readfiles(NAMES4[i])[1])
    mean5.append(readfiles(NAMES5[i])[0])
    se5.append(readfiles(NAMES5[i])[1])
    mean6.append(readfiles(NAMES6[i])[0])
    se6.append(readfiles(NAMES6[i])[1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Number of Levels', fontsize = 16)
ax.set_ylabel('Time Spent (s)', fontsize = 16)

ax.axis([0, 7, 0, 75])
linestyle = {"linestyle":"-", "linewidth":1, "markeredgewidth":1, "elinewidth":1, "capsize":2}
ax.errorbar(xaxis, mean1, yerr = se1, **linestyle, label='V-cycle Jacobi')
ax.errorbar(xaxis, mean2, yerr = se2, **linestyle, label='W-cycle Jacobi')
ax.errorbar(xaxis, mean3, yerr = se3, **linestyle, label='V-cycle MPI')
ax.errorbar(xaxis, mean4, yerr = se4, **linestyle, label='W-cycle MPI')
ax.errorbar(xaxis, mean5, yerr = se5, **linestyle, label='V-cycle OpenACC')
ax.errorbar(xaxis, mean6, yerr = se6, **linestyle, label='W-cycle OpenACC')
plt.title("512 by 512 grid", fontsize = 16)
ax.legend(loc = 'upper right')
plt.show()
