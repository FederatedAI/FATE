# -*- coding:utf-8 -*-
# @Author   : LuoJiahuan
# @File     : plot.py 
# @Time     : 2019/7/23 16:27

import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, make_interp_spline, BSpline
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

file_name = "./formatted_logs/yolo_test_map.csv"
with open(file_name) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    C5_E1_B1, C5_E5_B1, C5_E10_B1, C20_E1_B1, C20_E5_B1, C20_E10_B1, C1_E1_B1 = [list() for _ in range(7)]
    x = []
    c = []
    for i, row in enumerate(reader):
        x.append(i)
        C5_E1_B1.append(float(row[0]))
        C5_E5_B1.append(float(row[1]))
        C5_E10_B1.append(float(row[2]))
        C20_E1_B1.append(float(row[3]))
        C20_E5_B1.append(float(row[4]))
        C20_E10_B1.append(float(row[5]))
        try:
            C1_E1_B1.append(float(row[6]))
            c.append(i)
        except:
            pass
    f.close()
xnew = np.linspace(1, 1000, 100)
print(len(xnew))
# xnew = np.arange(1, 1000, 200)
cen = np.linspace(1, 866, 80)
# C5_E1_B1 = interp1d(x, C5_E1_B1, kind='cubic')(xnew)
# C5_E5_B1 = interp1d(x, C5_E5_B1, kind='cubic')(xnew)
# C5_E10_B1 = interp1d(x, C5_E10_B1, kind='cubic')(xnew)
# C20_E1_B1 = interp1d(x, C20_E1_B1, kind='cubic')(xnew)
# C20_E5_B1 = interp1d(x, C20_E5_B1, kind='cubic')(xnew)
# C20_E10_B1 = interp1d(x, C20_E10_B1, kind='cubic')(xnew)
# C1_E1_B1 = interp1d(c, C1_E1_B1, kind='cubic')(cen)
# print(len(C5_E1_B1_1))
# C5_E1_B1_1 = interp1d(xnew, C5_E1_B1_1, kind='cubic')(np.arange(1, 200, 0.2))
C5_E1_B1 = gaussian_filter1d(C5_E1_B1, sigma=8)
C5_E5_B1 = gaussian_filter1d(C5_E5_B1, sigma=8)
C5_E10_B1 = gaussian_filter1d(C5_E10_B1, sigma=8)
C20_E1_B1 = gaussian_filter1d(C20_E1_B1, sigma=8)
C20_E5_B1 = gaussian_filter1d(C20_E5_B1, sigma=8)
C20_E10_B1 = gaussian_filter1d(C20_E10_B1, sigma=8)
C1_E1_B1 = gaussian_filter1d(C1_E1_B1, sigma=8)
fig = plt.figure(dpi=320, figsize=(8, 8))
plt.title("Street YOLO", fontsize=18)
plt.xlabel("Rounds", fontsize=14)
plt.ylabel("Test mAP", fontsize=14)
plt.ylim((0.5, 0.875))
# plt.plot(x, C5_E1_B1, color='red', linewidth=2.0, linestyle='--', label='C=5 E=1 B=1')
# plt.plot(x, C5_E5_B1, color='orange', linewidth=2.0, linestyle='-', label='C=5 E=5 B=1')
# plt.plot(x, C5_E10_B1, color='blue', linewidth=2.0, linestyle='--', label='C=5 E=10 B=1')
plt.plot(x, C5_E1_B1, color='red', linewidth=2.0, linestyle='--', label='C=5 E=1 B=1')
plt.plot(x, C5_E5_B1, color='orange', linewidth=2.0, linestyle='-', label='C=5 E=5 B=1')
plt.plot(x, C5_E10_B1, color='blue', linewidth=2.0, linestyle='--', label='C=5 E=10 B=1')
plt.plot(x, C20_E1_B1, color='red', linewidth=2.0, linestyle='-', label='C=20 E=1 B=1')
plt.plot(x, C20_E5_B1, color='orange', linewidth=2.0, linestyle='--', label='C=20 E=5 B=1')
plt.plot(x, C20_E10_B1, color='blue', linewidth=2.0, linestyle='-', label='C=20 E=10 B=1')
plt.plot(c, C1_E1_B1, color='green', linewidth=2.0, linestyle='-', label='C=1 E=1 B=1')
# plt.plot(xnew, C5_E1_B1, color='red', linewidth=2.0, linestyle='--', label='C=5 E=1 B=1')
# plt.plot(xnew, C5_E5_B1, color='orange', linewidth=2.0, linestyle='-', label='C=5 E=5 B=1')
# plt.plot(xnew, C5_E10_B1, color='blue', linewidth=2.0, linestyle='--', label='C=5 E=10 B=1')
# plt.plot(xnew, C20_E1_B1, color='red', linewidth=2.0, linestyle='--', label='C=20 E=1 B=1')
# plt.plot(xnew, C20_E5_B1, color='orange', linewidth=2.0, linestyle='-', label='C=20 E=5 B=1')
# plt.plot(xnew, C20_E10_B1, color='blue', linewidth=2.0, linestyle='--', label='C=20 E=10 B=1')
# plt.plot(cen, C1_E1_B1, color='green', linewidth=2.0, linestyle='--', label='C=1 E=1 B=1')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc='lower right', fontsize=14)
# plt.show()
plt.savefig("centralized_smooth.png")
