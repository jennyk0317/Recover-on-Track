# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:12:05 2023

@author: jihyk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

duration = np.array([(11, 16, 18), (26, 17.9, 24.3), (41, 26.3, 38.7), (69.16, 63.9, 61.86)])

df = pd.DataFrame(duration.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Total Duration [s]')
plt.title('Boxplot of Total Exercise Duration, Beanbag Exercise')

#%% shoulder displacement XYZ

shoulder_displacement = np.array([(0.03, 0.048, 0.05), (0.057, 0.067, 0.063), (0.076, 0.054, 0.059), (0.0956, 0.06, 0.068)])

df = pd.DataFrame(shoulder_displacement.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Total shoulder_displacement [m]')
plt.title('Boxplot of Shoulder Displacement, Beanbag Exercise XYZ')

#%% shoulder displacement XY

shoulder_displacement = np.array([(0.0293, 0.025, 0.02), (0.015, 0.0379, 0.021), (0.018, 0.03, 0.042), (0.04, 0.045, 0.0498)])

df = pd.DataFrame(shoulder_displacement.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Total shoulder_displacement [m]')
plt.title('Boxplot of Shoulder Displacement, Beanbag Exercise XY')


#%% max speed

max_speed = np.array([(2.44, 2.52, 3.62), (1.35, 1.78, 0.995), (1.09, 0.452, 1.37), (0.76, 0.34, 1.93)])

df = pd.DataFrame(max_speed.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Maximum Speed [m/s]')
plt.title('Boxplot of Maximum Speed, Beanbag Exercise XYZ')

#%% mean speed

mean_speed = np.array([(0.334, 0.333, 0.37), (0.145, 0.22, 0.132), (0.105, 0.1085, 0.127), (0.04, 0.042, 0.112)])

df = pd.DataFrame(max_speed.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Mean Speed [m/s]')
plt.title('Boxplot of Mean Speed, Beanbag Exercise XYZ')


#%% elbow flexion

elbow_flexion = np.array([(99.87, 109, 119), (96.04, 93.4, 99.8), (96.3, 103.8, 91.44), (97, 92, 99.3)])

df = pd.DataFrame(elbow_flexion.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Elbow Flexion')
plt.title('Boxplot of Elbow Flexion, Beanbag Exercise XYZ')

#%% velocity peaks

vel_peaks = np.array([(26, 33, 36), (55, 40, 56), (96, 56, 91), (141, 131, 138)])

df = pd.DataFrame(vel_peaks.T, columns=['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])

plt.figure()
boxplot = df.boxplot(column = ['Mild Hi', 'Mild Lo', 'Mod Hi', 'Mod Lo'])
plt.ylabel('Number of Velocity Pekas')
plt.title('Boxplot of Smoothness, Beanbag Exercise XYZ')