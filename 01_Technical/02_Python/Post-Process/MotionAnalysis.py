# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:42:05 2023

open .npy file of joint coordinate frames and subsequently call the class DataFeatureCalculator to extract features

@author: jihyk
"""
import os

os.chdir('C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\Post-Process\\') #set current directory


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, filtfilt, spectrogram

from DataFeatureCalculator import DataFeatureCalculator

 
# function to scale data and create dataframe
def create_dataframe(X):
    
    '''sc = StandardScaler()
    x_coord = sc.fit_transfrom(X)[:,0]
    y_coord = sc.fit_transfrom(X)[:,1]
    z_coord = sc.fit_transfrom(X)[:,2]
    
    for i in range(X.shape[1]):'''
        
    
    X_ft = pd.DataFrame(X)
    X_ft.columns = ['x_0', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 
                    'y_0', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22', 'y_23', 'y_24', 
                    'z_0', 'z_11', 'z_12', 'z_13', 'z_14', 'z_15', 'z_16', 'z_17', 'z_18', 'z_19', 'z_20', 'z_21', 'z_22', 'z_23', 'z_24']
    
    return X_ft

# reshape coordinate data to create dataframe
def reshape_data(X):
    X_new = np.empty((X.shape[0],45))
    for i in range(X.shape[0]):
        X_new[i,0:15] = np.atleast_2d(X[i][:,0])
        X_new[i,15:30] = np.atleast_2d(X[i][:,1])
        X_new[i,30:45] = np.atleast_2d(X[i][:,2])
    return X_new


# Find frequency space of X and plot

def frequencySpace_Plot(signal, fs, name):
    
    sig_freq = fft(signal)
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(signal)
    ax1.set_title('X coordinate: time domain')
    ax2.plot(sig_freq)
    ax2.set_title('frequency domain')
    
    f, t, Sxx = spectrogram(signal, fs)
    plt.figure()
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title('Spectrogram of ' + name )
    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

#
def main(): 
    
    PLOT = True
    DISPLAYOUTPUT = True
    
    filename = 'G_Mod_L_3'
    
    UPPER_BODY_COORDS = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                         'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 
                         'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
                         'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                         'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP'] 
                            #total 15 landmarks for upperbody

    os.chdir('C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\Pose-Estimation\\' + filename +'\\') #set current directory

    results = np.load(filename + '_3D_filt.npy', allow_pickle=True) #load previously saved list of coordinates from pose estimation of video
    
    results = results[30:] #crop data if beg or end is funkay
    
    joint_3D_WORLD_list = np.empty([results.shape[0],33,3])
    time = np.empty([results.shape[0],1])
    
    k = 0
    fs = 30
    
    for i in range(results.shape[0]): 
        joint_3D_world = np.zeros((33,3), float)
        landmarks = results[i,:,:]
        time[i] = i * (1/fs)
        
        for j in range(33):
            joint_3D_world[j,0] = landmarks[j,0]
            joint_3D_world[j,1] = landmarks[j,1]
            joint_3D_world[j,2] = landmarks[j,2]
            
        '''landmarks = results[i].landmark
        
        for j in range(33):
            joint_3D_world[j,0] = landmarks[j].y
            joint_3D_world[j,1] = landmarks[j].x
            joint_3D_world[j,2] = landmarks[j].z'''
            
            
        joint_3D_WORLD_list[k,:,:] = joint_3D_world
        k += 1
        
    #reorganize and create dataframe
    data = np.empty([len(joint_3D_WORLD_list),15,3])
    data[:,0,:] = joint_3D_WORLD_list[:, 0, :]
    data[:,1:15,:] = joint_3D_WORLD_list[:, 11:25, :] #only take nose, 0, and upper coordinate]
    
    
    data_reorg = reshape_data(data)
    
    df = create_dataframe(data_reorg)
    
    #extract data features

    coordinates = joint_3D_WORLD_list
    data_features = DataFeatureCalculator(coordinates, df)
    
    movementTrajectory_Rwrist, movementTrajectory_Rwrist_filt, displacement_Rwrist, abs_displacement_Rwrist, displacement_Rwrist_XY, velocity_tot_filt, vel_peaks = data_features.getMovementTrajectoryWrist()
    movementTrajectory_RShoulder, movementTrajectory_RShoulder_filt, displacement_RShoulder, displacement_RShoulder_XY = data_features.getMovementTrajectoryShoulder()
    
    velocity_tot_filt = velocity_tot_filt #* 1000 #convert to mm
    vel_peaks_val = np.empty((vel_peaks[0].size,3))
    
    vel_peaks_val[:,0] = vel_peaks[0]
    vel_peaks_val[:,1] = vel_peaks[0]* (1/30)
    vel_peaks_val[:,2] = velocity_tot_filt[vel_peaks[0]].ravel()
    

    
    
    lElbowAngle, rElbowAngle = data_features.getElbowROM()
    
    #vel_tot, numVelPeaks, idxVelPeaks, velocity, acc = data_features.getNumVelPeaks()
    
    rShoulderAbAngle = data_features.getShoulderAbROM()
    
    if PLOT == True:
        

        #PLOT DISPLACEMENT AND VELOCITY
    
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
        ax1.plot(displacement_Rwrist) #zero centered 
        ax2.plot(abs_displacement_Rwrist)
        ax2.set_title('Total Displacement')
        ax3.plot(time, velocity_tot_filt[:], 'r')
        ax3.set_ylabel('velocity [m/s]')
        ax3.scatter(vel_peaks_val[:,1], vel_peaks_val[:,2], marker = 'o',  color = 'b')
        ax1.set_title(filename +' Movement Trajectory R Wrist XYZ')
        ax3.set_title('Total Velocity R Wrist, cutoff_freq: 12')
        
        # plot trajectory

        coordinate_x = movementTrajectory_Rwrist_filt[:,1]
        coordinate_y = movementTrajectory_Rwrist_filt[:,0]
        coordinate_z = movementTrajectory_Rwrist_filt[:,2]


        #fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax4.plot(time, coordinate_y)
        ax5.plot(time, coordinate_x)
        ax6.plot(time, coordinate_z)

        ax4.set_title(filename + 'RWrist x trajectory')
        ax5.set_title('RWrist y trajectory')
        ax6.set_title('RWrist z trajectory')
        
        #frequencySpace_Plot(coordinate_x, 30, 'X Trajectory')
        #frequencySpace_Plot(coordinate_y, 30, 'Y Trajectory')
        #frequencySpace_Plot(coordinate_x, 30, 'Z Trajectory')

        '''for ax in fig.get_axes():
            
            ax.set(xlabel='frame', ylabel='[m]')
            ax.label_outer()'''
        ''' #3d trajectory plot
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xs = df[coordinate_x], ys = df[coordinate_y], zs = df[coordinate_z])
        ax.plot(xs = df[coordinate_x], ys = df[coordinate_y], zs = df[coordinate_z])
        plt.title(filename + UPPER_BODY_COORDS[i-10] + ' 3D trajectory')'''
        
        # plot elbow angles
        ''' plt.figure()
        plt.plot(rElbowAngle)
        plt.title(filename+' Right Elbow Flexion Angle')
        plt.ylabel('Angle [deg]')'''

        # plot shoulder adduction angle

    '''plt.figure()
        plt.plot(rShoulderAbAngle)
        plt.title(filename + ' Right Shoulder Abduction Angle')
        plt.ylabel('Angle [deg]')'''

    if DISPLAYOUTPUT == True:
        print(filename)
        print('shoulder max displacement ', max(abs(displacement_RShoulder)))
        #print('wrist max displacement ', max(abs(abs_displacment_Rwrist)))
        
        print('shoulder max displacement XY ', max(abs(displacement_RShoulder_XY)))
        #print('wrist max displacement XY ', max(abs(abs_displacment_Rwrist)))
        
    
        print('maximum speed is ', max(abs(velocity_tot_filt)))
        print('mean velcity is ', np.mean(abs(velocity_tot_filt)))
        print('number of velocity peaks is ', vel_peaks_val.shape[0])
    
        
        print('minimum right elbow flexion angle is: ', min(rElbowAngle))
        print('maximum right elbow flexion angle is: ', max(rElbowAngle))
        print('minimum right shoulder horizontal adduction angle is: ', min(rShoulderAbAngle))
        print('maximum right shoulder horizontal adduction angle is: ', max(rShoulderAbAngle))
        
        duration = results.shape[0] / fs

        print('total duration is ', duration, ' seconds')
        
if __name__ == '__main__':
    main()


