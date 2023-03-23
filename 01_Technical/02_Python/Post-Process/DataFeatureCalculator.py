# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:22:04 2023

class for calculating data features

@author: jihyk
"""

import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema, find_peaks
import scipy.fftpack

def lPassFilter(signal, f_cutoff):

    fs = 30 #fps, hz
    order = 2
    nyq = 0.5 * fs #15, smallest rate zou can sample a signal without undersampling
    if nyq > f_cutoff:
        b, a = butter(order, f_cutoff, fs = fs, btype = 'low', analog = False)
        filtSignal = filtfilt(b, a, signal.T)
    else:
        print('ERROR: Cutoff frequency is larger than Nyquist Frequency')

    return filtSignal

def calcJointAngle(a, b, c):
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


class DataFeatureCalculator:
    '''this class take the raw input data from Mediapipe pose estimation and extracts desired biomechanical features'''
    
    def __init__(self, coordinates, df):
        #instance variable in class constructor
        self.coordinates = coordinates
        self.num_frames = coordinates.shape[0]
        self.num_coordinates = coordinates.shape[1]
        self.df = df
        self.fs = 30
        
 
    
    def getElbowROM(self):
        
        lElbowAngle = np.empty([self.num_frames, 1])
        rElbowAngle = np.empty([self.num_frames, 1])

        
        for i in range(lElbowAngle.shape[0]):
            lShoulder = self.coordinates[i,11,0:2]
            rShoulder = self.coordinates[i,12,0:2]
            lElbow = self.coordinates[i,13,0:2]
            rElbow = self.coordinates[i,14,0:2]
            lWrist = self.coordinates[i,15,0:2]
            rWrist = self.coordinates[i,16,0:2]
            
            
            lElbowAngle[i] = 180-calcJointAngle(lShoulder, lElbow, lWrist)
            rElbowAngle[i] = 180-calcJointAngle(rShoulder, rElbow, rWrist)
            
            #print('lShoulder, lElbow, lWrist, lElbowAngle', lShoulder, lElbow, lWrist, calcJointAngle(lShoulder, lElbow, lWrist))
            #print('rShoulder, rElbow, rWrist, rElbowAngle', rShoulder, rElbow, rWrist, calcJointAngle(rShoulder, rElbow, rWrist))
                    
        return lElbowAngle, rElbowAngle
    
    
        
    def getShoulderAbROM(self):
        
        rShoulderAbAngle = np.empty([self.num_frames, 1])
        
        for i in range(rShoulderAbAngle.shape[0]):
            lShoulder = self.coordinates[i,11,[0,2]] #we want to just look from the top plane, x and z coordinate only
            rShoulder = self.coordinates[i,12,[0,2]]
            rElbow = self.coordinates[i,14,[0,2]]
            
            rShoulderAbAngle[i] = 180 - calcJointAngle(lShoulder, rShoulder, rElbow)
        
        return rShoulderAbAngle
    
    def getMovementTrajectoryWrist(self):
        
               
        movementTrajectory = np.empty([self.num_frames, 3])
        abs_displacement = np.empty([self.num_frames, 3])
        abs_displacement_tot = np.empty([self.num_frames, 1])
        abs_displacement_xy = np.empty([self.num_frames, 1])
        velocity_tot = np.empty([self.num_frames, 1])
        velocity_tot_filt = np.empty([self.num_frames, 1])
        
        velocity_tot[0] = 0
        fs = 30


        
        for i in range(self.num_frames):
            movementTrajectory[i,:] = self.coordinates[i,16,0:3] #r wrist index 16
            
        movementTrajectory_filt = lPassFilter(movementTrajectory, 8).T #filter movement trajectory
        movementTrajectory_filt[:,2] = lPassFilter(movementTrajectory[:,2], 3).T
            
        for i in range(self.num_frames):
            abs_displacement[i,0] =  movementTrajectory_filt[i,0] - movementTrajectory_filt[0,0]#np.mean(movementTrajectory_filt[:,0])
            abs_displacement[i,1] =  movementTrajectory_filt[i,1] - movementTrajectory_filt[0,1]#np.mean(movementTrajectory_filt[:,1])
            abs_displacement[i,2] =  movementTrajectory_filt[i,2] - movementTrajectory_filt[0,2]#np.mean(movementTrajectory_filt[:,2])
            
        #abs_displacement = lPassFilter(abs_displacement, 12).T
        #abs_displacement[:,2] = lPassFilter(abs_displacement[:,2], 3).T
            
        abs_displacement_tot = np.sqrt(abs_displacement[:,0]**2 + abs_displacement[:,1]**2 + abs_displacement[:,2]**2)
        abs_displacement_xy = np.sqrt(abs_displacement[:,0]**2 + abs_displacement[:,1]**2)
        
        #abs_displacement_tot = lPassFilter(abs_displacement_tot, 12).T #low pass filter.. again
        abs_displacement_tot = abs_displacement_tot - np.mean(abs_displacement_tot) #zero cetner
     
        for i in range(1, self.num_frames):
            velocity_tot[i] = (abs_displacement_tot[i] - abs_displacement_tot[i-1])/(1/fs)
            
        velocity_tot_filt = lPassFilter(velocity_tot, 8).T
        
        vel_peaks = find_peaks(velocity_tot_filt.ravel() , distance = 4.5, prominence = 2e-02) #
        
       
        

        return movementTrajectory, movementTrajectory_filt, abs_displacement, abs_displacement_tot, abs_displacement_xy, velocity_tot_filt, vel_peaks
    
    def getMovementTrajectoryShoulder(self):
               
        movementTrajectory = np.empty([self.num_frames, 3])
        abs_displacement = np.empty([self.num_frames, 3])
        abs_displacement_tot = np.empty([self.num_frames, 1])
        abs_displacement_xy = np.empty([self.num_frames, 1])
        
        for i in range(self.num_frames):
            movementTrajectory[i,:] = self.coordinates[i,12,0:3] #r shoulder index 1
        movementTrajectory_filt = lPassFilter(movementTrajectory, 8).T
        movementTrajectory_filt[:,2] = lPassFilter(movementTrajectory[:,2], 4).T
        
        for i in range(self.num_frames):        
            abs_displacement[i,0] =  movementTrajectory_filt[i,0] - np.mean(movementTrajectory_filt[:,0])
            abs_displacement[i,1] =  movementTrajectory_filt[i,1] - np.mean(movementTrajectory_filt[:,1])
            abs_displacement[i,2] =  movementTrajectory_filt[i,2] - np.mean(movementTrajectory_filt[:,2])
            
        #abs_displacement = lPassFilter(abs_displacement, 12).T
        #abs_displacement[:,2] = lPassFilter(abs_displacement[:,2], 3).T
            
        abs_displacement_tot = np.sqrt(abs_displacement[:,0]**2 + abs_displacement[:,1]**2 + abs_displacement[:,2]**2)
        abs_displacement_xy = np.sqrt(abs_displacement[:,0]**2 + abs_displacement[:,1]**2)
            
        #abs_displacement_tot = lPassFilter(abs_displacement_tot, 12).T
        abs_displacement_tot = abs_displacement_tot - np.mean(abs_displacement_tot)
            
        
        return movementTrajectory, movementTrajectory_filt, abs_displacement_tot, abs_displacement_xy
    
    def getNumVelPeaks(self):
        
        FPS = 30
        
        #end effector index = 16
        
        dt = float(1/FPS)
        df = self.df
        pos_x = df['x_16']
        pos_y = df['y_16']
        pos_z = df['z_16']
        numVelPeaks = 0
        idxVelPeaks = []
        numVelPeaksXY = 0
        idxVelPeaksXY = []
        
        time = np.empty([self.num_frames, 1])
        velocity = np.empty([self.num_frames, 3])
        vel_tot = np.empty([self.num_frames, 1])
        acc = np.empty([self.num_frames, 3])
        acc_tot = np.empty([self.num_frames, 1])


        for i in range(len(time)):
            if i == 0:
                time[i] = 0
            else:
                time[i] = time[i-1] + dt
        
        for i in range(len(velocity)):
            if i == 0:
                velocity[i,:] = [0,0,0]
                vel_tot[i] = 0
                acc[i,:] = 0
                acc_tot[i] = 0 
            else:
                velocity[i,0] = (pos_x[i] - pos_x[i-1])/dt
                velocity[i,1] = (pos_y[i] - pos_y[i-1])/dt
                velocity[i,2] = (pos_z[i] - pos_z[i-1])/dt
                vel_tot[i] = np.sqrt(velocity[i,0]**2 + velocity[i,1]**2 + velocity[i,2]**2)
                
                acc[i,0] = (velocity[i,0] - velocity[i-1,0])/dt
                acc[i,1] = (velocity[i,1] - velocity[i-1,1])/dt
                acc[i,2] = (velocity[i,2] - velocity[i-1,2])/dt
                acc_tot[i] = (vel_tot[i] - vel_tot[i-1])/dt
        
        for i in range(1, len(vel_tot)-1):
            #print('i, vel_tot[i-1], vel_tot[i], vel_tot[i+1]', i, vel_tot[i-1], vel_tot[i], vel_tot[i+1])
            if (vel_tot[i] < vel_tot[i-1] and vel_tot[i] < vel_tot[i+1]) or (vel_tot[i] > vel_tot[i-1] and vel_tot[i] > vel_tot[i+1]):
                numVelPeaks += 1
                idxVelPeaks.append([i, vel_tot[i]])

                                        
        
        return vel_tot, numVelPeaks, idxVelPeaks, velocity, acc
    
    '''def getSmoothness(self):
        
        return smoothness
    
    def getActivityStart(self):
        
        return acitivyStartTime'''
        
        
