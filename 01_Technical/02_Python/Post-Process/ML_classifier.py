# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:46:27 2023

take .npy files that contain x (joint 3d coord) data and y label
@author: jihyk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, validation_curve


data = np.load('jk_test.npy', allow_pickle = True )


UPPER_BODY_COORDS = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                     'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 
                     'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
                     'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                     'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP'] 
                        #total 15 landmarks for upperbody

#%% function to scale data and create dataframe
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

#%% reshape coordinate data to create dataframe
def reshape_data(X):
    X_new = np.empty((X.shape[0],45))
    for i in range(X.shape[0]):
        X_new[i,0:15] = np.atleast_2d(X[i][:,0])
        X_new[i,15:30] = np.atleast_2d(X[i][:,1])
        X_new[i,30:45] = np.atleast_2d(X[i][:,2])
    return X_new
    
#%%

data_reorg = reshape_data(data)

df = create_dataframe(data_reorg)

#%% calculate joint angles for feature extraction

def calc_joint_angle(a, b, c):
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

#%%

def find_coord_and_calc_joint_angle(df, joint_x, joint_y, joint_z):
    angle_array = np.empty((df.shape[0],1))
    
    
    for i in range(df.shape[0]):
        a = np.array([df['x_' + str(joint_x)][i], df['y_' + str(joint_x)][i], df['z_' + str(joint_x)][i]])
        b = np.array([df['x_' + str(joint_y)][i], df['y_' + str(joint_y)][i], df['z_' + str(joint_y)][i]])
        c = np.array([df['x_' + str(joint_z)][i], df['y_' + str(joint_z)][i], df['z_' + str(joint_z)][i]])
        angle_array[i] = calc_joint_angle(a, b, c)
    
    
    return angle_array



#%%
hip_mid_point = np.array([(df['x_24'] + df['x_23'] )/ 2, (df['y_24'] + df['y_23'] )/ 2, (df['z_24'] + df['z_23'] )/ 2]).T
vert_from_hip_mid = np.array([hip_mid_point[:,0], hip_mid_point[:,1]+0.5, hip_mid_point[:,2]]).T
shoulder_mid_point = np.array([(df['x_11'] + df['x_12'] )/ 2, (df['y_11'] + df['y_12'] )/ 2, (df['z_11'] + df['z_12'] )/ 2]).T

df['x_hip_ctr'] = hip_mid_point[:,0].tolist()
df['y_hip_ctr'] = hip_mid_point[:,1].tolist()
df['z_hip_ctr'] = hip_mid_point[:,2].tolist()

df['x_shoulder_ctr'] = shoulder_mid_point[:,0].tolist()
df['y_shoulder_ctr'] = shoulder_mid_point[:,1].tolist()
df['z_shoulder_ctr'] = shoulder_mid_point[:,2].tolist()

df['x_hip_vert'] = vert_from_hip_mid[:,0].tolist()
df['y_hip_vert'] = vert_from_hip_mid[:,1].tolist()
df['z_hip_vert'] = vert_from_hip_mid[:,2].tolist()

#%% joint angle arrays in degrees

l_elbow_angle = find_coord_and_calc_joint_angle(df, 15, 13, 11)
r_elbow_angle = find_coord_and_calc_joint_angle(df, 16, 14, 12)
l_shoulder_angle = find_coord_and_calc_joint_angle(df, 24, 12, 14)
r_shoulder_angle = find_coord_and_calc_joint_angle(df, 23, 11, 13)

df['l_elbow_angle'] = l_elbow_angle[:,0].tolist()
df['r_elbow_angle'] = r_elbow_angle[:,0].tolist()
df['l_shoulder_angle'] = l_shoulder_angle[:,0].tolist()
df['r_shoulder_angle'] = r_shoulder_angle[:,0].tolist()


#%% there is a tilt in the mediapipe prediction so lets account for that, count the initial torso position as 0 deg

torso_angle = find_coord_and_calc_joint_angle(df, 'hip_vert', 'hip_ctr', 'shoulder_ctr')

zero_torso = torso_angle[0]

torso_angle_calib = np.subtract(torso_angle, zero_torso) #kinda sus about these results 


#%% plot trajectory
i = 13

coordinate_x = 'x_' + str(i)
coordinate_y = 'y_' + str(i)
coordinate_z = 'z_' + str(i)


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df[coordinate_x])
ax2.plot(df[coordinate_y])
ax3.plot(df[coordinate_z])
#fig.title(UPPER_BODY_COORDS[i] + ' x trajectory')
ax1.set_title(UPPER_BODY_COORDS[i-10] + ' x trajectory')
ax2.set_title(UPPER_BODY_COORDS[i-10] + ' y trajectory')
ax3.set_title(UPPER_BODY_COORDS[i-10] + ' z trajectory')

for ax in fig.get_axes():
    
    ax.set(xlabel='frame', ylabel='[m]')
    ax.label_outer()

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xs = df[coordinate_x], ys = df[coordinate_y], zs = df[coordinate_z])
ax.plot(xs = df[coordinate_x], ys = df[coordinate_y], zs = df[coordinate_z])
plt.title(UPPER_BODY_COORDS[i-10] + ' 3D trajectory')


#%% plot joint angle

subject = 'l_elbow_angle'
subject2 = 'r_elbow_angle'

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(df[subject])
ax1.set_title(subject)
ax2.plot(df[subject2])
ax2.set_title(subject2)


#%% label data

y_label = '2' # '1' for moderate, '2' for mild

y_label_data = np.ones(df.shape[0])

if y_label == '2':
    y_label_data = np.add(y_label_data, 1)

df['y_label'] = y_label_data.tolist()

#%%

X_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

#%% select classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier


#clf = DecisionTreeClassifier(random_state=0)
#clf = RandomForestClassifier(random_state=0)
#clf = LogisticRegression(random_state=0)
#clf = LinearSVC(random_state = 0)
clf = MLPClassifier(random_state = 0, max_iter=300, solver = 'adam', alpha = 0.0001)
#clf = KNeighborsClassifier(n_neighbors = 5)

#%% train predict and calculate scores


import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

start = time.time()
clf.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

print(cross_val_score(clf, X_train, y_train, cv=5))

#%%
y_hat= clf.predict(X_test_ft)


y_hat = pd.DataFrame(y_hat)
accuracy = accuracy_score(y_test_ft, y_hat) #number of correct predictions
precision = precision_score(y_test_ft, y_hat, average ='weighted') #number of positive that acutally belong to positive
recall = recall_score(y_test_ft, y_hat, average ='weighted') #number of correct positives made of all positive 
f1_score = f1_score(y_test_ft, y_hat, average ='weighted') #2*precision*recal/(precision+recall)
#scores = cross_val_score(clf, X_train_sc, y_train.values.flatten(), cv=5,
 #                       scoring = "f1")
 
print('precision: ', precision, '\nrecall: ', recall, '\nf1_score: ', f1_score,'\n')

