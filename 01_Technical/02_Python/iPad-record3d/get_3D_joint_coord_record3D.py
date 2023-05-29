# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:59:25 2023

@author: jihyk

open npy of mediapipe output, add depth to each frame
create a gif of the plotted output

also save npy
"""
#%% import libaries
import numpy as np
import cv2
import os
import liblzfse  # https://pypi.org/project/pyliblzfse/
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.interpolate import griddata

    
# load all functions

#loading .depth files
def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    #depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img

# lets look at segment length

def calculate_segment_length(point1, point2):
    # Calculate the differences between coordinates
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]

    # Calculate the squared differences
    squared_distance = dx**2 + dy**2 + dz**2

    # Calculate the length by taking the square root of the squared distance
    segment_length = math.sqrt(squared_distance)

    return segment_length

#calculate upper arm length 

def calculate_rupperarm_rforearm_length(coord_list):
    r_upperarm_length = []
    r_forearm_length = []
    
    for i in range(len(coord_list)):
        r_shoulder = coord_list[i][12,:]
        r_elbow = coord_list[i][14,:]
        r_wrist = coord_list[i][16,:]
        
        r_upperarm = calculate_segment_length(r_shoulder, r_elbow)
        r_forearm = calculate_segment_length(r_elbow, r_wrist)
        
        #print(f"In frame {i}, the upperarm length is {r_upperarm} and forearm length is {r_forearm}")
        
        r_upperarm_length.append(r_upperarm)
        r_forearm_length.append(r_forearm)
        
    return r_upperarm_length, r_forearm_length

# clean depth image to get rid of shadowing effect through interpolation

def filter_nan(image):
    # Create a mask to identify NaN values
    mask = np.isnan(image)
    
    # Get coordinates of non-NaN values
    coords = np.array(np.nonzero(~mask)).T
    
    # Get values of non-NaN pixels
    values = image[~mask]
    
    # Interpolate NaN values using nearest neighbors
    interpolated = griddata(coords, values, np.argwhere(mask), method='nearest')
    
    # Update the image array with interpolated values
    filtered_image = image.copy()
    filtered_image[mask] = interpolated.squeeze()
    
    return filtered_image

# functions for organizing data and plotting segment length

def combine_and_remove_duplicates(arr1, arr2):
    combined_array = np.concatenate((arr1, arr2))
    unique_elements = np.unique(combined_array)
    return unique_elements.tolist()

def plot_segment_length(r_upperarm_length, r_forearm_length):
    mean_upperarm_length = np.nanmean(r_upperarm_length)
    range_upperarm = 0.1 * mean_upperarm_length
    
    mean_forearm_length = np.nanmean(r_forearm_length)
    range_forearm = 0.1 * mean_forearm_length
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(r_upperarm_length, label = 'Segment Length')
    ax1.axhline(y =mean_upperarm_length, color = 'r', label = 'Mean')
    ax1.axhline(y = mean_upperarm_length + range_upperarm, color='r', linestyle='dotted', label='+10% Range')
    ax1.axhline(y =mean_upperarm_length - range_upperarm, color='r', linestyle='dotted', label='-10% Range')
    ax1.set_title('Right Upper Arm Length')
    ax1.legend()
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Segment Length [m]')
    
    ax2.plot(r_forearm_length, label = 'Segment Length')
    ax2.axhline(mean_forearm_length, color = 'r', label = 'Mean')
    ax2.axhline(mean_forearm_length + range_forearm, color='r', linestyle='dotted', label='+10% Range')
    ax2.axhline(mean_forearm_length - range_forearm, color='r', linestyle='dotted', label='-10% Range')
    ax2.set_title('Right Forearm Length')
    ax2.legend()
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Segment Length [m]')
    
    return ax1

#%% open file with coordinate data and create list of data

trialname = 'K_Mild_H_1-cup-lidar'

path = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\iPad-record3d\\'# + trialname + '\\'
#os.chdir(path)

joint_3D_WORLD_list = []
joint_3D_PIXEL_list = []

filename = trialname + '_color'
removed_frames = []

results = np.load(filename + '.npy', allow_pickle = True)

for i in range(results.size): 
    joint_3D_world = np.zeros((33,3), float)
    
    if results[i] is None: 
        removed_frames.append(i)
        print('removed frame: ', i)
        continue
    else:
        landmarks = results[i].landmark
    
    for j in range(33):
        joint_3D_world[j,0] = landmarks[j].y
        joint_3D_world[j,1] = landmarks[j].x
        
        
    joint_3D_WORLD_list.append(joint_3D_world)
    
#%% convert world coordinate data to pixel coordinate and remove outliers and add z value to world coordinates

RECORD3D_LIDAR = True
RECORD3D_TRUEDEPTH = False
FILTER_NAN = True

if RECORD3D_LIDAR:
    width = 256
    height = 192
    rotate_dir = cv2.ROTATE_90_COUNTERCLOCKWISE
    fps = 60
    
elif RECORD3D_TRUEDEPTH:
    width = 640
    height = 480
    rotate_dir = cv2.ROTATE_90_CLOCKWISE
    fps = 30

#for k in range(results.size):
k = 0
depth_error = []

while k < len(joint_3D_WORLD_list): #cap.isOpened():
    
    if k in removed_frames:
        print(f"Skipping frame {k} due to removal")
        k += 1
        continue

    #print(f"Processing frame {k}")

    joint_3D_pixel = np.zeros((17,3), float)
    depth_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\LiDAR\K_Mild_H_1-cup\\rgbd\\' + str(k) +  '.depth'
    color_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\LiDAR\K_Mild_H_1-cup\\rgbd\\' + str(k) +  '.jpg'
    
    depth_img = load_depth(depth_filepath) #this is depth value in meters
    depth_img = cv2.rotate(depth_img, rotate_dir)
    
    color_img = cv2.imread(color_filepath)
    color_img = cv2.rotate(color_img, rotate_dir)
    
    if FILTER_NAN: 
        depth_img = filter_nan(depth_img)
        
    frame_height, frame_width = depth_img.shape
    #cv2.imshow('Depth', depth_img)
    #cv2.waitKey(0)
    
    for l in range(17): #if joint is outside frame, name as nan otherwise convert to pixel coordinates
            
        joint_3D_pixel[l,0] = int(joint_3D_WORLD_list[k][l][0] * frame_height)
        joint_3D_pixel[l,1] = int(joint_3D_WORLD_list[k][l][1] * frame_width)
        
        z_coord = (depth_img[int(joint_3D_pixel[l,0]), int(joint_3D_pixel[l,1])])
        if l != 15 and l != 13 and  z_coord > 3.5: #here we are looking at right arm values only 'impaired side for dry run'
            #print(f'Unexpected depth reading of z = {z_coord} in frame {k}, joint{l}')
            
            '''figs, axs = plt.subplots(1, 2)
            im1 = axs[0].imshow(depth_img)
            axs[0].set_title(f'Unexpected depth reading of z = {z_coord} in frame {k}, joint{l}')
            axs[0].scatter(joint_3D_pixel[l,1], joint_3D_pixel[l,0], color = 'r') 
            plt.colorbar(im1, ax=axs[0])
            
            axs[1].imshow(color_img)
            axs[1].set_title(f'color image, frame {k}')
            axs[1].scatter(joint_3D_pixel[l,1], joint_3D_pixel[l,0], color = 'r') '''
            
            
            depth_error.append(k)
            joint_3D_pixel[l,2] = np.nan
            
        joint_3D_pixel[l,2] = z_coord #from depth map [world coord z]
        joint_3D_WORLD_list[k][l,2] = z_coord #from depth map
        

      
    joint_3D_PIXEL_list.append(joint_3D_pixel)

    k += 1
    
#np.save( trialname + '_3D_filt.npy', joint_3D_WORLD_list, allow_pickle = True)

#so nan was happening due to the shadowing effects....


#%% count and clean number of nan frames - no prediction for upper arm/forearm and unexpected predictions with wrong depth values

def remove_nan_and_depth_errors(coord_list, depth_error):
    
    r_upperarm_length, r_forearm_length = calculate_rupperarm_rforearm_length(coord_list)
    
    nan_upperarm = np.isnan(r_upperarm_length)
    nan_count_upperarm = np.sum(nan_upperarm)
    nan_forearm = np.isnan(r_forearm_length)
    nan_count_forearm = np.sum(nan_forearm)

    nan_indices_forearm = np.where(nan_forearm)
    nan_indices_upperarm = np.where(nan_upperarm)

    nan_indices_forearm_reshaped = np.reshape(nan_indices_forearm[0], (-1,))
    nan_indices_upperarm_reshaped = np.reshape(nan_indices_upperarm[0], (-1,))
    
    nan_indices_combined = combine_and_remove_duplicates(nan_indices_forearm_reshaped, nan_indices_upperarm_reshaped)
    nan_count_combined = len(nan_indices_combined)
    
    indices_remove = combine_and_remove_duplicates(nan_indices_combined, depth_error)
    
    print(f"Number of NaN values, Upperarm: {nan_count_upperarm} Forearm: {nan_count_forearm}, combined: {nan_count_combined}")
    
    coord_list_clean = [item for index, item in enumerate(coord_list) if index not in indices_remove]
    
    return coord_list_clean


joint_3D_WORLD_list_clean = remove_nan_and_depth_errors(joint_3D_WORLD_list, depth_error)
joint_3D_PIXEL_list_clean = remove_nan_and_depth_errors(joint_3D_PIXEL_list, depth_error)

r_upperarm_length_clean, r_forearm_length_clean = calculate_rupperarm_rforearm_length(joint_3D_WORLD_list_clean)

#%%

r_upperarm_length_clean_pxl, r_forearm_length_clean_pxl = calculate_rupperarm_rforearm_length(joint_3D_PIXEL_list_clean)


#%% plot segment length before and after cleaning

#plot_segment_length(r_upperarm_length, r_forearm_length)
plot_segment_length(r_upperarm_length_clean, r_forearm_length_clean)

plot_segment_length(r_upperarm_length_clean_pxl, r_forearm_length_clean_pxl)


#%% cut beginning of video - post clapping (click on when beginning stabilizes and exit window)
from mpl_point_clicker import clicker

ax = plot_segment_length(r_upperarm_length_clean, r_forearm_length_clean) #cleaned nan and removed depth prediction errors
ax2 = plot_segment_length(r_upperarm_length_clean_pxl, r_forearm_length_clean_pxl)

klicker = clicker(ax, ["event"], markers=["x"])
klicker_pxl = clicker(ax2, ["event"], markers=["x"])

#%% run this to crop and save .npy of cleaned coordinates 
start_frame = (klicker.get_positions())['event']

start_frame = int(start_frame[0][0])

start_frame_pxl = (klicker_pxl.get_positions())['event']

start_frame_pxl = int(start_frame_pxl[0][0])

joint_3D_WORLD_list_final = joint_3D_WORLD_list_clean[start_frame:]
joint_3D_PIXEL_list_final = joint_3D_PIXEL_list_clean[start_frame_pxl:]


r_upperarm_length_final, r_forearm_length_final = calculate_rupperarm_rforearm_length(joint_3D_WORLD_list_final)
plot_segment_length(r_upperarm_length_final, r_forearm_length_final)

r_upperarm_length_pxl_final, r_forearm_length_pxl_final = calculate_rupperarm_rforearm_length(joint_3D_PIXEL_list_final)
plot_segment_length(r_upperarm_length_pxl_final, r_forearm_length_pxl_final)

np.save( trialname + '_3D_clean.npy', joint_3D_WORLD_list_final, allow_pickle = True)
np.save( trialname + '_3D_clean_pxl.npy', joint_3D_PIXEL_list_final, allow_pickle = True)



#%%
os.chdir('C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\TrueDepth\\cup\\K_Mild_H_1-cup\\rgbd\\')


for j in range(len(nan_indices_combined[0])):
    i = nan_indices_combined[0][j]
    color_image = cv2.imread(f'{i}.jpg')
    color_image = cv2.rotate(color_image, rotate_dir)
    
    
    depth_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\TrueDepth\cup\K_Mild_H_1-cup\\rgbd\\' + str(i) +  '.depth'
    depth_img = load_depth(depth_filepath) #this is depth value in meters
    depth_img = cv2.rotate(depth_img, rotate_dir)
    #depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR).astype(np.uint8) #create 3 channels for concatenation
    
    #concatenated_image = cv2.vconcat([color_image, depth_img])
    
    cv2.imshow('Image with NaN', depth_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#%% plot specific frames (click on frame you want to plot and exit the window)
ax = plot_segment_length(r_upperarm_length, r_forearm_length) #cleaned nan and removed depth prediction errors

klicker = clicker(ax, ["event"], markers=["x"])
#%%
frame = (klicker.get_positions())['event']

frame = int(frame[0][0])

depth_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\TrueDepth\cup\K_Mild_H_1-cup\\rgbd\\' + str(frame) +  '.depth'
color_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\TrueDepth\cup\K_Mild_H_1-cup\\rgbd\\' + str(frame) +  '.jpg'

depth_img = load_depth(depth_filepath) #this is depth value in meters
depth_img = cv2.rotate(depth_img, rotate_dir)

depth_img_filled = filter_nan(depth_img)

color_img = cv2.imread(color_filepath)
color_img = cv2.rotate(color_img, rotate_dir)

figs, axs = plt.subplots(1, 3)
im1 = axs[0].imshow(depth_img)
#axs[0].set_title(f'Unexpected depth reading of z = {z_coord} in frame {k}, joint{l}')
#axs[0].scatter(joint_3D_PIXEL_list[frame][:,1], joint_3D_PIXEL_list[frame][:,0], color = 'r') 
plt.colorbar(im1, ax=axs[0])

axs[1].imshow(depth_img_filled)
axs[2].imshow(color_img)
#axs[1].set_title(f'color image, frame {k}')
#axs[1].scatter(joint_3D_PIXEL_list[frame][:,1], joint_3D_PIXEL_list[frame][:,0], color = 'r') 

#%% show depth image filtered w. neighborhood filling

for j in range(len(joint_3D_WORLD_list_clean)):
    
    i = nan_indices_combined[j]
    
    depth_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\03_DryRun\\DryRun-02052023\\Record3D\\TrueDepth\cup\K_Mild_H_1-cup\\rgbd\\' + str(i) +  '.depth'
    depth_img = load_depth(depth_filepath) #this is depth value in meters
    depth_img = cv2.rotate(depth_img, rotate_dir)
    depth_img_filt = filter_nan(depth_img)
    
    concatenated_image = cv2.vconcat([depth_img, depth_img_filt])
    
    cv2.imshow('Depth Image Cleaned', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%% generate gif of 2d pose (gross)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

coordinates = joint_3D_WORLD_list_final

steps = len(coordinates)


fig, ax = plt.subplots()
marker_size = 50

def animate(i):
   fig.clear()
   ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
   ax.set_xlim(0, 1.0)
   ax.set_ylim(-1.0, 0)
   ax.set_xticks([])
   ax.set_yticks([])

   ax.scatter(coordinates[i][0,1], -coordinates[i][0,0],  c='r', s = 400, alpha= 0.5) #nose point
   ax.scatter(coordinates[i][(11,12,14,16),1], -coordinates[i][(11,12,14,16),0],  c='b', s = 40, alpha= 0.8) #upper body
   ax.plot(coordinates[i][(11,12,14,16),1], -coordinates[i][(11,12,14,16),0]) #connect upper body points with lines
   ax.scatter(coordinates[i][16,1], -coordinates[i][16,0], c='g', s = 50, alpha= 0.7) #plot hand as green

ax.grid(False)
ax.set_title(trialname + ' 2D Pose Estimation')

if steps > 1000:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000))
    
    ani.save(trialname + '_1_2_animation2D.gif', writer='pillow')
    
    ani2 = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000, steps))
    
    ani2.save(trialname + '_2_2_animation2D.gif', writer='pillow')

else:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = steps)
    
    ani.save(trialname + '_animation2D.gif', writer='pillow')

#%% generate gif of 2d pose (gross) top down

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

coordinates = joint_3D_WORLD_list_final

steps = len(coordinates)


fig, ax = plt.subplots()
marker_size = 50

def animate(i):
   fig.clear()
   ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)
   ax.set_xlim(0, 1.0)
   ax.set_ylim(-1.0, 0)
   ax.set_xticks([])
   ax.set_yticks([])

   ax.scatter(coordinates[i][0,2], -coordinates[i][0,0],  c='r', s = 400, alpha= 0.5) #nose point
   ax.scatter(coordinates[i][(11,12,14,16),2], -coordinates[i][(11,12,14,16),0],  c='b', s = 40, alpha= 0.8) #upper body
   ax.plot(coordinates[i][(11,12,14,16),2], -coordinates[i][(11,12,14,16),0]) #connect upper body points with lines
   ax.scatter(coordinates[i][16,2], -coordinates[i][16,0], c='g', s = 50, alpha= 0.7) #plot hand as green

ax.grid(False)
ax.set_title(trialname + ' 2D Pose Estimation')

if steps > 1000:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000))
    
    ani.save(trialname + '_1_2_animation2D.gif', writer='pillow')
    
    ani2 = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000, steps))
    
    ani2.save(trialname + '_2_2_animation2D.gif', writer='pillow')

else:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = steps)
    
    ani.save(trialname + '_animation2D.gif', writer='pillow')

#%% generate gif of gross human pose
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

coordinates = joint_3D_WORLD_list_final

steps = len(coordinates)


fig, ax = plt.subplots()
marker_size = 50

def animate(i):
   fig.clear()
   ax = fig.add_subplot(111, aspect='equal', autoscale_on=True, projection = '3d')
   ax.set_xlim3d(0, 1)
   ax.set_ylim3d(0, -1)
   ax.view_init(elev= -180, azim = 90, vertical_axis = 'z')
   ax.set_zlim3d(0,3)
   ax.set_xticks([])
   ax.set_yticks([])
   ax.set_zticks([])
   ax.set_title(trialname + ' : 3D Pose Estimation')

   ax.scatter(coordinates[i][0,1], -coordinates[i][0,0], coordinates[i][0,2], c='r', s = 200, alpha= 0.5)
   ax.scatter(coordinates[i][(11,12,14),1], -coordinates[i][(11,12,14),0], coordinates[i][(11,12,14),2], c= 'b', s = 30, alpha = 0.8)
   ax.plot(coordinates[i][(11,12,14,16),1], -coordinates[i][(11,12,14,16),0], coordinates[i][(11,12,14,16),2], c='k')
   ax.scatter(coordinates[i][16,1], -coordinates[i][16,0], coordinates[i][16,2], c='g', s = 50, alpha= 0.7)

ax.grid(False)

#ani = animation.FuncAnimation(fig, animate, interval=33, frames=range(steps))

#ani.save(trialname + '_animation3D_gross.gif', writer='pillow')

if steps > 1000:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000))
    
    ani.save(trialname + '_1_2_animation3D_gross.gif', writer='pillow')
    
    ani2 = animation.FuncAnimation(fig, animate, interval=33, frames = range(1000, steps))
    
    ani2.save(trialname + '_2_2animation3D_gross.gif', writer='pillow')

else:
    ani = animation.FuncAnimation(fig, animate, interval=33, frames = steps)
    
    ani.save(trialname + '_animation3D_gross.gif', writer='pillow')

