# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:15:11 2023

@author: jihyk
"""

import json
import os

os.chdir('C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\iPad-record3d\\') #set current directory


with open('metadata-depth.json') as user_file:
  file_contents = user_file.read()
  
print(file_contents)

parsed_json = json.loads(file_contents)

K = parsed_json['K']
initPose = parsed_json['initPose']

#%% convert to depth image

import numpy as np
import cv2
import open3d as o3d

# Load the point cloud from a PLY file
pcd = o3d.io.read_point_cloud("0000000.ply")

# Convert the point cloud to a numpy array
points = np.asarray(pcd.points)

# Define the camera intrinsics and extrinsics
K = np.array(K).reshape((3, 3))  # Camera intrinsics
tx, ty, tz, qx, qy, qz, qw = initPose  # Camera extrinsics

# Convert quaternion to rotation matrix
R = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
              [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
              [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]])
t = np.array([[tx], [ty], [tz]])
initPose = np.hstack((R, t))

# Determine the dimensions of the depth image
width = parsed_json['dw'] #640  # Width of the color image
height = parsed_json['dh'] #480  # Height of the color image

# Calculate the depth image
depth_image = np.zeros((height, width))
fx, fy = K[0, 0], K[1, 1]  # Focal length along x and y axes
cx, cy = K[0, 2], K[1, 2]  # Principal point coordinates
for point in points:
    # Apply the camera extrinsics
    point_hom = np.append(point, 1)
    point_cam = np.dot(initPose, point_hom)
    x, y, z = point_cam[:3]
    
    # Apply the camera intrinsics
    u = int(np.round(fx * x / z + cx))
    v = int(np.round(fy * y / z + cy))
    
    # Store the depth value in the depth image
    if 0 <= u < width and 0 <= v < height:
        if depth_image[v, u] == 0 or depth_image[v, u] > z:
            depth_image[v, u] = z

# Display the depth image
cv2.imshow("Depth Image", depth_image / np.max(depth_image))
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

# then just type in following

PATH2EXR = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\iPad-record3d\\0.exr'

img = cv2.imread(PATH2EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
'''
you might have to disable following flags, if you are reading a semantic map/label then because it will convert it into binary map so check both lines and see what you need
''' 
#img = cv2.imread(PATH2EXR) 
 
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

depth = img2[:,:,2]/3

cv2.imshow('depth',depth)
cv2.waitKey(0)

#%%

import OpenEXR
import Imath
import numpy as np

# Open the EXR file
exr_file = OpenEXR.InputFile('C:\\Users\\jihyk\\Downloads\\File_000\\K_Mild_H_1-cup\\EXR_RGBD\\depth\\10.exr')

# Get the header information
header = exr_file.header()

# Get the size of the image
data_window = header['dataWindow']
width = data_window.max.x - data_window.min.x + 1
height = data_window.max.y - data_window.min.y + 1

# Get the channel names
channels = header['channels'].keys()

# Extract the depth channel
depth_channel = 'Z' # Replace with the name of the depth channel in your EXR file
depth = np.fromstring(exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
depth = np.reshape(depth, (height, width))

# Now you can use the 'depth' array to work with the depth information from the EXR file


#%%

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv

hdr = cv.imread(PATH2EXR, cv.IMREAD_UNCHANGED)


cv.namedWindow('hdr',
               cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)


hdr = cv.normalize(hdr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

hdr = cv2.rotate(hdr, cv2.ROTATE_90_COUNTERCLOCKWISE)



print('hdr: min {} | max {}'.format(hdr.min(), hdr.max()))


cv.imshow('hdr', hdr)

cv.waitKey(0)
cv.destroyAllWindows()

#%% read ply export with camera extrinsics and intrinsics from metadata 

import open3d as o3d
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud('0000001-TD.ply')


# Define the rotation matrix (90 degrees around z-axis)
theta = np.radians(270)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

# Apply the rotation to the point cloud
pcd.rotate(R)


# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Define the camera intrinsic parameters
K = parsed_json['K']
fx = K[0]
fy = K[4]
cx = K[2]
cy = K[5]

# Define the camera extrinsic parameters
initPose = parsed_json['initPose']
tx, ty, tz, qx, qy, qz, qw = initPose

# Convert quaternion to rotation matrix
R = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
              [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
              [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]])
t = np.array([[tx], [ty], [tz]])
extrinsics = np.hstack((R, t))

# Create an OffscreenRenderer object
render = o3d.visualization.rendering.OffscreenRenderer(640, 480)

# Set up the camera with intrinsic and extrinsic parameters
center = np.array([0, 0, 0], dtype=np.float32)
focal_length = np.array([1593.8209228515625, 1593.8209228515625, 0], dtype=np.float32)
render.setup_camera(122, center, focal_length, extrinsics, np.array([0, 1, 0], dtype=np.float32))


# Render the image
image = render.render_to_image()

# Show the rendered image
plt.imshow(np.asarray(image))
plt.show()



#%% 

import pandas as pd
from pyntcloud import PyntCloud

cloud = PyntCloud.from_file('0000000.ply')