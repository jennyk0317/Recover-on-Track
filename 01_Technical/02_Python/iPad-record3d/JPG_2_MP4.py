# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:54:29 2022
convert png-s exported using rs-convert to mp4 
@author: jihyk
"""

import cv2
import os
import pyrealsense2 as rs
from natsort import natsorted 

with_depth = False
only_depth = False

RECORD3D_LIDAR = True
RECORD3D_TRUEDEPTH = True



trialname = 'K_Mild_H_1-cup_lidar'

image_folder = 'C:/Users/jihyk/Dropbox (Partners HealthCare)/Recover-on-Track/03_DryRun/DryRun-02052023/Record3D/LiDAR/K_Mild_H_1-cup/rgbd/'

image_depth_folder = 'C:/Users/jihyk/Dropbox (Partners HealthCare)/Recover-on-Track/03_DryRun/RealSense/Output/' + trialname + '/depth/'

alpha = 0.7
beta = (1.0 - alpha)

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

dirFiles = os.listdir(image_folder)

images = [img for img in dirFiles if img.endswith(".jpg")]
images = natsorted(images) #sort 1, 2, 3, ... instead of 1, 10, 100, etc
#images_depth = [img_depth for img_depth in os.listdir(image_depth_folder) if img_depth.endswith(".png")]

frame = cv2.imread(os.path.join(image_folder, images[0]))
frame = cv2.rotate(frame, rotate_dir)
height, width, layers = frame.shape

hole_filling = rs.hole_filling_filter()

i = 0

if with_depth:
    video_name = trialname + '_color_n_depth.mp4'
    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        src1 = cv2.imread(os.path.join(image_folder, image))
        src1 = cv2.rotate(src1, rotate_dir)
        src2 = cv2.imread(os.path.join(image_depth_folder, images_depth[i]))
        src2 = cv2.rotate(src2, rotate_dir)
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
       
        video.write(dst)
        i += 1
        
elif only_depth:
    video_name = trialname + '_depth_medianfilt11.mp4'    
    video = cv2.VideoWriter(video_name, 0, 60, (width,height))

    for image in images:
        src1 = cv2.imread(os.path.join(image_depth_folder, images_depth[i]))
        src1 = cv2.medianBlur(src1, 11)
        #src1 = hole_filling.process(src1[:,:,0])
        i += 1
        video.write(src1)
        
else:
    video_name = trialname + '_color.mp4'
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))
    


    for image in images:
        
        src1 = cv2.imread(os.path.join(image_folder, image))
        src1 = cv2.rotate(src1, rotate_dir)
        src1 = cv2.resize(src1, (width, height))  # For a LiDAR 3D Video
        
        video.write(src1)
        

cv2.destroyAllWindows()
video.release()