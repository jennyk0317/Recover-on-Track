# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:18:19 2023

@author: jihyk
"""

import numpy as np
import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/


def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    #depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


if __name__ == '__main__':
    depth_filepath = 'C:\\Users\\jihyk\\Dropbox (Partners HealthCare)\\Recover-on-Track\\01_Technical\\02_Python\\iPad-record3d\\2023-05-01--15-39-50\\rgbd\\0.depth'
    depth_img = load_depth(depth_filepath) #this is depth value in meters
    #depth_img = depth_img[:,:]/3

    depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow('Depth', depth_img)
    cv2.waitKey(0)