# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:49:57 2022

@author: jihyk
"""

import cv2
import mediapipe as mp
import numpy as np
import sys

import math
import plotly.express as px
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import plotly.graph_objects as go

_PRESENCE_THRESHOLD = 0.8 #0.5
_VISIBILITY_THRESHOLD = 0.8 #0.5


def plot_landmarks(
    landmark_list,
    connections=None,
):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            #landmark.x = np.nan
            #landmark.y = np.nan
            #landmark.z = np.nan
            print('low visiblity or presence', landmark.visibility, landmark.presence)
        
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

pose = mp_pose.Pose(enable_segmentation = False, smooth_segmentation = True,
                                model_complexity=2, min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5, 
                                smooth_landmarks=True, static_image_mode = False)

holistic = mp_holistic.Holistic(enable_segmentation = False, smooth_segmentation = True,
                                refine_face_landmarks = False, model_complexity=2, 
                                min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                                smooth_landmarks=True, static_image_mode = False)


cap = cv2.VideoCapture(sys.argv[1])

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_smooth_annotated_complexity_2_mintrack_05_mindet_05_holistic.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'P', '4', 'V'), 60, (frame_width, frame_height))

joint_3D_WORLD_list = []
k = 0

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2.flip(image,1)
    image.flags.writeable = False
    results = holistic.process(image)
    #print(results.pose_landmarks)
    #print('thats it')
    joint_3D_WORLD_list.append(results.pose_landmarks)
    
    '''plt.figure(k+1)
    ax = plt.axes(projection = '3d')
    ax.scatter(joint_3D_world[:,0],joint_3D_world[:,1],joint_3D_world[:,2])
    ax.view_init(elev= -45, azim = 270)
    
    k += 1'''  


    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    
    
    '''mp_drawing.draw_landmarks(
       image,
       results.face_landmarks,
       mp_holistic.FACEMESH_CONTOURS,
       landmark_drawing_spec=None,
       connection_drawing_spec=mp_drawing_styles
       .get_default_face_mesh_contours_style())'''
    
    '''mp_drawing.draw_landmarks(
       image,
       results.left_hand_landmarks,
       mp_holistic.HAND_CONNECTIONS,
       landmark_drawing_spec=None,
       connection_drawing_spec=mp_drawing_styles
       .get_default_hand_connections_style())
    
    mp_drawing.draw_landmarks(
       image,
       results.right_hand_landmarks,
       mp_holistic.HAND_CONNECTIONS,
       landmark_drawing_spec=None,
       connection_drawing_spec=mp_drawing_styles
       .get_default_hand_connections_style())
     
    mp_drawing.draw_landmarks(
       image,
       results.pose_landmarks,
       mp_holistic.POSE_CONNECTIONS,
       landmark_drawing_spec=mp_drawing_styles
       .get_default_pose_landmarks_style())'''
    
    # Plot pose world landmarks.
    #mp_drawing.plot_landmarks(
     #   results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(image)
    #cv2.imshow('MediaPipe Pose', image)
    
#plot_landmarks(
 #   results.pose_world_landmarks,  mp_pose.POSE_CONNECTIONS)
#holistic.close()
np.save(f'{inflnm}.npy', joint_3D_WORLD_list, allow_pickle = True)

cap.release()
out.release()