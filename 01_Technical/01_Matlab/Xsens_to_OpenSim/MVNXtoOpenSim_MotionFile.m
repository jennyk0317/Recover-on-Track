%% Introduction
% This script was created to generate a motion file .mot to be open in
% OpenSim with Xsens kinematic data. In this code a conversion between
% the angles defined in the Xsens software and OpenSim software are made so,
% Xsens kinematic data can be used as motion data to animate a model in
% OpenSim.

% It is important to notice that this code was built to match two models in
% OpenSim, a full body model, developed by Rojagopal 2015 (available in 
% OpenSim website), and a lower body model Gait2392_Simbody, available in 
% the OpenSim libraries.
% Also these models are optimazed for analysis of gait.
% To use other models or other more complicated tasks, might be required to
% adjust the polarity of the parameters and the chosen angles.

%CopyRight, Maria Cabral, Product Specilaist @ Xsens (www.xsens.com)

%% Import Xsens Data
main_mvnx_v2

% Clear workspace before start
clearvars -except tree filename
close all
clc

%% !!! Select which OpenSim model to use!!!
OpenSim_model = 'Rojagopal_2015'; %Gait2392_Simbody or Rojagopal_2015


%% OpenSim model characteristics
% Variables defined in the Rojagopal2015 OpenSim model. These variables cn
% be adjusted if different OpenSim models will be used.
jointName_OpenSim = {'pelvis_tilt','pelvis_list','pelvis_rotation','pelvis_tx','pelvis_ty','pelvis_tz','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r','hip_flexion_l','hip_adduction_l','hip_rotation_l', 'knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l','lumbar_extension','lumbar_bending','lumbar_rotation','arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r','arm_flex_l','arm_add_l','arm_rot_l','elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l'};

% Adjust the signal of the joints according to OpenSim direction of
% rotation
if strcmp(OpenSim_model,'Rojagopal_2015')
    signs = ones(numel(jointName_OpenSim),1);
    signs(1) = -1;
    signs(6) = -1;
    signs(8) = -1;
    signs(15) = -1;
    signs(21) = -1;
    signs(25) = -1;
    signs(32) = -1;
elseif strcmp(OpenSim_model,'Gait2392_Simbody')
    signs = ones(numel(jointName_OpenSim),1);
    signs(1) = -1;
    signs(6) = -1;
    signs(8) = -1;
    signs(10) = -1;
    signs(13) = -1;
    signs(15) = -1;
    signs(17) = -1;
    signs(20) = -1;
    signs(21) = -1;
else
    error('This model is not implemented for this script.')
end


%% Create joint table of all joints and joint angles from IMU data
% Conversion from Xsens to OpenSim angles (OpenSim vs Xsens)
nSamples = size(tree.frame,2);
nJoints = size(tree.jointData,2);
% Joint Angles Name
for i = 1:1:nJoints
    joint_names{i} = tree.jointData(i).label;
end
% Ergonomic Joint Angles Name
for i = 1:1:size(tree.ergonomicJointAngle,2)
    erg_names{i} = tree.ergonomicJointAngle(i).label;
end

% Orientation of the pelvis in space
[pelvis_list, pelvis_tilt, pelvis_rotation] = q_to_euler(tree.segmentData(1).orientation);

% Position of the pelvis in space
pelvis_tx = tree.segmentData(1).position(:,1);
pelvis_ty = tree.segmentData(1).position(:,3);
pelvis_tz = tree.segmentData(1).position(:,2);


% Lower Body - Right Hip Angles
indx = find(strcmp(joint_names,'jRightHip'));
if indx > 0
    hip_flexion_r = tree.jointData(indx).jointAngle(:,3);
    hip_adduction_r = tree.jointData(indx).jointAngle(:,1);
    hip_rotation_r =  tree.jointData(indx).jointAngle(:,2);
else
    hip_flexion_r = zeros(nSamples,1);
    hip_adduction_r = zeros(nSamples,1);
    hip_rotation_r =  zeros(nSamples,1);
end
% Lower Body - Right Knee Angles
indx = find(strcmp(joint_names,'jRightKnee'));
if indx > 0
    knee_angle_r =  tree.jointData(indx).jointAngle(:,3);
else
    knee_angle_r = zeros(nSamples,1);
end
% Lower Body - Right Ankle Angles
indx = find(strcmp(joint_names,'jRightAnkle'));
if indx > 0
    ankle_angle_r =  tree.jointData(indx).jointAngle(:,3);
    subtalar_angle_r = tree.jointData(indx).jointAngle(:,2);
else
    ankle_angle_r = zeros(nSamples,1);
    subtalar_angle_r = zeros(nSamples,1);
end

% Lower Body - Right Foot Angles
%subtalar_angle_r =  % Not defined in Xsens

indx = find(strcmp(joint_names,'jRightBallFoot'));
if indx > 0
    mtp_angle_r =  tree.jointData(indx).jointAngle(:,3);
else
    mtp_angle_r = zeros(nSamples,1);
end

% Lower Body - Left Hip Angles
indx = find(strcmp(joint_names,'jLeftHip'));
if indx > 0
    hip_flexion_l = tree.jointData(indx).jointAngle(:,3);
    hip_adduction_l = tree.jointData(indx).jointAngle(:,1);
    hip_rotation_l =  tree.jointData(indx).jointAngle(:,2);
else
    hip_flexion_l = zeros(nSamples,1);
    hip_adduction_l = zeros(nSamples,1);
    hip_rotation_l =  zeros(nSamples,1);
end
% Lower Body - Left Knee Angles
indx = find(strcmp(joint_names,'jLeftKnee'));
if indx > 0
    knee_angle_l = tree.jointData(indx).jointAngle(:,3);
else
    knee_angle_l = zeros(nSamples,1);
end
% Lower Body - Left Ankle Angles
indx = find(strcmp(joint_names,'jLeftAnkle'));
if indx > 0
    ankle_angle_l = tree.jointData(indx).jointAngle(:,3);
    subtalar_angle_l = tree.jointData(indx).jointAngle(:,2);
else
    ankle_angle_l = zeros(nSamples,1);
    subtalar_angle_l = zeros(nSamples,1);
end
% Lower Body - Left Foot Angles
%subtalar_angle_l = % Not defined in Xsens
indx = find(strcmp(joint_names,'jLeftBallFoot'));
if indx > 0
    mtp_angle_l = tree.jointData(indx).jointAngle(:,3);
else
    mtp_angle_l = zeros(nSamples,1);
end

% Torso Angles

if find(strcmp(joint_names,'jL5S1'))
    indx = find(strcmp(joint_names,'jL5S1'));
    lumbar_extension = tree.jointData(indx).jointAngle(:,3);
    lumbar_bending = tree.jointData(indx).jointAngle(:,1);
    lumbar_rotation =  tree.jointData(indx).jointAngle(:,2);
else
    lumbar_extension = zeros(nSamples,1);
    lumbar_bending = zeros(nSamples,1);
    lumbar_rotation =  zeros(nSamples,1);
end

% lumbar_extension = -pelvis_tilt;
% lumbar_bending = -pelvis_list;
% lumbar_rotation = -pelvis_rotation;


% Upper Body - Right Shoulder Angles
indx = find(strcmp(joint_names,'jRightShoulder'));
if indx > 0
    arm_flex_r = tree.jointData(indx).jointAngle(:,3);
    arm_add_r = tree.jointData(indx).jointAngle(:,1);
    arm_rot_r = tree.jointData(indx).jointAngle(:,2);
else
    arm_flex_r = zeros(nSamples,1);
    arm_add_r = zeros(nSamples,1);
    arm_rot_r = zeros(nSamples,1);
end
% Upper Body - Right Elbow Angles
indx = find(strcmp(joint_names,'jRightElbow'));
if indx > 0
    elbow_flex_r = tree.jointData(indx).jointAngle(:,3);
    pro_sup_r = tree.jointData(indx).jointAngle(:,2);
else
    elbow_flex_r = zeros(nSamples,1);
    pro_sup_r = zeros(nSamples,1);
end
% Upper Body - Right Wrist Angles
indx = find(strcmp(joint_names,'jRightWrist'));
if indx > 0
    wrist_flex_r = tree.jointData(indx).jointAngle(:,3);
    wrist_dev_r = tree.jointData(indx).jointAngle(:,2);
else
    wrist_flex_r = zeros(nSamples,1);
    wrist_dev_r = zeros(nSamples,1);
end
% Upper Body - Left Shoulder Angles
indx = find(strcmp(joint_names,'jLeftShoulder'));
if indx > 0
    arm_flex_l = tree.jointData(indx).jointAngle(:,3);
    arm_add_l = tree.jointData(indx).jointAngle(:,1);
    arm_rot_l = tree.jointData(indx).jointAngle(:,2);
else
    arm_flex_l = zeros(nSamples,1);
    arm_add_l = zeros(nSamples,1);
    arm_rot_l = zeros(nSamples,1);
end
% Upper Body - Left Elbow Angles
indx = find(strcmp(joint_names,'jLeftElbow'));
if indx > 0
    elbow_flex_l = tree.jointData(indx).jointAngle(:,3);
    pro_sup_l = tree.jointData(indx).jointAngle(:,2);
else
    elbow_flex_l = zeros(nSamples,1);
    pro_sup_l = zeros(nSamples,1);
end
% Upper Body - Left Wrist Angles
indx = find(strcmp(joint_names,'jLeftWrist'));
if indx > 0
    wrist_flex_l = tree.jointData(indx).jointAngle(:,3);
    wrist_dev_l = tree.jointData(indx).jointAngle(:,2);
else
    wrist_flex_l = zeros(nSamples,1);
    wrist_dev_l = zeros(nSamples,1);
end

% Time
for i=1:1:size(tree.frame,2)
    time(i) = str2num(tree.frame(i).time)*0.001;
end

Angles_Matrix = [pelvis_tilt';pelvis_list';pelvis_rotation';pelvis_tx';pelvis_ty';pelvis_tz';hip_flexion_r';hip_adduction_r';hip_rotation_r';knee_angle_r';ankle_angle_r';subtalar_angle_r';mtp_angle_r';hip_flexion_l';hip_adduction_l';hip_rotation_l';knee_angle_l';ankle_angle_l';subtalar_angle_l';mtp_angle_l';lumbar_extension';lumbar_bending';lumbar_rotation';arm_flex_r';arm_add_r';arm_rot_r';elbow_flex_r';pro_sup_r';wrist_flex_r';wrist_dev_r';arm_flex_l';arm_add_l';arm_rot_l';elbow_flex_l';pro_sup_l';wrist_flex_l';wrist_dev_l'];
Angles_Matrix = signs.*(Angles_Matrix);
Angles_Matrix =[time; Angles_Matrix];

%% Generate motion file

[numRows,numCols] = size(Angles_Matrix');
ncols = num2str(numCols);
nrows = num2str(numRows);
% Create base joint angle matrix
splitfilename = strsplit(filename,'.');
splitfilename = splitfilename{1};
fileID = fopen(strcat(splitfilename,'_Xsens_jointangle_q','.mot'),'w');
fprintf(fileID,strcat(['first trial\nnRows=',nrows,'\nnColumns=',ncols,'\n\n']));
fprintf(fileID,'# SIMM Motion File Header:\n');
nrange = num2str([min(min(Angles_Matrix)) max(max(Angles_Matrix))]); 
otherdata = num2str(1); %always 1 since time vec exists
jointNameStrings = ['pelvis_tilt  pelvis_list  pelvis_rotation  pelvis_tx  pelvis_ty  pelvis_tz  hip_flexion_r  hip_adduction_r  hip_rotation_r  knee_angle_r  ankle_angle_r  subtalar_angle_r  mtp_angle_r  hip_flexion_l  hip_adduction_l  hip_rotation_l  knee_angle_l  ankle_angle_l  subtalar_angle_l  mtp_angle_l  lumbar_extension  lumbar_bending  lumbar_rotation  arm_flex_r  arm_add_r  arm_rot_r  elbow_flex_r  pro_sup_r  wrist_flex_r  wrist_dev_r  arm_flex_l  arm_add_l  arm_rot_l  elbow_flex_l  pro_sup_l  wrist_flex_l  wrist_dev_l'];

fprintf(fileID,strcat(['name ',splitfilename, '\ndatacolumns ',ncols,'\ndatarows ', nrows,'\notherdata ',otherdata,'\nrange ',nrange,'\nendheader\n']));
fprintf(fileID,strcat(['time ',jointNameStrings,'\n']));
fprintf(fileID,[repmat('%5.4f\t',1,size(Angles_Matrix,1)),'\n'],Angles_Matrix);
fprintf(fileID,'\n');
fclose(fileID);

