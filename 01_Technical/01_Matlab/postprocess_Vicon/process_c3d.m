clc
clear all

dir('C:\Users\jihyk\Dropbox (Partners HealthCare)\Recover-on-Track\03_DryRun\DryRun02\Vicon');

[file,path] = uigetfile('*.c3d');

acq = btkReadAcquisition(file);

markers = btkGetMarkers(acq);

angles = btkGetAngles(acq);