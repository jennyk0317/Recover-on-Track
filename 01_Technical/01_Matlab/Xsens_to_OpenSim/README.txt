***How to import Xsens files in OpenSim:***
To use this MATLAB scripts it is required the OpenSim 4.1 version.

**MVN
1 - Open your recording in MVN Studio
2 - HD reprocess the data with the "reset pelvis orientation" box checked and realocate the right foot to 0,0,0
3 - After reprocess, export the file in .mvnx format. Make sure that all the variables you would like are checked. 

**MATLAB
1 - Open the matlab scripts called "MVNXtoOpenSim_MotionFile.m"
2 - Change the OpenSim model name for the corresponding model name you want to use
3 - Run the code.
4 - A pop up window will show up and you should select from which mvnx file you want to create a motion file. 
5 - This MATLAB code will create a motion file which is compatible with the format accepted in OpenSim.

**OpenSim
1 - Open your model in OpenSim, available in the OpenSim libraries or downloaded from the OpenSim website. This model should be the same model that you have selected in the MATLAB script.
2 - Load the motion on your model, by selecting the new motion file created by your MATLAB script. 
3 - Click play and analyzse the motion in OpenSim.

Remarks: 
- It is important to notice that any additional steps performed in OpenSim, such as scaling the model, inverse dynamics, etc, are not here explained. To do so, you should go to OpenSim support material.
- With this scripts we only provide a conversion from the kinematics in the MVN model and the kinematics in the OpenSim model. 

