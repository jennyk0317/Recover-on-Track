function [roll_bymanualXsens, pitch_bymanualXsens, yaw_bymanualXsens] = q_to_euler(quaternion)

% Quaternions to Euler
%pelvis segment 1
q0 = quaternion(:,1);
q1 = quaternion(:,2);
q2 = quaternion(:,3);
q3 = quaternion(:,4);

roll_bymanualXsens = (180/pi)*atan((2*q2.*q3+2.*q0.*q1)./(2.*q0.*q0 + 2.*q3.*q3 -1));
pitch_bymanualXsens = -(180/pi)*asin(2.*q1.*q3-2.*q0.*q2);
yaw_bymanualXsens = (180/pi)*atan((2.*q1.*q2+2.*q0.*q3)./(2.*q0.*q0 + 2.*q1.*q1 -1));

