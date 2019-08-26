clear 
close all

dt = dlmread('../data/data_my_gyr_t.txt');         
data_x = dlmread('../data/data_my_gyr_x.txt'); 
data_y= dlmread('../data/data_my_gyr_y.txt'); 
data_z = dlmread('../data/data_my_gyr_z.txt'); 
data_avr=(data_x+data_y+data_z)./3;
data_draw=[data_x data_y data_z,data_avr] ;
data_draw=data_draw.*(pi/180/3600);

data_sim_x= dlmread('../data/data_my_sim_acc_x.txt'); 
data_sim_y= dlmread('../data/data_my_sim_acc_y.txt'); 
data_sim_z= dlmread('../data/data_my_sim_acc_z.txt'); 
data_sim_avr=(data_sim_x+data_sim_y+data_sim_z)./3;
data_sim_draw=[data_sim_x data_sim_y data_sim_z,data_sim_avr] ;
data_sim_draw=data_sim_draw.*(pi/180/3600);


figure
% loglog(dt, data_draw , 'o');
loglog(dt, data_draw ,'-','LineWidth',3);
title('陀螺仪标定');
xlabel('time:sec');                
ylabel('Sigma:rad/s');             
% legend('x','y','z');      
grid on;          
% hold on;                           
% loglog(dt, data_sim_draw , '-');
