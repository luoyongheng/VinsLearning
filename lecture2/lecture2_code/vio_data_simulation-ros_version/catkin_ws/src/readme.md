注意：
这里的vio_sim功能包用于产生IMU的仿真数据集，生成imu.bag
kalibr_allan有一个bagconvert功能包，用于把imu.bag转换成matlab格式的imu.mat，因此需要matlab相关库的依赖，在ubuntu下需要安装matlab即可，但是实际运行matlab脚本报错，不知何故。
imu_utils功能包和code_utils联合使用，作为allan方差标定的替代方案