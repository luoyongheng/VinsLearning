//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};
int main()
{

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z);
    }

    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();
    /* your code begin */
    Eigen::MatrixXd D(2*(end_frame_id-start_frame_id),4);
    Eigen::VectorXd b(Eigen::VectorXd::Zero(2*(end_frame_id-start_frame_id)));

    for(int i=start_frame_id;i<poseNums;i++){
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d tcw = -Rcw * camera_pose[i].twc;
        Eigen::Matrix4d T=Eigen::Matrix4d::Zero();
        T.topLeftCorner(3,3) = Rcw;
        T.topRightCorner(3,1) = tcw;
        T(3,3) = 1;
        D.block((i-start_frame_id)*2,0,1,4) = camera_pose[i].uv[0]*T.row(2) - T.row(0);
        D.block((i-start_frame_id)*2+1,0,1,4) = camera_pose[i].uv[1]*T.row(2) - T.row(1);
    }
    std::cout<<D<<std::endl;
    Eigen::MatrixXd DTD = D.transpose()*D;
    ///llt直接求解
//    Eigen::Vector4d y=DTD.llt().solve(D.transpose() * b);// llt分解要求D是正定阵

    ///SVD直接求解
//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D,Eigen::ComputeThinU | Eigen::ComputeThinV);//构建最小二乘问题
//    Eigen::Vector4d y = svd.singularValues();
//    y=svd.solve(b);
    //std::cout<<y.transpose()<<std::endl;

    ///通过求DTD的特征向量
//    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(DTD);
//    std::cout<<std::endl;
//    std::cout<<eigenSolver.eigenvalues()<<std::endl;
//    Eigen::Matrix4d U = eigenSolver.eigenvectors();
//    std::cout<<std::endl;
//    std::cout<<U<<std::endl;
//    Eigen::Vector4d leastEigenVec = U.col(0);
//    P_est = leastEigenVec.head(3)/leastEigenVec[3];

    ///通过求D的奇异向量
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D,Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix4d V = svd.matrixV();
    std::cout<<std::endl;
    std::cout<<V<<std::endl;
    Eigen::Vector4d leastEigenVec = V.col(3);
    P_est = leastEigenVec.head(3)/leastEigenVec[3];

    /* your code end */

    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
    return 0;
}
