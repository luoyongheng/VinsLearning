//
// Created by luoyongheng on 19-6-17.
//
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;
int main()
{
    AngleAxisd rotVec=AngleAxisd(M_PI/4,Vector3d(0,0,1));
    Matrix3d rotM=rotVec.toRotationMatrix();
    //题中的omiga不是指李代数，而是：omiga=theta*mu
    //从而theta=sqrt(omiga*omiga)=0.0374
    //mu=omiga/theta=0.267,0.535,0.8
    AngleAxisd omiga=AngleAxisd(0.0374,Vector3d(0.267,0.535,0.8));
    Matrix3d omigaR=omiga.toRotationMatrix();
    cout<<rotM*omigaR<<endl;

    Quaterniond rotQ(rotVec);
    Quaterniond deltaQ(1,0.005,0.01,0.015);
    Quaterniond qres=rotQ*deltaQ;
    cout<<qres.toRotationMatrix()<<endl;
    return 0;
}
