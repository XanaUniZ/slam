/**
* This file is part of Mini-SLAM
*
* Copyright (C) 2021 Juan J. Gómez Rodríguez and Juan D. Tardós, University of Zaragoza.
*
* Mini-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Mini-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with Mini-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "FishEye.h"
#include <unsupported/Eigen/Polynomials>

#define fx vParameters_[0]
#define fy vParameters_[1]
#define cx vParameters_[2]
#define cy vParameters_[3]
#define k1 vParameters_[4]
#define k2 vParameters_[5]
#define k3 vParameters_[6]
#define k4 vParameters_[7]

void FishEye::project(const Eigen::Vector3f& p3D, Eigen::Vector2f& p2D){
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    float r = sqrt(p3D.x()*p3D.x() + p3D.y()*p3D.y());
    float theta = atan(r/p3D.z());
    float theta3 = theta*theta*theta;
    float theta5 = theta3*theta*theta;
    float theta7 = theta5*theta*theta;
    float theta9 = theta7*theta*theta;
    float d = theta + k1*theta3 + k2*theta5 + k3*theta7 + k4*theta9; 

    p2D(0) = fx * d * (p3D.x() / r) + cx;
    p2D(1) = fy * d * (p3D.y() / r) + cy;
}

void FishEye::unproject(const Eigen::Vector2f& p2D, Eigen::Vector3f& p3D) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */
    float mx = (p2D.x()-cx)/fx;
    float my = (p2D.y()-cy)/fy;
    float r = sqrt(mx*mx + my*my);
    float theta = solve_theta(r, k1, k2, k3, k4);

    p3D(0) = sin(theta)*mx/r;
    p3D(1) = sin(theta)*my/r;
    p3D(2) = cos(theta);
}

void FishEye::projectJac(const Eigen::Vector3f& p3D, Eigen::Matrix<float,2,3>& Jac) {
    /*
     * Your code for Lab 3 - Task 5 here!
     */

    // Jac(0,0) = fx / p3D(2);
    // Jac(0,1) = 0.f;
    // Jac(0,2) = -fx * p3D(0) / (p3D(2) * p3D(2));

    // Jac(1,0) = 0.f;
    // Jac(1,1) = fy / p3D(2);
    // Jac(1,2) = -fy * p3D(1) / (p3D(2) * p3D(2));

    float x =p3D(0);
    float y =p3D(1);
    float z =p3D(2);
    float r = sqrt(p3D.x()*p3D.x() + p3D.y()*p3D.y());
    float sq_sum = (x*x + z*z + y*y);
    float theta = atan(r/p3D.z());
    float theta3 = theta*theta*theta;
    float theta5 = theta3*theta*theta;
    float theta7 = theta5*theta*theta;
    float theta9 = theta7*theta*theta;
    float d = theta + k1*theta3 + k2*theta5 + k3*theta7 + k4*theta9; 

    // Partial d_theta theta
    float theta2 = theta*theta;
    float theta4 = theta2*theta*theta;
    float theta6 = theta4*theta*theta;
    float theta8 = theta6*theta*theta;
    float partial_d_theta_theta = 1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8;

    // Partial theta
    float patial_theta_x = (z*x)/(r*sq_sum);
    float patial_theta_y = (z*y)/(r*sq_sum);
    float patial_theta_z = -r/sq_sum;

    // Partial d_theta xyz
    float partial_d_theta_x = partial_d_theta_theta * patial_theta_x;
    float partial_d_theta_y = partial_d_theta_theta * patial_theta_y;
    float partial_d_theta_z = partial_d_theta_theta * patial_theta_z;

    // Partial u xyz
    float partial_u_x = partial_d_theta_x * fx * (x/r) + d*(fx*y*y)/std::pow(r, 3/2);
    float partial_u_y = partial_d_theta_y * fx * (x/r) - d*(2*fx*x*y)/std::pow(r, 3/2);
    float partial_u_z = partial_d_theta_z * fx * (x/r);

    // Partial v xyz
    float partial_v_x = partial_d_theta_x * fy * (y/r) - d*(2*fy*x*y)/std::pow(r, 3/2);
    float partial_v_y = partial_d_theta_y * fy * (y/r) + d*(fx*x*x)/std::pow(r, 3/2);
    float partial_v_z = partial_d_theta_z * fy * (y/r);

    Jac(0,0) = partial_u_x;
    Jac(0,1) = partial_u_y;
    Jac(0,2) = partial_u_z;

    Jac(1,0) = partial_v_x;
    Jac(1,1) = partial_v_y;
    Jac(1,2) = partial_v_z;

}

void FishEye::unprojectJac(const Eigen::Vector2f& p2D, Eigen::Matrix<float,3,2>& Jac) {
    /*
     * Left Empty
     */
}

float FishEye::solve_theta(float r, float _k1, float _k2, float _k3, float _k4) {
    Eigen::VectorXf coeffs(10);
    coeffs << -r, 1, 0, _k1, 0, _k2, 0, _k3, 0, _k4; // Represents: θ^9*k4 + θ^7*k3 + θ^5*k2 + θ^3*k1 + θ - r = 0
    
    Eigen::PolynomialSolver<float, 9> solver(coeffs);
    std::vector<float> roots;
    solver.realRoots(roots);
    
    // Return the smallest positive real root (physical solution)
    for (float root : roots) {
        if (root > 0) return root;
    }
    return r; // Fallback to initial guess
}