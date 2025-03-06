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

/*
 * Author: Juan J. Gómez Rodríguez (jjgomez@unizar.es)
 *
 * Implementation of the FishEye camera model with 4 parameters
 */

#ifndef JJSLAM_FishEye_H
#define JJSLAM_FishEye_H

#include "CameraModel.h"

#include <assert.h>
#include <vector>

#include <opencv2/opencv.hpp>


class FishEye : public CameraModel{
public:
    FishEye() {
        // 4 Camera Intr + 4 Dist Coeff
        vParameters_.resize(4+4);
    }

    /*
     * Constructor with a vector of parameters that corresponds to:
     *      [fx, fy, cx, cy]
     */
    FishEye(const std::vector<float> _vParameters) : CameraModel(_vParameters) {
        // 4 Camera Intr + 4 Dist Coeff
        assert(vParameters_.size() == 4+4);
    }

    /*
     * Implementation of the FishEye projection function
     */
    void project(const Eigen::Vector3f& p3D, Eigen::Vector2f& p2D);

    /*
     * Implementation of the FishEye unprojection function
     */
    void unproject(const Eigen::Vector2f& p2D, Eigen::Vector3f& p3D);

    /*
     * Implementation of the jacobian matrix of the FishEye projection function
     */
    void projectJac(const Eigen::Vector3f& p3D, Eigen::Matrix<float,2,3>& Jac);

    /*
     * Implementation of the jacobian matrix of the FishEye unprojection function
     */
    void unprojectJac(const Eigen::Vector2f& p2D, Eigen::Matrix<float,3,2>& Jac);

    float solve_theta(float r, float _k1, float _k2, float _k3, float _k4);
};


#endif //JJSLAM_FishEye_H