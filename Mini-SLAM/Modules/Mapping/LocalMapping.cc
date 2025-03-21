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

#include "Mapping/LocalMapping.h"
#include "Optimization/g2oBundleAdjustment.h"
#include "Matching/DescriptorMatching.h"
#include "Utils/Geometry.h"
#include <unordered_set>

using namespace std;

LocalMapping::LocalMapping() {

}

LocalMapping::LocalMapping(Settings& settings, std::shared_ptr<Map> pMap) {
    settings_ = settings;
    pMap_ = pMap;
}

void LocalMapping::doMapping(std::shared_ptr<KeyFrame> &pCurrKeyFrame, trackingResult* trackingRes) {
    //Keep input keyframe
    currKeyFrame_ = pCurrKeyFrame;

    if(!currKeyFrame_)
        return;

    //Remove redundant MapPoints
    mapPointCulling(trackingRes);

    //Triangulate new MapPoints
    triangulateNewMapPoints(trackingRes);

    checkDuplicatedMapPoints();

    //Run a local Bundle Adjustment
    localBundleAdjustment(pMap_.get(),currKeyFrame_->getId());
}

void LocalMapping::mapPointCulling(trackingResult* trackingRes) {
    /*
     * Your code for Lab 4 - Task 4 here!
     */
    // return;
    int min_n_obs = 2;
    int min_n_keyframes = 5;

    int n_keyframes = pMap_->getKeyFrames().size();

    long removed_points = 0;

    if (n_keyframes > min_n_keyframes){
        auto vMapPoints = pMap_->getMapPoints(); // Use reference to avoid copies
        std::unordered_set<ID> points_to_remove;

        // Iterate using range-based for loop
        for (auto pair : vMapPoints) {
            // std::cout << "Inside Loop " << std::endl;
            auto* pMP = pair.second.get();
            if (!pMP) continue; // Skip if nullptr (safety check)

            int n_obs = pMap_->getNumberOfObservations(pMP->getId());
            // std::cout << "n_obs: " << n_obs << std::endl;
            if (n_obs < min_n_obs) {
                // std::cout << "Inside removeMapPoint " << std::endl;
                // std::cout << "pMP->getId() " << pMP->getId() << std::endl;
                points_to_remove.insert(pMP->getId());
                std::cout << "HII!!!!!\n";
            }
        }

        removed_points = static_cast<double>(points_to_remove.size()) / static_cast<double>(vMapPoints.size());
        for (ID point : points_to_remove) {
            pMap_->removeMapPoint(point);
        }
        
    }

    trackingRes->culledPoints = removed_points;
}

void LocalMapping::triangulateNewMapPoints(trackingResult* trackingRes) {
    //Get a list of the best covisible KeyFrames with the current one
    vector<pair<ID,int>> vKeyFrameCovisible = pMap_->getCovisibleKeyFrames(currKeyFrame_->getId());

    vector<int> vMatches(currKeyFrame_->getMapPoints().size());

    //Get data from the current KeyFrame
    shared_ptr<CameraModel> calibration1 = currKeyFrame_->getCalibration();
    Sophus::SE3f T1w = currKeyFrame_->getPose();

    int nTriangulated = 0;
    long pointsBehind = 0;
    long highError = 0;
    long lowParallax = 0;
    long totalPoints = 0;
    for(pair<ID,int> pairKeyFrame_Obs : vKeyFrameCovisible){
        int commonObservations = pairKeyFrame_Obs.second;
        if(commonObservations < 20)
            continue;

        shared_ptr<KeyFrame> pKF = pMap_->getKeyFrame(pairKeyFrame_Obs.first);
        if(pKF->getId() == currKeyFrame_->getId())
            continue;

        //Check that baseline between KeyFrames is not too short
        Eigen::Vector3f vBaseLine = currKeyFrame_->getPose().inverse().translation() - pKF->getPose().inverse().translation();
        float medianDepth = pKF->computeSceneMedianDepth();
        float ratioBaseLineDepth = vBaseLine.norm() / medianDepth;

        if(ratioBaseLineDepth < 0.01){
            continue;
        }

        Sophus::SE3f T2w = pKF->getPose();

        Sophus::SE3f T21 = T2w*T1w.inverse();
        Eigen::Matrix<float,3,3> E = computeEssentialMatrixFromPose(T21);

        //Match features between the current and the covisible KeyFrame
        //TODO: this can be further improved using the orb vocabulary
        int nMatches = searchForTriangulation(currKeyFrame_.get(),pKF.get(),settings_.getMatchingForTriangulationTh(),
                settings_.getEpipolarTh(),E,vMatches);

        vector<cv::KeyPoint> vTriangulated1, vTriangulated2;
        vector<int> vMatches_;
        //Try to triangulate a new MapPoint with each match
        for(size_t i = 0; i < vMatches.size(); i++){
            if(vMatches[i] != -1){
                /*
                 * Your code for Lab 4 - Task 2 here!
                 * Note that the last KeyFrame inserted is stored at this->currKeyFrame_
                 */
                // float min_reprError = 5.991;
                totalPoints += 1;
                float min_reprError = 1.0;
                float minParallaxCos = 0.9998; // From YAML files

                // 1. Get the matched keypoints in each keyframe
                int idxMatch = vMatches[i];
                const cv::KeyPoint& kp1 = currKeyFrame_->getKeyPoint(i);
                const cv::KeyPoint& kp2 = pKF->getKeyPoint(idxMatch);

                // 2. Convert pixel coordinates to normalized rays using each camera’s calibration
                Eigen::Vector3f ray1 = calibration1->unproject(kp1.pt.x, kp1.pt.y);
                shared_ptr<CameraModel> calibration2 = pKF->getCalibration();
                Eigen::Vector3f ray2 = calibration2->unproject(kp2.pt.x, kp2.pt.y);
                
                // 3. Triangulate the 3D point in world coordinates
                Eigen::Vector3f p3D, p3D_c1, p3D_c2;
                triangulate(ray1, ray2, T1w, T2w, p3D);
                p3D_c1 = T1w * p3D;
                p3D_c2 = T2w * p3D;

                //Check that the point has been triangulated in front of the cameras (possitive depth)
                if((p3D_c1(2) < 0.0f) || (p3D_c2(2) < 0.0f)){
                    pointsBehind += 1;
                    continue; // Point behind at least one camera
                }

                // 5. Check reprojection error in each keyframe:
                cv::Point2f uv1 = calibration1->project(p3D_c1);
                cv::Point2f uv2 = calibration2->project(p3D_c2);
                cv::Point2f kp1Copy = kp1.pt;
                cv::Point2f kp2Copy = kp1.pt;
                float repError_c1 = squaredReprojectionError(kp1Copy,uv1) > min_reprError;
                float repError_c2 = squaredReprojectionError(kp2Copy,uv2) > min_reprError;
                if((repError_c1 > min_reprError) || (repError_c2 > min_reprError))
                {
                    highError += 1;
                    continue;  // Reprojection error too large
                }

                // 6. Check the parallax of the triangulated point
                Eigen::Vector3f normal1 = p3D_c1;
                Eigen::Vector3f normal2 = p3D_c1 - (T21.inverse().translation());
                float cosParallaxPoint = cosRayParallax(normal1,normal2);
                if(cosParallaxPoint < minParallaxCos)
                {
                    lowParallax += 1;
                    continue;  // Reprojection error too large
                }

                // 7. Create and add a new MapPoint if everything is valid
                shared_ptr<MapPoint> pMP(new MapPoint(p3D));

                // 8. Register observations in both keyframes
                currKeyFrame_->setMapPoint(static_cast<int>(i), pMP);
                pKF->setMapPoint(idxMatch, pMP);
                pMap_->insertMapPoint(pMP);

                // 9. Register observations in map
                pMap_->addObservation(currKeyFrame_->getId(), pMP->getId(), static_cast<int>(i));
                pMap_->addObservation(pKF->getId(), pMP->getId(), idxMatch);
                nTriangulated += 1;

            }
        }
    }

    trackingRes->pointsBehind += pointsBehind;
    trackingRes->highError += highError;
    trackingRes->lowParallax += lowParallax;
    trackingRes->nTriangulated += nTriangulated;
    trackingRes->totalPoints += totalPoints;
}

void LocalMapping::checkDuplicatedMapPoints() {
    vector<pair<ID,int>> vKFcovisible = pMap_->getCovisibleKeyFrames(currKeyFrame_->getId());
    vector<shared_ptr<MapPoint>> vCurrMapPoints = currKeyFrame_->getMapPoints();

    for(int i = 0; i < vKFcovisible.size(); i++){
        if(vKFcovisible[i].first == currKeyFrame_->getId())
            continue;
        int nFused = fuse(pMap_->getKeyFrame(vKFcovisible[i].first),settings_.getMatchingFuseTh(),vCurrMapPoints,pMap_.get());
        pMap_->checkKeyFrame(vKFcovisible[i].first);
        pMap_->checkKeyFrame(currKeyFrame_->getId());
    }
}
