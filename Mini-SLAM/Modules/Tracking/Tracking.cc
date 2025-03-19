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


#include "Tracking.h"

#include "Features/FAST.h"
#include "Features/ORB.h"

#include "Map/KeyFrame.h"
#include "Map/MapPoint.h"
#include "Matching/DescriptorMatching.h"

#include "Optimization/g2oBundleAdjustment.h"

using namespace std;

Tracking::Tracking(){}

Tracking::Tracking(Settings& settings, std::shared_ptr<FrameVisualizer>& visualizer,
                    std::shared_ptr<MapVisualizer>& mapVisualizer, std::shared_ptr<Map> map) {
    currFrame_ = Frame(settings.getFeaturesPerImage(),settings.getGridCols(),settings.getGridRows(),
                       settings.getImCols(),settings.getImRows(), settings.getNumberOfScales(), settings.getScaleFactor(),
                       settings.getCalibration(),settings.getDistortionParameters());
    prevFrame_ = Frame(settings.getFeaturesPerImage(),settings.getGridCols(),settings.getGridRows(),
                       settings.getImCols(),settings.getImRows(),settings.getNumberOfScales(), settings.getScaleFactor(),
                       settings.getCalibration(),settings.getDistortionParameters());

    featExtractor_ = shared_ptr<Feature>(new FAST(settings.getNumberOfScales(),settings.getScaleFactor(),settings.getFeaturesPerImage()*2,20,7));
    descExtractor_ = shared_ptr<Descriptor>(new ORB(settings.getNumberOfScales(),settings.getScaleFactor()));

    vMatches_ = vector<int>(settings.getFeaturesPerImage());

    vPrevMatched_ = vector<cv::Point2f>(settings.getFeaturesPerImage());

    status_ = NOT_INITIALIZED;
    bFirstIm_ = true;
    bMotionModel_ = false;

    monoInitializer_ = MonocularMapInitializer(settings.getFeaturesPerImage(),settings.getCalibration(),settings.getEpipolarTh(),settings.getMinCos());

    visualizer_ = visualizer;
    mapVisualizer_ = mapVisualizer;

    pMap_ = map;

    nLastKeyFrameId = 0;
    nFramesFromLastKF_ = 0;

    bInserted = false;

    settings_ = settings;

    std::cout << "Opening Vocab" << std::endl;
    //  OrbVocabulary voc("ORBvoc.txt"); // XAVI: use this
    OrbVocabulary voc("small_voc.yml.gz");

    dbowDB_ = OrbDatabase(voc, false, 0);                   
}

std::vector<cv::Mat> changeStructure(const cv::Mat &plain) {
    std::vector<cv::Mat> out;
    out.resize(plain.rows);
    for (int i = 0; i < plain.rows; ++i) {
      out[i] = plain.row(i).clone();
    }
    return out;
  }

bool Tracking::doTracking(const cv::Mat &im, Sophus::SE3f &Tcw) {
    currIm_ = im.clone();

    //Update previous frame
    if(status_ != NOT_INITIALIZED)
        prevFrame_.assign(currFrame_);

    currFrame_.setIm(currIm_);

    //Extract features in the current image
    extractFeatures(im);

    visualizer_->drawCurrentFeatures(currFrame_.getKeyPointsDistorted(),currIm_);

    //If no map is initialized, perform monocular initialization
    if(status_ == NOT_INITIALIZED){
        if(monocularMapInitialization()){
            status_ = GOOD;
            Tcw = currFrame_.getPose();

            //Update motion model
            updateMotionModel();

            return true;
        }
        else{
            return false;
        }
    }
    //SLAM is initialized and tracking was good, track new frame
    else if(status_ == GOOD){
        //Mapping may has added/deleted MapPoints
        updateLastMapPoints();
        if(cameraTracking()){
            if(trackLocalMap()){
                //Check if we need to insert a new KeyFrame into the system
                if(needNewKeyFrame()){
                    promoteCurrentFrameToKeyFrame();
                }

                //Update motion model
                updateMotionModel();

                Tcw = currFrame_.getPose();

                visualizer_->drawCurrentFrame(currFrame_);

                return true;
            }
            else{
                status_ = LOST;
                return false;
            }
        }
        else{
            status_ = LOST;
            return false;
        }
    }
    //Camera tracking failed last frame, try to rellocalise
    else{
        //Not implemented yet
        //XAVI: HERE WE DO THE RELOCALIZATION
        // return false;

        // Get the closest keyframe in the database
        std::cout << "ENTERING RELOCALIZATION" << std::endl;

        bool relocSuccess = relocalize();

        if (relocSuccess && trackLocalMap()) {
            // Promote to KeyFrame and update visualization
            std::cout << "Before KeyFRame promotion" << std::endl;
            promoteCurrentFrameToKeyFrame();
            std::cout << "KeyFRame promotion done!" << std::endl;
            updateMotionModel();
            std::cout << "updateMotionModel done!" << std::endl;
            visualizer_->drawCurrentFrame(currFrame_);
            std::cout << "drawCurrentFrame done!" << std::endl;


            cv::waitKey(0);
            return true;
        }

        std::cout << "WARNING: FAILED RELOCALIZATION" << std::endl;
        return false;
    }
}

bool goodMapPoint(
    const Sophus::SE3f& currFramePose,
    const Sophus::SE3f& keyframePose,
    const Eigen::Vector3f& mapPointPos,
    const cv::KeyPoint& currKp,
    const cv::KeyPoint& kfKp,
    const std::shared_ptr<CameraModel>& currCalib,
    const std::shared_ptr<CameraModel>& kfCalib,
    float maxReprojError = 5.991f,
    float minParallaxCos = 0.9998f) 
{
    // Convert to double precision for Eigen operations
    const Sophus::SE3d currPoseD = currFramePose.cast<double>();
    const Sophus::SE3d kfPoseD = keyframePose.cast<double>();
    const Eigen::Vector3d mapPointPosD = mapPointPos.cast<double>();

    // 1. Create non-const copies for camera projection
    Eigen::Vector3d P_curr = currPoseD.inverse() * mapPointPosD;  // Remove const
    Eigen::Vector3d P_kf = kfPoseD.inverse() * mapPointPosD;     // Remove const

    // Front-checking
    if(P_curr.z() <= 0 || P_kf.z() <= 0) return false;

    // 2. Project using non-const positions
    Eigen::Vector2d projCurr, projKF;
    try {
        projCurr = currCalib->project(P_curr);  // Now passes non-const reference
        projKF = kfCalib->project(P_kf);        // Now passes non-const reference
    } catch (const std::exception& e) {
        return false;
    }

    // Convert to OpenCV format
    const cv::Point2f projCurrCV(projCurr.x(), projCurr.y());
    const cv::Point2f projKFCV(projKF.x(), projKF.y());

    // 3. Calculate reprojection errors
    const float errCurr = cv::norm(currKp.pt - projCurrCV);
    const float errKF = cv::norm(kfKp.pt - projKFCV);
    if(errCurr > maxReprojError || errKF > maxReprojError) return false;

    // 4. Parallax check with normalized rays
    const Eigen::Vector3d camCurr = currPoseD.inverse().translation();
    const Eigen::Vector3d camKF = kfPoseD.inverse().translation();
    const Eigen::Vector3d rayCurr = (mapPointPosD - camCurr).normalized();
    const Eigen::Vector3d rayKF = (mapPointPosD - camKF).normalized();
    
    return rayCurr.dot(rayKF) < minParallaxCos;
}

bool Tracking::relocalize(){
    // Get camera matrix
    shared_ptr<CameraModel> calibration = currFrame_.getCalibration();
    float fx = calibration->getParameter(0);
    float fy = calibration->getParameter(1);
    float cx = calibration->getParameter(2);
    float cy = calibration->getParameter(3);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 
        fx, 0,  cx,
        0,  fy, cy,
        0,  0,  1);
    cv::Mat distCoeffs   = cv::Mat::zeros(4, 1, CV_32F);  // or your real distortion

    // Query the DB
    QueryResults ret;
    cv::Mat curr_desc = currFrame_.getDescriptors();
    dbowDB_.query(changeStructure(curr_desc), ret, 4);

    // Transpose the current descriptor, as the following code
    // assumes Ndesc x descDim
    // curr_desc = curr_desc.t();

    bool relocSuccess = false;

    // Iterate through each top candidate
    for (const auto& result : ret) {
        // Retrieve candidate KeyFrame
        std::shared_ptr<KeyFrame> candidateKF = pMap_->getKeyFrames()[result.Id];
        std::vector<std::shared_ptr<MapPoint>> mapPoints = candidateKF->getMapPoints();


        //////////////////////////////////////////////////////////////////////////////
        // NNDR Matching
        //////////////////////////////////////////////////////////////////////////////
        // Collect valid MapPoints and their descriptors from the candidate KeyFrame
        std::vector<std::shared_ptr<MapPoint>> validMapPoints;
        cv::Mat kfDescriptors;
        std::vector<int> originalIndices;
        for (size_t i = 0; i < mapPoints.size(); ++i) {
            auto mp = mapPoints[i];
            if (mp) {
                validMapPoints.push_back(mp);
                originalIndices.push_back(i);
                kfDescriptors.push_back(candidateKF->getDescriptors().row(i));
            }
        }
        if (validMapPoints.empty()) continue;

        // Match current frame descriptors with candidate's MapPoints using ratio test
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(curr_desc, kfDescriptors, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i].size() < 2) continue;
            const cv::DMatch& m1 = knnMatches[i][0];
            const cv::DMatch& m2 = knnMatches[i][1];
            if (m1.distance < 0.9 * m2.distance) {
                goodMatches.push_back(m1);
            }
        }
        if (goodMatches.size() < 4) continue;

        std::cout << "Matching done!" << std::endl;

        //////////////////////////////////////////////////////////////////////////////
        // PnP
        //////////////////////////////////////////////////////////////////////////////
        // Collect 2D-3D correspondences
        std::vector<cv::Point3f> pts3D;
        std::vector<cv::Point2f> pts2D;
        for (const auto& match : goodMatches) {
            // 2D match
            pts2D.push_back(currFrame_.getKeyPoint(match.queryIdx).pt);
            
            // 3D match
            auto& mp = validMapPoints[match.trainIdx];
            Eigen::Vector3f eigenPos = mp->getWorldPosition();
            pts3D.push_back(cv::Point3f(
                eigenPos.x(), 
                eigenPos.y(), 
                eigenPos.z()
            ));
        }

        // Solve PnP using RANSAC
        cv::Mat rvec, tvec, inliers;
        bool pnpSuccess = cv::solvePnPRansac(
            pts3D, pts2D, cameraMatrix, distCoeffs,
            rvec, tvec, false, 100, 8.0, 0.99, inliers
        );
        std::cout << "PnP done!" << std::endl;

        //////////////////////////////////////////////////////////////////////////////
        // Adding MapPoints
        //////////////////////////////////////////////////////////////////////////////
        // Check if PnP was successful with enough inliers
        if (pnpSuccess && inliers.rows >= 50) {
            // SET THE POSE
            // Convert rotation vector to matrix and create Sophus pose
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            Eigen::Matrix3f R_eigen;
            Eigen::Vector3f t_eigen;
            R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
                       R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
                       R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
            t_eigen << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
            Sophus::SE3f Tcw(R_eigen, t_eigen);

            // Update current frame pose and MapPoints
            currFrame_.setPose(Tcw);

            // SET THE MapPoints TO THE FRAME
            for (int i = 0; i < inliers.rows; ++i) {
                int idx = inliers.at<int>(i);
                const auto& match = goodMatches[idx];
                const int origIdx = originalIndices[match.trainIdx];
                if(goodMapPoint(
                    Tcw,                                // Current frame pose
                    candidateKF->getPose(),             // Candidate KF pose
                    validMapPoints[match.trainIdx]->getWorldPosition(),
                    currFrame_.getKeyPoint(match.queryIdx),
                    candidateKF->getKeyPoint(origIdx),  // Original KF keypoint
                    currFrame_.getCalibration(),
                    candidateKF->getCalibration()
                )) {
                    currFrame_.setMapPoint(match.queryIdx, validMapPoints[match.trainIdx]);
                    // pMap->addObservation(pKF->getId(),pMP->getId(),match.queryIdx);
                }
            }
            
            // FINAL COMPROBATIONS & VIZ
            currFrame_.checkAllMapPointsAreGood();
            mapVisualizer_->updateCurrentPose(Tcw);
            std::cout << "PnP succeed!" << std::endl;
            return true; // Exit loop after successful relocalization
        }
    }

    return false;
}

void Tracking::updateLastMapPoints() {
    if(bInserted){
        vector<shared_ptr<MapPoint>> vMps = pMap_->getKeyFrame(nLastKeyFrameId)->getMapPoints();
        Sophus::SE3f Tcw = pMap_->getKeyFrame(nLastKeyFrameId)->getPose();
        prevFrame_.setPose(Tcw);

        for(size_t i = 0; i < vMps.size(); i++){
            if(vMps[i]){
                prevFrame_.setMapPoint(i,vMps[i]);
            }
            else{
                prevFrame_.setMapPoint(i,nullptr);
            }
        }

        bInserted = false;
    }
}

void Tracking::extractFeatures(const cv::Mat &im) {
    //Extracf image features
    featExtractor_->extract(im,currFrame_.getKeyPointsDistorted());

    //Compute descriptors to extracted features
    descExtractor_->describe(im,currFrame_.getKeyPointsDistorted(),currFrame_.getDescriptors());

    //Distribute keys and undistort them
    currFrame_.distributeFeatures();
}

bool Tracking::monocularMapInitialization() {
    //Set first frame received as the reference frame
    if(bFirstIm_){
        monoInitializer_.changeReference(currFrame_.getKeyPoints());
        prevFrame_.assign(currFrame_);

        bFirstIm_ = false;

        visualizer_->setReferenceFrame(prevFrame_.getKeyPointsDistorted(),currIm_);

        for(size_t i = 0; i < vPrevMatched_.size(); i++){
            vPrevMatched_[i] = prevFrame_.getKeyPoint(i).pt;
        }

        return false;
    }

    //Find matches between previous and current frame
    int nMatches = searchForInitializaion(prevFrame_,currFrame_,settings_.getMatchingInitTh(),vMatches_,vPrevMatched_);

    // visualizer_->drawFrameMatches(currFrame_.getKeyPointsDistorted(),currIm_,vMatches_);
    // cv::waitKey(0);

    //If not enough matches found, updtate reference frame
    if(nMatches < 70){
        monoInitializer_.changeReference(currFrame_.getKeyPoints());
        prevFrame_.assign(currFrame_);

        visualizer_->setReferenceFrame(prevFrame_.getKeyPointsDistorted(),currIm_);

        for(size_t i = 0; i < vPrevMatched_.size(); i++){
            vPrevMatched_[i] = prevFrame_.getKeyPoint(i).pt;
        }

        return false;
    }

    //Try to initialize by finding an Essential matrix
    Sophus::SE3f Tcw;
    vector<Eigen::Vector3f> v3DPoints;
    v3DPoints.reserve(vMatches_.capacity());
    vector<bool> vTriangulated(vMatches_.capacity(),false);
    if(!monoInitializer_.initialize(currFrame_.getKeyPoints(), vMatches_, nMatches, Tcw, v3DPoints, vTriangulated)){
        return false;
    }

    //Get map scale
    vector<float> vDepths;
    for(int i = 0; i < vTriangulated.size(); i++){
        if(vTriangulated[i])
            vDepths.push_back(v3DPoints[i](2));
    }

    nth_element(vDepths.begin(),vDepths.begin()+vDepths.size()/2,vDepths.end());
    const float scale = vDepths[vDepths.size()/2];

    //Create map
    Tcw.translation() = Tcw.translation() / scale;

    currFrame_.setPose(Tcw);

    int nTriangulated = 0;

    for(size_t i = 0; i < vTriangulated.size(); i++){
        if(vTriangulated[i]){
            Eigen::Vector3f v = v3DPoints[i] / scale;
            shared_ptr<MapPoint> pMP(new MapPoint(v));

            prevFrame_.setMapPoint(i,pMP);
            currFrame_.setMapPoint(vMatches_[i],pMP);

            pMap_->insertMapPoint(pMP);

            nTriangulated++;
        }
    }

    cout << "Map initialized with " << nTriangulated << " MapPoints" << endl;

    shared_ptr<KeyFrame> kf0(new KeyFrame(prevFrame_));
    shared_ptr<KeyFrame> kf1(new KeyFrame(currFrame_));

    pMap_->insertKeyFrame(kf0);
    pMap_->insertKeyFrame(kf1);

    //Set observations into the map
    vector<shared_ptr<MapPoint>>& vMapPoints = kf0->getMapPoints();
    for(size_t i = 0; i < vMapPoints.size(); i++){
        auto pMP = vMapPoints[i];
        if(pMP){
            //Add observation
            pMap_->addObservation(0,pMP->getId(),i);
            pMap_->addObservation(1,pMP->getId(),vMatches_[i]);
        }
    }

    //Run a Bundle Adjustment to refine the solution
    // std::cout << "Before BA" << std::endl;
    bundleAdjustment(pMap_.get());
    // std::cout << "After BA" << std::endl;

    Tcw = kf1->getPose();
    currFrame_.setPose(Tcw);

    updateMotionModel();

    pLastKeyFrame_ = kf1;
    nLastKeyFrameId = kf1->getId();

    mapVisualizer_->updateCurrentPose(Tcw);

    bInserted = true;

    return true;
}

bool Tracking::cameraTracking() {
    //Set pose estimation for the current frame with the motion model
    Sophus::SE3f currPose;
    if(bMotionModel_){
        currPose = motionModel_ * prevFrame_.getPose();
    }
    else{
        currPose = prevFrame_.getPose();
        bMotionModel_ = true;
    }

    currFrame_.setPose(currPose);

    //Match features between current and previous frame
    int nMatches = guidedMatching(prevFrame_,currFrame_,settings_.getMatchingGuidedTh(),vMatches_,1);

    if(nMatches < 20){
        nMatches = guidedMatching(prevFrame_,currFrame_,settings_.getMatchingGuidedTh(),vMatches_,2);
    }

    //Run a pose optimization
    nFeatTracked_ = poseOnlyOptimization(currFrame_);

    currFrame_.checkAllMapPointsAreGood();

    //Update MapDrawer
    currPose = currFrame_.getPose();
    mapVisualizer_->updateCurrentPose(currPose);

    //We enforce a minimum of 20 MapPoint matches to consider the estimation as good
    return nFeatTracked_ >= 20;
}

bool Tracking::trackLocalMap() {
    //Get local map from the last KeyFrame
    currFrame_.checkAllMapPointsAreGood();

    set<ID> sLocalMapPoints, sLocalKeyFrames, sFixedKeyFrames;
    pMap_->getLocalMapOfKeyFrame(nLastKeyFrameId,sLocalMapPoints,sLocalKeyFrames,sFixedKeyFrames);

    //Keep a record of the already tracked map points
    vector<shared_ptr<MapPoint>>& vTrackedMapPoints = currFrame_.getMapPoints();
    unordered_set<ID> sTrackedMapPoints;
    for(auto pMP : vTrackedMapPoints){
        if(pMP)
            sTrackedMapPoints.insert(pMP->getId());
    }

    //Project local map points into the current Frame
    vector<shared_ptr<MapPoint>> vMapPointsToMatch;
    vMapPointsToMatch.reserve(sLocalKeyFrames.size());
    for(auto nMapPointId : sLocalMapPoints){
        //Check if this local MapPoint is already been tracked
        if(sTrackedMapPoints.count(nMapPointId) != 0){
            continue;
        }

        vMapPointsToMatch.push_back(pMap_->getMapPoint(nMapPointId));
    }


    int nMatches = searchWithProjection(currFrame_,settings_.getMatchingByProjectionTh(),vMapPointsToMatch);


    //Run a pose optimization
    nFeatTracked_ = poseOnlyOptimization(currFrame_);

    currFrame_.checkAllMapPointsAreGood();

    //Update MapDrawer
    Sophus::SE3f currPose = currFrame_.getPose();
    mapVisualizer_->updateCurrentPose(currPose);

    //We enforce a minimum of 20 MapPoint matches to consider the estimation as good
    return nFeatTracked_ >= 20;
}

bool Tracking::needNewKeyFrame() {
    /*
     * Your code for Lab 4 - Task 1 here!
     */
    int max_frames_between_KF = 5;
    int min_feat_tracked = 90;
    nFramesFromLastKF_ += 1;

    std::cout << "nFeatTracked_:\t" << nFeatTracked_ << std::endl;
    std::cout << "nFramesFromLastKF_:\t" << nFramesFromLastKF_ << std::endl;
    if ((nFeatTracked_  < min_feat_tracked) ||
    (nFramesFromLastKF_ > max_frames_between_KF)){
        nFramesFromLastKF_ = 0;
        // std::cout << "Desc size: " << currFrame_.getDescriptors().size() << std::endl;
        // std::cout << "Desc size: " << changeStructure(currFrame_.getDescriptors()).size() << std::endl;
        return true;
    }

    return false;
}

void Tracking::promoteCurrentFrameToKeyFrame() {
    // Add frame to the DB
    dbowDB_.add(changeStructure(currFrame_.getDescriptors()));
    //Promote current frame to KeyFrame
    pLastKeyFrame_ = shared_ptr<KeyFrame>(new KeyFrame(currFrame_));

    //Insert KeyFrame into the map
    pMap_->insertKeyFrame(pLastKeyFrame_);

    //Add all obsevations into the map
    nLastKeyFrameId = pLastKeyFrame_->getId();
    vector<shared_ptr<MapPoint>>& vMapPoints = pLastKeyFrame_->getMapPoints();
    for(int i = 0; i < vMapPoints.size(); i++){
        MapPoint* pMP = vMapPoints[i].get();
        if(pMP)
            pMap_->addObservation(nLastKeyFrameId,pMP->getId(),i);
    }
    pMap_->checkKeyFrame(pLastKeyFrame_->getId());

    bInserted = true;
}

std::shared_ptr<KeyFrame> Tracking::getLastKeyFrame() {
    shared_ptr<KeyFrame> toReturn = pLastKeyFrame_;
    pLastKeyFrame_ = nullptr;

    return toReturn;
}

void Tracking::updateMotionModel() {
    motionModel_ = currFrame_.getPose() * prevFrame_.getPose().inverse();
}
