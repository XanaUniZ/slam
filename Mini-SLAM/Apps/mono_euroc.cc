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
 * A demo showing the Mini-SLAM library processing a sequence of the EuRoC dataset
 */

#include "DatasetLoader/EurocVisualLoader.h"
#include "System/MiniSLAM.h"
#include "Tracking/Tracking.h"
  #include <opencv2/core.hpp>

#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv){
    //Check program parameters are good
    if(argc != 3){
        cerr << "[Error]: you need to invoke the program with 2 parameters: " << endl;
        cerr << "\t./mono_euroc <dataset_path> <timestamps_file>" << endl;
        cerr << "Finishing execution..." << endl;
        return -1;
    }

    //Load dataset sequence
    string datasetPath = argv[1];
    string timestampsFile = argv[2];
    EurocVisualLoader sequence(datasetPath, timestampsFile, datasetPath + "/mav0/state_groundtruth_estimate0/data.csv");

    //Create SLAM system
    MiniSLAM SLAM("Data/EuRoC.yaml");

    //File to store the trajectory
    ofstream trajectoryFile ("trajectory.txt");
    if (!trajectoryFile.is_open()){
        cerr << "[Error]: could not open the trajectory file, aborting..." << endl;
        return -2;
    }

    //Process the sequence
    cv::Mat currIm;
    double currTs;
    trackingResult trackRes;
    initTrackingRes(&trackRes);
    
    std::vector<double> behindVect, errorVect, parallaxVect, triangVect;
    std::vector<long> culledVect; 

    for(int i = 200; i < sequence.getLenght(); i++){
    // for(int i = 500; i < sequence.getLenght(); i++){
        sequence.getLeftImage(i,currIm);
        sequence.getTimeStamp(i,currTs);

        Sophus::SE3f Tcw;
        if(SLAM.processImage(currIm, Tcw, currTs, &trackRes)){
            trackRes.nFrames += 1;
            Sophus::SE3f Twc = Tcw.inverse();
            //Save predicted pose to the file
            trajectoryFile << setprecision(17) << currTs*1e9 << "," << setprecision(7) << Twc.translation()(0) << ",";
            trajectoryFile << Twc.translation()(1) << "," << Twc.translation()(2) << ",";
            trajectoryFile << Twc.unit_quaternion().x() << "," << Twc.unit_quaternion().y() << ",";
            trajectoryFile << Twc.unit_quaternion().z() << "," << Twc.unit_quaternion().w() << endl;
        }
        // printTrackingRes(trackRes); 
        std::cout << "\033[1;32mNumber of Frames: \033[0m" << trackRes.nFrames << std::endl;
        std::cout << "\033[1;32mNumber of KeyFrames: \033[0m" << trackRes.nKeyframes << std::endl;
        std::cout << "\033[1;32mNumber of MapPoints: \033[0m" << trackRes.nMapPoints << std::endl;
        if (trackRes.isKF){
            behindVect.push_back(trackRes.pctPointsBehind);
            errorVect.push_back(trackRes.pcthighError);
            parallaxVect.push_back(trackRes.pctlowParallax);
            triangVect.push_back(trackRes.pctnTriangulated);
            culledVect.push_back(trackRes.culledPoints);

            resetTrackingRes(&trackRes);

            std::cout << "\033[1;31mRemoved Behind-> \033[0m" << "\033[1;31mMean: \033[0m" << setprecision(4) << calculateMean(behindVect) << "\033[1;31m Std: \033[0m" << setprecision(4) << calculateStdev(behindVect) << std::endl;
            std::cout << "\033[1;31mRemoved High Error-> \033[0m" << "\033[1;31mMean: \033[0m" << setprecision(4) << calculateMean(errorVect) << "\033[1;31m Std: \033[0m" << setprecision(4) << calculateStdev(errorVect) << std::endl;
            std::cout << "\033[1;31mRemoved Low Par-> \033[0m" << "\033[1;31mMean: \033[0m" << setprecision(4) << calculateMean(parallaxVect) << "\033[1;31m Std: \033[0m" << setprecision(4) << calculateStdev(parallaxVect) << std::endl;
            std::cout << "\033[1;31mRemoved Added-> \033[0m" << "\033[1;31mMean: \033[0m" << setprecision(4) << calculateMean(triangVect) << "\033[1;31m Std: \033[0m" << setprecision(4) << calculateStdev(triangVect) << std::endl;
            std::cout << "\033[1;31mCulled-> \033[0m" << "\033[1;31mMean: \033[0m" << setprecision(4) << calculateMean(culledVect) << "\033[1;31m Std: \033[0m" << setprecision(4) << calculateStdev(culledVect) << std::endl;
        }
    }

    trajectoryFile.close();
    cv::waitKey(0);

    return 0;
}