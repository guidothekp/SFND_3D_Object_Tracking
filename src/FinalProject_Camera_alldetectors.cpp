
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

struct CameraResult {
    double cameraTTC;
    std::string detector;
    std::string descriptor;
};

struct FrameResult {
    double lidarTTC;
    std::vector<CameraResult> results;
    size_t index;
    size_t runCount;
};

void processResults(const std::vector<FrameResult> & results) {
    for (int i = 0; i < results.size(); i ++) {
        const FrameResult & fr = results[i];
        const auto & cresults = fr.results;
        std::cout << "frame #: " << fr.index;
        std::cout << ", ran = " << fr.runCount;
        std::cout << ", lidarTTC = " << fr.lidarTTC << std::endl;
        for (auto it = cresults.cbegin() ; it != cresults.cend(); it ++) {
            std::cout << "\t(" << it->detector << "-" << it->descriptor << ")";
            std::cout << "," << it->cameraTTC << std::endl;
        }
    }
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */
    std::vector<FrameResult> frameResults; //results by frame
    // data location
    string dataPath = "../";
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        //std::cout << "image index: " << imgIndex << std::endl;
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame = img;
        //frame.cameraImg = img;
        if (dataBuffer.size() == dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);
        if (DEBUG)
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        if (DEBUG)
            cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        if (DEBUG)
            cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            const std::string message = std::to_string(imgIndex);
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), 
                    //cv::Size(2000, 2000), true);
                cv::Size(1200, 1200), message, false);
        }
        bVis = false;

        if (DEBUG)
            cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;


        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        //continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        const std::vector<std::string> detectors = {"SHITOMASI"};
        const std::vector<std::string> descriptors = {"ORB"};
        for (const auto & detectorType : detectors) {
            for (const auto & mainDescriptorType : descriptors) {
                //string detectorType = "SHITOMASI";
                double detectTime = 0;
                bool show = false;
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detectTime = detKeypointsShiTomasi(keypoints, imgGray, show);
                }
                else if (detectorType.compare("HARRIS") == 0) 
                {
                    detectTime = detKeypointsHarris(keypoints, imgGray, show);
                } else {
                    detectTime = detKeypointsModern(keypoints, imgGray, detectorType, show);
                }
                //metrics.detectionTimes.push_back(detectTime);
                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;

                if (DEBUG)
                    cout << "#5 : DETECT KEYPOINTS done" << endl;


                /* EXTRACT KEYPOINT DESCRIPTORS */
                cv::Mat descriptors;
                //string mainDescriptorType = "BRISK";
                string descriptorType = mainDescriptorType; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
                double descriptorTime = 
                    descKeypoints((dataBuffer.end() - 1)->keypoints, 
                            (dataBuffer.end() - 1)->cameraImg, 
                            descriptors, descriptorType);

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                if (DEBUG)
                    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    FrameResult frameResult;
                    frameResult.index = imgIndex;
                    frameResult.runCount = 0;

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
                    string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                            (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                            matches, descriptorType, matcherType, selectorType, true);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    if (DEBUG)
                        cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


                    /* TRACK 3D OBJECT BOUNDING BOXES */

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
                    map<int, int> bbBestMatches;
                    matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                    if (DEBUG)
                        cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                    /* COMPUTE TTC ON OBJECT IN FRONT */

                    // loop over all BB match pairs
                    for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                    {
                        // find bounding boxes associates with current match
                        BoundingBox *prevBB, *currBB;
                        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                            {
                                currBB = &(*it2);
                            }
                        }

                        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                            {
                                prevBB = &(*it2);
                            }
                        }
                        //std::cout << frameResult.index << ", " 
                         //   << frameResult.runCount << std::endl;
                        // compute TTC for current match
                        if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                        {
                            //std::cout << "processing\n" << std::endl;
                            //// STUDENT ASSIGNMENT
                            //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                            double ttcLidar; 
                            computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                            frameResult.lidarTTC = ttcLidar;
                            frameResult.runCount ++;
                            //// EOF STUDENT ASSIGNMENT

                            //// STUDENT ASSIGNMENT
                            //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                            //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                            double ttcCamera;
                            clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                            computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                            CameraResult cr;
                            cr.detector = detectorType;
                            cr.descriptor = mainDescriptorType;
                            cr.cameraTTC = ttcCamera;
                            frameResult.results.push_back(cr);
                            /*
                               std::cout << detectorType << ", " << mainDescriptorType << 
                               ", " <<
                               ttcCamera << ", " << ttcLidar
                               << std::endl;
                               */
                            //// EOF STUDENT ASSIGNMENT

                            bVis = false;
                            if (bVis)
                            {
                                const std::string num = std::to_string(imgIndex);
                                cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                                showLidarImgOverlay(visImg, currBB->lidarPoints, 
                                        P_rect_00, R_rect_00, RT, &visImg);
                                showLidarTopview(currBB->lidarPoints, cv::Size(4.0, 20.0), 
                                        cv::Size(1000, 1000), num, false);
                                cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

                                char str[200];
                                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                                putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                                if (SHOW_WINDOWS) {
                                    string windowName = "Final Results : TTC -- " + num;
                                    cv::namedWindow(windowName, 4);
                                    cv::imshow(windowName, visImg);
                                }
                                cv::imwrite("../results/TTC_" + num + ".jpg", visImg);
                                //cout << "Press key to continue to next frame" << endl;
                                cv::waitKey(0);
                            }
                            bVis = false;

                        } // eof TTC computation
                        else {
                            /*std::cout << "skipped: " << imgIndex << ", ";
                            std::cout << "current lidar count = ";
                            std::cout << currBB->lidarPoints.size();
                            std::cout << ", previous lidar count = ";
                            std::cout <<prevBB->lidarPoints.size();
                            std::cout << std::endl;
                            */
                        }
                    } // eof loop over all BB matches            
                    frameResults.push_back(frameResult);
                }
            }
        }

    } // eof loop over all images
    //std::cout << "press something to end...\n";
    //cv::waitKey(0);
    processResults(frameResults);
    return 0;
}