#include <numeric>
#include "matching2D.hpp"
#include "timer.hpp"

using namespace std;
bool LOG(0);

/*
double toMilliSecond(double time) {
    return 1000 * time;
}
*/

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
        std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, bool crossCheck)
{
    // configure matcher
    //bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        //int normType = cv::NORM_HAMMING;
        int normType = descriptorType.compare("DES_BINARY") == 0 
            ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> matchesList;
        matcher->knnMatch(descSource, descRef, matchesList, 2);
        int accepted(0), total(matchesList.size());
        for (const auto & matchesEntry : matchesList) {
            if (matchesEntry[0].distance < 0.8 * matchesEntry[1].distance) {
                matches.push_back(matchesEntry[0]);
                accepted ++;
            }
        }
        //std::cout << "accepted: " << accepted << " ,discarded: " << (total - accepted) << ", % discarded: " \
            << (100.0 * accepted/total) << std::endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (!descriptorType.compare("ORB")) 
    {
        extractor = cv::ORB::create();
        
    } 
    else if (!descriptorType.compare("BRIEF")) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (!descriptorType.compare("AKAZE")) {
        extractor = cv::AKAZE::create();
    } else if (!descriptorType.compare("FREAK")) {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (!descriptorType.compare("SIFT")) {
        extractor = cv::SIFT::create();
    }

    // perform feature description
    //double t = (double)cv::getTickCount();
    Timer timer; 
    extractor->compute(img, keypoints, descriptors);
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    if (!descriptorType.compare("SIFT")) {
         descriptors.convertTo(descriptors, CV_8U);
    }
    milliseconds t = timer.delta();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    if(LOG)
    cout << descriptorType << " descriptor extraction in " << t  << " ms" << endl;
    return t;
}

void showKeyPoints(const cv::Mat & img, const std::vector<cv::KeyPoint> & keypoints, 
        const std::string & windowName, const bool show) {
    if (!show) {
        return;
    }
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

void applyNonMaximumSuppression(const cv::Mat & img, std::vector<cv::KeyPoint> & keypoints, 
        const int & minResponse, const double & maxOverlap, const int & apertureSize) {
    for (size_t row = 0; row < img.rows; row ++) {
        for (size_t column = 0; column < img.cols; column ++) {
            int response = (int) img.at<float>(row, column);
            if (response <= minResponse) continue;
            cv::KeyPoint newKeyPoint(cv::Point2f(column, row), 2 * apertureSize, -1, response);
            bool overlap = false;
            for (auto keyPoint : keypoints) {
                double keypointOverlap = cv::KeyPoint::overlap(newKeyPoint, keyPoint);
                if (keypointOverlap <= maxOverlap) continue;
                overlap = true;
                if (response > keyPoint.response) {
                    keyPoint = newKeyPoint;
                    break;
                }
            }
            if (overlap) continue;
            keypoints.push_back(newKeyPoint);
        }
    }
}

double detKeypointsHarris(std::vector<cv::KeyPoint> & keypoints, const cv::Mat & img, bool bVis) {
    int blockSize = 2;
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    Timer timer;
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::noArray());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    double maxOverlap = 0.0;
    applyNonMaximumSuppression(dst_norm, keypoints, minResponse, maxOverlap, apertureSize);
    milliseconds t = timer.delta();
    //cout << "Harris keypoint detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if(LOG)
    cout << "Harris keypoint detection with n=" << keypoints.size() << " keypoints in " << t << " ms" << endl;
    showKeyPoints(img, keypoints, "Harris Corner Detector Results", bVis);
    return t;
}


double detKeypointsModern(std::vector<cv::KeyPoint> & keypoints, 
        const cv::Mat & img, const std::string & detectorType, bool bVis) {
    Timer timer;
    if (!detectorType.compare("BRISK")) {
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    } else if (!detectorType.compare("FAST")) {
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        detector->detect(img, keypoints);
    } else if (!detectorType.compare("AKAZE")) {
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        detector->detect(img, keypoints);
    } else if (!detectorType.compare("ORB")) {
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        detector->detect(img, keypoints);
    } else if (!detectorType.compare("SIFT")) {
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        detector->detect(img, keypoints);
    }
    milliseconds t = timer.delta();
    //cout << detectorType << " keypoint detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if(LOG)
     cout << detectorType << " keypoint detection with n=" << keypoints.size() << " keypoints in " << t << " ms" << endl;
    showKeyPoints(img, keypoints, detectorType + " Corner Detector Results", bVis);
    return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, const cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    //double t = (double)cv::getTickCount();
    Timer timer;
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    milliseconds t = timer.delta();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if(LOG)
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << t << " ms" << endl;

        std::string windowName = "Shi-Tomasi Corner Detector Results";
        showKeyPoints(img, keypoints, windowName, bVis);
    // visualize results
    /*
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }*/
    return t;
}
