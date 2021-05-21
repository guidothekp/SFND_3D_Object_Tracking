
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;
using BoundingBoxID = int;
using KeyPointPtr = cv::KeyPoint*;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
 * The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
 * However, you can make this function work for other sizes too.
 * For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
        std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
        std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}

using queryIndex = int; 
using trainIndex = int; 
void convertMatchesToBBoxes(const std::vector<cv::DMatch> & matches, 
        const std::unordered_map<queryIndex, BoundingBoxID> & queryMapping, 
        const std::unordered_map<trainIndex, BoundingBoxID> & trainingMapping, 
        std::map<BoundingBoxID, BoundingBoxID> & bbBestMatches) {
    std::unordered_map<queryIndex, std::vector<cv::DMatch*>*> queryIndexToMatches;
    //assume that the data is not clean and a single queryIndex maps to
    //multiple matches.
    for (const auto & match : matches) {
        queryIndex q = match.queryIdx;
        if (queryMapping.find(q) == queryMapping.end()) {
            continue;
        }
        std::vector<cv::DMatch*>* sv;
        if (queryIndexToMatches.find(q) == queryIndexToMatches.end()) {
            std::vector<cv::DMatch*>* sv = new std::vector<cv::DMatch*>;
            queryIndexToMatches[q] = sv;
        }
        sv = queryIndexToMatches.at(q); //cannot fail
        sv->push_back(const_cast<cv::DMatch*>(&match));
    }
    for (auto it = queryIndexToMatches.begin();
            it != queryIndexToMatches.end(); it ++) {
        assert(it->second->size() <= 1);
    }
    std::cout << "query size = " << queryIndexToMatches.size() << std::endl;
    assert(queryIndexToMatches.size() == queryMapping.size());
    for (auto it = queryMapping.begin(); it != queryMapping.end(); it ++) {
        std::vector<cv::DMatch*>* match = queryIndexToMatches.at(it->first);
        trainIndex t = match->at(0)->trainIdx;
        if (trainingMapping.find(t) == trainingMapping.end()) {
            continue;
        }
        BoundingBoxID qB = it->second;
        BoundingBoxID tB = trainingMapping.at(t);
        bbBestMatches[qB] = tB;
    }
}
std::unordered_map<int, BoundingBoxID> prepareKeyPointBoundingBoxID(
        const DataFrame & frame);

/** 
 * 1. current frame has the matches with the previous one. 
 * 2. both frames also have bounding boxes for each image.
 * 3. we don't know which bounding box corresponds to which.
 * 4. we need to use the matches as the hint to figure out this match.
 * 5. we first need to map keypoints to bounding boxes in each frame. 
 * 6. in each frame, we will have a situation where a keypoint falls in
 * multiple bounding boxes. these we call problematic ones. the keypoints that
 * map to exactly one bounding box will be used immediately.
 * 7. if a keypoint falls in multiple bounding boxes, it gets assigned to the
 * one with the largest (if any) or first one (in case of tie).
 * 8. at this point, we have keypoints to boundingbox mapping.
 * 9. we remove from the matchees object all keypoints that do not have a bounding box.
 * 9. we then translate the matches to bounding boxes. this will give us an
 * object with the boundingbox matches.
 * 10. we fill the map with these matches and return.
 */
void matchBoundingBoxes(const std::vector<cv::DMatch> & matches, 
        std::map<int, int> &bbBestMatches, 
        const DataFrame & prevFrame, const DataFrame & currFrame)
{
    std::unordered_map<int, BoundingBoxID> prevMapping = 
        prepareKeyPointBoundingBoxID(prevFrame); //query
    std::unordered_map<int, BoundingBoxID> currMapping = 
        prepareKeyPointBoundingBoxID(currFrame); //train
    convertMatchesToBBoxes(matches, prevMapping, currMapping, bbBestMatches);
}

std::unordered_map<int, BoundingBoxID> prepareKeyPointBoundingBoxID(
        const DataFrame & frame) {
    const std::vector<cv::KeyPoint> & keyPoints = frame.keypoints;
    const std::vector<BoundingBox> & boundingBoxes = frame.boundingBoxes;
    std::unordered_map<cv::KeyPoint*, int> keyPointToIdx;
    std::unordered_map<cv::KeyPoint*, std::vector<BoundingBox*>*> mapping;
    std::unordered_map<BoundingBox*, size_t> bbCount;
    for (size_t i = 0; i < keyPoints.size(); i ++) {
        const cv::KeyPoint & kp = keyPoints[i];
        keyPointToIdx[const_cast<cv::KeyPoint*>(&kp)] = i;
        for (const BoundingBox & bb : boundingBoxes) {
            if (bb.roi.contains(kp.pt)) {
                cv::KeyPoint* kptr = const_cast<cv::KeyPoint*>(&kp);
                BoundingBox* bbptr = const_cast<BoundingBox*>(&bb);
                if (mapping.find(kptr) == mapping.end()) {
                    std::vector<BoundingBox*>* sb = new std::vector<BoundingBox*>;
                    mapping[kptr] = sb;
                }
                std::vector<BoundingBox*>* sb = mapping.at(kptr);
                sb->push_back(bbptr);
                mapping.at(kptr) = sb;
                if (bbCount.find(bbptr) == bbCount.end()) {
                    bbCount[bbptr] = 0;
                }
                bbCount[bbptr] += 1;
            }
        }
    }
    //at this point we have a map of keypoint to matching bounding boxes.
    std::unordered_map<cv::KeyPoint*, BoundingBoxID> rmap;
    std::vector<cv::KeyPoint*> toRemove;
    std::vector<cv::KeyPoint*> conflicts;
    std::cout << "mapping size = " << mapping.size() << std::endl;
    for (auto it = mapping.begin(); it != mapping.end(); it ++) {
        std::vector<BoundingBox*>* sv = it->second;
        BoundingBox* bbptr = sv->at(0);
        if (sv->size() == 1) {
            rmap[it->first]  = bbptr->boxID;
            toRemove.push_back(it->first);
        } else {
            conflicts.push_back(it->first);
        }
    }
    assert(mapping.size() == toRemove.size() + conflicts.size());
    //we are now left with those that match multiple bounding boxes.
    for (cv::KeyPoint* kp : conflicts) {
        std::vector<BoundingBox*>* sv = mapping.at(kp);
        assert(sv->size() > 1);
        size_t max = 0;
        BoundingBox* maxBox;
        for (auto it2 = sv->begin(); it2 != sv->end(); it2 ++) {
            BoundingBox* bb = *it2;
            size_t count = bbCount.at(bb);
            if (count > max) {
                max = count;
                maxBox = bb;
            }
        }
        rmap[kp] = maxBox->boxID;
        toRemove.push_back(kp);
    }
    assert(mapping.size() == toRemove.size());
    for (auto key : toRemove) {
        std::vector<BoundingBox*>* sv = mapping.at(key); //cannot fail here, use at instead of []
        mapping.erase(key);
        sv->clear();
        delete(sv);
    }
    assert(mapping.size() == 0);
    conflicts.clear();
    toRemove.clear();
    std::unordered_map<int, BoundingBoxID> rv;
    for (auto it = rmap.begin(); it != rmap.end(); it ++) {
        rv[keyPointToIdx[it->first]] = it->second;
    }
    return rv;
}
