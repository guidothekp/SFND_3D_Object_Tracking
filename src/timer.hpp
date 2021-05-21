#include <opencv2/core.hpp>

#ifndef __TIMER__
#define __TIMER__

typedef double milliseconds;
struct Timer {
    double start, end;
    Timer () {
        start = (double) cv::getTickCount();
    }
    //return how much time has passed since start
    milliseconds delta() {
        return 1000 * ((double)cv::getTickCount() - start) / cv::getTickFrequency();
    }
};

#endif
