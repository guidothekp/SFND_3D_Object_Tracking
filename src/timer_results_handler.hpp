#include "interfaces.hpp"
#include "metrics.hpp"

#include <vector>
#include <string>

namespace tdUtils {
    struct TimerData {
        std::string detector;
        std::string descriptor;
        std::string subType;
        SummaryStatistics<double> detectorStatistics;
        SummaryStatistics<double> descriptorStatistics;
        Sum<double> sum;
    };

    std::string createHeaders() {
        std::string rv;
        rv += "DETECTOR";
        rv = rv + " ," + "DESCRIPTOR";
        //rv = rv + " , " + "SUBTYPE";
        const std::vector<std::string> types
        {"DETECTION_TIME", "DESCRIPTOR_TIME", "TOTAL_TIME"};
        const std::vector<std::string> subTypes
        { "MEAN", "SD"};
        for (auto t : types) {
            for (auto s : subTypes) {
                rv = rv + ", " + t + "_" + s;
            }
        }
        return rv;
    };

    TimerData convertMetricsToTimerData(const Metrics & metrics) {
        TimerData td;
        //td.subType = metrics.matcherSubType;
        td.detector = metrics.detectorType;
        td.descriptor = metrics.descriptorType;
        td.detectorStatistics = statistics(metrics.detectionTimes);
        td.descriptorStatistics = statistics(metrics.descriptorTimes);
        td.sum.statistics({td.detectorStatistics, td.descriptorStatistics});
        return td;
    }

    template<typename T>
        void mergeVectors(std::vector<T> & to, 
                const std::vector<T> & from) {
            std::copy(from.cbegin(), from.cend(), std::back_inserter(to));
        }

    //when running keypoint and descriptor detections, we get multiple
    //measures because the same code runs when we use various matching
    //methods.
    std::vector<Metrics> merge(const std::vector<Metrics> & list) {
        std::vector<Metrics> rv;
        std::map<std::string, std::vector<Metrics>> map;
        for (auto iter = list.cbegin(); iter != list.cend(); iter ++) {
            std::string key = iter->detectorType + "_" + iter->descriptorType;
            if (map.find(key) == map.end()) {
                map[key] = std::vector<Metrics>{};
            } 
            map[key].push_back(*iter);
        }
        for (auto iter = map.cbegin(); iter != map.cend(); iter ++) {
            std::vector<Metrics> v = iter->second;
            Metrics m = v[0];
            for (int i = 1; i < v.size(); i ++) {
                Metrics mm = v[i];
                mergeVectors(m.detectionTimes, mm.detectionTimes);
                mergeVectors(m.detectionTimes, mm.detectionTimes);
            }
            rv.push_back(m);
        }
        return rv;
    }

    std::vector<Metrics> filter(const std::vector<Metrics> & list) {
        std::vector<Metrics> rv;
        std::map<std::string, Metrics> map;
        for (auto iter = list.cbegin(); iter != list.cend(); iter ++) {
            std::string key = iter->detectorType + "_" + iter->descriptorType;
            if (map.find(key) == map.end()) {
                map[key] = *iter;
            }
        }
        std::transform(map.cbegin(), map.cend(), std::back_inserter(rv), 
                [](std::pair<std::string, Metrics> pair) 
                {return pair.second;});
        return rv;
    }

    std::vector<TimerData> prepareData(const std::vector<Metrics> & metricsList) {
        const std::vector<Metrics> ml = merge(metricsList);
        std::vector<TimerData> data(ml.size());
        std::transform(ml.cbegin(), ml.cend(), data.begin(),
                convertMetricsToTimerData);
        sort(data.begin(), data.end(), 
                [](const TimerData & t1, const TimerData & t2) {
                return t1.sum.avg < t2.sum.avg;
                });
        return data;
    }

    std::string timerDataToString(const TimerData & td) {
        std::string str;
        str += td.detector;
        str += ", ";
        str += td.descriptor;
        str += ", ";
        //str += td.subType;
        //str += ", ";
        str += td.detectorStatistics.to_string();
        str += ", ";
        str += td.descriptorStatistics.to_string();
        str += ", ";
        str += td.sum.to_string();
        return str;
    }

    std::vector<std::string> convertToString(const std::vector<TimerData> & list) {
        std::vector<std::string> rv;//(list.size() + 1);
        rv.push_back(createHeaders());
        std::transform(list.cbegin(), list.cend(), std::back_inserter(rv), timerDataToString);
        return rv;
    }

    void printData(const std::vector<TimerData> & list, 
            const std::string & file) {
        std::vector<std::string> strings = convertToString(list);
        std::ofstream out;
        out.open(file);
        for (auto str : strings) 
            out << str << std::endl;
        out.close();
    }
};

class timerResultsHandler : public IHandler {
    public:
        void handle(
                const std::vector<Metrics> & metricsList,
                const std::string & filename) {
            std::vector<tdUtils::TimerData> timerData = tdUtils::prepareData(metricsList);
            tdUtils::printData(timerData, filename);
        };
        ~timerResultsHandler() {}
};
