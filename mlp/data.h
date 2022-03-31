#ifndef NEURAL_NETWORK_DATA_H
#define NEURAL_NETWORK_DATA_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

class Data{
public:
    Data(const std::string& train_path, const std::string& test_path, const int& width, const int& height);

    void get_nextTrainData(std::vector<float>& data, int& label);
    void get_nextTestData(std::vector<float>& data, int& label);

    void resetTrain() { _trainIdx = 0; }
    void resetTest() { _testIdx = 0; }

    int get_trainSize() { return _trains.size(); }
    int get_testSize() { return _tests.size(); }

private:
    /*
    std::vector<std::string> _trainPaths;
    std::vector<int> _trainLabels;
    */
    ///std::vector<std::pair<std::string, int>> _trains;
    std::vector<std::pair<std::vector<float>, int>> _trains;

    /*
    std::vector<std::string> _testPaths;
    std::vector<int> _testLabels;
    */
    ///std::vector<std::pair<std::string, int>> _tests;
    std::vector<std::pair<std::vector<float>, int>> _tests;


    int _trainIdx;
    int _testIdx;

    std::vector<std::string> get_directories(const std::string& s);


};




#endif //NEURAL_NETWORK_DATA_H
