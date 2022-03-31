#include "data.h"

Data::Data(const std::string &train_path, const std::string &test_path, const int& width, const int& height)
    : _trainIdx(0), _testIdx(0)
{
    std::vector<std::string> trainMain = get_directories(train_path);
    std::vector<std::string> testMain = get_directories(test_path);
    if(trainMain.size() != testMain.size())
        throw std::out_of_range("Dataset is incorrect!");

    std::sort(trainMain.begin(), trainMain.end());
    std::sort(testMain.begin(), testMain.end());

    int idx = 0;
    for(auto& trainM: trainMain) {
        for (const auto &entry : std::filesystem::directory_iterator(trainM)) {
            cv::Mat img = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
            cv::resize(img, img, cv::Size(width, height));
            std::vector<float> data;
            for(int i=0; i<width; i++){
                for(int j=0; j<height; j++){
                    data.push_back( (float(img.at<u_char>(cv::Point(j, i)))/255.0) * 0.99 + 0.01 );
                }
            }
            _trains.push_back(std::pair<std::vector<float>, int>(data, idx));
        }
        idx++;
    }

    idx = 0;
    for(auto& testM: testMain) {
        for (const auto &entry : std::filesystem::directory_iterator(testM)) {
            cv::Mat img = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
            cv::resize(img, img, cv::Size(width, height));
            std::vector<float> data;
            for(int i=0; i<width; i++){
                for(int j=0; j<height; j++){
                    data.push_back( (float(img.at<u_char>(cv::Point(j, i)))/255.0) * 0.99 + 0.01 );
                }
            }
            _tests.push_back(std::pair<std::vector<float>, int>(data, idx));
        }
        idx++;
    }

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(_tests), std::end(_tests), rng);
    std::shuffle(std::begin(_trains), std::end(_trains), rng);

}

void Data::get_nextTrainData(std::vector<float>& data, int& label) {
    if(_trainIdx < _trains.size()){
        data = _trains.at(_trainIdx).first;
        label = _trains.at(_trainIdx).second;
        _trainIdx++;
    }else{
        label = -1;
    }
}

void Data::get_nextTestData(std::vector<float>& data, int& label) {
    if(_testIdx < _tests.size()){
        data = _tests.at(_testIdx).first;
        label = _tests.at(_testIdx).second;
        _testIdx++;
    }else{
        label = -1;
    }
}

std::vector<std::string> Data::get_directories(const std::string& s) {
    std::vector<std::string> r;
    for(auto& p : std::filesystem::recursive_directory_iterator(s))
        if (p.is_directory())
            r.push_back(p.path().string());
    return r;
}


