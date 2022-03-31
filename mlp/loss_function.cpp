#include "loss_function.h"

/// MSE
std::vector<float> MSELoss::get_value(const std::vector<float> &target, const std::vector<float> &out) {
    std::vector<float> res(target.size());
    for(int i=0; i<target.size(); i++){
        res.at(i) = (target.at(i) - out.at(i)) * (target.at(i) - out.at(i)) * 0.5 ;
    }
    return res;
}

std::vector<float> MSELoss::get_derivative(const std::vector<float> &target, const std::vector<float> &out) {
    std::vector<float> res(target.size());
    for(int i=0; i<target.size(); i++){
        res.at(i) = ( out.at(i) - target.at(i) );
    }
    return res;
}

/// Softmax
std::vector<float> Softmax::get_value(const std::vector<float> &target, const std::vector<float> &out) {
    double max = *std::max_element(out.begin(), out.end());

    double sum = 0;
    for(auto &v: out){ sum += (std::exp(v-max)); }

    std::vector<float> props;
    for(auto &val: out){
        props.push_back( std::exp(val - max - std::log(sum)) );
    }

    return props;
}

std::vector<float> Softmax::get_derivative(const std::vector<float> &target, const std::vector<float> &out) {
    std::vector<float> res = get_value(target, out);
    for(auto& r: res){
        r = r*(1-r);
    }
    return res;
}

/// CrossEntropy
std::vector<float> CrossEntropy::get_value(const std::vector<float> &target, const std::vector<float> &out) {
    /// Softmax props
    double max = *std::max_element(out.begin(), out.end());
    double sum = 0;
    for(auto &v: out){ sum += (std::exp(v-max)); }
    std::vector<float> props;
    for(auto &val: out)
        props.push_back( std::exp(val - max - std::log(sum)) );

    /// Loss
    std::vector<float> res(target.size());
    for(int i=0; i<target.size(); i++){
        res.at(i) = ( (-1) * target.at(i) * std::log(props.at(i)) );
    }
    return res;
}

std::vector<float> CrossEntropy::get_derivative(const std::vector<float> &target, const std::vector<float> &out) {
    std::vector<float> res = Softmax().get_value(target, out);
    for(size_t i=0; i<target.size(); i++){
        res.at(i) = res.at(i) - target.at(i);
    }
    return res;
}


