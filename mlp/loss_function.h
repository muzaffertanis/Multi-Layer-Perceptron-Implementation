#ifndef NEURAL_NETWORK_LOSS_FUNCTION_H
#define NEURAL_NETWORK_LOSS_FUNCTION_H

#include <vector>
#include <complex>
#include <iostream>

class LossFunc{
public:
    LossFunc() {};
    virtual std::vector<float> get_value(const std::vector<float>& target, const std::vector<float>& out) = 0;
    virtual std::vector<float> get_derivative(const std::vector<float>& target, const std::vector<float>& out) = 0;
};


class MSELoss : LossFunc{
public:
    MSELoss() {};
    virtual std::vector<float> get_value(const std::vector<float>& target, const std::vector<float>& out);
    virtual std::vector<float> get_derivative(const std::vector<float>& target, const std::vector<float>& out);

};

class Softmax : LossFunc{
public:
    Softmax() {}
    virtual std::vector<float> get_value(const std::vector<float>& target, const std::vector<float>& out);
    virtual std::vector<float> get_derivative(const std::vector<float>& target, const std::vector<float>& out);

};

class CrossEntropy : LossFunc{
public:
    CrossEntropy() {}
    virtual std::vector<float> get_value(const std::vector<float>& target, const std::vector<float>& out);
    virtual std::vector<float> get_derivative(const std::vector<float>& target, const std::vector<float>& out);

};



#endif //NEURAL_NETWORK_LOSS_FUNCTION_H
