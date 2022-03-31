#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <iostream>

std::vector<float> multiply(const std::vector<std::vector<float>>& mat1, const std::vector<float>& mat2);

std::vector<float> getColumn(const std::vector<std::vector<float>>& mat, const int& i);
float unitMultiply(const std::vector<float>& vec1, const std::vector<float>& vec2);
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);

std::vector<std::vector<float>> multiply(const std::vector<float>& vec1, const std::vector<float>& vec2);

#endif //NEURAL_NETWORK_MATRIX_H