#include "matrix.h"

std::vector<float> multiply(const std::vector<std::vector<float>>& mat1, const std::vector<float>& mat2) {
    std::vector<float> res;
    if (mat1.front().size() != mat2.size()){
        std::cout << mat1.size() << "x" << mat1.front().size() << " * " << "1x" << mat2.size() << "\n";
        throw std::out_of_range("Matrix!!");
    }
	for (auto& row1 : mat1) {
		float r = 0;
		for (size_t i = 0; i < row1.size(); i++) {
			r += row1.at(i) * mat2.at(i);
		}
		res.push_back(r);
	}
	return res;
}

std::vector<float> getColumn(const std::vector<std::vector<float>>& mat, const int& i){
    std::vector<float> res;
    for(size_t j=0; j<mat.size(); j++)
        res.push_back( mat.at(j).at(i) );
    return res;
}
float unitMultiply(const std::vector<float>& vec1, const std::vector<float>& vec2){
    if(vec1.size() != vec2.size())
        throw std::out_of_range("Matrix..");
    float res = 0.0;
    for(size_t i=0; i<vec1.size(); i++)
        res += ( vec1.at(i) * vec2.at(i) );
    return res;
}
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2){
    if (mat1.front().size() != mat2.size()){
        std::cout << mat1.size() << "x" << mat1.front().size() << " * " << mat2.size() << "x" << mat2.front().size() << "\n";
        throw std::out_of_range("Matrix!!");
    }
    std::vector<std::vector<float>> res;
    for(const auto& row1: mat1){
        int repeat = 0, i = 0;
        std::vector<float> rowRes;
        while(repeat < mat2.front().size()){
            rowRes.push_back( unitMultiply(row1, getColumn(mat2, i)) );
            repeat++;
            if(i == mat2.front().size())
                i = 0;
            i++;
        }
        res.push_back(rowRes);
    }
    return res;
}

std::vector<std::vector<float>> multiply(const std::vector<float>& vec1, const std::vector<float>& vec2){
    if(vec1.size() != vec2.size())
        throw std::out_of_range("Multiply!!\n");
    std::vector<std::vector<float>> mat1(vec1.size(), std::vector<float>(1));
    for(size_t i=0; i<mat1.size(); i++){
        mat1.at(i).at(0) = vec1.at(i);
    }
    ///std::cout << "->mat1:" << mat1.size() << "x" << mat1.front().size() << "\n";
    std::vector<std::vector<float>> mat2(1, vec2);
    std::vector<std::vector<float>> res;
    for(const auto& row1: mat1){
        int repeat = 0, i = 0;
        std::vector<float> rowRes;
        while(repeat < mat2.front().size()){
            rowRes.push_back( unitMultiply(row1, getColumn(mat2, i)) );
            repeat++;
            if(i == mat2.front().size())
                i = 0;
            i++;
        }
        res.push_back(rowRes);
    }
    return res;
}
