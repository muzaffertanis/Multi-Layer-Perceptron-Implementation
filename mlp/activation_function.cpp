#include "activation_function.h"
#include <iostream>

float Sigmoid::get_value(const float& input){
	return (float)1/((float)1+exp(float(-input)));
	//return input;
}

float Sigmoid::get_derivative(const float& input) {
    /*
	float val = get_value(input);
	std::cout << "AF'--->" << val << " * " << " (1 - " << val << ")\n";
	return val * (1 - val);
*/
    //std::cout << "AF'--->" << input << " * " << " (1 - " << input << ")\n";
    return input * (1.0 - input);
}

std::string Sigmoid::get_name() {
    return "sigmoid";
}