#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

#include <random>

class ActFunc {
public:
	ActFunc() {};
	virtual float get_value(const float& input) = 0;
	virtual float get_derivative(const float& input) = 0;
	virtual std::string get_name() = 0;

};

class Sigmoid : public ActFunc {
public:
	Sigmoid() {}
	virtual float get_value(const float& input);
	virtual float get_derivative(const float& input);
    virtual std::string get_name();

};



#endif