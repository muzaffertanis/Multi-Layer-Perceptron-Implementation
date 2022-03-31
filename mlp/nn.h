#ifndef NN
#define NN

#include "layer.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

/// neural net
class Net {
public:
    Net(){}
	Net(int input, float learning_rate) 
		: _input(input), _learningRate(learning_rate) 
	{}

	/// insert layer
	void insertLayer(const std::string& layerName, 
					 const int& outs,
					 std::shared_ptr<ActFunc> activationFunc);
    void insertLayer(const std::string& layerName,
                     const int& outs,
                     std::shared_ptr<ActFunc> activationFunc,
                     const std::vector<std::vector<float>>& initialWeights,
                     const std::vector<float>& initBias);

    /// prints
	void printLayers();
	void printParameters();


	/// forward
	std::vector<float> forward(std::vector<float> input);

	/// error calculation
	std::vector<float> loss(const std::vector<float>& targets, const std::vector<float>& outs);

	void setError(const std::vector<float>& error);

	/// backward
    std::vector<std::vector<std::vector<float>>> backward(const std::vector<float>& inputs);

	/// update
	void update(const std::vector<std::vector<std::vector<float>>>& allDeltaWeights);

	/// save
	void save(const std::string& path);

	/// load
	void load(const std::string& path);


private:
	std::vector<Layer> _layers;
    int _input;
	float _learningRate;

};



#endif