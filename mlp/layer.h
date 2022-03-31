#ifndef LAYER
#define LAYER

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>

#include "activation_function.h"
#include "matrix.h"
#include "loss_function.h"

class Node {
public:
    Node() = default;
    explicit Node(const int& ins) {
		/// weight initialization
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(-0.3, 0.3);
        for(size_t i=0; i<ins; i++){
            _weights.push_back(dist(mt));
        }
		/// bias initialization
		_bias = dist(mt);
    }
    explicit Node(const int& ins, const std::vector<float>& initialWeights, const float& bias=0) {
        /// set weights
        for(size_t i=0; i<ins; i++){
            _weights.push_back(initialWeights.at(i));
        }
        /// set biases
        _bias = bias;
    }
    std::vector<float> get_weights() const { return _weights; }
	float get_bias() const { return _bias; }

    void update(const std::vector<float>& deltaWeights);

private:
    std::vector<float> _weights;
	float _bias;

};

class Layer {
public:
    /// layerName, layerID, inputNumber, nodeNumber, actFunc
	Layer(std::string layerName,
		  const int& layerID,
		  const int& inputNumber,
		  const int& nodeNumber,
		  std::shared_ptr<ActFunc> actFunc)
		: _layerName(std::move(layerName)), _layerID(layerID), _activationFunction(actFunc)
	{
        for(size_t i=0; i<nodeNumber; i++)
            _nodes.push_back(Node(inputNumber));

        _outs.resize(nodeNumber);
        _derivatives.resize(nodeNumber);
    }
    Layer(std::string layerName,
          const int& layerID,
          const int& inputNumber,
          const int& nodeNumber,
          std::shared_ptr<ActFunc> actFunc,
          const std::vector<std::vector<float>>& initialWeights,
          const std::vector<float>& initBias)
            : _layerName(std::move(layerName)), _layerID(layerID), _activationFunction(actFunc)
    {
        for(size_t i=0; i<nodeNumber; i++)
            _nodes.push_back(Node(inputNumber, initialWeights.at(i), initBias.at(i)));

        _outs.resize(nodeNumber);
        _derivatives.resize(nodeNumber);
    }

    /// Getter
	std::string get_layerName() const { return _layerName; }
	std::vector<Node> get_nodes() const;
	int get_nodesSize() const;
	std::string get_activation_name() const { return _activationFunction->get_name(); }

	float get_out(const int& nodeID) const { return _outs.at(nodeID); }
	float get_error(const int& nodeID) const { return _errors_derives.at(nodeID); }

	float get_act_derivative(const int& nodeID) const { return _derivatives.at(nodeID); }
	float get_weight(const int& nodeID, const int& wID) const { return _nodes.at(nodeID).get_weights().at(wID); }

	/// forward
	std::vector<float> forward(const std::vector<float>& input);

	/// calculates final layer errors
	std::vector<float> calculateErrors(const std::vector<float>& target, const std::vector<float>& outs);
	void setErrors(const std::vector<float>& errors); /// batch training

    std::vector<std::vector<float>> backward(const Layer& nextLayer);
    std::vector<std::vector<float>> backward(const Layer& nextLayer, const Layer& preLayer);
    std::vector<std::vector<float>> backward(const std::vector<float>& inputs, const Layer& preLayer);

    /// update weights
    void update(const std::vector<std::vector<float>>& deltaWeights);


private:
	const std::string _layerName;
	const int _layerID;
	std::vector<Node> _nodes;
	std::shared_ptr<ActFunc> _activationFunction;

	std::vector<float> _outs;           /// _ins_outs[node_id(i)]    = vector of ins and outs       -> { in(i), out(i) } [out(i) = AF(in(i))]
	std::vector<float> _derivatives;    /// _derivatives[node_id(i)] = the derivative of act. func. -> { dout(i)/din(i) }
	std::vector<float> _errors_derives; /// _errors[j]               = error                        -> { dEo1/dout(i) }
    /// Size: [node_size][next_layer_out_size]
    /// { dEo1/dout(1), dEo2/dout(1) } node_1
    /// { dEo1/dout(2), dEo2/dout(2) } node_2
    /// { dEo1/dout(3), dEo2/dout(3) } node_3

	/// Activation function stuffs
	float activation(const float& value);
	float activationDerivative(const float& value);

};





#endif