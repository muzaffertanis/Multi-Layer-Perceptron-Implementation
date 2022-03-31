#include "layer.h"

void Node::update(const std::vector<float> &deltaWeights) {
    for(size_t i=0; i<_weights.size(); i++){
        _weights.at(i) += deltaWeights.at(i);
    }
}

std::vector<Node> Layer::get_nodes() const { return _nodes; }
int Layer::get_nodesSize() const { return int(_nodes.size()); }

float Layer::activation(const float &value) {return _activationFunction->get_value(value);}
float Layer::activationDerivative(const float &value) {return _activationFunction->get_derivative(value);}

std::vector<float> Layer::forward(const std::vector<float> &input) {
    std::vector<float> result;
    std::vector<std::vector<float>> mat1;
    for(auto& node: _nodes){
        mat1.push_back(node.get_weights());
    }
    result = multiply(mat1, input);
    for(size_t i=0; i<_nodes.size(); i++){
        /// bias
        result.at(i) += _nodes.at(i).get_bias();
        /// result
        result.at(i) = activation( result.at(i) );
        /// _outs -> AF(_ins)
        _outs.at(i) = result.at(i);
    }
    return result;
}

std::vector<float> Layer::calculateErrors(const std::vector<float> &target, const std::vector<float> &outs) {
    if(_nodes.size() != outs.size() || outs.size() != target.size())
        throw std::out_of_range("Sizes must be same!");

    /// loss derivative function
    auto lossFunc = MSELoss();
    //auto lossFunc = CrossEntropy();
    _errors_derives = lossFunc.get_derivative(target, outs);

    /// activation derivative
    for(size_t i=0; i<_nodes.size(); i++) {
        _derivatives.at(i) = (activationDerivative(_outs.at(i)));
    }

    /// return loss
    return lossFunc.get_value(target, outs);
}
void Layer::setErrors(const std::vector<float> &errors) {
    _errors_derives.clear();
    for(size_t i=0; i<_nodes.size(); i++){
        _errors_derives.push_back(errors.at(i));
    }
}

/// output layer
std::vector<std::vector<float>> Layer::backward(const Layer &nextLayer) {
    /// delta weights calculation
    std::vector<std::vector<float>> deltaWeights(_nodes.size()); // [node_id[weight_id]
    for(int i=0; i<_nodes.size(); i++){
        std::vector<float> weights = _nodes.at(i).get_weights();
        deltaWeights.at(i).resize(weights.size());
        for(int j=0; j<weights.size(); j++){
            deltaWeights.at(i).at(j) = _errors_derives.at(i) * get_act_derivative(i) * nextLayer.get_out(j);
        }
    }
    return deltaWeights;
}

/// mid layers
std::vector<std::vector<float>> Layer::backward(const Layer &nextLayer, const Layer &preLayer) {
    /// error calculation
    _errors_derives.clear();
    for(int i=0; i<_nodes.size(); i++){
        float nodeError = 0;
        for(int k=0; k<preLayer.get_nodesSize(); k++){
            nodeError += preLayer.get_act_derivative(k) * preLayer.get_weight(k, i) * preLayer.get_error(k);
        }
        _errors_derives.push_back(nodeError);
        _derivatives.at(i) = ( activationDerivative(_outs.at(i)) );
    }
    /// delta weights calculation
    std::vector<std::vector<float>> deltaWeights(_nodes.size()); // [node_id[weight_id]
    for(int i=0; i<_nodes.size(); i++){
        std::vector<float> weights = _nodes.at(i).get_weights();
        deltaWeights.at(i).resize(weights.size());
        for(int j=0; j<weights.size(); j++){
            deltaWeights.at(i).at(j) = _errors_derives.at(i) * get_act_derivative(i) * nextLayer.get_out(j);
        }
    }
    return deltaWeights;
}

/// first layer
std::vector<std::vector<float>> Layer::backward(const std::vector<float> &inputs, const Layer &preLayer) {
    /// error calculation
    _errors_derives.clear();
    for(int i=0; i<_nodes.size(); i++){
        float nodeError = 0;
        for(int k=0; k<preLayer.get_nodesSize(); k++){
            nodeError += preLayer.get_act_derivative(k) * preLayer.get_weight(k, i) * preLayer.get_error(k);
        }
        _errors_derives.push_back(nodeError);
        _derivatives.at(i) = ( activationDerivative(_outs.at(i)) );
    }
    /// delta weights calculation
    std::vector<std::vector<float>> deltaWeights(_nodes.size());
    for(int i=0; i<_nodes.size(); i++){
        std::vector<float> weights = _nodes.at(i).get_weights();
        deltaWeights.at(i).resize(weights.size());
        for(int j=0; j<weights.size(); j++){
            deltaWeights.at(i).at(j) = get_error(i) * get_act_derivative(i ) * inputs.at(j);
        }
    }
    return deltaWeights;
}

void Layer::update(const std::vector<std::vector<float>> &deltaWeights) {
    for(int i=0; i<_nodes.size(); i++){
        _nodes.at(i).update(deltaWeights.at(i));
    }
}

