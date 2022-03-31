#include "nn.h"

void Net::insertLayer(const std::string& layerName,
				      const int& outs,
				      std::shared_ptr<ActFunc> activationFunc)
{
	/// random weights and bias initialization!!
	if(_layers.empty()){
	    Layer newLayer(layerName, 0, _input, outs, activationFunc);
		_layers.push_back(newLayer);
	}else{
        Layer newLayer(layerName, int(_layers.size()), int(_layers.back().get_nodesSize()), outs, activationFunc);
		_layers.push_back(newLayer);
    }
}
void Net::insertLayer(const std::string &layerName, const int &outs, std::shared_ptr<ActFunc> activationFunc,
                      const std::vector<std::vector<float>> &initialWeights, const std::vector<float>& initBias) {
    if(_layers.empty()){
        Layer newLayer(layerName, 0, _input, outs, activationFunc, initialWeights, initBias);
        _layers.push_back(newLayer);
    }else{
        Layer newLayer(layerName, int(_layers.size()), int(_layers.back().get_nodesSize()), outs, activationFunc, initialWeights, initBias);
        _layers.push_back(newLayer);
    }
}

void Net::printLayers() {
	for (auto& layer : _layers) {
		std::cout << layer.get_layerName() << "\n";
	}
}

void Net::printParameters() {
    std::cout << "\n";
    std::cout << "input: " << _input << "\n";
    for(auto& layer : _layers){
        std::cout << "layer: " << layer.get_layerName() << "\n";
        int node_id = 1;
        for(auto& node: layer.get_nodes()){
            std::cout << "Node id: " << node_id << " ||\t";
            int in = 1;
            for(auto& weight : node.get_weights()){
                std::cout << "weight(" << node_id << "," << in++ << ") = " << weight << "\t";
            }
            std::cout << "\n";
            node_id++;
        }
        std::cout << "\n";
    }
}

std::vector<float> Net::forward(std::vector<float> input) {
	if (_input != input.size()) {
		throw std::out_of_range("Invalid number of input!");
	}
    for(auto& layer: _layers){
        input = layer.forward(input);
    }
    return input;
}

std::vector<float> Net::loss(const std::vector<float> &targets, const std::vector<float>& outs) {
    /// check
    if (targets.size() != outs.size())
        throw std::out_of_range("Size of the target and output must be same!");
    if (_layers.back().get_nodesSize() != outs.size())
        throw std::out_of_range("Invalid output!");
    /// output error calculation
    std::vector<float> errors = _layers.back().calculateErrors(targets, outs);
    return errors;
}

void Net::setError(const std::vector<float> &error) {
    _layers.back().setErrors(error);
}

std::vector<std::vector<std::vector<float>>> Net::backward(const std::vector<float>& inputs) {
    std::vector<std::vector<std::vector<float>>> allDeltaWeights(_layers.size());
    for(int i=int(_layers.size())-1; i>=0; i--){
        std::vector<std::vector<float>> deltaWeights;
        if(_layers.at(i).get_layerName() == "output"){
            deltaWeights = _layers.at(i).backward(_layers.at(i-1));
        }else if(i == 0){
            deltaWeights = _layers.at(i).backward(inputs, _layers.at(i+1));
        }else{
            deltaWeights = _layers.at(i).backward( _layers.at(i-1), _layers.at(i+1) );
        }
        for(auto& del: deltaWeights){
            for(auto& d: del){
                d *= (-1 * _learningRate);
            }
        }
        allDeltaWeights.at(i) = deltaWeights;
    }
    return allDeltaWeights;
}

void Net::update(const std::vector<std::vector<std::vector<float>>> &allDeltaWeights) {
    /// update
    for(size_t i=0; i<_layers.size(); i++){
        _layers.at(i).update(allDeltaWeights.at(i));
    }
}

void Net::save(const std::string &path) {
    std::ofstream outFile(path);
    if(!outFile)
        std::cout << "File could not create!! >>" << path << "\n";

    outFile << _input << "\n" << _learningRate << "\n" << _layers.size() << "\n";
    for(const auto& layer: _layers){
        outFile << layer.get_layerName() << "\n";
        outFile << layer.get_nodesSize() << "\n";
        outFile << layer.get_activation_name() << "\n";
        for(const auto& node: layer.get_nodes()){
            for(const auto& w: node.get_weights()){
                outFile << w << ";";
            }
            outFile << "\n";
        }
        for(const auto& node: layer.get_nodes()){
            outFile << node.get_bias() << ";";
        }
        outFile << "\n";
    }
    outFile.close();

}

void Net::load(const std::string &path) {
    std::ifstream inFile(path);
    if(!inFile)
        std::cout << "File could not open!! >>" << path << "\n";

    std::string fileLine;
    std::getline(inFile, fileLine);
    _input = std::stoi(fileLine);
    std::getline(inFile, fileLine);
    _learningRate = std::stof(fileLine);
    std::getline(inFile, fileLine);
    int layerSize = std::stoi(fileLine);
    int layerCounter = 0;
    //while(!inFile.eof()){
    while(layerCounter != layerSize){
        layerCounter++;
        // layerName
        std::string layerName;
        std::getline(inFile, layerName);
        // nodeSize
        std::getline(inFile, fileLine);
        int nodeSize = std::stoi(fileLine);
        // activationName
        std::string actName;
        std::getline(inFile, actName);
        // <weights> node_0_weights \n node_1_weights \n ..
        std::vector<std::vector<float>> layerWeights;
        for(int i=0; i<nodeSize; i++){
            std::getline(inFile, fileLine);
            std::vector<float> nodeWeights;
            int k = 0;
            while(k < fileLine.size()){
                int start = k;
                while(fileLine.at(k) != ';') {
                    k++;
                }
                nodeWeights.push_back( std::stof( fileLine.substr(start, k-start) ) );
                k++;
            }
            layerWeights.push_back(nodeWeights);
        }
        // <biases>
        std::vector<float> layerBiases;
        std::getline(inFile, fileLine);
        int k=0;
        while(k < fileLine.size()){
            int start = k;
            while(fileLine.at(k) != ';'){
                k++;
            }
            layerBiases.push_back( std::stof(fileLine.substr(start, k-start)) );
            k++;
        }

        auto sigmoid{std::make_shared<Sigmoid>()};
        insertLayer(layerName, nodeSize, sigmoid, layerWeights, layerBiases);
    }

    inFile.close();
}