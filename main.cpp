#include <iostream>
#include "mlp/nn.h"
#include "mlp/data.h"
#include "mlp/matrix.h"

int main() {

    /// MNIST Dataset
    Data data("../MNIST Dataset JPG format/MNIST - JPG - training/",
              "../MNIST Dataset JPG format/MNIST - JPG - testing/",
              28, 28);

    std::vector<float> in;
    int label;
    data.get_nextTrainData(in, label);

    std::cout << "Train Size: " << data.get_trainSize() << "\n";
    std::cout << "Test Size: " << data.get_testSize() << "\n";

    /// new net
    std::string filePath = "../nets/MNIST_mlp.net";
    auto sigmo{std::make_shared<Sigmoid>()};
    Net mnistNet(784, 0.02);
    mnistNet.insertLayer("hidden_1", 20, sigmo);
    mnistNet.insertLayer("output", 10, sigmo);
    mnistNet.printParameters();

    /// training loop
    int epochSize = 10; size_t i = 0;
    while(i<epochSize){
        i++;
        std::cout << "EPOCH: " << i << "\n";
        for(size_t j=0; j<data.get_trainSize(); j++){
            std::vector<float> input;
            int label;
            data.get_nextTrainData(input, label);
            if(label == -1){
                break;
            }

            std::vector<float> target(10, 0.01);
            target.at(label) = 0.99;

            auto out = mnistNet.forward(input);
            auto error = mnistNet.loss(target, out);

            mnistNet.update(mnistNet.backward(input));

            /// save
            if(j%5000 == 0){
                mnistNet.save(filePath);
                float eTotal = 0; for(auto& e: error){ eTotal += e; } std::cout << "Total Error: " << eTotal << "\n";
                std::cout << "(" << j << ") Model saved..\n";
            }
        }
        data.resetTrain();

        /// test
        /*
        int counter = 0;
        for(size_t j=0; j<data.get_testSize(); j++){
            std::vector<float> input; int label;
            data.get_nextTestData(input, label);

            auto out = mnistNet.forward(input);
            int predict = int(std::max_element(out.begin(),out.end()) - out.begin());
            if(predict == label){ counter++; }
        }
        std::cout << "Test Accuracy: [" << counter << "/" << data.get_testSize() << "] %";
        if(counter != 0)
            std::cout << (float(counter)/float(data.get_testSize()))*100.0 << "\n";
        else
            std::cout << "0\n";
        data.resetTest();
        */
    }

    /// load and test
    Net mnistNet_test;
    mnistNet_test.load("../nets/MNIST_mlp.net");
    int counter = 0;
    for(size_t j=0; j<data.get_testSize(); j++){
        std::vector<float> input; int label;
        data.get_nextTestData(input, label);

        std::vector<float> test(10, 0);

        auto out = mnistNet_test.forward(input);
        int predict = std::max_element(out.begin(),out.end()) - out.begin();
        if(predict == label){ counter++; test.at(predict) = 1; }
    }
    std::cout << "Test Accuracy: [" << counter << "/" << data.get_testSize() << "] %";
    if(counter != 0)
        std::cout << (float(counter)/float(data.get_testSize()))*100.0 << "\n";
    else
        std::cout << "0\n";


	return 1;
}