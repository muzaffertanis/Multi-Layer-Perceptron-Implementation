# Multi-Layer-Perceptron-Implementation
C++ Multi Layer Perceptron implementation. The implementation can be used as a header only library.

## Basic usage
```c++
/// new net
std::string filePath = "../nets/MNIST_mlp.net"; // Model save path
auto sigmo{std::make_shared<Sigmoid>()}; // Activation function
Net mnistNet(784, 0.02); // input size and learning rate
mnistNet.insertLayer("hidden_1", 20, sigmo); // add layer (hidden)
mnistNet.insertLayer("output", 10, sigmo); // add layer (output)
mnistNet.printParameters();
```
---

# Results
As an example, I tried to train a classifier model on MNIST handwritten dataset.

The model has 20 hidden layers.

After 10 epoch, the model has reached %94.29 accuracy.
