#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NNMatrix.h"
#include "data_loader.h"
#include <vector>
#include <string>

using namespace std;

enum ActivationType {
    SIGMOID,
    RELU,
    TANH
};

struct Layer {
    NNMatrix weights;
    NNMatrix biases;
    NNMatrix z;
    NNMatrix a;
    Layer(int current_nodes, int previous_nodes);
};

struct NeuralNetwork {
    vector<Layer> layers;
    double learningRate;
    ActivationType hidden_activation;

    // 🌟 新增：梯度累加池，用于支持 Mini-Batch 和 BGD
    vector<NNMatrix> weight_gradients_acc;
    vector<NNMatrix> bias_gradients_acc;
    int accumulated_samples;

    NeuralNetwork(vector<int> topology, double lr = 0.01, ActivationType act = SIGMOID);
    
    NNMatrix forward(NNMatrix input);
    void train(NNMatrix input, NNMatrix target);
    int predict(NNMatrix input); 
    double get_accuracy(const MNISTData& data);
    void save_model(string filename);
    void load_model(string filename);

    // 🌟 新增：底层优化器核心方法
    void reset_gradients();
    void accumulate_gradients(NNMatrix input, NNMatrix target);
    void apply_gradients();
};

#endif