#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NNMatrix.h"
#include "data_loader.h"
#include <vector>
#include <string>

using namespace std;

// 预留激活函数类型，方便后续扩展
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
    ActivationType hidden_activation; // 记录隐藏层激活函数类型

    NeuralNetwork(vector<int> topology, double lr = 0.01, ActivationType act = SIGMOID);
    
    NNMatrix forward(NNMatrix input);
    void train(NNMatrix input, NNMatrix target);
    int predict(NNMatrix input); 

    // --- 核心：测试集准确率评估 ---
    double get_accuracy(const MNISTData& data);

    // --- 模型持久化 ---
    void save_model(string filename);
    void load_model(string filename);
};

#endif