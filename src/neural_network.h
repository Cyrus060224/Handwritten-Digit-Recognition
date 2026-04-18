#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NNMatrix.h"
#include <vector>
#include <string> // 新增：为了处理文件名

using namespace std;

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

    NeuralNetwork(vector<int> topology, double lr = 0.01);
    
    NNMatrix forward(NNMatrix input);
    void train(NNMatrix input, NNMatrix target);

    int predict(NNMatrix input); 
    double get_accuracy(const vector<NNMatrix>& test_images, const vector<NNMatrix>& test_labels);
    double calculate_mse(const vector<NNMatrix>& inputs, const vector<NNMatrix>& targets);

    // --- 新增：要求 (2)-② 保存与读取网络参数 ---
    void save_model(string filename);
    void load_model(string filename);
};

#endif