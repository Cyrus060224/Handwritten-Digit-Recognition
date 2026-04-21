#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NNMatrix.h"
#include "data_loader.h"
#include <vector>
#include <string>

using namespace std;

enum ActivationType { SIGMOID, RELU, TANH };

struct Layer {
    NNMatrix weights;
    NNMatrix biases;
    NNMatrix z;
    NNMatrix a;
    NNMatrix dropout_mask; // 🌟 必须在头文件里声明

    Layer(int current_nodes, int previous_nodes);
};

struct NeuralNetwork {
    vector<Layer> layers;
    double learningRate;
    ActivationType hidden_activation;
    double keep_rate; // 🌟 必须在头文件里声明

    vector<NNMatrix> weight_gradients_acc;
    vector<NNMatrix> bias_gradients_acc;
    int accumulated_samples;

    // 构造函数新增 keep_rate 参数
    NeuralNetwork(vector<int> topology, double lr = 0.01, ActivationType act = SIGMOID, double keep_rate = 1.0);
    
    NNMatrix forward(NNMatrix input, bool is_training = false); 
    void train(NNMatrix input, NNMatrix target);
    int predict(NNMatrix input); 
    double get_accuracy(const MNISTData& data);
    void save_model(string filename);
    void load_model(string filename);

    void reset_gradients();
    void accumulate_gradients(NNMatrix input, NNMatrix target);
    void apply_gradients();
};

#endif