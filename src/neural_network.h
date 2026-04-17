#pragma once
#include "NNMatrix.h"
#include "activations.h"
#include <string>

struct Layer {
    NNMatrix* weights;
    NNMatrix* bias;
    NNMatrix* z; 
    NNMatrix* a; 
    
    Layer(int input_size, int output_size);
    ~Layer();
};

struct NeuralNetwork {
    int layer_count;
    Layer** layers;
    Activation activation;

    NeuralNetwork(int* sizes, int count, Activation act);
    ~NeuralNetwork();
};

NNMatrix* forward_propagate(NeuralNetwork* net, NNMatrix* input);
void back_propagate(NeuralNetwork* net, NNMatrix* input, int label);
void save_network(NeuralNetwork* net, const std::string& filename);
void load_network(NeuralNetwork* net, const std::string& filename);