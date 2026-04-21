#include "neural_network.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include "activations.h" 

using namespace std;

Layer::Layer(int current_nodes, int previous_nodes) 
    : weights(current_nodes, previous_nodes), 
      biases(current_nodes, 1),
      z(current_nodes, 1), 
      a(current_nodes, 1),
      dropout_mask(current_nodes, 1) 
{
    weights.randomize();
    biases.randomize();
}

NeuralNetwork::NeuralNetwork(vector<int> topology, double lr, ActivationType act, double keep_rate) 
    : learningRate(lr), hidden_activation(act), keep_rate(keep_rate) {
    for (size_t i = 1; i < topology.size(); i++) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
    reset_gradients();
    backup_layers = layers; // 🌟 初始化时保存初代快照 
}

// 🌟 贪心快照的具体实现（vector 自动深拷贝）
void NeuralNetwork::save_checkpoint() { 
    backup_layers = layers; 
}

void NeuralNetwork::load_checkpoint() { 
    layers = backup_layers; 
}

NNMatrix NeuralNetwork::forward(NNMatrix input, bool is_training) {
    NNMatrix current_activation = input;
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].z = NNMatrix::multiply(layers[i].weights, current_activation);
        layers[i].z.add(layers[i].biases);
        layers[i].a = layers[i].z;

        if (i == layers.size() - 1) {
            layers[i].a.map(sigmoid); 
        } else {
            if (hidden_activation == RELU) layers[i].a.map(relu);
            else if (hidden_activation == TANH) layers[i].a.map(tanh_act);
            else layers[i].a.map(sigmoid);
            
            if (is_training && keep_rate < 1.0) {
                layers[i].dropout_mask = NNMatrix::generate_dropout_mask(layers[i].a.rows, layers[i].a.cols, keep_rate);
                layers[i].a.multiply_elements(layers[i].dropout_mask);
            }
        }
        current_activation = layers[i].a;
    }
    return current_activation;
}

void NeuralNetwork::reset_gradients() {
    weight_gradients_acc.clear();
    bias_gradients_acc.clear();
    for (const auto& L : layers) {
        weight_gradients_acc.push_back(NNMatrix(L.weights.rows, L.weights.cols));
        bias_gradients_acc.push_back(NNMatrix(L.biases.rows, L.biases.cols));
    }
    accumulated_samples = 0;
}

void NeuralNetwork::accumulate_gradients(NNMatrix input, NNMatrix target) {
    NNMatrix output = forward(input, true); 
    NNMatrix errors = NNMatrix::subtract(target, output);

    // 🌟 修复警告：显式转换为 int，防止 size_t 减法溢出
    for (int i = (int)layers.size() - 1; i >= 0; i--) {
        NNMatrix gradients = layers[i].z; 
        
        if (i == (int)layers.size() - 1) {
            gradients.map(sigmoid_derivative);
        } else {
            if (hidden_activation == RELU) gradients.map(relu_derivative);
            else if (hidden_activation == TANH) gradients.map(tanh_derivative);
            else gradients.map(sigmoid_derivative);
            
            if (keep_rate < 1.0) {
                gradients.multiply_elements(layers[i].dropout_mask);
            }
        }
        
        gradients.multiply_elements(errors);
        
        NNMatrix prev_activation_T = (i == 0) ? input.transpose() : layers[i - 1].a.transpose();
        NNMatrix deltas = NNMatrix::multiply(gradients, prev_activation_T);

        weight_gradients_acc[i].add(deltas);
        bias_gradients_acc[i].add(gradients);

        NNMatrix weights_T = layers[i].weights.transpose();
        errors = NNMatrix::multiply(weights_T, errors);
    }
    accumulated_samples++;
}

void NeuralNetwork::apply_gradients() {
    if (accumulated_samples == 0) return;

    double lr = learningRate / (double)accumulated_samples;
    
    for (size_t i = 0; i < layers.size(); i++) { 
        
        weight_gradients_acc[i].map([lr](double x) { return x * lr; });
        bias_gradients_acc[i].map([lr](double x) { return x * lr; });

        layers[i].weights.add(weight_gradients_acc[i]);
        layers[i].biases.add(bias_gradients_acc[i]);
    }
    reset_gradients(); 
}

void NeuralNetwork::train(NNMatrix input, NNMatrix target) {
    accumulate_gradients(input, target);
    apply_gradients();
}

int NeuralNetwork::predict(NNMatrix input) {
    NNMatrix output = forward(input, false); // 🌟 预测时禁用 Dropout
    int maxIndex = 0;
    double maxValue = output.data[0][0];
    for (int i = 1; i < 10; i++) {
        if (output.data[i][0] > maxValue) {
            maxValue = output.data[i][0];
            maxIndex = i;
        }
    }
    return maxIndex;
}

double NeuralNetwork::get_accuracy(const MNISTData& data) {
    if (data.images.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < data.images.size(); i++) {
        int prediction = predict(data.images[i]);
        int label = 0;
        for (int r = 0; r < 10; r++) {
            if (data.labels[i].data[r][0] > 0.5) {
                label = r;
                break;
            }
        }
        if (prediction == label) correct++;
    }
    return (double)correct / data.images.size();
}

void NeuralNetwork::load_model(string filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Model file not found: " << filename << endl;
        return;
    }
    int layer_count;
    in >> layer_count;
    layers.clear();
    for (int i = 0; i < layer_count; i++) {
        int r, c;
        in >> r >> c;
        Layer L(r, c);
        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c; col++) {
                in >> L.weights.data[row][col];
            }
        }
        for (int row = 0; row < r; row++) {
            in >> L.biases.data[row][0];
        }
        layers.push_back(L);
    }
    in.close();
    cout << "Model loaded successfully. Weights restored!" << endl;
}

void NeuralNetwork::save_model(string filename) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Cannot create model file: " << filename << endl;
        return;
    }
    out << layers.size() << "\n";
    for (const auto& L : layers) {
        out << L.weights.rows << " " << L.weights.cols << "\n";
        for (int r = 0; r < L.weights.rows; r++) {
            for (int c = 0; c < L.weights.cols; c++) {
                out << L.weights.data[r][c] << " ";
            }
        }
        out << "\n";
        for (int r = 0; r < L.biases.rows; r++) {
            out << L.biases.data[r][0] << " ";
        }
        out << "\n";
    }
    out.close();
    cout << "Success: Model weights permanently saved to: " << filename << endl;
}