#include "neural_network.h"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);
}

Layer::Layer(int current_nodes, int previous_nodes) 
    : weights(current_nodes, previous_nodes), 
      biases(current_nodes, 1),
      z(current_nodes, 1), 
      a(current_nodes, 1) 
{
    weights.randomize();
    biases.randomize();
}

NeuralNetwork::NeuralNetwork(vector<int> topology, double lr, ActivationType act) 
    : learningRate(lr), hidden_activation(act) {
    for (size_t i = 1; i < topology.size(); i++) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
}

NNMatrix NeuralNetwork::forward(NNMatrix input) {
    NNMatrix current_activation = input;
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].z = NNMatrix::multiply(layers[i].weights, current_activation);
        layers[i].z.add(layers[i].biases);
        layers[i].a = layers[i].z;
        layers[i].a.map(sigmoid); // 目前统一使用 Sigmoid
        current_activation = layers[i].a;
    }
    return current_activation;
}

void NeuralNetwork::train(NNMatrix input, NNMatrix target) {
    NNMatrix output = forward(input);
    NNMatrix errors = NNMatrix::subtract(target, output);

    for (int i = layers.size() - 1; i >= 0; i--) {
        NNMatrix gradients = layers[i].z; 
        gradients.map(sigmoid_derivative);
        gradients.multiply_elements(errors);
        
        // 应用学习率
        for (int r = 0; r < gradients.rows; r++) {
            for (int c = 0; c < gradients.cols; c++) {
                gradients.data[r][c] *= learningRate;
            }
        }

        NNMatrix prev_activation_T = (i == 0) ? input.transpose() : layers[i - 1].a.transpose();
        NNMatrix deltas = NNMatrix::multiply(gradients, prev_activation_T);

        layers[i].weights.add(deltas);
        layers[i].biases.add(gradients);

        NNMatrix weights_T = layers[i].weights.transpose();
        errors = NNMatrix::multiply(weights_T, errors);
    }
}

int NeuralNetwork::predict(NNMatrix input) {
    NNMatrix output = forward(input);
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