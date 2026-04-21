#include "neural_network.h"
#include <cmath>
#include <fstream>  // 新增：用于文件读写
#include <iostream> // 新增：用于打印保存状态

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

NeuralNetwork::NeuralNetwork(vector<int> topology, double lr) : learningRate(lr) {
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
        layers[i].a.map(sigmoid); 
        current_activation = layers[i].a;
    }
    return current_activation;
}

void NeuralNetwork::train(NNMatrix input, NNMatrix target) {
    forward(input);

    NNMatrix output = layers.back().a;
    NNMatrix output_error = NNMatrix::subtract(output, target);
    
    NNMatrix delta = layers.back().z;
    delta.map(sigmoid_derivative);
    delta.multiply_elements(output_error);

    for (int i = layers.size() - 1; i >= 0; i--) {
        NNMatrix prev_a = (i == 0) ? input : layers[i - 1].a;
        NNMatrix prev_a_T = prev_a.transpose();
        NNMatrix weight_gradient = NNMatrix::multiply(delta, prev_a_T);
        
        for (int r = 0; r < layers[i].weights.rows; r++) {
            for (int c = 0; c < layers[i].weights.cols; c++) {
                layers[i].weights.data[r][c] -= learningRate * weight_gradient.data[r][c];
            }
        }

        for (int r = 0; r < layers[i].biases.rows; r++) {
            layers[i].biases.data[r][0] -= learningRate * delta.data[r][0];
        }

        if (i > 0) {
            NNMatrix w_T = layers[i].weights.transpose();
            NNMatrix next_delta = NNMatrix::multiply(w_T, delta);
            NNMatrix z_prev_derivative = layers[i - 1].z;
            z_prev_derivative.map(sigmoid_derivative);
            next_delta.multiply_elements(z_prev_derivative);
            delta = next_delta;
        }
    }
}

int NeuralNetwork::predict(NNMatrix input) {
    NNMatrix output = forward(input);
    int max_index = 0;
    double max_value = output.data[0][0];
    for (int i = 1; i < output.rows; i++) {
        if (output.data[i][0] > max_value) {
            max_value = output.data[i][0];
            max_index = i;
        }
    }
    return max_index;
}

double NeuralNetwork::get_accuracy(const vector<NNMatrix>& test_images, const vector<NNMatrix>& test_labels) {
    int correct_count = 0;
    for (size_t i = 0; i < test_images.size(); i++) {
        int prediction = predict(test_images[i]);
        int truth = 0;
        for (int j = 0; j < test_labels[i].rows; j++) {
            if (test_labels[i].data[j][0] == 1.0) {
                truth = j;
                break;
            }
        }
        if (prediction == truth) {
            correct_count++;
        }
    }
    return (double)correct_count / test_images.size();
}

double NeuralNetwork::calculate_mse(const vector<NNMatrix>& inputs, const vector<NNMatrix>& targets) {
    double total_mse = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) {
        NNMatrix output = forward(inputs[i]);
        double sample_mse = 0.0;
        for (int j = 0; j < output.rows; j++) {
            double error = targets[i].data[j][0] - output.data[j][0];
            sample_mse += error * error;
        }
        total_mse += sample_mse / output.rows;
    }
    return total_mse / inputs.size();
}

// --- 新增：将权重和偏置写入文本文件 ---
void NeuralNetwork::save_model(string filename) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Cannot create model file " << filename << endl;
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

// --- 新增：从文本文件恢复权重和偏置 ---
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
        for (int i_r = 0; i_r < r; i_r++) {
            for (int i_c = 0; i_c < c; i_c++) {
                in >> L.weights.data[i_r][i_c];
            }
        }
        for (int i_r = 0; i_r < r; i_r++) {
            in >> L.biases.data[i_r][0];
        }
        layers.push_back(L);
    }
    in.close();
    cout << "Model loaded successfully. Weights restored!" << endl;
}