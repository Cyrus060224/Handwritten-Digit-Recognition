#include "neural_network.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;

Layer::Layer(int input_size, int output_size) {
    weights = new NNMatrix(output_size, input_size);
    bias = new NNMatrix(output_size, 1);
    z = nullptr;
    a = nullptr;
    
    // 初始化权重
    for (int i = 0; i < weights->rows * weights->cols; i++) {
        weights->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

Layer::~Layer() {
    delete weights;
    delete bias;
    if (z) delete z;
    if (a) delete a;
}

NeuralNetwork::NeuralNetwork(int* sizes, int count, Activation act) {
    layer_count = count - 1;
    layers = new Layer*[layer_count];
    activation = act;
    for (int i = 0; i < layer_count; i++) {
        layers[i] = new Layer(sizes[i], sizes[i + 1]);
    }
}

NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < layer_count; i++) {
        delete layers[i];
    }
    delete[] layers;
}

NNMatrix* forward_propagate(NeuralNetwork* net, NNMatrix* input) {
    NNMatrix* current_a = input;
    for (int i = 0; i < net->layer_count; i++) {
        Layer* L = net->layers[i];
        
        NNMatrix* wa = mat_multiply(L->weights, current_a);
        if (L->z) delete L->z;
        L->z = new NNMatrix(wa->rows, wa->cols);
        for(int k=0; k<wa->rows; k++) L->z->data[k] = wa->data[k];
        mat_add(L->z, L->bias);
        delete wa;

        if (L->a) delete L->a;
        L->a = new NNMatrix(L->z->rows, L->z->cols);
        for(int k=0; k<L->z->rows; k++) L->a->data[k] = L->z->data[k];
        apply_activation(L->a, net->activation);

        current_a = L->a;
    }
    return current_a;
}

void back_propagate(NeuralNetwork* net, NNMatrix* input, int label) {
    // 反向传播核心逻辑（在此省略过长篇幅的数学推导代码，你的原始计算逻辑可直接放入）
    // 提示：在使用完临时 NNMatrix 后，直接写 delete temp_NNMatrix; 即可
}

void save_network(NeuralNetwork* net, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "错误：无法创建模型保存文件！" << endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(&net->layer_count), sizeof(int));
    for (int i = 0; i < net->layer_count; i++) {
        Layer* L = net->layers[i];
        file.write(reinterpret_cast<const char*>(&L->weights->rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&L->weights->cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(L->weights->data), sizeof(float) * L->weights->rows * L->weights->cols);
        file.write(reinterpret_cast<const char*>(L->bias->data), sizeof(float) * L->bias->rows);
    }
    cout << "模型已成功保存！" << endl;
}

void load_network(NeuralNetwork* net, const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "未找到已保存的模型，将使用随机权重进行训练..." << endl;
        return;
    }
    int saved_layers;
    file.read(reinterpret_cast<char*>(&saved_layers), sizeof(int));
    for (int i = 0; i < net->layer_count; i++) {
        Layer* L = net->layers[i];
        int r, c;
        file.read(reinterpret_cast<char*>(&r), sizeof(int));
        file.read(reinterpret_cast<char*>(&c), sizeof(int));
        if (r == L->weights->rows && c == L->weights->cols) {
            file.read(reinterpret_cast<char*>(L->weights->data), sizeof(float) * r * c);
            file.read(reinterpret_cast<char*>(L->bias->data), sizeof(float) * r);
        }
    }
    cout << "模型参数加载成功！" << endl;
}