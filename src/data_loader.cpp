#include "data_loader.h"
#include <fstream>
#include <iostream>

using namespace std;

// 翻转字节序 (大端转小端)
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

void load_mnist_images(const string& filename, int count, vector<NNMatrix*>& images) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "无法打开图像文件: " << filename << endl;
        return;
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    for (int i = 0; i < count; i++) {
        images[i] = new NNMatrix(784, 1);
        for (int j = 0; j < 784; j++) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i]->data[j] = pixel / 255.0f; // 归一化
        }
    }
}

void load_mnist_labels(const string& filename, int count, vector<int>& labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "无法打开标签文件: " << filename << endl;
        return;
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    for (int i = 0; i < count; i++) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
}