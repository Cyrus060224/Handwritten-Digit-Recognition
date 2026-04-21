#include "NNMatrix.h"
#include <random>

using namespace std;

NNMatrix::NNMatrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0.0)) {}

// 在 NNMatrix.cpp 中找到这个函数并修改
void NNMatrix::randomize() {
    static mt19937 gen(42); 
    // 🌟 核心修改：Xavier 初始化。不要用固定的 0.5，要根据输入的节点数缩小方差
    double limit = sqrt(1.0 / (double)cols); 
    normal_distribution<double> dist(0.0, limit); // 让权重变得极小，防止 Sigmoid 饱和

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}
NNMatrix NNMatrix::multiply(const NNMatrix& a, const NNMatrix& b) {
    NNMatrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int k = 0; k < a.cols; k++) {
            for (int j = 0; j < b.cols; j++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

void NNMatrix::add(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += other.data[i][j];
        }
    }
}

NNMatrix NNMatrix::subtract(const NNMatrix& a, const NNMatrix& b) {
    NNMatrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

void NNMatrix::multiply_elements(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] *= other.data[i][j];
        }
    }
}

void NNMatrix::map(function<double(double)> func) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] = func(this->data[i][j]);
        }
    }
}

NNMatrix NNMatrix::transpose() const {
    NNMatrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = this->data[i][j];
        }
    }
    return result;
}

// 将这段代码直接加在 NNMatrix.cpp 的最下面

NNMatrix NNMatrix::generate_dropout_mask(int r, int c, double keep_probability) {
    NNMatrix mask(r, c);
    static mt19937 gen(random_device{}()); // 随机数引擎
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            // 🌟 核心：如果保留，则放大 1/p 倍；如果丢弃，则为 0
            if (dist(gen) < keep_probability) {
                mask.data[i][j] = 1.0 / keep_probability;
            } else {
                mask.data[i][j] = 0.0;
            }
        }
    }
    return mask;
}