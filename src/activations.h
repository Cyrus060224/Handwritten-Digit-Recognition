#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

// ==========================================
// 1. Sigmoid 激活函数
// ==========================================
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ==========================================
// 2. ReLU 激活函数
// ==========================================
inline double relu(double x) {
    return x > 0 ? x : 0.0;
}

inline double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// ==========================================
// 3. Tanh 激活函数
// ==========================================
inline double tanh_act(double x) {
    return tanh(x);
}

inline double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

#endif