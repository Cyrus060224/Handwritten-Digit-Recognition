/**
 * @file activations.h
 * @brief 激活函数及其导数定义
 *
 * [Requirement 5] 隐藏层激活函数：sigmoid、relu、tanh 三选一
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

/**
 * @brief Sigmoid 激活函数
 * @param x 输入值
 * @return σ(x) = 1/(1+e^(-x))
 */
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Sigmoid 导数
 * @param x 激活前的原始值
 * @return σ'(x) = σ(x) * (1 - σ(x))
 */
inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

/**
 * @brief ReLU 激活函数
 * @param x 输入值
 * @return max(0, x)
 */
inline double relu(double x) {
    return x > 0 ? x : 0.0;
}

/**
 * @brief ReLU 导数
 * @param x 激活前的原始值
 * @return 1 (x>0) 或 0 (x<=0)
 */
inline double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

/**
 * @brief Tanh 激活函数
 * @param x 输入值
 * @return tanh(x)
 */
inline double tanh_act(double x) {
    return tanh(x);
}

/**
 * @brief Tanh 导数
 * @param x 激活前的原始值
 * @return 1 - tanh^2(x)
 */
inline double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

#endif
