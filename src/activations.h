/**
 * @file activations.h
 * @brief 激活函数及其导数定义
 *
 * =========================================================================
 * [Requirement 5] 系统可设置隐藏层激活函数：logistic(sigmoid)、tanh、relu
 * =========================================================================
 *
 * 本文件实现了三种常用激活函数及其导数，用于神经网络隐藏层和输出层。
 * 所有函数均声明为 inline 以避免函数调用开销。
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

// ==========================================
// 1. Sigmoid (Logistic) 激活函数
// ==========================================
// [Requirement 5] 激活函数选项之一：logistic
//
// 数学定义:
//   σ(x) = 1 / (1 + e^(-x))
//
// 导数推导（链式法则关键）:
//   dσ/dx = σ(x) * (1 - σ(x))
//
// 推导过程:
//   令 u = 1 + e^(-x)
//   σ(x) = u^(-1)
//   dσ/dx = -u^(-2) * du/dx
//         = -u^(-2) * (-e^(-x))
//         = e^(-x) / (1 + e^(-x))^2
//         = [1/(1+e^(-x))] * [e^(-x)/(1+e^(-x))]
//         = σ(x) * (1 - σ(x))  ■
//
// 特性:
//   - 输出范围: (0, 1)，适合用于输出层做概率解释
//   - 缺点: 输入绝对值过大时梯度趋近于 0（梯度消失问题）
// ==========================================

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
 * @param x 激活前的原始值（z 值，非激活后的值）
 * @return σ'(x) = σ(x) * (1 - σ(x))
 * @note 在反向传播中，此函数接收 z 值（即加权和），先通过 sigmoid 得到激活值，
 *       再计算 σ * (1-σ) 作为局部梯度
 */
inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ==========================================
// 2. ReLU (Rectified Linear Unit) 激活函数
// ==========================================
// [Requirement 5] 激活函数选项之一：relu
//
// 数学定义:
//   f(x) = max(0, x)
//
// 导数推导:
//   df/dx = 1,  if x > 0
//          = 0,  if x <= 0
//   （在 x=0 处不可导，工程上约定为 0）
//
// 特性:
//   - 计算极其高效（只需一次比较）
//   - 解决了梯度消失问题（正半区梯度恒为 1）
//   - 推荐用于隐藏层（本项目默认隐藏层激活函数）
// ==========================================

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

// ==========================================
// 3. Tanh (双曲正切) 激活函数
// ==========================================
// [Requirement 5] 激活函数选项之一：tanh
//
// 数学定义:
//   tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
//
// 导数推导:
//   d/dx tanh(x) = 1 - tanh^2(x)
//
// 推导过程（利用商法则）:
//   tanh(x) = sinh(x) / cosh(x)
//   d/dx tanh(x) = (cosh^2(x) - sinh^2(x)) / cosh^2(x)
//                = 1 / cosh^2(x)
//                = 1 - tanh^2(x)  ■
//
// 特性:
//   - 输出范围: (-1, 1)，零中心化，有利于下一层学习
//   - 相比 Sigmoid，梯度消失问题稍轻
// ==========================================

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
