/**
 * @file NNMatrix.cpp
 * @brief 纯 C++ 手写矩阵引擎 - 运算实现
 *
 * =========================================================================
 * [Requirement 1] 实现多层前馈神经网络的核心算法 - 手动实现底层矩阵运算
 * =========================================================================
 *
 * 本文件实现了 NNMatrix 的所有运算方法。所有计算均不使用任何
 * 第三方线性代数库（如 Eigen/BLAS），完全基于 std::vector 手动实现。
 */

#include "NNMatrix.h"
#include <random>

using namespace std;

/**
 * @brief 构造函数：创建 r x c 的零矩阵
 *
 * 初始化所有元素为 0.0，为后续的随机初始化或数据填充做准备。
 */
NNMatrix::NNMatrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0.0)) {}

/**
 * @brief Xavier/Glorot 权重初始化
 *
 * =========================================================================
 * [Requirement 1] 贪心策略 - 基于贪心的权重初始化策略
 * =========================================================================
 *
 * 数学推导:
 *   Xavier 初始化的核心思想：让每一层的激活值方差与输入方差保持一致，
 *   防止信号在前向传播过程中指数级放大（爆炸）或缩小（消失）。
 *
 *   对于第 l 层的激活值 a[l]:
 *     z[l] = W[l] * a[l-1]          (加权和)
 *     a[l] = f(z[l])                 (激活函数)
 *
 *   假设 W 和 a[l-1] 独立同分布，均值为 0:
 *     Var(z[l]) = n_in * Var(W) * Var(a[l-1])
 *
 *   要使 Var(z[l]) = Var(a[l-1])，需满足:
 *     n_in * Var(W) = 1
 *     Var(W) = 1 / n_in
 *
 *   其中 n_in = cols = 前一层神经元数（fan-in）
 *
 *   使用正态分布: W ~ N(0, sigma), sigma = sqrt(1/n_in)
 *
 * 为什么这对 Sigmoid 至关重要:
 *   Sigmoid 在 |x| > 5 时梯度趋近于 0。如果权重初始化过大，
 *   深层神经元的 z 值会迅速超出 [-5, 5] 范围，导致梯度消失。
 *   Xavier 初始化通过将权重控制在小范围内，确保初始阶段信号可以正常传播。
 *
 * 固定种子 gen(42) 的作用:
 *   确保每次运行程序时产生相同的初始权重，便于实验可重复性。
 */
void NNMatrix::randomize() {
    static mt19937 gen(42);
    // Xavier 初始化标准差
    double limit = sqrt(1.0 / (double)cols);
    normal_distribution<double> dist(0.0, limit);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

/**
 * @brief 矩阵乘法: C = A × B
 *
 * 数学公式:
 *   C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
 *
 * 复杂度: O(M × K × N)，其中 A 为 M×K，B 为 K×N
 *
 * 循环顺序说明 (i-k-j):
 *   此顺序使内层循环访问 B 的行（连续内存），
 *   同时 result.data[i][j] 在 k 循环中保持不变，
 *   有利于 CPU 缓存命中。
 *
 * 在神经网络前向传播中的应用:
 *   第 l 层: z[l] = W[l] * a[l-1]
 *   其中 W[l] 维度为 (n_l × n_{l-1}), a[l-1] 为 (n_{l-1} × 1)
 *   结果 z[l] 维度为 (n_l × 1)
 */
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

/**
 * @brief 矩阵加法（原地）: this = this + other
 *
 * 在神经网络中的应用:
 *   z[l] = W[l] * a[l-1] + b[l]
 *   即先做矩阵乘法得到加权和，再加上偏置项。
 *
 * 复杂度: O(rows × cols)
 */
void NNMatrix::add(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += other.data[i][j];
        }
    }
}

/**
 * @brief 矩阵减法: C = A - B
 *
 * 在神经网络中的应用:
 *   计算输出层误差（MSE 损失的梯度）:
 *   δ[L] = ∂Loss/∂z[L] = (output - target) ⊙ f'(z[L])
 *   其中 output - target 由本函数计算。
 *
 * 复杂度: O(rows × cols)
 */
NNMatrix NNMatrix::subtract(const NNMatrix& a, const NNMatrix& b) {
    NNMatrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

/**
 * @brief Hadamard 积（逐元素相乘，原地）: this = this ⊙ other
 *
 * 数学公式:
 *   C[i][j] = A[i][j] * B[i][j]
 *
 * 在神经网络中的两大应用:
 *
 * 1. 反向传播中的链式法则:
 *    当前层的误差信号需要通过激活函数的导数进行调制:
 *    δ[l] = (W[l+1]^T * δ[l+1]) ⊙ f'(z[l])
 *
 * 2. Dropout 正则化:
 *    a[l] = a[l] ⊙ mask[l]
 *    其中 mask[l] 是随机生成的 0/1 掩码矩阵
 *
 * 复杂度: O(rows × cols)
 */
void NNMatrix::multiply_elements(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] *= other.data[i][j];
        }
    }
}

/**
 * @brief 逐元素函数映射（原地）: this[i][j] = func(this[i][j])
 *
 * 使用 std::function<double(double)> 允许传入任意标量函数，
 * 包括激活函数、导数函数、lambda 表达式等。
 *
 * 在神经网络中的典型用法:
 *   1. 前向传播: layers[i].a.map(sigmoid);   // 应用激活函数
 *   2. 反向传播: gradients.map(sigmoid_derivative); // 应用导数
 *   3. 权重更新: weight_gradients_acc[i].map([lr](double x){ return x * lr; });
 *
 * 复杂度: O(rows × cols)
 */
void NNMatrix::map(function<double(double)> func) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] = func(this->data[i][j]);
        }
    }
}

/**
 * @brief 矩阵转置: C = A^T
 *
 * 数学公式:
 *   C[j][i] = A[i][j]
 *
 * 在神经网络反向传播中的关键作用:
 *   误差从第 l+1 层传播到第 l 层时，需要乘以转置权重矩阵:
 *
 *   δ[l] = (W[l+1]^T × δ[l+1]) ⊙ f'(z[l])
 *
 *   展开说明:
 *     W[l+1] 维度: (n_{l+1} × n_l)
 *     W[l+1]^T 维度: (n_l × n_{l+1})
 *     δ[l+1] 维度: (n_{l+1} × 1)
 *     结果: (n_l × n_{l+1}) × (n_{l+1} × 1) = (n_l × 1) ✓
 *
 * 复杂度: O(rows × cols)
 */
NNMatrix NNMatrix::transpose() const {
    NNMatrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = this->data[i][j];
        }
    }
    return result;
}

/**
 * @brief 生成 Inverted Dropout 随机掩码矩阵
 *
 * =========================================================================
 * [Requirement 10] 随机丢弃（Dropout）- 防止过拟合的网络结构正则化方法
 * =========================================================================
 *
 * Dropout 的动机:
 *   在训练过程中，每次迭代随机"丢弃"一部分神经元（将其输出置为 0），
 *   强制网络不依赖于任何特定的神经元组合，从而提高泛化能力。
 *   这等价于在训练时对网络结构进行 L2 正则化。
 *
 * Inverted Dropout 的核心改进:
 *   传统 Dropout 在测试时需要将所有权重乘以保留概率 p，
 *   而 Inverted Dropout 在训练时就将保留的神经元输出放大 1/p 倍，
 *   测试时无需任何修改，直接使用完整网络。
 *
 * 数学推导:
 *   设神经元输出为 x，掩码为 m:
 *     m = 1/p,  with probability p  (保留)
 *     m = 0,    with probability 1-p (丢弃)
 *
 *   期望输出:
 *     E[x * m] = p * (x * 1/p) + (1-p) * 0 = x  ✓
 *
 *   即：经过 Dropout 后的期望输出等于原始输出，
 *   测试时无需做任何缩放。
 *
 * 使用 random_device 种子的原因:
 *   每次调用都需要真正的随机性，与 Xavier 初始化的固定种子不同。
 *
 * @param r 掩码行数（对应神经元数量）
 * @param c 掩码列数（对应样本数量）
 * @param keep_probability 保留概率 p ∈ (0, 1]
 * @return 掩码矩阵，元素值为 1/p 或 0
 */
NNMatrix NNMatrix::generate_dropout_mask(int r, int c, double keep_probability) {
    NNMatrix mask(r, c);
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (dist(gen) < keep_probability) {
                mask.data[i][j] = 1.0 / keep_probability; // Inverted Dropout: 放大 1/p
            } else {
                mask.data[i][j] = 0.0; // 丢弃该神经元
            }
        }
    }
    return mask;
}
