/**
 * @file NNMatrix.h
 * @brief 纯 C++ 手写矩阵引擎 - 神经网络底层数据结构
 *
 * =========================================================================
 * [Requirement 1] 实现多层前馈神经网络的核心算法 - 手动实现底层矩阵运算
 * =========================================================================
 *
 * NNMatrix 是整个神经网络的计算基石。所有前向传播的矩阵乘法、
 * 反向传播的梯度计算、权重更新等操作，都依赖于此结构体提供的运算能力。
 *
 * 设计原则:
 *   - 零外部依赖：不使用 Eigen/BLAS/NumPy，仅用 std::vector<vector<double>>
 *   - 列向量约定：每一列代表一个样本，每一行代表一个特征/神经元
 *   - 双精度浮点：使用 double 而非 float，保证训练过程中的数值稳定性
 */

#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <vector>
#include <functional>

using namespace std;

/**
 * @struct NNMatrix
 * @brief 神经网络专用矩阵结构体
 *
 * [Requirement 1] 纯手写矩阵引擎，不依赖任何线性代数库
 *
 * 矩阵布局说明:
 *   - data[i][j] 表示第 i 行、第 j 列的元素
 *   - 对于权重矩阵: rows = 当前层神经元数, cols = 前一层神经元数
 *   - 对于激活矩阵: rows = 神经元数, cols = 1 (列向量)
 */
struct NNMatrix {
    int rows;                       ///< 矩阵行数（神经元数量或样本特征数）
    int cols;                       ///< 矩阵列数（样本数量）
    vector<vector<double>> data;    ///< 二维数据存储

    /**
     * @brief 构造函数：创建指定维度的零矩阵
     * @param r 行数
     * @param c 列数
     */
    NNMatrix(int r, int c);

    /**
     * @brief Xavier/Glorot 初始化
     *
     * [Requirement 1] 贪心策略的权重初始化部分
     *
     * 数学原理:
     *   Xavier 初始化通过限制权重方差，防止激活值在前向传播过程中
     *   爆炸或消失。对于 Sigmoid/Tanh 激活函数尤其关键。
     *
     *   方差计算: Var(W) = 1 / n_in
     *   其中 n_in = cols = 前一层神经元数（fan-in）
     *
     *   使用正态分布 N(0, limit)，其中 limit = sqrt(1/cols)
     */
    void randomize();

    /**
     * @brief 矩阵乘法 C = A * B
     * @param a 左操作数矩阵 (M x K)
     * @param b 右操作数矩阵 (K x N)
     * @return 结果矩阵 C (M x N)
     *
     * 数学定义:
     *   C[i][j] = sum(A[i][k] * B[k][j], k = 0..K-1)
     *
     * 循环顺序: i-k-j（缓存友好，减少 cache miss）
     *
     * 在神经网络中的应用:
     *   z = W * a + b （加权求和）
     *   其中 W 是权重矩阵，a 是前一层激活输出
     */
    static NNMatrix multiply(const NNMatrix& a, const NNMatrix& b);

    /**
     * @brief 矩阵加法 (原地操作): this += other
     * @param other 加数矩阵（需与 this 同维度）
     *
     * 在神经网络中的应用:
     *   z += b （加上偏置项）
     */
    void add(const NNMatrix& other);

    /**
     * @brief 矩阵减法: C = A - B
     * @param a 被减数矩阵
     * @param b 减数矩阵
     * @return 差矩阵 C（需与 A、B 同维度）
     *
     * 在神经网络中的应用:
     *   error = target - output （计算输出层误差）
     */
    static NNMatrix subtract(const NNMatrix& a, const NNMatrix& b);

    /**
     * @brief Hadamard 积（逐元素相乘，原地操作）: this = this ⊙ other
     * @param other 乘数矩阵
     *
     * 数学定义:
     *   (A ⊙ B)[i][j] = A[i][j] * B[i][j]
     *
     * 在神经网络中的应用:
     *   1. 反向传播中：gradient ⊙ activation_derivative
     *   2. Dropout 中：activation ⊙ dropout_mask
     */
    void multiply_elements(const NNMatrix& other);

    /**
     * @brief 逐元素映射变换（原地操作）
     * @param func 应用到每个元素的标量函数
     *
     * 在神经网络中的应用:
     *   1. 激活函数：z.map(sigmoid) 或 z.map(relu)
     *   2. 导数计算：z.map(sigmoid_derivative)
     *   3. 梯度缩放：gradient.map([lr](x){ return x * lr; })
     */
    void map(function<double(double)> func);

    /**
     * @brief 矩阵转置: C = A^T
     * @return 转置后的矩阵 C
     *
     * 数学定义:
     *   C[j][i] = A[i][j]
     *
     * 在神经网络中的应用:
     *   反向传播中传播误差时需要使用转置权重矩阵:
     *   error_prev = W^T * error_current
     */
    NNMatrix transpose() const;

    /**
     * @brief 生成 Inverted Dropout 随机掩码矩阵
     *
     * [Requirement 10] 随机丢弃（Dropout）- 防止过拟合的正则化方法
     *
     * Inverted Dropout 原理:
     *   - 训练时以概率 p 保留神经元，以概率 (1-p) 丢弃
     *   - 被保留的神经元输出放大 1/p 倍（Inverted 的核心）
     *   - 测试时不做任何操作，直接使用全部神经元
     *
     * 数学定义:
     *   mask[i][j] = 1/p,  with probability p
     *                = 0,    with probability (1-p)
     *
     * 期望保持:
     *   E[x * mask] = p * (x/p) + (1-p) * 0 = x  ✓
     *
     * @param r 掩码行数
     * @param c 掩码列数
     * @param keep_probability 保留概率 p（例如 0.8 表示 80% 的神经元被保留）
     * @return 随机掩码矩阵
     */
    static NNMatrix generate_dropout_mask(int r, int c, double keep_probability);

};

#endif
