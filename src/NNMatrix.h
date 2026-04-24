/**
 * @file NNMatrix.h
 * @brief 纯 C++ 手写矩阵引擎 - 神经网络底层数据结构
 *
 * [Requirement 1] 纯手写矩阵运算，不依赖任何第三方线性代数库
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
 * 矩阵布局: 列向量约定，每列代表一个样本
 * 数据以 std::vector<vector<double>> 存储，使用双精度浮点保证数值稳定性
 */
struct NNMatrix {
    int rows;
    int cols;
    vector<vector<double>> data;

    NNMatrix(int r, int c);

    /**
     * @brief Xavier/Glorot 初始化
     *
     * 通过限制权重方差防止激活值在前向传播中爆炸或消失。
     * 方差: Var(W) = 1 / fan_in，使用正态分布 N(0, sqrt(1/cols))
     */
    void randomize();

    /**
     * @brief 矩阵乘法: C = A × B
     * @param a 左操作数矩阵 (M × K)
     * @param b 右操作数矩阵 (K × N)
     * @return 结果矩阵 C (M × N)
     *
     * 循环顺序采用 i-k-j 以优化缓存命中率
     */
    static NNMatrix multiply(const NNMatrix& a, const NNMatrix& b);

    /**
     * @brief 矩阵加法（原地）: this += other
     * @param other 加数矩阵
     */
    void add(const NNMatrix& other);

    /**
     * @brief 矩阵减法: C = A - B
     * @param a 被减数矩阵
     * @param b 减数矩阵
     * @return 差矩阵 C
     */
    static NNMatrix subtract(const NNMatrix& a, const NNMatrix& b);

    /**
     * @brief Hadamard 积（逐元素相乘，原地）: this = this ⊙ other
     * @param other 乘数矩阵
     */
    void multiply_elements(const NNMatrix& other);

    /**
     * @brief 逐元素映射变换（原地）
     * @param func 应用到每个元素的标量函数
     */
    void map(function<double(double)> func);

    /**
     * @brief 矩阵转置: C = A^T
     * @return 转置后的矩阵
     */
    NNMatrix transpose() const;

    /**
     * @brief 生成 Inverted Dropout 随机掩码矩阵
     *
     * [Requirement 10] Dropout 正则化 - 防止过拟合的网络结构正则化方法
     *
     * 训练时以概率 p 保留神经元并放大 1/p 倍（Inverted Dropout），
     * 测试时无需任何操作，直接使用完整网络。
     *
     * @param r 掩码行数
     * @param c 掩码列数
     * @param keep_probability 保留概率 p
     * @return 随机掩码矩阵，元素值为 1/p 或 0
     */
    static NNMatrix generate_dropout_mask(int r, int c, double keep_probability);
};

#endif
