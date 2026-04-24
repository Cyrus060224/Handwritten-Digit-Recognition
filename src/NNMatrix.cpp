/**
 * @file NNMatrix.cpp
 * @brief 矩阵引擎运算实现
 *
 * [Requirement 1] 所有矩阵运算均基于 std::vector 手动实现
 */

#include "NNMatrix.h"
#include <random>

using namespace std;

/**
 * @brief 构造 r x c 的零矩阵
 */
NNMatrix::NNMatrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0.0)) {}

/**
 * @brief Xavier 权重初始化
 *
 * 使用正态分布 N(0, limit) 初始化权重，其中 limit = sqrt(1/cols)。
 * 通过控制权重方差，防止深层网络中激活值指数级放大或缩小。
 * 固定种子 gen(42) 保证实验可重复性。
 */
void NNMatrix::randomize() {
    static mt19937 gen(42);
    double limit = sqrt(1.0 / (double)cols);
    normal_distribution<double> dist(0.0, limit);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

/**
 * @brief 矩阵乘法: C[i][j] = Σ(k) A[i][k] * B[k][j]
 *
 * 采用 i-k-j 循环顺序优化缓存命中率。
 * 复杂度: O(M × K × N)
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
 * @brief 矩阵加法（原地）: this[i][j] += other[i][j]
 */
void NNMatrix::add(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += other.data[i][j];
        }
    }
}

/**
 * @brief 矩阵减法: C[i][j] = A[i][j] - B[i][j]
 * @return 差矩阵
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
 * @brief Hadamard 积（逐元素相乘，原地）: this[i][j] *= other[i][j]
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
 * @param func 标量函数
 */
void NNMatrix::map(function<double(double)> func) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] = func(this->data[i][j]);
        }
    }
}

/**
 * @brief 矩阵转置: C[j][i] = A[i][j]
 * @return 转置矩阵
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
 * @brief 生成 Inverted Dropout 随机掩码
 *
 * [Requirement 10] Dropout 正则化
 *
 * 以概率 p 将掩码元素设为 1/p（保留并放大），否则设为 0（丢弃）。
 * 使用 random_device 种子确保每次调用的随机性。
 *
 * @param r 掩码行数
 * @param c 掩码列数
 * @param keep_probability 保留概率 p
 * @return 掩码矩阵
 */
NNMatrix NNMatrix::generate_dropout_mask(int r, int c, double keep_probability) {
    NNMatrix mask(r, c);
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (dist(gen) < keep_probability) {
                mask.data[i][j] = 1.0 / keep_probability;
            } else {
                mask.data[i][j] = 0.0;
            }
        }
    }
    return mask;
}
