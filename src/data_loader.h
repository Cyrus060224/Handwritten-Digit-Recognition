/**
 * @file data_loader.h
 * @brief MNIST 数据集加载器
 *
 * =========================================================================
 * [Requirement 2] 训练和测试数据从文件中读取，进行数据预处理
 * =========================================================================
 *
 * MNIST 数据格式:
 *   - 图像文件: IDX3-UBYTE 格式 (魔数 + 图像数量 + 行数 + 列数 + 像素数据)
 *   - 标签文件: IDX1-UBYTE 格式 (魔数 + 标签数量 + 标签数据)
 *   - 所有多字节整数以大端序 (Big-Endian) 存储
 *   - 本项目数据集: 60000 张训练图像 + 10000 张测试图像，每张 28x28 像素
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "NNMatrix.h"
#include <string>
#include <vector>

using namespace std;

/**
 * @struct MNISTData
 * @brief MNIST 数据集容器
 *
 * 数据结构:
 *   - images: 每个元素是一个 784x1 的列向量，表示一张 28x28 的展平图像
 *             像素值已归一化到 [0, 1] 范围
 *   - labels: 每个元素是一个 10x1 的 one-hot 编码向量
 *             例如数字 3 对应 [0,0,0,1,0,0,0,0,0,0]^T
 */
struct MNISTData {
    vector<NNMatrix> images;   ///< 图像数据向量，每个为 784x1 矩阵
    vector<NNMatrix> labels;   ///< 标签数据向量，每个为 10x1 one-hot 矩阵
};

/**
 * @class DataLoader
 * @brief MNIST 二进制文件解析器
 *
 * [Requirement 2] 从 IDX 格式文件中读取训练/测试数据
 *
 * 支持的 MNIST 文件:
 *   - train-images.idx3-ubyte  (训练集图像)
 *   - train-labels.idx1-ubyte  (训练集标签)
 *   - t10k-images.idx3-ubyte   (测试集图像)
 *   - t10k-labels.idx1-ubyte   (测试集标签)
 */
class DataLoader {
public:
    /**
     * @brief 加载 MNIST 数据集
     * @param image_path 图像二进制文件路径
     * @param label_path 标签二进制文件路径
     * @return 包含归一化图像和 one-hot 标签的 MNISTData 结构
     */
    static MNISTData load_mnist(string image_path, string label_path);
private:
    /**
     * @brief 大端序转小端序 (Big-Endian to Little-Endian)
     * @param i 大端序整数
     * @return 小端序整数
     *
     * MNIST IDX 文件使用大端序存储整数，
     * 而 x86 架构使用小端序，因此需要字节反转。
     */
    static int reverse_int(int i);
};

#endif
