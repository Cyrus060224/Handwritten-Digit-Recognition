/**
 * @file data_loader.h
 * @brief MNIST 数据集加载器
 *
 * [Requirement 2] 从 IDX 二进制格式文件读取 MNIST 训练/测试数据
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
 * images:  每个元素为 784x1 列向量，像素值已归一化到 [0, 1]
 * labels:  每个元素为 10x1 one-hot 编码向量
 */
struct MNISTData {
    vector<NNMatrix> images;
    vector<NNMatrix> labels;
};

/**
 * @class DataLoader
 * @brief MNIST 二进制文件解析器
 *
 * 解析 IDX3-UBYTE（图像）和 IDX1-UBYTE（标签）格式文件。
 * 所有多字节整数以大端序存储，需进行字节序转换。
 */
class DataLoader {
public:
    /**
     * @brief 加载 MNIST 数据集
     * @param image_path 图像二进制文件路径
     * @param label_path 标签二进制文件路径
     * @return 归一化图像和 one-hot 标签
     */
    static MNISTData load_mnist(string image_path, string label_path);
private:
    /**
     * @brief 大端序转小端序
     * @param i 大端序整数
     * @return 小端序整数
     */
    static int reverse_int(int i);
};

#endif
