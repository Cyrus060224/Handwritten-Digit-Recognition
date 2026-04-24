/**
 * @file data_loader.cpp
 * @brief MNIST 数据集加载器实现
 *
 * [Requirement 2] 读取 IDX 二进制文件，进行数据预处理
 *
 * 预处理步骤:
 *   1. 读取大端序二进制文件头
 *   2. 字节序转换（大端 → 小端）
 *   3. 像素值归一化: [0, 255] → [0.0, 1.0]
 *   4. 标签 one-hot 编码
 */

#include "data_loader.h"
#include <fstream>

using namespace std;

/**
 * @brief 大端序转小端序
 *
 * 通过位操作将 4 字节整数的字节顺序反转。
 * x86/x64 架构使用小端序，而 MNIST IDX 文件使用大端序。
 *
 * @param i 从文件读入的大端序整数
 * @return 字节反转后的正确整数
 */
int DataLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * @brief 加载 MNIST 数据集
 *
 * 读取图像和标签二进制文件，将像素归一化到 [0, 1]，
 * 并将标签转换为 10 维 one-hot 向量。
 *
 * @param image_path 图像文件路径
 * @param label_path 标签文件路径
 * @return MNISTData
 */
MNISTData DataLoader::load_mnist(string image_path, string label_path) {
    MNISTData data;
    ifstream img_file(image_path, ios::binary);
    ifstream lbl_file(label_path, ios::binary);

    if (img_file.is_open() && lbl_file.is_open()) {
        int magic = 0, count = 0, rows = 0, cols = 0;

        // 读取图像文件头
        img_file.read((char*)&magic, 4);
        img_file.read((char*)&count, 4);
        count = reverse_int(count);
        img_file.read((char*)&rows, 4);
        img_file.read((char*)&cols, 4);

        // 读取标签文件头
        lbl_file.read((char*)&magic, 4);
        lbl_file.read((char*)&magic, 4);

        // 逐样本读取
        for (int i = 0; i < count; i++) {
            NNMatrix img(784, 1);
            for (int r = 0; r < 784; r++) {
                unsigned char pixel = 0;
                img_file.read((char*)&pixel, 1);
                img.data[r][0] = (double)pixel / 255.0;
            }
            data.images.push_back(img);

            unsigned char label = 0;
            lbl_file.read((char*)&label, 1);
            NNMatrix lbl(10, 1);
            lbl.data[(int)label][0] = 1.0;
            data.labels.push_back(lbl);
        }
    }
    return data;
}
