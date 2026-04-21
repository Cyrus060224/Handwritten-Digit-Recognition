/**
 * @file data_loader.cpp
 * @brief MNIST 数据集加载器实现
 *
 * =========================================================================
 * [Requirement 2] 训练和测试数据从文件中读取，进行数据预处理
 * =========================================================================
 *
 * 数据预处理步骤:
 *   1. 读取 IDX3-UBYTE 二进制格式文件（大端序）
 *   2. 字节序转换（大端 → 小端）
 *   3. 像素值归一化: [0, 255] → [0.0, 1.0]
 *   4. 标签 one-hot 编码: 标量标签 → 10 维向量
 */

#include "data_loader.h"
#include <fstream>

using namespace std;

/**
 * @brief 大端序转小端序
 *
 * MNIST IDX 文件格式说明:
 *   所有多字节整数（magic number, count, rows, cols）均以
 *   MSB-first (Big-Endian) 顺序存储。
 *
 *   例如 4 字节整数 0x00000803 在文件中存储为:
 *     [00] [00] [08] [03]  (大端序)
 *
 *   而 x86/x64 架构使用 Little-Endian，读取后为:
 *     0x03080000 (错误)
 *
 *   本函数通过位操作将字节反转:
 *     原始: c1 c2 c3 c4  (c1=最低位字节, c4=最高位字节)
 *     反转后: c4 c3 c2 c1 → c1<<24 | c2<<16 | c3<<8 | c4
 *
 * @param i 从文件直接读入的 4 字节整数（在小端机上字节顺序错误）
 * @return 字节反转后的正确整数
 */
int DataLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;             // 提取最低位字节
    c2 = (i >> 8) & 255;      // 提取第 2 位字节
    c3 = (i >> 16) & 255;     // 提取第 3 位字节
    c4 = (i >> 24) & 255;     // 提取最高位字节
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * @brief 加载 MNIST 数据集
 *
 * =========================================================================
 * [Requirement 2] 从文件中读取训练和测试数据，进行数据预处理
 * =========================================================================
 *
 * IDX3-UBYTE 图像文件格式:
 *   Offset  Type      Description
 *   0       32-bit int  Magic Number (0x00000803)
 *   4       32-bit int  图像数量 (60000 for train, 10000 for test)
 *   8       32-bit int  行数 (28)
 *   12      32-bit int  列数 (28)
 *   16      ubyte       像素数据 (逐行扫描，共 rows*cols 字节)
 *
 * IDX1-UBYTE 标签文件格式:
 *   Offset  Type      Description
 *   0       32-bit int  Magic Number (0x00000801)
 *   4       32-bit int  标签数量
 *   8       ubyte       标签数据 (0-9)
 *
 * 预处理:
 *   - 像素归一化: pixel / 255.0 ∈ [0.0, 1.0]
 *   - One-hot 编码: label=3 → [0,0,0,1,0,0,0,0,0,0]
 *
 * @param image_path 图像二进制文件路径
 * @param label_path 标签二进制文件路径
 * @return 预处理后的 MNISTData
 */
MNISTData DataLoader::load_mnist(string image_path, string label_path) {
    MNISTData data;
    ifstream img_file(image_path, ios::binary);
    ifstream lbl_file(label_path, ios::binary);

    if (img_file.is_open() && lbl_file.is_open()) {
        int magic = 0, count = 0, rows = 0, cols = 0;

        // --- 读取图像文件头 ---
        img_file.read((char*)&magic, 4);   // 魔数验证
        img_file.read((char*)&count, 4);   // 图像数量
        count = reverse_int(count);        // 大端序转换
        img_file.read((char*)&rows, 4);    // 行数 (28)
        img_file.read((char*)&cols, 4);    // 列数 (28)

        // --- 读取标签文件头 ---
        lbl_file.read((char*)&magic, 4);   // 魔数
        lbl_file.read((char*)&magic, 4);   // 标签数量（此处未使用，因为与图像数量一致）

        // --- 逐样本读取 ---
        for (int i = 0; i < count; i++) {
            // 读取一张图像: 28*28 = 784 像素
            NNMatrix img(784, 1);
            for (int r = 0; r < 784; r++) {
                unsigned char pixel = 0;
                img_file.read((char*)&pixel, 1);
                // 预处理: 归一化到 [0, 1]
                img.data[r][0] = (double)pixel / 255.0;
            }
            data.images.push_back(img);

            // 读取对应标签并转为 one-hot 编码
            unsigned char label = 0;
            lbl_file.read((char*)&label, 1);
            NNMatrix lbl(10, 1);
            lbl.data[(int)label][0] = 1.0;  // one-hot: 正确类别位置为 1
            data.labels.push_back(lbl);
        }
    }
    return data;
}
