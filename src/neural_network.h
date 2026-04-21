/**
 * @file neural_network.h
 * @brief 多层前馈神经网络核心定义
 *
 * =========================================================================
 * [Requirement 1] 实现多层前馈神经网络和反向传播算法的核心算法
 * [Requirement 4] 系统可设置网络的层数及每层神经元个数
 * [Requirement 5] 系统可设置隐藏层激活函数
 * [Requirement 10] 提供防止过拟合的方法：早停、随机丢弃
 * =========================================================================
 *
 * 本文件定义了神经网络的层结构 (Layer) 和网络整体结构 (NeuralNetwork)。
 * 核心功能包括:
 *   - 前向传播 (forward)
 *   - 反向传播 (accumulate_gradients)
 *   - 梯度应用 (apply_gradients)
 *   - 模型持久化 (save_model / load_model)
 *   - 贪心快照 (save_checkpoint / load_checkpoint)
 *   - Dropout 正则化
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "NNMatrix.h"
#include "data_loader.h"
#include <vector>
#include <string>

using namespace std;

/**
 * @enum ActivationType
 * @brief 激活函数类型枚举
 *
 * [Requirement 5] 系统可设置隐藏层激活函数：logistic(sigmoid)、tanh、relu
 */
enum ActivationType { SIGMOID, RELU, TANH };

/**
 * @struct Layer
 * @brief 神经网络单层结构
 *
 * [Requirement 11] 输出网络结构：Input layer → Hidden layer → Output layer
 *
 * 每层包含:
 *   - weights: 权重矩阵 W (current_nodes × previous_nodes)
 *   - biases:  偏置向量 b (current_nodes × 1)
 *   - z:       加权和 (z = W*a_prev + b)，激活前的值
 *   - a:       激活输出 (a = f(z))
 *   - dropout_mask: Dropout 掩码（训练时使用）
 */
struct Layer {
    NNMatrix weights;       ///< 权重矩阵: [n_current × n_previous]
    NNMatrix biases;        ///< 偏置向量: [n_current × 1]
    NNMatrix z;             ///< 加权输入: z = W * a_prev + b
    NNMatrix a;             ///< 激活输出: a = f(z)
    NNMatrix dropout_mask;  ///< [Requirement 10] Dropout 随机掩码矩阵

    /**
     * @brief 构造新层并进行 Xavier 初始化
     * @param current_nodes 当前层神经元数
     * @param previous_nodes 前一层神经元数
     */
    Layer(int current_nodes, int previous_nodes);
};

/**
 * @struct NeuralNetwork
 * @brief 多层前馈神经网络
 *
 * [Requirement 1] 手动实现反向传播算法
 * [Requirement 4] 可设置网络层数及每层神经元个数 (如 784→128→10)
 * [Requirement 10] 支持早停和 Dropout 防过拟合
 *
 * 网络拓扑示例:
 *   topology = {784, 128, 10}
 *   → 输入层: 784 节点 (28×28 展平)
 *   → 隐藏层: 128 节点
 *   → 输出层: 10 节点 (数字 0-9)
 */
struct NeuralNetwork {
    vector<Layer> layers;             ///< 网络各层
    vector<Layer> backup_layers;      ///< [Requirement 1] 贪心策略的权重快照备份

    double learningRate;              ///< 学习率 (可自适应调整)
    ActivationType hidden_activation; ///< [Requirement 5] 隐藏层激活函数类型
    double keep_rate;                 ///< [Requirement 10] Dropout 保留概率 (1.0 = 不使用 Dropout)

    // --- 梯度累加器（支持 Mini-Batch / BGD） ---
    // [Requirement 3] 支持 SGD / Mini-Batch / BGD 三种优化方案
    vector<NNMatrix> weight_gradients_acc;  ///< 权重梯度累加器
    vector<NNMatrix> bias_gradients_acc;    ///< 偏置梯度累加器
    int accumulated_samples;                ///< 当前批次已累积的样本数

    /**
     * @brief 构造神经网络
     * @param topology 网络拓扑向量，如 {784, 128, 10}
     * @param lr 初始学习率
     * @param act 隐藏层激活函数类型
     * @param keep_rate Dropout 保留概率 (1.0 表示不使用 Dropout)
     */
    NeuralNetwork(vector<int> topology, double lr = 0.01,ActivationType act = SIGMOID, double keep_rate = 1.0);

    /**
     * @brief 前向传播
     * @param input 输入向量 (784×1)
     * @param is_training 是否为训练模式（影响 Dropout）
     * @return 输出层激活值 (10×1)
     */
    NNMatrix forward(NNMatrix input, bool is_training = false);

    /**
     * @brief 单样本训练（前向+反向+更新）
     * @param input 输入向量
     * @param target 目标 one-hot 向量
     */
    void train(NNMatrix input, NNMatrix target);

    /**
     * @brief 推理预测
     * @param input 输入向量
     * @return 预测的数字类别 (0-9)
     */
    int predict(NNMatrix input);

    /**
     * @brief 在数据集上评估准确率
     *
     * [Requirement 12] 系统具有训练模块和测试模块，得到模型准确度评估
     *
     * @param data 测试数据集
     * @return 分类准确率 (0.0 ~ 1.0)
     */
    double get_accuracy(const MNISTData& data);

    /**
     * @brief 将模型权重保存到文件
     *
     * [Requirement 2] 将网络参数存储于文件
     *
     * @param filename 输出文件路径
     */
    void save_model(string filename);

    /**
     * @brief 从文件加载模型权重
     *
     * [Requirement 2] 从文件中读取参数构造网络
     *
     * @param filename 模型文件路径
     */
    void load_model(string filename);

    /**
     * @brief 重置梯度累加器（新批次开始时调用）
     */
    void reset_gradients();

    /**
     * @brief 累积单个样本的梯度
     *
     * [Requirement 1] 手动实现反向传播 (Backpropagation, BP)
     *
     * 此函数执行完整的反向传播算法，计算损失函数对权重和偏置的梯度，
     * 并将梯度累加到梯度累加器中。
     *
     * @param input 输入向量
     * @param target 目标 one-hot 向量
     */
    void accumulate_gradients(NNMatrix input, NNMatrix target);

    /**
     * @brief 应用累积的梯度更新权重
     *
     * [Requirement 3] 根据 accumulated_samples 支持不同优化方案:
     *   - SGD:      accumulated_samples = 1，每样本更新
     *   - Mini-Batch: accumulated_samples = 128，每 128 样本更新
     *   - BGD:      accumulated_samples = 全部样本，每轮更新
     */
    void apply_gradients();

    // --- [Requirement 1] 贪心策略快照接口 ---
    void save_checkpoint();   ///< 保存当前权重快照（贪心回溯用）
    void load_checkpoint();   ///< 恢复权重快照（贪心回溯用）
};

#endif
