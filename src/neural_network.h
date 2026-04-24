/**
 * @file neural_network.h
 * @brief 多层前馈神经网络核心定义
 *
 * [Requirement 1] 前馈神经网络与反向传播算法
 * [Requirement 4] 可配置网络层数及每层神经元个数
 * [Requirement 5] 可配置隐藏层激活函数
 * [Requirement 10] Dropout 与早停防过拟合
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
 * @brief 激活函数类型
 *
 * [Requirement 5] sigmoid / relu / tanh
 */
enum ActivationType { SIGMOID, RELU, TANH };

/**
 * @struct Layer
 * @brief 神经网络单层结构
 *
 * 包含权重矩阵、偏置向量、前向传播的中间变量（z 和 a）以及 Dropout 掩码。
 */
struct Layer {
    NNMatrix weights;
    NNMatrix biases;
    NNMatrix z;
    NNMatrix a;
    NNMatrix dropout_mask;

    /**
     * @brief 构造新层，执行 Xavier 初始化
     * @param current_nodes 当前层神经元数
     * @param previous_nodes 前一层神经元数
     */
    Layer(int current_nodes, int previous_nodes);
};

/**
 * @struct NeuralNetwork
 * @brief 多层前馈神经网络
 *
 * 支持任意层数的全连接前馈网络，默认拓扑示例: {784, 128, 10}
 */
struct NeuralNetwork {
    vector<Layer> layers;
    vector<Layer> backup_layers;  ///< 贪心策略权重快照

    double learningRate;
    ActivationType hidden_activation;
    double keep_rate;  ///< Dropout 保留概率 (1.0 = 不使用)

    // 梯度累加器（支持 Mini-Batch / BGD）
    vector<NNMatrix> weight_gradients_acc;
    vector<NNMatrix> bias_gradients_acc;
    int accumulated_samples;

    /**
     * @brief 构造神经网络
     * @param topology 网络拓扑向量，如 {784, 128, 10}
     * @param lr 初始学习率
     * @param act 隐藏层激活函数类型
     * @param keep_rate Dropout 保留概率
     */
    NeuralNetwork(vector<int> topology, double lr = 0.01, ActivationType act = SIGMOID, double keep_rate = 1.0);

    /**
     * @brief 前向传播
     * @param input 输入向量 (784×1)
     * @param is_training 训练模式标志（决定是否应用 Dropout）
     * @return 输出层激活值 (10×1)
     */
    NNMatrix forward(NNMatrix input, bool is_training = false);

    /**
     * @brief 单样本训练（前向 + 反向 + 更新）
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
     * @brief 评估数据集分类准确率
     *
     * [Requirement 12] 测试模块 - 模型准确度评估
     *
     * @param data 评估数据集
     * @return 准确率 (0.0 ~ 1.0)
     */
    double get_accuracy(const MNISTData& data);

    /**
     * @brief 将模型权重保存到文件
     *
     * [Requirement 2] 网络参数持久化
     *
     * @param filename 输出文件路径
     */
    void save_model(string filename);

    /**
     * @brief 从文件加载模型权重
     *
     * [Requirement 2] 从文件读取参数构造网络
     *
     * @param filename 模型文件路径
     */
    void load_model(string filename);

    /**
     * @brief 重置梯度累加器
     */
    void reset_gradients();

    /**
     * @brief 累积单个样本的梯度（执行反向传播）
     *
     * [Requirement 1] 手动实现反向传播算法
     *
     * @param input 输入向量
     * @param target 目标 one-hot 向量
     */
    void accumulate_gradients(NNMatrix input, NNMatrix target);

    /**
     * @brief 应用累积的梯度更新权重
     *
     * [Requirement 3] 支持 SGD / Mini-Batch / BGD 三种优化方案
     */
    void apply_gradients();

    /**
     * @brief 保存当前权重快照（贪心回溯用）
     */
    void save_checkpoint();

    /**
     * @brief 恢复权重快照（贪心回溯用）
     */
    void load_checkpoint();
};

#endif
