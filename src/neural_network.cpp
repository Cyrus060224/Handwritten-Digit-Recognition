/**
 * @file neural_network.cpp
 * @brief 多层前馈神经网络核心实现
 *
 * [Requirement 1] 手动实现前向传播与反向传播算法
 */

#include "neural_network.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include "activations.h"

using namespace std;

/**
 * @brief 构造单层网络，初始化权重和偏置
 *
 * 维度: weights(current_nodes × previous_nodes), biases(current_nodes × 1)
 * 构造后执行 Xavier 初始化。
 */
Layer::Layer(int current_nodes, int previous_nodes)
    : weights(current_nodes, previous_nodes),
      biases(current_nodes, 1),
      z(current_nodes, 1),
      a(current_nodes, 1),
      dropout_mask(current_nodes, 1)
{
    weights.randomize();
    biases.randomize();
}

/**
 * @brief 构造多层前馈神经网络
 *
 * [Requirement 4] 根据拓扑向量构建网络层。
 * 示例: {784, 128, 10} → 输入层784 → 隐藏层128 → 输出层10
 *
 * @param topology 网络拓扑向量
 * @param lr 初始学习率
 * @param act 隐藏层激活函数类型
 * @param keep_rate Dropout 保留概率
 */
NeuralNetwork::NeuralNetwork(vector<int> topology, double lr,
                              ActivationType act, double keep_rate)
    : learningRate(lr), hidden_activation(act), keep_rate(keep_rate)
{
    for (size_t i = 1; i < topology.size(); i++) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
    reset_gradients();
    backup_layers = layers;
}

/**
 * @brief 贪心策略 -- 保存权重快照
 *
 * [Requirement 1] 在应用梯度更新前备份当前权重。
 * vector 赋值自动执行深拷贝。
 */
void NeuralNetwork::save_checkpoint() {
    backup_layers = layers;
}

/**
 * @brief 贪心策略 -- 恢复权重快照
 *
 * 当 Greedy Bold Driver 检测到损失恶化时，回滚到快照状态。
 */
void NeuralNetwork::load_checkpoint() {
    layers = backup_layers;
}

/**
 * @brief 前向传播 (Forward Propagation)
 *
 * [Requirement 1] 逐层计算 z[l] = W[l] × a[l-1] + b[l]，再应用激活函数。
 * 输出层固定使用 Sigmoid，隐藏层根据配置选择。
 * 训练模式下隐藏层应用 Dropout（输出层不使用）。
 *
 * @param input 输入向量 (784×1)
 * @param is_training 训练模式标志
 * @return 输出层激活值 (10×1)
 */
NNMatrix NeuralNetwork::forward(NNMatrix input, bool is_training) {
    NNMatrix current_activation = input;

    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].z = NNMatrix::multiply(layers[i].weights, current_activation);
        layers[i].z.add(layers[i].biases);
        layers[i].a = layers[i].z;

        if (i == layers.size() - 1) {
            layers[i].a.map(sigmoid);
        } else {
            if (hidden_activation == RELU) {
                layers[i].a.map(relu);
            } else if (hidden_activation == TANH) {
                layers[i].a.map(tanh_act);
            } else {
                layers[i].a.map(sigmoid);
            }

            if (is_training && keep_rate < 1.0) {
                layers[i].dropout_mask = NNMatrix::generate_dropout_mask(
                    layers[i].a.rows, layers[i].a.cols, keep_rate);
                layers[i].a.multiply_elements(layers[i].dropout_mask);
            }
        }
        current_activation = layers[i].a;
    }
    return current_activation;
}

/**
 * @brief 重置梯度累加器
 *
 * 将权重和偏置的梯度累加器清零，并重置样本计数器。
 */
void NeuralNetwork::reset_gradients() {
    weight_gradients_acc.clear();
    bias_gradients_acc.clear();

    for (const auto& L : layers) {
        weight_gradients_acc.push_back(NNMatrix(L.weights.rows, L.weights.cols));
        bias_gradients_acc.push_back(NNMatrix(L.biases.rows, L.biases.cols));
    }
    accumulated_samples = 0;
}

/**
 * @brief 反向传播 (Backpropagation) -- 核心算法
 *
 * [Requirement 1] 手动实现反向传播，计算损失对权重和偏置的梯度。
 *
 * 核心逻辑:
 *   1. 前向传播得到输出，计算输出层误差: errors = target - output
 *   2. 从输出层倒序遍历到输入层，逐层执行:
 *      a. 将当前层的 z 值替换为激活函数导数 f'(z)
 *      b. 若使用 Dropout，乘以掩码修正
 *      c. 与误差做 Hadamard 积得到误差项: gradients = errors ⊙ f'(z)
 *      d. 计算权重梯度（外积）: deltas = gradients × a[l-1]^T
 *      e. 累加权重和偏置梯度到累加器
 *      f. 将误差通过转置权重传播到前一层: errors = W[l]^T × errors
 *
 * Dropout 情况下（第 316-318 行），反向传播需乘以相同掩码保持梯度一致性。
 *
 * @param input 输入向量 (784×1)
 * @param target 目标 one-hot 向量 (10×1)
 */
void NeuralNetwork::accumulate_gradients(NNMatrix input, NNMatrix target) {
    NNMatrix output = forward(input, true);
    NNMatrix errors = NNMatrix::subtract(target, output);

    for (int i = (int)layers.size() - 1; i >= 0; i--) {
        NNMatrix gradients = layers[i].z;

        if (i == (int)layers.size() - 1) {
            gradients.map(sigmoid_derivative);
        } else {
            if (hidden_activation == RELU) {
                gradients.map(relu_derivative);
            } else if (hidden_activation == TANH) {
                gradients.map(tanh_derivative);
            } else {
                gradients.map(sigmoid_derivative);
            }

            if (keep_rate < 1.0) {
                gradients.multiply_elements(layers[i].dropout_mask);
            }
        }

        gradients.multiply_elements(errors);

        NNMatrix prev_activation_T = (i == 0) ? input.transpose() : layers[i - 1].a.transpose();
        NNMatrix deltas = NNMatrix::multiply(gradients, prev_activation_T);

        weight_gradients_acc[i].add(deltas);
        bias_gradients_acc[i].add(gradients);

        NNMatrix weights_T = layers[i].weights.transpose();
        errors = NNMatrix::multiply(weights_T, errors);
    }
    accumulated_samples++;
}

/**
 * @brief 应用累积的梯度更新网络权重
 *
 * [Requirement 3] 根据 accumulated_samples 区分三种优化方案:
 *   - SGD:      n=1，每样本更新
 *   - Mini-Batch: n=128，每128样本更新
 *   - BGD:      n=全部样本，每轮更新
 *
 * 更新公式: W[l] += lr/n × 累积梯度
 */
void NeuralNetwork::apply_gradients() {
    if (accumulated_samples == 0) return;

    double lr = learningRate / (double)accumulated_samples;

    for (size_t i = 0; i < layers.size(); i++) {
        weight_gradients_acc[i].map([lr](double x) { return x * lr; });
        bias_gradients_acc[i].map([lr](double x) { return x * lr; });

        layers[i].weights.add(weight_gradients_acc[i]);
        layers[i].biases.add(bias_gradients_acc[i]);
    }
    reset_gradients();
}

/**
 * @brief 单样本训练（前向 + 反向 + 更新）
 *
 * 适用于 SGD 模式。Mini-Batch 和 BGD 模式下分别调用
 * accumulate_gradients 和 apply_gradients。
 */
void NeuralNetwork::train(NNMatrix input, NNMatrix target) {
    accumulate_gradients(input, target);
    apply_gradients();
}

/**
 * @brief 推理预测
 *
 * 前向传播后，返回输出层激活值最大的索引作为预测结果 (argmax)。
 *
 * @param input 输入向量
 * @return 预测数字 (0-9)
 */
int NeuralNetwork::predict(NNMatrix input) {
    NNMatrix output = forward(input, false);
    int maxIndex = 0;
    double maxValue = output.data[0][0];

    for (int i = 1; i < 10; i++) {
        if (output.data[i][0] > maxValue) {
            maxValue = output.data[i][0];
            maxIndex = i;
        }
    }
    return maxIndex;
}

/**
 * @brief 评估分类准确率
 *
 * [Requirement 12] 遍历数据集，逐样本预测并与 one-hot 标签对比，
 * 统计正确预测数占总样本数的比例。
 *
 * @param data 评估数据集
 * @return 准确率 (0.0 ~ 1.0)
 */
double NeuralNetwork::get_accuracy(const MNISTData& data) {
    if (data.images.empty()) return 0.0;

    int correct = 0;
    for (size_t i = 0; i < data.images.size(); i++) {
        int prediction = predict(data.images[i]);
        int label = 0;

        for (int r = 0; r < 10; r++) {
            if (data.labels[i].data[r][0] > 0.5) {
                label = r;
                break;
            }
        }

        if (prediction == label) {
            correct++;
        }
    }
    return (double)correct / data.images.size();
}

/**
 * @brief 从文件加载模型权重
 *
 * [Requirement 2] 文件格式:
 *   第1行: 层数 L
 *   每层: 行数 列数 → 权重矩阵 → 偏置向量
 *
 * @param filename 模型文件路径
 */
void NeuralNetwork::load_model(string filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error: Model file not found: " << filename << endl;
        return;
    }

    int layer_count;
    in >> layer_count;
    layers.clear();

    for (int i = 0; i < layer_count; i++) {
        int r, c;
        in >> r >> c;
        Layer L(r, c);

        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c; col++) {
                in >> L.weights.data[row][col];
            }
        }
        for (int row = 0; row < r; row++) {
            in >> L.biases.data[row][0];
        }
        layers.push_back(L);
    }
    in.close();
    cout << "Model loaded successfully. Weights restored!" << endl;
}

/**
 * @brief 将模型权重保存到文件
 *
 * [Requirement 2] 按层数 → 维度 → 权重矩阵 → 偏置向量的顺序写入文件。
 *
 * @param filename 输出文件路径
 */
void NeuralNetwork::save_model(string filename) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Cannot create model file: " << filename << endl;
        return;
    }

    out << layers.size() << "\n";
    for (const auto& L : layers) {
        out << L.weights.rows << " " << L.weights.cols << "\n";
        for (int r = 0; r < L.weights.rows; r++) {
            for (int c = 0; c < L.weights.cols; c++) {
                out << L.weights.data[r][c] << " ";
            }
            out << "\n";
        }
        for (int r = 0; r < L.biases.rows; r++) {
            out << L.biases.data[r][0] << " ";
        }
        out << "\n";
    }
    out.close();
    cout << "Success: Model weights permanently saved to: " << filename << endl;
}
