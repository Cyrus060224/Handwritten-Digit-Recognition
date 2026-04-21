/**
 * @file neural_network.cpp
 * @brief 多层前馈神经网络核心实现
 *
 * =========================================================================
 * [Requirement 1] 实现多层前馈神经网络和反向传播算法的核心算法
 * =========================================================================
 *
 * 本文件是神经网络的"大脑"，包含:
 *   1. 前向传播 (Forward Propagation)
 *   2. 反向传播 (Backpropagation, BP) -- 链式法则的精确实现
 *   3. 梯度更新 (Gradient Descent)
 *   4. 模型持久化 (Save/Load)
 *   5. 贪心策略快照 (Checkpoint)
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
 * 维度说明:
 *   weights: (current_nodes × previous_nodes)
 *   biases:  (current_nodes × 1)
 *   z, a:    (current_nodes × 1) -- 列向量
 *
 * 初始化后立即调用 randomize() 进行 Xavier 初始化。
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
 * =========================================================================
 * [Requirement 4] 系统可设置网络的层数及每层神经元个数
 * =========================================================================
 *
 * 网络构建过程:
 *   输入 topology = {n0, n1, n2, ..., nL}
 *   创建 L 层:
 *     Layer 1: (n1 × n0) -- 第一隐藏层
 *     Layer 2: (n2 × n1) -- 第二隐藏层 (可选)
 *     ...
 *     Layer L: (nL × n_{L-1}) -- 输出层
 *
 * 示例: topology = {784, 128, 10}
 *   Layer 1: weights(128×784), biases(128×1)
 *   Layer 2: weights(10×128), biases(10×1)
 *
 * @param topology 网络拓扑向量 {输入层节点数, 隐藏层节点数, ..., 输出层节点数}
 * @param lr 初始学习率
 * @param act 隐藏层激活函数 (SIGMOID / RELU / TANH)
 * @param keep_rate Dropout 保留概率
 */
NeuralNetwork::NeuralNetwork(vector<int> topology, double lr,
                              ActivationType act, double keep_rate)
    : learningRate(lr), hidden_activation(act), keep_rate(keep_rate)
{
    // 根据拓扑向量创建各层
    for (size_t i = 1; i < topology.size(); i++) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
    reset_gradients();  // 初始化梯度累加器
    backup_layers = layers; // 初始化贪心快照
}

/**
 * @brief 贪心策略 -- 保存权重快照
 *
 * =========================================================================
 * [Requirement 1] 贪心算法的融合 -- 基于贪心的权重回溯机制
 * =========================================================================
 *
 * 在应用梯度更新之前，将当前所有层的权重完整备份。
 * 如果更新后损失显著上升，则回滚到此次快照。
 *
 * vector 的赋值运算符会自动执行深拷贝，因此 backup_layers
 * 是 layers 的完整独立副本。
 */
void NeuralNetwork::save_checkpoint() {
    backup_layers = layers;
}

/**
 * @brief 贪心策略 -- 恢复权重快照（回滚）
 *
 * 当 Greedy Bold Driver 检测到损失恶化时，调用此函数
 * 将网络权重恢复到更新前的状态，防止网络发散。
 */
void NeuralNetwork::load_checkpoint() {
    layers = backup_layers;
}

/**
 * @brief 前向传播 (Forward Propagation)
 *
 * =========================================================================
 * [Requirement 1] 多层前馈神经网络的前向传播算法
 * =========================================================================
 *
 * 数学推导:
 *   对于第 l 层 (l = 1, 2, ..., L):
 *
 *     z[l] = W[l] × a[l-1] + b[l]     (线性变换)
 *     a[l] = f_l(z[l])                 (非线性激活)
 *
 *   其中:
 *     - a[0] = input (输入层)
 *     - f_L = sigmoid (输出层固定使用 Sigmoid)
 *     - f_l (l < L) 由 hidden_activation 决定
 *
 *   输出层 a[L] 是一个 10 维向量，每个元素表示对应数字的概率估计。
 *
 * Dropout 处理 (仅在训练模式且 keep_rate < 1.0 时):
 *   a[l] = a[l] ⊙ mask[l]
 *   其中 mask[l] 是随机生成的 Inverted Dropout 掩码
 *   注意: 输出层不使用 Dropout
 *
 * @param input 输入向量 (784×1)，归一化像素值
 * @param is_training 训练模式标志（决定是否应用 Dropout）
 * @return 输出层激活值 (10×1)，表示各数字类别的预测概率
 */
NNMatrix NeuralNetwork::forward(NNMatrix input, bool is_training) {
    NNMatrix current_activation = input;

    for (size_t i = 0; i < layers.size(); i++) {
        // 线性变换: z[l] = W[l] * a[l-1]
        layers[i].z = NNMatrix::multiply(layers[i].weights, current_activation);
        // 加上偏置: z[l] += b[l]
        layers[i].z.add(layers[i].biases);
        layers[i].a = layers[i].z;

        // 应用激活函数
        if (i == layers.size() - 1) {
            // 输出层：固定使用 Sigmoid
            // [Requirement 5] 输出层激活函数
            layers[i].a.map(sigmoid);
        } else {
            // 隐藏层：根据配置选择激活函数
            // [Requirement 5] 系统可设置隐藏层激活函数
            if (hidden_activation == RELU) {
                layers[i].a.map(relu);
            } else if (hidden_activation == TANH) {
                layers[i].a.map(tanh_act);
            } else {
                layers[i].a.map(sigmoid);
            }

            // [Requirement 10] Dropout 正则化（仅在隐藏层且训练模式）
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
 * 在每个新批次开始时调用，将所有梯度累加器归零，
 * 并将已累积样本数清零。
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
 * @brief 反向传播 (Backpropagation, BP) -- 核心算法
 *
 * =========================================================================
 * [Requirement 1] 手动实现反向传播算法，基于数学推导，理解链式法则
 * =========================================================================
 *
 * =========================================================================
 * 完整的数学推导
 * =========================================================================
 *
 * 损失函数 (均方误差 MSE):
 *   L = 1/2 * ||target - a[L]||^2
 *     = 1/2 * Σ(j=1 to 10) (target[j] - a[L][j])^2
 *
 * -------------------------------------------------------------------------
 * 步骤 1: 输出层误差 (l = L)
 * -------------------------------------------------------------------------
 *
 *   根据链式法则:
 *     δ[L] = ∂L/∂z[L] = ∂L/∂a[L] · ∂a[L]/∂z[L]
 *
 *   其中:
 *     ∂L/a[L] = a[L] - target          (MSE 对激活值的导数)
 *     ∂a[L]/∂z[L] = sigmoid'(z[L])       (Sigmoid 导数)
 *
 *   所以:
 *     δ[L] = (a[L] - target) ⊙ sigmoid'(z[L])
 *          = -(target - a[L]) ⊙ sigmoid'(z[L])
 *
 *   代码中: errors = target - output = -(a[L] - target)
 *          gradients = sigmoid'(z[L])
 *          gradients = gradients ⊙ errors = -δ[L]
 *
 *   注意代码中的符号: errors = target - output (与上述推导相差一个负号)
 *   因此 gradients 实际存储的是 -δ，但在权重更新时:
 *     W += lr * (-gradients) * a_prev^T
 *   负号被吸收，最终效果等价于:
 *     W -= lr * δ * a_prev^T  ✓
 *
 * -------------------------------------------------------------------------
 * 步骤 2: 隐藏层误差反向传播 (l = L-1, L-2, ..., 1)
 * -------------------------------------------------------------------------
 *
 *   对于隐藏层 l:
 *     δ[l] = ∂L/z[l]
 *
 *   根据链式法则:
 *     δ[l] = ∂L/∂z[l+1] · ∂z[l+1]/∂a[l] · ∂a[l]/∂z[l]
 *          = (W[l+1]^T × δ[l+1]) ⊙ f_l'(z[l])
 *
 *   写成矩阵形式:
 *     δ[l] = (W[l+1]^T × δ[l+1]) ⊙ f_l'(z[l])
 *
 *   代码实现:
 *     errors = W[l+1]^T × errors   (传播误差到前一层)
 *     gradients = f_l'(z[l])       (计算当前层激活导数)
 *     gradients = gradients ⊙ errors
 *
 * -------------------------------------------------------------------------
 * 步骤 3: 计算权重和偏置梯度
 * -------------------------------------------------------------------------
 *
 *   对于第 l 层:
 *     ∂L/∂W[l] = δ[l] × a[l-1]^T     (外积)
 *     ∂L/∂b[l] = δ[l]
 *
 *   代码中（注意符号约定）:
 *     deltas = gradients × a[l-1]^T    (即 -∂L/∂W[l])
 *     weight_gradients_acc[l] += deltas
 *     bias_gradients_acc[l] += gradients  (即 -∂L/∂b[l])
 *
 *   在 apply_gradients() 中:
 *     W[l] += lr * weight_gradients_acc[l] / n
 *   等价于:
 *     W[l] -= lr * (1/n) * Σ ∂L/∂W[l]   ✓
 *
 * -------------------------------------------------------------------------
 * 步骤 4: Dropout 梯度修正
 * -------------------------------------------------------------------------
 *
 *   [Requirement 10] 使用 Dropout 时，反向传播需要乘以相同的掩码:
 *     δ[l] = δ[l] ⊙ mask[l]
 *
 *   这是因为 Inverted Dropout 在前向时乘以了 mask，
 *   反向时也需要乘以相同的 mask 来保持梯度一致性。
 *
 * @param input 输入向量 (784×1)
 * @param target 目标 one-hot 向量 (10×1)
 */
void NeuralNetwork::accumulate_gradients(NNMatrix input, NNMatrix target) {
    // 前向传播，同时保存中间变量 (z, a) 供反向传播使用
    NNMatrix output = forward(input, true);

    // 计算输出误差: errors = target - output
    // 这等价于 -(a[L] - target)，即负的 MSE 梯度
    NNMatrix errors = NNMatrix::subtract(target, output);

    // --- 从输出层向输入层反向传播 ---
    for (int i = (int)layers.size() - 1; i >= 0; i--) {
        // 从 z 值开始计算梯度
        NNMatrix gradients = layers[i].z;

        // 应用激活函数导数（链式法则的关键步骤）
        if (i == (int)layers.size() - 1) {
            // 输出层使用 Sigmoid 导数
            gradients.map(sigmoid_derivative);
        } else {
            // 隐藏层根据配置选择导数
            if (hidden_activation == RELU) {
                gradients.map(relu_derivative);
            } else if (hidden_activation == TANH) {
                gradients.map(tanh_derivative);
            } else {
                gradients.map(sigmoid_derivative);
            }

            // [Requirement 10] Dropout 掩码修正
            if (keep_rate < 1.0) {
                gradients.multiply_elements(layers[i].dropout_mask);
            }
        }

        // 链式法则: 将误差与激活导数相乘
        // gradients = errors ⊙ f'(z[l]) = -δ[l]
        gradients.multiply_elements(errors);

        // 计算权重梯度: ∂L/W[l] = δ[l] × a[l-1]^T
        // prev_activation_T = a[l-1]^T
        NNMatrix prev_activation_T = (i == 0) ? input.transpose() : layers[i - 1].a.transpose();
        // deltas = gradients × a[l-1]^T (累积梯度)
        NNMatrix deltas = NNMatrix::multiply(gradients, prev_activation_T);

        // 累加到梯度累加器
        weight_gradients_acc[i].add(deltas);
        bias_gradients_acc[i].add(gradients);

        // 将误差传播到前一层: errors = W[l]^T × errors
        NNMatrix weights_T = layers[i].weights.transpose();
        errors = NNMatrix::multiply(weights_T, errors);
    }
    accumulated_samples++;
}

/**
 * @brief 应用累积的梯度更新网络权重
 *
 * =========================================================================
 * [Requirement 3] 提供不同的优化方案: BGD / SGD / Mini-Batch
 * =========================================================================
 *
 * 权重更新公式:
 *   W[l] ← W[l] + (lr / n) * Σ(k=1 to n) (-∂L/∂W[l])_k
 *
 *   其中:
 *     - n = accumulated_samples (当前批次样本数)
 *     - lr = learningRate (学习率)
 *     - 累加器中存储的是 -∂L/W[l] 的和
 *
 * 三种优化方案的区别仅在于 accumulated_samples 的值:
 *   - SGD:      每处理 1 个样本就调用一次，n = 1
 *   - Mini-Batch: 每处理 128 个样本调用一次，n = 128
 *   - BGD:      每处理完全部训练样本调用一次，n = 60000
 *
 * 更新步骤:
 *   1. 将累加梯度乘以 lr/n（平均梯度 × 学习率）
 *   2. W[l] += 缩放后的梯度
 *   3. b[l] += 缩放后的偏置梯度
 *   4. 重置累加器
 */
void NeuralNetwork::apply_gradients() {
    if (accumulated_samples == 0) return;

    // 计算有效学习率: lr_effective = lr / n
    double lr = learningRate / (double)accumulated_samples;

    for (size_t i = 0; i < layers.size(); i++) {
        // 缩放梯度: gradient *= lr/n
        weight_gradients_acc[i].map([lr](double x) { return x * lr; });
        bias_gradients_acc[i].map([lr](double x) { return x * lr; });

        // 应用更新: W += scaled_gradient
        layers[i].weights.add(weight_gradients_acc[i]);
        layers[i].biases.add(bias_gradients_acc[i]);
    }
    reset_gradients(); // 重置累加器，准备下一批次
}

/**
 * @brief 单样本训练（前向 + 反向 + 更新）
 *
 * 这是 SGD 模式的核心调用方式：每个样本立即更新权重。
 * 在 Mini-Batch 和 BGD 模式下，此函数不被直接调用，
 * 而是由 main.cpp 中的训练循环分别调用 accumulate_gradients 和 apply_gradients。
 */
void NeuralNetwork::train(NNMatrix input, NNMatrix target) {
    accumulate_gradients(input, target);
    apply_gradients();
}

/**
 * @brief 推理预测：返回预测的数字类别
 *
 * 前向传播后，取输出层 10 个神经元中激活值最大的索引作为预测结果。
 * 这对应于 argmax(output) 操作。
 *
 * @param input 输入向量 (784×1)
 * @return 预测的数字 (0-9)
 */
int NeuralNetwork::predict(NNMatrix input) {
    NNMatrix output = forward(input, false); // 推理模式，不应用 Dropout
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
 * @brief 在数据集上评估分类准确率
 *
 * =========================================================================
 * [Requirement 12] 系统具有训练模块和测试模块，得到模型准确度评估
 * =========================================================================
 *
 * 遍历数据集中每个样本，用 predict() 进行推理，
 * 统计预测正确的样本数，计算准确率 = 正确数 / 总数。
 *
 * @param data 评估数据集（训练集或测试集）
 * @return 准确率 (0.0 ~ 1.0)
 */
double NeuralNetwork::get_accuracy(const MNISTData& data) {
    if (data.images.empty()) return 0.0;

    int correct = 0;
    for (size_t i = 0; i < data.images.size(); i++) {
        int prediction = predict(data.images[i]);
        int label = 0;

        // 从 one-hot 向量中恢复原始标签
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
 * =========================================================================
 * [Requirement 2] 从文件中读取参数构造网络
 * =========================================================================
 *
 * 文件格式:
 *   第 1 行: 层数 L
 *   对于每层 l:
 *     第 1 行: 行数 列数
 *     接下来 rows 行: 权重矩阵数据 (每行 cols 个值)
 *     接下来 1 行: 偏置向量数据 (rows 个值)
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

        // 读取权重矩阵
        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c; col++) {
                in >> L.weights.data[row][col];
            }
        }
        // 读取偏置向量
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
 * =========================================================================
 * [Requirement 2] 将网络参数存储于文件
 * =========================================================================
 *
 * 保存格式与 load_model 对应，确保模型可持久化和恢复。
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
        // 保存权重矩阵
        for (int r = 0; r < L.weights.rows; r++) {
            for (int c = 0; c < L.weights.cols; c++) {
                out << L.weights.data[r][c] << " ";
            }
            out << "\n";
        }
        // 保存偏置向量
        for (int r = 0; r < L.biases.rows; r++) {
            out << L.biases.data[r][0] << " ";
        }
        out << "\n";
    }
    out.close();
    cout << "Success: Model weights permanently saved to: " << filename << endl;
}
