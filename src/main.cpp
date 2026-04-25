/**
 * @file main.cpp
 * @brief NoobNetwork 主程序 -- GUI 交互、训练循环、AutoML 搜索
 *
 * [Requirement 3]  SGD / Mini-Batch / BGD 优化方案选择
 * [Requirement 4]  可配置网络层数及每层神经元个数
 * [Requirement 6]  可配置最大迭代次数、早停
 * [Requirement 7]  K 折交叉验证
 * [Requirement 8]  网格搜索 / 随机搜索超参数优化
 * [Requirement 9]  损失曲线和准确率曲线实时可视化
 * [Requirement 10] 早停 + Dropout + Greedy Bold Driver 自适应学习率
 * [Requirement 12] 训练模块和测试模块（性能评估）
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "neural_network.h"
#include "data_loader.h"
#include "activations.h"
#include <sys/stat.h>

using namespace std;

/**
 * @brief 检查文件是否存在
 * @param name 文件路径
 * @return 文件存在则返回 true
 */
bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/**
 * @brief 图像预处理
 *
 * [Requirement 2] 将手绘图像预处理为与 MNIST 数据集一致的格式。
 *
 * 处理流程:
 *   1. 边界框裁剪：定位所有非白色像素的最小包围矩形
 *   2. 等比例缩放：将较大边缩放到 20px
 *   3. 居中填充：将缩放后的图像放置到 28x28 白色画布中央
 *
 * @param original_img 原始手绘图像
 * @return 预处理后的 28x28 图像
 */
Image preprocess_image(Image original_img) {
    Color* pixels = LoadImageColors(original_img);
    int minX = original_img.width;
    int minY = original_img.height;
    int maxX = 0;
    int maxY = 0;
    bool hasInk = false;

    //遍历像素，找到所有非白色像素的边界框
    for (int y = 0; y < original_img.height; y++) {
        for (int x = 0; x < original_img.width; x++) {
            Color c = pixels[y * original_img.width + x];
            if (c.r < 128) {
                hasInk = true;
                //更新bbbox边界
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    UnloadImageColors(pixels);

    // 如果没有任何非白色像素，返回一个空白的 28x28 图像
    if (!hasInk) {
        return GenImageColor(28, 28, WHITE);
    }

    Rectangle bbox = {
        (float)minX, (float)minY,
        (float)(maxX - minX + 1), (float)(maxY - minY + 1)
    };

    Image crop_img = ImageCopy(original_img);
    ImageCrop(&crop_img, bbox);

    int new_w, new_h;
    if (crop_img.width > crop_img.height) {
        new_w = 20;
        new_h = (int)((20.0f / crop_img.width) * crop_img.height);
    } else {
        new_h = 20;
        new_w = (int)((20.0f / crop_img.height) * crop_img.width);
    }

    if (new_w <= 0) new_w = 1;
    if (new_h <= 0) new_h = 1;
    ImageResize(&crop_img, new_w, new_h);

    // 创建一个 28x28 的白色画布，并将缩放后的图像居中绘制上去
    Image final_img = GenImageColor(28, 28, WHITE);
    int offsetX = (28 - new_w) / 2;
    int offsetY = (28 - new_h) / 2;
    ImageDraw(&final_img, crop_img,
        { 0, 0, (float)new_w, (float)new_h },
        { (float)offsetX, (float)offsetY, (float)new_w, (float)new_h }, WHITE);

    // 清理中间图像资源
    UnloadImage(crop_img);
    return final_img;
}

/**
 * @brief K 折交叉验证索引生成器
 *
 * [Requirement 7] K 折交叉验证
 *
 * 将数据集随机打乱后均分为 K 份，每次取 1 份作为验证集，
 * 其余 K-1 份作为训练集，返回 K 组训练/验证索引。
 *
 * @param total_samples 总样本数
 * @param k 折数
 * @return K 个 Fold 的训练/验证索引集合
 */
struct FoldIndices {
    vector<int> train_indices;
    vector<int> val_indices;
};

// 生成 K 折索引：对样本索引随机打乱后按位置切分为 K 份，
// 每份作为一次验证集，其余作为训练集（注意：使用固定种子 1337，结果可复现）
vector<FoldIndices> get_k_fold_indices(int total_samples, int k) {
    vector<int> all_indices(total_samples);
    for(int i = 0; i < total_samples; i++) {
        all_indices[i] = i;
    }

    static mt19937 g(1337);
    shuffle(all_indices.begin(), all_indices.end(), g);

    vector<FoldIndices> folds(k);
    int fold_size = total_samples / k;

    for(int i = 0; i < k; i++) {
        int start = i * fold_size;
        int end = (i == k - 1) ? total_samples : (i + 1) * fold_size;

        for(int j = 0; j < total_samples; j++) {
            if(j >= start && j < end) {
                folds[i].val_indices.push_back(all_indices[j]);
            } else {
                folds[i].train_indices.push_back(all_indices[j]);
            }
        }
    }
    return folds;
}

/**
 * @brief AutoML 搜索空间配置项
 *
 * [Requirement 8] 超参数搜索 - 网格搜索 / 随机搜索
 *
 * 每个配置包含学习率、隐藏层神经元数、激活函数类型和交叉验证平均分。
 */
struct SearchConfig {
    double lr;
    int h1;
    ActivationType act;
    string name;
    float score = 0.0f;

    SearchConfig(double l, int h, ActivationType a, string d) : lr(l), h1(h), act(a), name(d) {}
};

// ============================================================
// 程序入口
// ============================================================
int main() {
    const int screenWidth = 1200;
    const int screenHeight = 900;
    InitWindow(screenWidth, screenHeight, "NoobNetwork - Pro Complete Edition");
    SetTargetFPS(60);

    // 字体加载
    Font customFont = GetFontDefault();
    string font_path = "data/ui_font.ttf";

    if (file_exists(font_path)) {
        customFont = LoadFontEx(font_path.c_str(), 32, 0, 0);
        SetTextureFilter(customFont.texture, TEXTURE_FILTER_BILINEAR);
        GuiSetFont(customFont);
    } else {
        cout << "Warning: Custom font not found at " << font_path << ". Using default." << endl;
    }

    GuiSetStyle(DEFAULT, TEXT_SIZE, 15);

    // --- UI 控制变量 ---

    // [Requirement 4] 网络结构配置
    int h1Nodes = 128;
    int h2Nodes = 0;
    bool h1EditMode = false;
    bool h2EditMode = false;

    // [Requirement 6] 训练迭代控制
    bool maxEpochEdit = false;
    int maxEpochs = 5;

    // [Requirement 10] Dropout 配置
    float dropoutRate = 0.0f;

    // [Requirement 6] 早停配置
    bool enableEarlyStopping = true;

    // [Requirement 1] 贪心策略开关
    bool enableGreedy = true;

    // [Requirement 7] K 折交叉验证配置
    int k_folds = 5;
    bool kFoldEdit = false;

    // [Requirement 8] AutoML 搜索范围
    int activeSearchType = 0; // 0=Grid, 1=Random
    int lrMinInt = 1;
    int lrMaxInt = 5;
    int hMin = 64;
    int hMax = 256;
    bool lrMinEdit = false;
    bool lrMaxEdit = false;
    bool hMinEdit = false;
    bool hMaxEdit = false;

    // AutoML 运行时状态
    bool isAutoSearching = false;
    int searchIdx = 0;
    int foldIdx = 0;
    vector<SearchConfig> searchSpace;
    string searchStatus = "System Idle";
    vector<FoldIndices> currentFolds;

    // --- 神经网络初始化 ---
    // [Requirement 4] 默认拓扑: 784 → 128 → 10
    vector<int> topology = {784, h1Nodes, 10};
    NeuralNetwork nn(topology, 0.01);
    bool isModelLoaded = false;
    string model_path = "data/trained_model.txt";

    // [Requirement 2] 启动时尝试加载已保存的模型
    if (file_exists(model_path)) {
        nn.load_model(model_path);
        isModelLoaded = true;
    }

    MNISTData train_data;
    MNISTData test_data;
    bool isTrainingState = false;
    int currentEpoch = 0;
    int currentImgIdx = 0;
    int totalSamples = 0;
    float testAccuracy = 0.0f;
    float trainingProgress = 0.0f;
    int recognizedDigit = -1;

    // [Requirement 5] 激活函数和优化器选择
    int activeAct = 0; // 0=Sigmoid, 1=ReLU, 2=Tanh
    int activeOpt = 1; // 0=SGD, 1=Mini-Batch, 2=BGD

    // --- Greedy Bold Driver 状态变量 ---
    double previous_loss = 999999.0;
    double smoothed_loss = 0.0;
    double accumulated_loss_for_gradient = 0.0;

    // --- 实时曲线监控变量 ---
    vector<float> lossHistory;//历史损失记录
    vector<float> accHistory;//历史准确率记录
    const int maxPoints = 200;//曲线显示的最大点数
    int metric_counter = 0;//用于控制曲线采样频率的计数器
    int metric_correct = 0;//当前采样周期内分类正确的样本数
    double metric_loss_accum = 0.0;//当前采样周期内累计的损失值
    float displayLoss = 0.0f;//用于显示的当前损失值（经过平滑处理）
    float displayAcc = 0.0f;//用于显示的当前准确率（经过平滑处理）

    // --- 早停监控变量 ---
    float best_val_loss = 9999.0f;//验证集最佳损失
    int patience_counter = 0;//早停耐心计数器
    const int MAX_PATIENCE = 5;//允许的连续验证损失不下降的最大周期数
    bool triggered_early_stop = false;//早停触发标志

    // 绘图画布初始化
    RenderTexture2D canvas = LoadRenderTexture(440, 440);
    BeginTextureMode(canvas);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    // ============================================================
    // 主渲染循环
    // ============================================================
    while (!WindowShouldClose()) {
        //标记当前是否处于训练或 AutoML 搜索状态
        bool isProcessing = isTrainingState || isAutoSearching;

        // --- 训练循环 ---
        if (isProcessing && !train_data.images.empty()) {
            // [Requirement 3] 每帧处理样本上限，防止界面卡顿
            int frame_processing_size = 250;

            // [Requirement 3] 根据优化器确定批次大小
            int target_batch_size = (activeOpt == 0) ? 1 :
                                    ((activeOpt == 1) ? 128 :
                                    (int)train_data.images.size());

            for (int b = 0; b < frame_processing_size && (isTrainingState || isAutoSearching); b++) {
                // 获取当前样本索引（根据是否在 AutoML 搜索阶段选择训练集或全局索引）
                int realIdx = isAutoSearching ? currentFolds[foldIdx].train_indices[currentImgIdx] : currentImgIdx;

                // 获取输入和目标输出
                NNMatrix input = train_data.images[realIdx];//输入图像（已预处理为 784x1 向量）
                NNMatrix target = train_data.labels[realIdx];//目标标签（one-hot 编码的 10x1 向量）

                // 前向传播
                NNMatrix output = nn.forward(input, true);
                NNMatrix diff = NNMatrix::subtract(target, output);

                // 计算 MSE 损失
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) {
                    sample_loss += diff.data[r][0] * diff.data[r][0];
                }

                accumulated_loss_for_gradient += sample_loss;// 累积损失用于 Greedy Bold Driver 策略
                metric_loss_accum += sample_loss;// 累积损失用于实时曲线监控

                // 统计分类正确数
                int maxIdx = 0;
                int trueIdx = 0;
                double maxVal = output.data[0][0];
                for(int i = 0; i < 10; i++) {
                    // 找到输出层中值最大的索引（预测数字）
                    if(output.data[i][0] > maxVal) {
                        maxVal = output.data[i][0];
                        maxIdx = i;
                    }
                    // 找到目标标签中值为 1 的索引（真实数字）
                    if(target.data[i][0] > 0.5) {
                        trueIdx = i;
                    }
                }

                if(maxIdx == trueIdx) {
                    metric_correct++;// 预测正确，分类正确数加一
                }

                // 累积梯度（执行反向传播）
                nn.accumulate_gradients(input, target);

                // --- 批次权重更新与 Greedy Bold Driver ---
                if (nn.accumulated_samples >= target_batch_size) {

                    // [Requirement 1] 贪心快照
                    if (enableGreedy && !isAutoSearching) {
                        nn.save_checkpoint();// 在应用梯度更新前保存当前权重状态，以便在损失上升时回滚
                    }
                    
                    // 应用梯度更新
                    nn.apply_gradients();

                    // 计算平均批次损失
                    double avg_update_loss = accumulated_loss_for_gradient / (double)target_batch_size;
                    accumulated_loss_for_gradient = 0.0;

                    // 指数移动平均平滑损失 (EMA, alpha=0.05)
                    if (smoothed_loss == 0.0) {
                        smoothed_loss = avg_update_loss;
                    } else {
                        smoothed_loss = 0.05 * avg_update_loss + 0.95 * smoothed_loss;
                    }

                    // --- Greedy Bold Driver 贪心自适应学习率策略 ---
                    //
                    // 核心逻辑: 检查损失是否下降。
                    //   下降/轻微波动 → 接受更新，增大学习率
                    //   明显上升 → 撤销更新（回滚权重），缩小学习率
                    // 5% 噪声容忍度: 允许 smoothed_loss <= previous_loss * 1.05 不触发回滚
                    if (!isAutoSearching && previous_loss != 999999.0) {
                        if (smoothed_loss <= previous_loss * 1.05) {
                            if (enableGreedy) {
                                nn.learningRate *= 1.05;
                            } else {
                                nn.learningRate *= 1.01;
                            }
                            previous_loss = smoothed_loss;
                        } else {
                            if (enableGreedy) {
                                // 权重回滚 + 学习率减半
                                nn.load_checkpoint();
                                nn.learningRate *= 0.5;
                                smoothed_loss = previous_loss;
                            } else {
                                //不会滚回，但会轻微减小学习率
                                nn.learningRate *= 0.99;
                                previous_loss = smoothed_loss;
                            }
                        }

                        // 学习率越界保护
                        if (nn.learningRate > 0.1) nn.learningRate = 0.1;
                        if (nn.learningRate < 0.0001) nn.learningRate = 0.0001;
                    } else {
                        previous_loss = smoothed_loss;
                    }
                }

                // --- 曲线采样 ---
                metric_counter++;
                if (metric_counter >= 300) {
                    // 每 300 个样本采样一次，计算平均损失和准确率，并更新显示值（带平滑处理）
                    float avg_metric_loss = (float)(metric_loss_accum / 300.0);
                    displayAcc = (float)metric_correct / 300.0f;

                    // 平滑显示损失，避免曲线过于波动
                    if (displayLoss == 0.0f) {
                        displayLoss = avg_metric_loss;
                    } else {
                        displayLoss = 0.1f * avg_metric_loss + 0.9f * displayLoss;
                    }
                    
                    // 将当前显示的损失和准确率添加到历史记录中，用于绘制曲线
                    lossHistory.push_back(displayLoss);
                    accHistory.push_back(displayAcc);

                    // 限制历史记录长度，保持曲线显示的流畅性
                    if (lossHistory.size() > maxPoints) {
                        lossHistory.erase(lossHistory.begin());
                        accHistory.erase(accHistory.begin());
                    }

                    // 重置计数器和累积值，开始下一个采样周期
                    metric_counter = 0;
                    metric_correct = 0;
                    metric_loss_accum = 0.0;
                }

                // 批次游标推进
                currentImgIdx++;
                // [Requirement 3] 在 AutoML 搜索阶段，训练样本数量受当前 fold 的训练集大小限制；否则使用全局样本数量
                int limit = isAutoSearching ? (int)currentFolds[foldIdx].train_indices.size() : totalSamples;

                if (currentImgIdx >= limit) {
                    currentImgIdx = 0;
                    currentEpoch++;

                    if (nn.accumulated_samples > 0) {
                        nn.apply_gradients();// 应用剩余的梯度更新
                        accumulated_loss_for_gradient = 0.0;
                    }

                    // --- 早停 (Early Stopping) ---
                    // [Requirement 10] 监控平滑损失，连续 MAX_PATIENCE 个 epoch 无改善则终止训练
                    if (isTrainingState && enableEarlyStopping && !isAutoSearching) {
                        if (smoothed_loss < best_val_loss - 0.0001f) {
                            best_val_loss = smoothed_loss;
                            patience_counter = 0;
                        } else {
                            patience_counter++;
                        }

                        if (patience_counter >= MAX_PATIENCE) {
                            isTrainingState = false;
                            triggered_early_stop = true;
                            nn.save_model(model_path);
                            isModelLoaded = true;
                            break;
                        }
                    }

                    // --- AutoML K 折交叉验证 ---
                    if (isAutoSearching && currentEpoch >= 1) {
                        // 计算当前 fold 的验证准确率
                        float vAcc = 0;
                        for(int vIdx : currentFolds[foldIdx].val_indices) {
                            //获取真实标签索引
                            int trueLabelIdx = 0;
                            for(int i = 0; i < 10; i++) {
                                if(train_data.labels[vIdx].data[i][0] > 0.5f) {
                                    trueLabelIdx = i;
                                    break;
                                }
                            }
                            // 使用当前模型进行预测，并统计正确预测的数量
                            if(nn.predict(train_data.images[vIdx]) == trueLabelIdx) {
                                vAcc += 1.0f;
                            }
                        }
                        // 将当前 fold 的准确率累积到对应配置的总分中，平均分计算方式为：当前 fold 的准确率 / fold 数量
                        searchSpace[searchIdx].score += (vAcc / (float)currentFolds[foldIdx].val_indices.size()) / (float)k_folds;

                        foldIdx++;
                        if (foldIdx >= k_folds) {
                            foldIdx = 0;
                            searchIdx++;// 进入下一个配置的搜索

                            if (searchIdx >= (int)searchSpace.size()) {
                                // AutoML 搜索完成，找到最佳配置
                                isAutoSearching = false;
                                // 在搜索空间中找到得分最高的配置
                                int bestIdx = 0;
                                for(int i = 1; i < (int)searchSpace.size(); i++){
                                    if(searchSpace[i].score > searchSpace[bestIdx].score) {
                                        bestIdx = i;
                                    }
                                }
                                // 更新搜索状态显示，展示最佳配置和对应的准确率
                                searchStatus = string(TextFormat("Done! Best: %s (Acc: %.1f%%)", searchSpace[bestIdx].name.c_str(), searchSpace[bestIdx].score * 100.0f));
                            }
                        }

                        // 切换到下一个配置
                        if (isAutoSearching) {
                            nn = NeuralNetwork({784, searchSpace[searchIdx].h1, 10}, searchSpace[searchIdx].lr, searchSpace[searchIdx].act, 1.0);
                            currentEpoch = 0;
                        }
                        break;
                    }
                    // 手动训练完成
                    else if (isTrainingState && currentEpoch >= maxEpochs) {
                        isTrainingState = false;
                        nn.save_model(model_path);
                        isModelLoaded = true;
                    }
                }
            }
            if (isTrainingState) {
                trainingProgress = (float)((double)(currentEpoch * totalSamples + currentImgIdx) / (double)(maxEpochs * totalSamples));
            }
        }

        // ============================================================
        // 渲染与 UI 绘制
        // ============================================================
        BeginDrawing();
        ClearBackground(GetColor((unsigned int)GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
        DrawText("NoobNetwork: MNIST Pro Complete", 35, 20, 24, DARKGRAY);

        // --- 左侧：画板区域 ---
        GuiGroupBox({ 35, 60, 460, 530 }, "Drawing Canvas & Tools");
        Rectangle canvasArea = { 45, 80, 440, 440 };
        DrawTextureRec(canvas.texture,
            { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height },
            { canvasArea.x, canvasArea.y }, WHITE);
        DrawRectangleLinesEx(canvasArea, 2, GRAY);

        // 手绘输入
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), canvasArea)) {
            BeginTextureMode(canvas);
            DrawCircleV({ GetMousePosition().x - canvasArea.x, GetMousePosition().y - canvasArea.y }, 15.0f, BLACK);
            EndTextureMode();
        }

        // [Requirement 12] 识别按钮
        if (GuiButton({ 45, 535, 210, 40 }, "RECOGNIZE DIGIT")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture);
                ImageFlipVertical(&img);
                Image final_img = preprocess_image(img);//预处理手绘图像

                // 将预处理后的图像转换为神经网络输入格式（784x1 向量，像素值归一化到 [0,1]）
                NNMatrix input(784, 1);
                Color* p = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) {
                    input.data[i][0] = (255.0f - p[i].r) / 255.0f;
                }
                recognizedDigit = nn.predict(input);// 使用当前模型进行识别

                // 清理图像资源
                UnloadImageColors(p);
                UnloadImage(final_img);
                UnloadImage(img);
            }
        }
        if (GuiButton({ 275, 535, 210, 40 }, "CLEAR CANVAS")) {
            BeginTextureMode(canvas);
            ClearBackground(RAYWHITE);
            EndTextureMode();
            recognizedDigit = -1;
        }

        // --- 右侧：控制台区域 ---
        float cpX = 520.0f;
        GuiGroupBox({ cpX, 60, 650, 530 }, "Configuration & AutoML");

        float cY = 85.0f;
        float rowHeight = 40.0f;

        // [Requirement 5] 激活函数选择
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Activation:");
        GuiToggleGroup({ cpX + 100, cY, 70, 30 }, "SIGMOID;RELU;TANH", &activeAct);

        // [Requirement 3] 优化器选择
        GuiLabel({ cpX + 320, cY, 70, 30 }, "Optimizer:");
        GuiToggleGroup({ cpX + 390, cY, 80, 30 }, "SGD;MINI-BATCH;BGD", &activeOpt);

        cY += rowHeight;
        // [Requirement 4] 隐藏层神经元配置
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Hidden L1:");
        if (GuiValueBox({ cpX + 100, cY, 70, 30 }, NULL, &h1Nodes, 10, 512, h1EditMode)) h1EditMode = !h1EditMode;
        GuiLabel({ cpX + 180, cY, 80, 30 }, "Hidden L2:");
        if (GuiValueBox({ cpX + 260, cY, 70, 30 }, NULL, &h2Nodes, 0, 512, h2EditMode)) h2EditMode = !h2EditMode;

        // [Requirement 10] Dropout 配置
        GuiLabel({ cpX + 340, cY, 70, 30 }, "Dropout:");
        GuiSliderBar({ cpX + 410, cY, 120, 30 }, NULL, NULL, &dropoutRate, 0.0f, 0.5f);
        DrawText(TextFormat("%.2f", dropoutRate), (int)cpX + 540, (int)cY + 8, 16, DARKBLUE);

        cY += rowHeight;
        // [Requirement 6] 最大迭代次数
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Max Epochs:");
        if (GuiValueBox({ cpX + 100, cY, 70, 30 }, NULL, &maxEpochs, 1, 100, maxEpochEdit)) maxEpochEdit = !maxEpochEdit;

        // [Requirement 10] 早停开关
        GuiCheckBox({ cpX + 185, cY + 5, 20, 20 }, "Early Stop", &enableEarlyStopping);

        // [Requirement 1] Greedy Bold Driver 开关
        GuiCheckBox({ cpX + 340, cY + 5, 20, 20 }, "Greedy Bold Driver (Req.10)", &enableGreedy);

        cY += rowHeight + 10.0f;
        // 手动训练按钮
        if (GuiButton({ cpX + 20, cY, 290, 40 }, isTrainingState ? "STOP" : "START MANUAL TRAINING")) {
            if (!isTrainingState && !isAutoSearching) {
                if (train_data.images.empty()) {
                    BeginDrawing();
                    ClearBackground(RAYWHITE);
                    DrawText("Loading Data...", screenWidth/2-100, screenHeight/2, 24, DARKGRAY);
                    EndDrawing();
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                if (!train_data.images.empty()) {
                    isTrainingState = true;
                    triggered_early_stop = false;

                    // [Requirement 4] 根据 UI 配置构建网络拓扑
                    vector<int> newTopology = {784, h1Nodes};
                    if (h2Nodes > 0) newTopology.push_back(h2Nodes);
                    newTopology.push_back(10);

                    // [Requirement 10] keep_rate = 1.0 - dropoutRate
                    nn = NeuralNetwork(newTopology, 0.01, (ActivationType)activeAct, 1.0 - (double)dropoutRate);

                    // 重置所有训练状态
                    currentEpoch = 0;
                    currentImgIdx = 0;
                    totalSamples = (int)train_data.images.size();
                    previous_loss = 999999.0;
                    smoothed_loss = 0.0;
                    accumulated_loss_for_gradient = 0.0;

                    lossHistory.clear();
                    accHistory.clear();
                    metric_counter = 0;
                    metric_correct = 0;
                    metric_loss_accum = 0.0;
                    displayLoss = 0.0f;
                    displayAcc = 0.0f;
                    best_val_loss = 9999.0f;
                    patience_counter = 0;
                }
            } else {
                isTrainingState = false;
            }
        }

        // [Requirement 12] 性能测试
        if (GuiButton({ cpX + 330, cY, 290, 40 }, "RUN PERFORMANCE TEST (10K)")) {
            if (test_data.images.empty()) {
                BeginDrawing();
                ClearBackground(RAYWHITE);
                DrawText("Loading Data...", screenWidth/2-100, screenHeight/2, 24, DARKGRAY);
                EndDrawing();
                test_data = DataLoader::load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
            }
            if (!test_data.images.empty()) {
                testAccuracy = (float)nn.get_accuracy(test_data);
            }
        }

        cY += 50.0f;
        // 状态栏
        DrawRectangle(cpX + 20, cY, 600, 45, Fade(LIGHTGRAY, 0.3f));
        DrawText(TextFormat("Test Acc: %.2f%%", testAccuracy * 100.0f), (int)cpX + 30, (int)cY + 15, 18, DARKGRAY);
        DrawText(TextFormat("LR: %.6f", nn.learningRate), (int)cpX + 200, (int)cY + 15, 18, DARKGRAY);
        //早停状态显示：如果触发早停，显示 "EARLY STOP!"；否则显示模型加载状态
        if (triggered_early_stop) {
            DrawText("EARLY STOP!", (int)cpX + 380, (int)cY + 15, 18, ORANGE);
        } else {
            DrawText(isModelLoaded ? "READY" : "EMPTY", (int)cpX + 380, (int)cY + 15, 18, isModelLoaded ? DARKGREEN : MAROON);
        }
        //识别结果
        if (recognizedDigit != -1) {
            DrawText(TextFormat("DIGIT: %d", recognizedDigit), (int)cpX + 480, (int)cY + 15, 20, DARKBLUE);
        }

        // --- AutoML 配置区 ---
        cY += 65.0f;
        DrawLine((int)cpX + 20, (int)cY, (int)cpX + 620, (int)cY, GRAY);
        cY += 15.0f;
        DrawText("AutoML Search Space Configuration", (int)cpX + 20, (int)cY, 16, DARKBLUE);

        cY += 30.0f;
        //学习率范围
        GuiLabel({ cpX + 20, cY, 80, 30 }, "LR Range:");
        if (GuiValueBox({ cpX + 100, cY, 60, 30 }, NULL, &lrMinInt, 1, 100, lrMinEdit)) lrMinEdit = !lrMinEdit;
        DrawText("-", (int)cpX + 165, (int)cY + 5, 20, GRAY);
        if (GuiValueBox({ cpX + 180, cY, 60, 30 }, NULL, &lrMaxInt, 1, 100, lrMaxEdit)) lrMaxEdit = !lrMaxEdit;

        // 隐藏层神经元范围
        GuiLabel({ cpX + 260, cY, 80, 30 }, "H1 Range:");
        if (GuiValueBox({ cpX + 340, cY, 60, 30 }, NULL, &hMin, 10, 512, hMinEdit)) hMinEdit = !hMinEdit;
        DrawText("-", (int)cpX + 405, (int)cY + 5, 20, GRAY);
        if (GuiValueBox({ cpX + 420, cY, 60, 30 }, NULL, &hMax, 10, 512, hMaxEdit)) hMaxEdit = !hMaxEdit;

        cY += rowHeight;
        // [Requirement 8] 搜索策略选择
        GuiLabel({ cpX + 20, cY, 70, 30 }, "Strategy:");
        GuiToggleGroup({ cpX + 90, cY, 90, 30 }, "GRID;RANDOM", &activeSearchType);

        // [Requirement 7] K 折数选择
        GuiLabel({ cpX + 290, cY, 60, 30 }, "K-Fold:");
        if (GuiValueBox({ cpX + 350, cY, 80, 30 }, NULL, &k_folds, 2, 10, kFoldEdit)) kFoldEdit = !kFoldEdit;

        cY += rowHeight + 5;
        // AutoML 启动按钮
        if (GuiButton({ cpX + 20, cY, 240, 40 }, isAutoSearching ? "STOP SEARCH" : "RUN AUTO SEARCH")) {
            if (!isAutoSearching && !isTrainingState) {
                if (train_data.images.empty()) {
                    BeginDrawing();
                    ClearBackground(RAYWHITE);
                    DrawText("Loading Data...", screenWidth/2-100, screenHeight/2, 24, DARKGRAY);
                    EndDrawing();
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                if (!train_data.images.empty()) {
                    searchSpace.clear();
                    float lr_min = (float)lrMinInt / 100.0f;
                    float lr_max = (float)lrMaxInt / 100.0f;

                    // [Requirement 8] 生成搜索空间
                    if (activeSearchType == 0) {
                        // 网格搜索: 取范围边界点
                        searchSpace.push_back(SearchConfig((double)lr_min, hMin, RELU, string(TextFormat("Grid LR:%.2f H:%d", lr_min, hMin))));
                        searchSpace.push_back(SearchConfig((double)lr_max, hMax, RELU, string(TextFormat("Grid LR:%.2f H:%d", lr_max, hMax))));
                    } else {
                        // 随机搜索
                        for(int i=0; i<3; i++) {
                            if (lr_max < lr_min) std::swap(lr_min, lr_max);
                            if (hMax < hMin) std::swap(hMin, hMax);

                            //随机学习率
                            float r_lr = lr_min;
                            if (lr_max > lr_min) {
                                r_lr = lr_min + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (lr_max - lr_min);
                            }

                            //随机隐藏层神经元数量
                            int r_h = hMin;
                            if (hMax >= hMin) {
                                int range = (hMax - hMin) + 1;
                                r_h = hMin + (range > 0 ? (rand() % range) : 0);
                            }
                            searchSpace.push_back(SearchConfig{(double)r_lr, r_h, RELU, string(TextFormat("Rand LR:%.2f H:%d", r_lr, r_h))});
                        }
                    }

                    // 初始化 AutoML 搜索状态
                    isAutoSearching = true;
                    searchIdx = 0;
                    foldIdx = 0;
                    currentEpoch = 0;
                    currentImgIdx = 0;
                    // 生成 K 折交叉验证的索引
                    currentFolds = get_k_fold_indices(60000, k_folds);

                    lossHistory.clear();
                    accHistory.clear();
                    // 根据第一个配置初始化神经网络
                    nn = NeuralNetwork({784, searchSpace[0].h1, 10}, searchSpace[0].lr, searchSpace[0].act, 1.0);
                }
            } else {
                isAutoSearching = false;
                searchStatus = "System Idle";
            }
        }

        // AutoML 搜索状态显示
        if (isAutoSearching) {
            searchStatus = string(TextFormat("Running: [%s] | Fold: %d/%d", searchSpace[searchIdx].name.c_str(), foldIdx + 1, k_folds));
        }
        DrawText(searchStatus.c_str(), (int)cpX + 280, (int)cY + 12, 16, isAutoSearching ? MAROON : DARKGRAY);

        // --- 底部图表：实时损失曲线和准确率曲线 ---
        GuiGroupBox({ 35, 600, 1135, 230 }, "Real-time Metrics Monitor: Loss (Maroon) & Accuracy (Blue)");
        Rectangle graphBox = { 50, 620, 1100, 190 };
        DrawRectangleRec(graphBox, Fade(RAYWHITE, 0.8f));
        DrawRectangleLinesEx(graphBox, 1, LIGHTGRAY);

        for(int i = 1; i < 4; i++) {
            float yLine = graphBox.y + i * (graphBox.height / 4.0f);
            DrawLineV({graphBox.x, yLine}, {graphBox.x + graphBox.width, yLine}, Fade(GRAY, 0.2f));
        }

        // 绘制曲线
        if (lossHistory.size() > 1) {
            //找出历史损失中的最大值，用于动态调整 Y 轴的缩放，确保曲线在图表中有足够的空间显示
            float maxLoss = 1.0f;
            for(float l : lossHistory) {
                if(l > maxLoss) maxLoss = l;
            }
            maxLoss *= 1.2f;

            float stepX = graphBox.width / (float)(maxPoints - 1);

            //绘制每两个相邻点之间的连线
            for (size_t i = 0; i < lossHistory.size() - 1; i++) {
                float px1 = graphBox.x + i * stepX;
                float px2 = graphBox.x + (i + 1) * stepX;

                // 损失曲线（栗色）
                float ly1 = graphBox.y + graphBox.height - (lossHistory[i] / maxLoss) * graphBox.height;
                float ly2 = graphBox.y + graphBox.height - (lossHistory[i+1] / maxLoss) * graphBox.height;
                if(ly1 < graphBox.y) ly1 = graphBox.y;
                if(ly2 < graphBox.y) ly2 = graphBox.y;
                DrawLineEx({px1, ly1}, {px2, ly2}, 2.0f, MAROON);

                // 准确率曲线（深蓝）
                float ay1 = graphBox.y + graphBox.height - (accHistory[i]) * graphBox.height;
                float ay2 = graphBox.y + graphBox.height - (accHistory[i+1]) * graphBox.height;
                if(ay1 < graphBox.y) ay1 = graphBox.y;
                if(ay2 < graphBox.y) ay2 = graphBox.y;
                DrawLineEx({px1, ay1}, {px2, ay2}, 2.0f, DARKBLUE);
            }
            // 显示最新的损失值和准确率
            DrawText(TextFormat("Loss: %.4f", lossHistory.back()), (int)graphBox.x + 10, (int)graphBox.y + 10, 15, MAROON);
            DrawText(TextFormat("Acc: %.1f%%", accHistory.back() * 100.0f), (int)graphBox.x + 10, (int)graphBox.y + 30, 15, DARKBLUE);
        } else if (!isProcessing) {
            // 如果没有数据且不在处理状态，显示提示信息
            DrawText("START TRAINING OR AUTOML TO RENDER CURVES", (int)(graphBox.x + graphBox.width/2 - 200), (int)(graphBox.y + graphBox.height/2 - 10), 20, GRAY);
        }

        // --- 底部进度条 ---
        if (isProcessing) {
            DrawRectangle(0, screenHeight - 45, screenWidth, 45, Fade(BLACK, 0.8f));
            //计算进度百分比：如果在 AutoML 搜索阶段，进度基于完成的 fold 数和配置数；否则基于当前 epoch 和样本索引
            float p = isAutoSearching ? ((float)(searchIdx * k_folds + foldIdx) / (float)(searchSpace.size() * k_folds) * 100.0f) : (trainingProgress * 100.0f);
            // 绘制进度条背景
            GuiProgressBar({ 35, screenHeight - 35, 1135, 20 }, NULL, NULL, &p, 0, 100);
            // 显示进度文本：如果在 AutoML 搜索阶段，显示当前配置和 fold；否则显示当前 epoch 和整体进度百分比
            if(!isAutoSearching) {
                DrawText(TextFormat("EPOCH: %d/%d   |   PROGRESS: %.1f%%", currentEpoch + 1, maxEpochs, p), 45, screenHeight - 33, 15, WHITE);
            } else {
                DrawText(TextFormat("AUTOML PROGRESS: %d / %d FOLDS COMPLETED", searchIdx * k_folds + foldIdx, (int)searchSpace.size() * k_folds), 45, screenHeight - 33, 15, WHITE);
            }
        }

        EndDrawing();
    }

    UnloadRenderTexture(canvas);// 卸载画布资源
    UnloadFont(customFont);// 卸载字体资源
    CloseWindow();// 关闭窗口，结束程序
    return 0;
}
