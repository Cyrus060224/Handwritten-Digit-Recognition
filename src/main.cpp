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

// --- 基础辅助函数 ---
bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// --- 图像预处理 ---
Image preprocess_image(Image original_img) {
    Color* pixels = LoadImageColors(original_img);
    int minX = original_img.width;
    int minY = original_img.height;
    int maxX = 0;
    int maxY = 0;
    bool hasInk = false;

    for (int y = 0; y < original_img.height; y++) {
        for (int x = 0; x < original_img.width; x++) {
            Color c = pixels[y * original_img.width + x];
            if (c.r < 128) {
                hasInk = true;
                if (x < minX) minX = x; 
                if (x > maxX) maxX = x;
                if (y < minY) minY = y; 
                if (y > maxY) maxY = y;
            }
        }
    }
    UnloadImageColors(pixels);
    
    if (!hasInk) {
        return GenImageColor(28, 28, WHITE);
    }

    Rectangle bbox = { 
        (float)minX, 
        (float)minY, 
        (float)(maxX - minX + 1), 
        (float)(maxY - minY + 1) 
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

    Image final_img = GenImageColor(28, 28, WHITE);
    int offsetX = (28 - new_w) / 2;
    int offsetY = (28 - new_h) / 2;
    ImageDraw(&final_img, crop_img, 
        { 0, 0, (float)new_w, (float)new_h }, 
        { (float)offsetX, (float)offsetY, (float)new_w, (float)new_h }, WHITE);
        
    UnloadImage(crop_img);
    return final_img;
}

// --- 交叉验证数据切分 ---
struct FoldIndices {
    vector<int> train_indices;
    vector<int> val_indices;
};

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

// --- AutoML 搜索配置结构 ---
struct SearchConfig {
    double lr;
    int h1;
    ActivationType act;
    string name;
    float score = 0.0f; 
};

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 900; 
    InitWindow(screenWidth, screenHeight, "NoobNetwork - Pro Complete Edition");
    SetTargetFPS(60);
    
    GuiSetStyle(DEFAULT, TEXT_SIZE, 15); 

    // --- UI 控制变量 ---
    int h1Nodes = 128;
    int h2Nodes = 0;      
    bool h1EditMode = false;
    bool h2EditMode = false;
    bool maxEpochEdit = false;
    float dropoutRate = 0.0f; 
    int maxEpochs = 5;
    bool enableEarlyStopping = true; 
    bool enableGreedy = true; // 🌟 需求 10: 贪心自适应学习率开关

    // --- AutoML 配置变量 ---
    int k_folds = 5;
    bool kFoldEdit = false;
    int activeSearchType = 0; 
    
    int lrMinInt = 1;
    int lrMaxInt = 5; 
    int hMin = 64;
    int hMax = 256;
    bool lrMinEdit = false;
    bool lrMaxEdit = false;
    bool hMinEdit = false;
    bool hMaxEdit = false;

    // --- AutoML 状态变量 ---
    bool isAutoSearching = false;
    int searchIdx = 0;
    int foldIdx = 0;
    vector<SearchConfig> searchSpace;
    string searchStatus = "System Idle";
    vector<FoldIndices> currentFolds;

    // --- 神经网络与底层变量 ---
    vector<int> topology = {784, h1Nodes, 10};
    NeuralNetwork nn(topology, 0.01);
    bool isModelLoaded = false;
    string model_path = "data/trained_model.txt";
    
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
    int activeAct = 0;
    int activeOpt = 1;

    double previous_loss = 999999.0;
    double smoothed_loss = 0.0;
    double accumulated_loss_for_gradient = 0.0; 
    
    // --- 曲线与监控大屏变量 ---
    vector<float> lossHistory; 
    vector<float> accHistory;
    const int maxPoints = 200;           
    int metric_counter = 0;
    int metric_correct = 0;             
    double metric_loss_accum = 0.0;     
    float displayLoss = 0.0f;
    float displayAcc = 0.0f;   

    float best_val_loss = 9999.0f;
    int patience_counter = 0;
    const int MAX_PATIENCE = 15; 
    bool triggered_early_stop = false;

    RenderTexture2D canvas = LoadRenderTexture(440, 440);
    BeginTextureMode(canvas); 
    ClearBackground(RAYWHITE); 
    EndTextureMode();

    while (!WindowShouldClose()) {
        
        bool isProcessing = isTrainingState || isAutoSearching;

        // ============================================================
        // 引擎训练循环 (完全展开，包含完整贪心策略)
        // ============================================================
        if (isProcessing && !train_data.images.empty()) {
            int frame_processing_size = 250; 
            int target_batch_size = (activeOpt == 0) ? 1 : ((activeOpt == 1) ? 128 : (int)train_data.images.size()); 

            for (int b = 0; b < frame_processing_size && (isTrainingState || isAutoSearching); b++) {
                
                int realIdx = isAutoSearching ? currentFolds[foldIdx].train_indices[currentImgIdx] : currentImgIdx;

                NNMatrix input = train_data.images[realIdx];
                NNMatrix target = train_data.labels[realIdx];
                
                NNMatrix output = nn.forward(input, true);
                NNMatrix diff = NNMatrix::subtract(target, output);
                
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) {
                    sample_loss += diff.data[r][0] * diff.data[r][0];
                }
                
                accumulated_loss_for_gradient += sample_loss;
                metric_loss_accum += sample_loss;

                int maxIdx = 0;
                int trueIdx = 0;
                double maxVal = output.data[0][0];
                for(int i = 0; i < 10; i++) {
                    if(output.data[i][0] > maxVal) { 
                        maxVal = output.data[i][0]; 
                        maxIdx = i; 
                    }
                    if(target.data[i][0] > 0.5) {
                        trueIdx = i;
                    }
                }
                
                if(maxIdx == trueIdx) {
                    metric_correct++;
                }

                nn.accumulate_gradients(input, target);
                
                // --- 批次权重更新与贪心策略判定 ---
                if (nn.accumulated_samples >= target_batch_size) {
                    
                    // 🌟 贪心快照备份 (在应用新梯度前)
                    if (enableGreedy && !isAutoSearching) {
                        nn.save_checkpoint();
                    }

                    nn.apply_gradients();
                    
                    double avg_update_loss = accumulated_loss_for_gradient / (double)target_batch_size;
                    accumulated_loss_for_gradient = 0.0; 

                    if (smoothed_loss == 0.0) {
                        smoothed_loss = avg_update_loss;
                    } else {
                        smoothed_loss = 0.05 * avg_update_loss + 0.95 * smoothed_loss;
                    }

                    // 🌟 严谨修复后的贪心自适应策略
                    if (!isAutoSearching && previous_loss != 999999.0) {
                        if (smoothed_loss <= previous_loss) {
                            // 损失下降 (赚了)：接收更新，尝试拉大学习率
                            if (enableGreedy) {
                                nn.learningRate *= 1.05; 
                            } else {
                                nn.learningRate *= 1.01;
                            }
                            previous_loss = smoothed_loss; // 只有成功才更新历史记录
                        } else {
                            // 损失上升或震荡 (亏了)
                            if (enableGreedy) {
                                nn.load_checkpoint(); // 🌟 撤销权重更新，防止发散
                                nn.learningRate *= 0.5; // 🌟 缩小步幅
                                smoothed_loss = previous_loss; // 🌟 撤销被恶化的 Loss 记录！
                            } else {
                                nn.learningRate *= 0.99;
                                previous_loss = smoothed_loss;
                            }
                        }
                        
                        // 学习率越界保护
                        if (nn.learningRate > 0.1) nn.learningRate = 0.1;
                        if (nn.learningRate < 0.0001) nn.learningRate = 0.0001;
                    } else {
                        // 初始化第一帧
                        previous_loss = smoothed_loss;
                    }
                }

                // UI 曲线独立采样器
                metric_counter++;
                if (metric_counter >= 300) {
                    float avg_metric_loss = (float)(metric_loss_accum / 300.0);
                    displayAcc = (float)metric_correct / 300.0f;
                    
                    if (displayLoss == 0.0f) {
                        displayLoss = avg_metric_loss;
                    } else {
                        displayLoss = 0.1f * avg_metric_loss + 0.9f * displayLoss; 
                    }

                    lossHistory.push_back(displayLoss);
                    accHistory.push_back(displayAcc);

                    if (lossHistory.size() > maxPoints) {
                        lossHistory.erase(lossHistory.begin());
                        accHistory.erase(accHistory.begin());
                    }

                    // 早停检测
                    if (isTrainingState && enableEarlyStopping) {
                        if (displayLoss < best_val_loss - 0.001f) {
                            best_val_loss = displayLoss;
                            patience_counter = 0; 
                        } else {
                            patience_counter++; 
                        }

                        if (patience_counter >= MAX_PATIENCE) {
                            isTrainingState = false;
                            triggered_early_stop = true;
                            nn.save_model(model_path);
                            isModelLoaded = true;
                        }
                    }

                    metric_counter = 0; 
                    metric_correct = 0; 
                    metric_loss_accum = 0.0;
                }

                // 批次游标推进
                currentImgIdx++;
                int limit = isAutoSearching ? (int)currentFolds[foldIdx].train_indices.size() : totalSamples;

                if (currentImgIdx >= limit) {
                    currentImgIdx = 0; 
                    currentEpoch++;
                    
                    if (nn.accumulated_samples > 0) { 
                        nn.apply_gradients(); 
                        accumulated_loss_for_gradient = 0.0; 
                    }
                    
                    if (isAutoSearching && currentEpoch >= 1) { 
                        float vAcc = 0;
                        for(int vIdx : currentFolds[foldIdx].val_indices) {
                            if(nn.predict(train_data.images[vIdx]) == nn.predict(train_data.images[vIdx])) {
                                vAcc += 1.0f;
                            }
                        }
                        searchSpace[searchIdx].score += (vAcc / (float)currentFolds[foldIdx].val_indices.size()) / (float)k_folds;
                        
                        foldIdx++;
                        if (foldIdx >= k_folds) {
                            foldIdx = 0; 
                            searchIdx++;
                            
                            if (searchIdx >= (int)searchSpace.size()) {
                                isAutoSearching = false; 
                                int bestIdx = 0;
                                for(int i = 1; i < (int)searchSpace.size(); i++){
                                    if(searchSpace[i].score > searchSpace[bestIdx].score) {
                                        bestIdx = i;
                                    }
                                }
                                searchStatus = string(TextFormat("Done! Best: %s (Acc: %.1f%%)", searchSpace[bestIdx].name.c_str(), searchSpace[bestIdx].score * 100.0f));
                            }
                        }
                        
                        if (isAutoSearching) {
                            nn = NeuralNetwork({784, searchSpace[searchIdx].h1, 10}, searchSpace[searchIdx].lr, searchSpace[searchIdx].act, 1.0);
                            currentEpoch = 0;
                        }
                        break;
                    } 
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
        // 渲染与对齐 UI 绘制
        // ============================================================
        BeginDrawing();
        ClearBackground(GetColor((unsigned int)GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
        DrawText("NoobNetwork: MNIST Pro Complete", 35, 20, 24, DARKGRAY);

        // --- 左侧：画板 ---
        GuiGroupBox({ 35, 60, 460, 530 }, "Drawing Canvas & Tools");
        Rectangle canvasArea = { 45, 80, 440, 440 };
        DrawTextureRec(canvas.texture, 
            { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height }, 
            { canvasArea.x, canvasArea.y }, WHITE);
        DrawRectangleLinesEx(canvasArea, 2, GRAY);
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), canvasArea)) {
            BeginTextureMode(canvas); 
            DrawCircleV({ GetMousePosition().x - canvasArea.x, GetMousePosition().y - canvasArea.y }, 15.0f, BLACK); 
            EndTextureMode();
        }

        if (GuiButton({ 45, 535, 210, 40 }, "RECOGNIZE DIGIT")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture); 
                ImageFlipVertical(&img); 
                Image final_img = preprocess_image(img);
                
                NNMatrix input(784, 1); 
                Color* p = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) {
                    input.data[i][0] = (255.0f - p[i].r) / 255.0f;
                }
                recognizedDigit = nn.predict(input);
                
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

        // --- 右侧：控制台 ---
        float cpX = 520.0f;
        GuiGroupBox({ cpX, 60, 650, 530 }, "Configuration & AutoML");

        float cY = 85.0f;
        float rowHeight = 40.0f; 

        // 按钮宽度精准控制
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Activation:");
        GuiToggleGroup({ cpX + 100, cY, 70, 30 }, "SIGMOID;RELU;TANH", &activeAct); 
        
        GuiLabel({ cpX + 320, cY, 70, 30 }, "Optimizer:");
        GuiToggleGroup({ cpX + 390, cY, 80, 30 }, "SGD;MINI-BATCH;BGD", &activeOpt); 

        cY += rowHeight;
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Hidden L1:");
        if (GuiValueBox({ cpX + 100, cY, 70, 30 }, NULL, &h1Nodes, 10, 512, h1EditMode)) h1EditMode = !h1EditMode;
        GuiLabel({ cpX + 180, cY, 80, 30 }, "Hidden L2:");
        if (GuiValueBox({ cpX + 260, cY, 70, 30 }, NULL, &h2Nodes, 0, 512, h2EditMode)) h2EditMode = !h2EditMode;
        
        GuiLabel({ cpX + 340, cY, 70, 30 }, "Dropout:");
        GuiSliderBar({ cpX + 410, cY, 120, 30 }, NULL, NULL, &dropoutRate, 0.0f, 0.5f);
        DrawText(TextFormat("%.2f", dropoutRate), (int)cpX + 540, (int)cY + 8, 16, DARKBLUE);

        cY += rowHeight;
        GuiLabel({ cpX + 20, cY, 80, 30 }, "Max Epochs:");
        if (GuiValueBox({ cpX + 100, cY, 70, 30 }, NULL, &maxEpochs, 1, 100, maxEpochEdit)) maxEpochEdit = !maxEpochEdit;
        GuiCheckBox({ cpX + 190, cY + 5, 20, 20 }, "Enable Early Stopping", &enableEarlyStopping);
        GuiCheckBox({ cpX + 420, cY + 5, 20, 20 }, "Greedy Bold Driver (Req.10)", &enableGreedy);

        cY += rowHeight + 10.0f;
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
                    
                    vector<int> newTopology = {784, h1Nodes};
                    if (h2Nodes > 0) newTopology.push_back(h2Nodes);
                    newTopology.push_back(10);
                    
                    nn = NeuralNetwork(newTopology, 0.01, (ActivationType)activeAct, 1.0 - (double)dropoutRate); 
                    
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
        DrawRectangle(cpX + 20, cY, 600, 45, Fade(LIGHTGRAY, 0.3f));
        DrawText(TextFormat("Test Acc: %.2f%%", testAccuracy * 100.0f), (int)cpX + 30, (int)cY + 15, 18, DARKGRAY);
        DrawText(TextFormat("LR: %.6f", nn.learningRate), (int)cpX + 200, (int)cY + 15, 18, DARKGRAY);
        if (triggered_early_stop) {
            DrawText("EARLY STOP!", (int)cpX + 380, (int)cY + 15, 18, ORANGE);
        } else {
            DrawText(isModelLoaded ? "READY" : "EMPTY", (int)cpX + 380, (int)cY + 15, 18, isModelLoaded ? DARKGREEN : MAROON);
        }
        if (recognizedDigit != -1) {
            DrawText(TextFormat("DIGIT: %d", recognizedDigit), (int)cpX + 480, (int)cY + 15, 20, DARKBLUE);
        }

        // --- AutoML 配置区 ---
        cY += 65.0f;
        DrawLine((int)cpX + 20, (int)cY, (int)cpX + 620, (int)cY, GRAY);
        cY += 15.0f;
        DrawText("AutoML Search Space Configuration", (int)cpX + 20, (int)cY, 16, DARKBLUE);
        
        cY += 30.0f;
        GuiLabel({ cpX + 20, cY, 80, 30 }, "LR Range:");
        if (GuiValueBox({ cpX + 100, cY, 60, 30 }, NULL, &lrMinInt, 1, 100, lrMinEdit)) lrMinEdit = !lrMinEdit; 
        DrawText("-", (int)cpX + 165, (int)cY + 5, 20, GRAY);
        if (GuiValueBox({ cpX + 180, cY, 60, 30 }, NULL, &lrMaxInt, 1, 100, lrMaxEdit)) lrMaxEdit = !lrMaxEdit;

        GuiLabel({ cpX + 260, cY, 80, 30 }, "H1 Range:");
        if (GuiValueBox({ cpX + 340, cY, 60, 30 }, NULL, &hMin, 10, 512, hMinEdit)) hMinEdit = !hMinEdit;
        DrawText("-", (int)cpX + 405, (int)cY + 5, 20, GRAY); 
        if (GuiValueBox({ cpX + 420, cY, 60, 30 }, NULL, &hMax, 10, 512, hMaxEdit)) hMaxEdit = !hMaxEdit;

        cY += rowHeight;
        GuiLabel({ cpX + 20, cY, 70, 30 }, "Strategy:");
        GuiToggleGroup({ cpX + 90, cY, 90, 30 }, "GRID;RANDOM", &activeSearchType);
        
        GuiLabel({ cpX + 290, cY, 60, 30 }, "K-Fold:");
        if (GuiValueBox({ cpX + 350, cY, 80, 30 }, NULL, &k_folds, 2, 10, kFoldEdit)) kFoldEdit = !kFoldEdit;

        cY += rowHeight + 5;
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

                    if (activeSearchType == 0) { 
                        searchSpace.push_back({(double)lr_min, hMin, RELU, string(TextFormat("Grid LR:%.2f H:%d", lr_min, hMin))});
                        searchSpace.push_back({(double)lr_max, hMax, RELU, string(TextFormat("Grid LR:%.2f H:%d", lr_max, hMax))});
                    } else { 
                        for(int i=0; i<3; i++) {
                            float r_lr = lr_min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(lr_max-lr_min)));
                            int r_h = hMin + rand() % ((hMax - hMin) + 1);
                            searchSpace.push_back({(double)r_lr, r_h, RELU, string(TextFormat("Rand LR:%.2f H:%d", r_lr, r_h))});
                        }
                    }
                    
                    isAutoSearching = true; 
                    searchIdx = 0; 
                    foldIdx = 0; 
                    currentEpoch = 0; 
                    currentImgIdx = 0;
                    currentFolds = get_k_fold_indices(60000, k_folds);
                    
                    lossHistory.clear(); 
                    accHistory.clear();
                    nn = NeuralNetwork({784, searchSpace[0].h1, 10}, searchSpace[0].lr, searchSpace[0].act, 1.0);
                }
            } else {
                isAutoSearching = false;
                searchStatus = "System Idle";
            }
        }
        
        if (isAutoSearching) {
            searchStatus = string(TextFormat("Running: [%s] | Fold: %d/%d", searchSpace[searchIdx].name.c_str(), foldIdx + 1, k_folds));
        }
        DrawText(searchStatus.c_str(), (int)cpX + 280, (int)cY + 12, 16, isAutoSearching ? MAROON : DARKGRAY);

        // --- 3. 底部图表大屏 ---
        GuiGroupBox({ 35, 600, 1135, 230 }, "Real-time Metrics Monitor: Loss (Maroon) & Accuracy (Blue)");
        Rectangle graphBox = { 50, 620, 1100, 190 };
        DrawRectangleRec(graphBox, Fade(RAYWHITE, 0.8f));
        DrawRectangleLinesEx(graphBox, 1, LIGHTGRAY);

        for(int i = 1; i < 4; i++) {
            float yLine = graphBox.y + i * (graphBox.height / 4.0f);
            DrawLineV({graphBox.x, yLine}, {graphBox.x + graphBox.width, yLine}, Fade(GRAY, 0.2f));
        }

        if (lossHistory.size() > 1) {
            float maxLoss = 1.0f; 
            for(float l : lossHistory) {
                if(l > maxLoss) maxLoss = l;
            }
            maxLoss *= 1.2f; 

            float stepX = graphBox.width / (float)(maxPoints - 1); 

            for (size_t i = 0; i < lossHistory.size() - 1; i++) {
                float px1 = graphBox.x + i * stepX;
                float px2 = graphBox.x + (i + 1) * stepX;
                
                float ly1 = graphBox.y + graphBox.height - (lossHistory[i] / maxLoss) * graphBox.height;
                float ly2 = graphBox.y + graphBox.height - (lossHistory[i+1] / maxLoss) * graphBox.height;
                if(ly1 < graphBox.y) ly1 = graphBox.y; 
                if(ly2 < graphBox.y) ly2 = graphBox.y;
                DrawLineEx({px1, ly1}, {px2, ly2}, 2.0f, MAROON);

                float ay1 = graphBox.y + graphBox.height - (accHistory[i]) * graphBox.height;
                float ay2 = graphBox.y + graphBox.height - (accHistory[i+1]) * graphBox.height;
                if(ay1 < graphBox.y) ay1 = graphBox.y; 
                if(ay2 < graphBox.y) ay2 = graphBox.y;
                DrawLineEx({px1, ay1}, {px2, ay2}, 2.0f, DARKBLUE);
            }
            DrawText(TextFormat("Loss: %.4f", lossHistory.back()), (int)graphBox.x + 10, (int)graphBox.y + 10, 15, MAROON);
            DrawText(TextFormat("Acc: %.1f%%", accHistory.back() * 100.0f), (int)graphBox.x + 10, (int)graphBox.y + 30, 15, DARKBLUE);
        } else if (!isProcessing) {
            DrawText("START TRAINING OR AUTOML TO RENDER CURVES", (int)(graphBox.x + graphBox.width/2 - 200), (int)(graphBox.y + graphBox.height/2 - 10), 20, GRAY);
        }

        // --- 底部进度条 ---
        if (isProcessing) {
            DrawRectangle(0, screenHeight - 45, screenWidth, 45, Fade(BLACK, 0.8f));
            float p = isAutoSearching ? ((float)(searchIdx * k_folds + foldIdx) / (float)(searchSpace.size() * k_folds) * 100.0f) : (trainingProgress * 100.0f); 
            GuiProgressBar({ 35, screenHeight - 35, 1135, 20 }, NULL, NULL, &p, 0, 100);
            
            if(!isAutoSearching) {
                DrawText(TextFormat("EPOCH: %d/%d   |   PROGRESS: %.1f%%", currentEpoch + 1, maxEpochs, p), 45, screenHeight - 33, 15, WHITE);
            } else {
                DrawText(TextFormat("AUTOML PROGRESS: %d / %d FOLDS COMPLETED", searchIdx * k_folds + foldIdx, (int)searchSpace.size() * k_folds), 45, screenHeight - 33, 15, WHITE);
            }
        }
        
        EndDrawing();
    }
    
    UnloadRenderTexture(canvas); 
    CloseWindow(); 
    return 0;
}