#include <iostream>
#include <string>
#include <vector>
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "neural_network.h"
#include "data_loader.h"
#include "activations.h"
#include <sys/stat.h>

using namespace std;

bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

Image preprocess_image(Image original_img) {
    Color* pixels = LoadImageColors(original_img);
    int minX = original_img.width, minY = original_img.height;
    int maxX = 0, maxY = 0;
    bool hasInk = false;

    for (int y = 0; y < original_img.height; y++) {
        for (int x = 0; x < original_img.width; x++) {
            Color c = pixels[y * original_img.width + x];
            if (c.r < 128) {
                hasInk = true;
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
            }
        }
    }
    UnloadImageColors(pixels);
    if (!hasInk) return GenImageColor(28, 28, WHITE);

    Rectangle bbox = { (float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1) };
    Image crop_img = ImageCopy(original_img);
    ImageCrop(&crop_img, bbox);

    int new_w, new_h;
    if (crop_img.width > crop_img.height) {
        new_w = 20; new_h = (int)((20.0f / crop_img.width) * crop_img.height);
    } else {
        new_h = 20; new_w = (int)((20.0f / crop_img.height) * crop_img.width);
    }
    if (new_w <= 0) new_w = 1; if (new_h <= 0) new_h = 1;
    ImageResize(&crop_img, new_w, new_h);

    Image final_img = GenImageColor(28, 28, WHITE);
    int offsetX = (28 - new_w) / 2;
    int offsetY = (28 - new_h) / 2;
    ImageDraw(&final_img, crop_img, { 0, 0, (float)new_w, (float)new_h }, { (float)offsetX, (float)offsetY, (float)new_w, (float)new_h }, WHITE);
    UnloadImage(crop_img);
    return final_img;
}

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 900; 
    InitWindow(screenWidth, screenHeight, "NoobNetwork - Pro Monitor Suite");
    SetTargetFPS(60);
    GuiSetStyle(DEFAULT, TEXT_SIZE, 18); 

    int h1Nodes = 128, h2Nodes = 0;      
    bool h1EditMode = false, h2EditMode = false;
    float dropoutRate = 0.0f; 

    vector<int> topology = {784, h1Nodes, 10};
    NeuralNetwork nn(topology, 0.01);
    bool isModelLoaded = false;
    string model_path = "data/trained_model.txt";
    if (file_exists(model_path)) { nn.load_model(model_path); isModelLoaded = true; }

    RenderTexture2D canvas = LoadRenderTexture(440, 440);
    BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode();

    MNISTData train_data, test_data;
    bool isTrainingState = false;
    int currentEpoch = 0, currentImageIdx = 0, totalEpochs = 5, totalSamples = 0;
    float trainingProgress = 0.0f, testAccuracy = 0.0f;
    int recognizedDigit = -1, activeActivation = 0, activeOptimizer = 1;

    double previous_loss = 999999.0, smoothed_loss = 0.0;
    double accumulated_loss_for_gradient = 0.0; 
    
    // 🌟 核心修改：UI图表专属的“独立采样器”
    vector<float> lossHistory; 
    vector<float> accuracyHistory;
    const int maxPoints = 200;           // 图表容量：刚好填满一整个屏幕对应 1 个 Epoch
    int metric_counter = 0;             // 采样计数器
    int metric_correct = 0;             // 局部准确数
    double metric_loss_accum = 0.0;     // 局部误差池
    float current_display_loss = 0.0f;  // UI显示的平滑Loss
    float current_display_acc = 0.0f;   // UI显示的准确率

    while (!WindowShouldClose()) {
        if (isTrainingState && !train_data.images.empty()) {
            int frame_processing_size = 250; 
            int target_batch_size = (activeOptimizer == 0) ? 1 : ((activeOptimizer == 1) ? 128 : (int)train_data.images.size()); 

            double lr_inc = (activeOptimizer == 0) ? 1.0001 : ((activeOptimizer == 1) ? 1.01 : 1.10);
            double lr_dec = (activeOptimizer == 0) ? 0.9990 : ((activeOptimizer == 1) ? 0.99 : 0.90);

            for (int b = 0; b < frame_processing_size && isTrainingState; b++) {
                NNMatrix input = train_data.images[currentImageIdx];
                NNMatrix target = train_data.labels[currentImageIdx];
                
                NNMatrix output = nn.forward(input);
                NNMatrix diff = NNMatrix::subtract(target, output);
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) sample_loss += diff.data[r][0] * diff.data[r][0];
                
                // 兵分两路：给优化器算梯度的误差池
                accumulated_loss_for_gradient += sample_loss;
                
                // 兵分两路：给 UI 画图用的误差池
                metric_loss_accum += sample_loss;

                int maxIdx = 0, trueIdx = 0;
                double maxVal = output.data[0][0];
                for(int i = 0; i < 10; i++) {
                    if(output.data[i][0] > maxVal) { maxVal = output.data[i][0]; maxIdx = i; }
                    if(target.data[i][0] > 0.5) trueIdx = i;
                }
                if(maxIdx == trueIdx) metric_correct++;

                // --- 1. 底层模型权重更新逻辑 (受 Optimizer 控制) ---
                nn.accumulate_gradients(input, target);
                if (nn.accumulated_samples >= target_batch_size) {
                    nn.apply_gradients();
                    double avg_update_loss = accumulated_loss_for_gradient / (double)target_batch_size;
                    accumulated_loss_for_gradient = 0.0; 

                    if (smoothed_loss == 0.0) smoothed_loss = avg_update_loss;
                    else smoothed_loss = 0.05 * avg_update_loss + 0.95 * smoothed_loss;

                    if (previous_loss != 999999.0) {
                        if (smoothed_loss < previous_loss) nn.learningRate *= lr_inc;
                        else nn.learningRate *= lr_dec;
                    }
                    if (nn.learningRate > 0.1) nn.learningRate = 0.1;
                    if (nn.learningRate < 0.001) nn.learningRate = 0.001;
                    previous_loss = smoothed_loss;
                }

                // --- 2. 🌟 UI 曲线匀速采样逻辑 (完全独立，每 300 张图触发一次) ---
                metric_counter++;
                if (metric_counter >= 300) {
                    float avg_metric_loss = (float)(metric_loss_accum / 300.0);
                    current_display_acc = (float)metric_correct / 300.0f;
                    
                    if (current_display_loss == 0.0f) current_display_loss = avg_metric_loss;
                    else current_display_loss = 0.1f * avg_metric_loss + 0.9f * current_display_loss; // 曲线平滑处理

                    lossHistory.push_back(current_display_loss);
                    accuracyHistory.push_back(current_display_acc);

                    if (lossHistory.size() > maxPoints) lossHistory.erase(lossHistory.begin());
                    if (accuracyHistory.size() > maxPoints) accuracyHistory.erase(accuracyHistory.begin());

                    metric_counter = 0;
                    metric_correct = 0;
                    metric_loss_accum = 0.0;
                }

                // --- 游标推进 ---
                currentImageIdx++;
                if (currentImageIdx >= totalSamples) {
                    currentImageIdx = 0; currentEpoch++;
                    if (nn.accumulated_samples > 0) { nn.apply_gradients(); accumulated_loss_for_gradient = 0.0; }
                    if (currentEpoch >= totalEpochs) { isTrainingState = false; nn.save_model(model_path); isModelLoaded = true; }
                }
            }
            trainingProgress = (float)((double)(currentEpoch * totalSamples + currentImageIdx) / (double)(totalEpochs * totalSamples));
        }

        BeginDrawing();
        ClearBackground(GetColor((unsigned int)GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
        
        DrawText("MNIST Neural Network: Real-time Handwritten Digit Recognition", 45, 25, 26, DARKGRAY);

        // ==========================================
        // 区域 1：画板区
        // ==========================================
        GuiGroupBox({ 35, 75, 470, 550 }, "Drawing Canvas & Tools");
        Rectangle canvasArea = { 50, 100, 440, 440 };
        DrawTextureRec(canvas.texture, { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height }, { canvasArea.x, canvasArea.y }, WHITE);
        DrawRectangleLinesEx(canvasArea, 2, GRAY);
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), canvasArea)) {
            BeginTextureMode(canvas); DrawCircleV({ GetMousePosition().x - canvasArea.x, GetMousePosition().y - canvasArea.y }, 15.0f, BLACK); EndTextureMode();
        }

        if (GuiButton({ 50, 560, 210, 45 }, "RECOGNIZE DIGIT")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture); ImageFlipVertical(&img); Image final_img = preprocess_image(img);
                NNMatrix input(784, 1); Color* p = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) input.data[i][0] = (255.0f - p[i].r) / 255.0f;
                recognizedDigit = nn.predict(input);
                UnloadImageColors(p); UnloadImage(final_img); UnloadImage(img);
            }
        }
        if (GuiButton({ 280, 560, 210, 45 }, "CLEAR CANVAS")) { 
            BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode(); recognizedDigit = -1; 
        }

        // ==========================================
        // 区域 2：控制区
        // ==========================================
        float controlPanelX = 530.0f;
        GuiGroupBox({ controlPanelX, 75, 635, 550 }, "Control Center & Metrics");

        float currentY = 105.0f;
        float rowHeight = 50.0f; 

        GuiLabel({ controlPanelX + 20, currentY, 120, 35 }, "Activation:");
        GuiToggleGroup({ controlPanelX + 140, currentY, 155, 35 }, "SIGMOID;RELU;TANH", &activeActivation);

        currentY += rowHeight;
        GuiLabel({ controlPanelX + 20, currentY, 120, 35 }, "Optimizer:");
        GuiToggleGroup({ controlPanelX + 140, currentY, 155, 35 }, "SGD;MINI-BATCH;BGD", &activeOptimizer);

        currentY += rowHeight;
        GuiLabel({ controlPanelX + 20, currentY, 100, 35 }, "Hidden L1:");
        if (GuiValueBox({ controlPanelX + 120, currentY, 140, 35 }, NULL, &h1Nodes, 10, 512, h1EditMode)) h1EditMode = !h1EditMode;

        GuiLabel({ controlPanelX + 280, currentY, 100, 35 }, "Hidden L2:");
        if (GuiValueBox({ controlPanelX + 380, currentY, 140, 35 }, NULL, &h2Nodes, 0, 512, h2EditMode)) h2EditMode = !h2EditMode;
        
        currentY += rowHeight;
        GuiLabel({ controlPanelX + 20, currentY, 120, 35 }, "Dropout:");
        GuiSliderBar({ controlPanelX + 140, currentY, 320, 35 }, NULL, NULL, &dropoutRate, 0.0f, 0.5f);
        DrawText(TextFormat("%.2f", dropoutRate), (int)controlPanelX + 480, (int)currentY + 8, 20, DARKBLUE);

        currentY += rowHeight;
        string topoStr = "Architecture: 784 -> " + to_string(h1Nodes);
        if (h2Nodes > 0) topoStr += " -> " + to_string(h2Nodes);
        topoStr += " -> 10";
        DrawText(topoStr.c_str(), (int)controlPanelX + 20, (int)currentY + 5, 20, DARKGRAY);

        currentY += 45.0f;
        float btnWidth = 280.0f;
        if (GuiButton({ controlPanelX + 20, currentY, btnWidth, 45 }, isTrainingState ? "TRAINING..." : "START TRAINING")) {
            if (!isTrainingState) {
                if (train_data.images.empty()) {
                    BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Data...", 500, 400, 24, DARKGRAY); EndDrawing();
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                if (!train_data.images.empty()) { 
                    isTrainingState = true; 
                    vector<int> newTopology = {784, h1Nodes};
                    if (h2Nodes > 0) newTopology.push_back(h2Nodes);
                    newTopology.push_back(10);
                    
                    nn = NeuralNetwork(newTopology, 0.01, (ActivationType)activeActivation, 1.0 - (double)dropoutRate); 
                    currentEpoch = 0; currentImageIdx = 0; totalSamples = (int)train_data.images.size(); 
                    previous_loss = 999999.0; smoothed_loss = 0.0; accumulated_loss_for_gradient = 0.0;
                    
                    // 🌟 清空历史曲线与独立采样器
                    lossHistory.clear(); accuracyHistory.clear(); 
                    metric_counter = 0; metric_correct = 0; metric_loss_accum = 0.0;
                    current_display_loss = 0.0f; current_display_acc = 0.0f;
                }
            }
        }
        if (GuiButton({ controlPanelX + 330, currentY, btnWidth, 45 }, "RUN TEST (10K)")) {
            if (test_data.images.empty()) {
                BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Data...", 500, 400, 24, DARKGRAY); EndDrawing();
                test_data = DataLoader::load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
            }
            if (!test_data.images.empty()) testAccuracy = (float)nn.get_accuracy(test_data);
        }

        currentY += 65.0f;
        DrawText(TextFormat("Test Acc: %.2f%%", testAccuracy * 100.0f), (int)controlPanelX + 20, (int)currentY, 20, DARKGRAY);
        DrawText(TextFormat("LR: %.6f", nn.learningRate), (int)controlPanelX + 220, (int)currentY, 20, DARKGRAY);
        DrawText(isModelLoaded ? "Model: READY" : "Model: EMPTY", (int)controlPanelX + 440, (int)currentY, 20, isModelLoaded ? DARKGREEN : MAROON);

        currentY += 35.0f;
        DrawRectangle((int)controlPanelX + 20, (int)currentY, 595, 80, Fade(LIGHTGRAY, 0.4f));
        DrawRectangleLines((int)controlPanelX + 20, (int)currentY, 595, 80, GRAY);
        if (recognizedDigit != -1) {
            DrawText(TextFormat("%d", recognizedDigit), (int)controlPanelX + 290, (int)currentY - 5, 85, DARKBLUE);
            DrawText("PREDICTION RESULT", (int)controlPanelX + 35, (int)currentY + 30, 16, GRAY);
        } else {
            DrawText("AWAITING IMAGE INPUT...", (int)controlPanelX + 175, (int)currentY + 30, 20, GRAY);
        }

        // ==========================================
        // 🌟 区域 3：底部曲线监测大屏 (绝对流畅版)
        // ==========================================
        GuiGroupBox({ 35, 645, 1130, 200 }, "Real-time Training Metrics");
        Rectangle graphBox = { 50, 665, 1100, 165 };
        DrawRectangleRec(graphBox, Fade(RAYWHITE, 0.8f));
        DrawRectangleLinesEx(graphBox, 1, LIGHTGRAY);

        for(int i = 1; i < 4; i++) {
            float yLine = graphBox.y + i * (graphBox.height / 4.0f);
            DrawLineV({graphBox.x, yLine}, {graphBox.x + graphBox.width, yLine}, Fade(GRAY, 0.2f));
        }

        DrawText("Loss (Smooth)", (int)graphBox.x + 15, (int)graphBox.y + 10, 18, MAROON);
        DrawText("Accuracy", (int)graphBox.x + 15, (int)graphBox.y + 35, 18, DARKBLUE);
        if (!lossHistory.empty()) {
            DrawText(TextFormat("%.4f", lossHistory.back()), (int)graphBox.x + 140, (int)graphBox.y + 10, 18, MAROON);
            DrawText(TextFormat("%.1f%%", accuracyHistory.back() * 100.0f), (int)graphBox.x + 140, (int)graphBox.y + 35, 18, DARKBLUE);
        }

        // 🌟 核心：从左往右匀速画线，填满即滚动
        if (lossHistory.size() > 1) {
            float maxLoss = 1.0f; 
            for(float l : lossHistory) if(l > maxLoss) maxLoss = l;
            maxLoss *= 1.2f; 

            // 每个点的横向间隔固定，填满 maxPoints 时刚好触及右边缘
            float stepX = graphBox.width / (float)(maxPoints - 1); 

            for (size_t i = 0; i < lossHistory.size() - 1; i++) {
                // 不再使用右靠齐 offset，而是从 0 稳稳起步
                float px1 = graphBox.x + i * stepX;
                float px2 = graphBox.x + (i + 1) * stepX;
                
                float ly1 = graphBox.y + graphBox.height - (lossHistory[i] / maxLoss) * graphBox.height;
                float ly2 = graphBox.y + graphBox.height - (lossHistory[i+1] / maxLoss) * graphBox.height;
                if(ly1 < graphBox.y) ly1 = graphBox.y; if(ly2 < graphBox.y) ly2 = graphBox.y;
                DrawLineEx({px1, ly1}, {px2, ly2}, 2.5f, MAROON);

                float ay1 = graphBox.y + graphBox.height - (accuracyHistory[i]) * graphBox.height;
                float ay2 = graphBox.y + graphBox.height - (accuracyHistory[i+1]) * graphBox.height;
                if(ay1 < graphBox.y) ay1 = graphBox.y; if(ay2 < graphBox.y) ay2 = graphBox.y;
                DrawLineEx({px1, ay1}, {px2, ay2}, 2.5f, DARKBLUE);
            }
        } else if (!isTrainingState) {
            DrawText("START TRAINING TO RENDER CURVES", (int)(graphBox.x + graphBox.width/2 - 200), (int)(graphBox.y + graphBox.height/2 - 10), 22, GRAY);
        }

        if (isTrainingState) {
            DrawRectangle(0, screenHeight - 45, screenWidth, 45, Fade(BLACK, 0.8f));
            float p = trainingProgress * 100.0f;
            GuiProgressBar({ 35, screenHeight - 35, 1130, 20 }, NULL, NULL, &p, 0, 100);
            DrawText(TextFormat("EPOCH: %d/5   |   PROGRESS: %.1f%%", currentEpoch + 1, p), 45, screenHeight - 33, 16, WHITE);
        }
        
        EndDrawing();
    }
    UnloadRenderTexture(canvas); CloseWindow();
    return 0;
}