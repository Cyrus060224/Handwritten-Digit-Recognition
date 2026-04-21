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

// --- 辅助函数 ---
bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// 图像预处理流水线
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
    // 🌟 彻底扩充屏幕尺寸，走向宽屏专业软件风格
    const int screenWidth = 1200;
    const int screenHeight = 800; 
    InitWindow(screenWidth, screenHeight, "NoobNetwork - MNIST Professional Suite");
    SetTargetFPS(60);
    
    // 全局字体调大，避免细小文字堆叠
    GuiSetStyle(DEFAULT, TEXT_SIZE, 18); 

    // --- 1. 神经网络结构变量 ---
    int h1Nodes = 128;    
    int h2Nodes = 0;      
    bool h1EditMode = false;
    bool h2EditMode = false;

    // --- 2. 初始拓扑与模型实例化 ---
    vector<int> topology;
    topology.push_back(784);
    topology.push_back(h1Nodes);
    if (h2Nodes > 0) topology.push_back(h2Nodes);
    topology.push_back(10);

    NeuralNetwork nn(topology, 0.01);
    bool isModelLoaded = false;
    string model_path = "data/trained_model.txt";
    if (file_exists(model_path)) { nn.load_model(model_path); isModelLoaded = true; }

    // --- 3. 界面与训练状态变量 ---
    RenderTexture2D canvas = LoadRenderTexture(440, 440); // 画板稍微调大一点
    BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode();

    MNISTData train_data;
    MNISTData test_data;
    bool isTrainingState = false;
    int currentEpoch = 0, currentImageIdx = 0, totalEpochs = 5, totalSamples = 0;
    float trainingProgress = 0.0f, testAccuracy = 0.0f;
    int recognizedDigit = -1;
    
    int activeActivation = 0; // 0:Sigmoid, 1:ReLU, 2:Tanh
    int activeOptimizer = 1;  // 0:SGD, 1:Mini-Batch, 2:BGD

    double previous_loss = 999999.0, smoothed_loss = 0.0;
    double current_batch_display_loss = 0.0;
    double accumulated_loss = 0.0; 

    while (!WindowShouldClose()) {
        // --- 训练引擎核心逻辑 ---
        if (isTrainingState && !train_data.images.empty()) {
            int frame_processing_size = 250; 
            
            int target_batch_size = 1;
            if (activeOptimizer == 0) target_batch_size = 1;      
            else if (activeOptimizer == 1) target_batch_size = 128;
            else target_batch_size = train_data.images.size();     

            double lr_inc = (activeOptimizer == 0) ? 1.0001 : ((activeOptimizer == 1) ? 1.01 : 1.10);
            double lr_dec = (activeOptimizer == 0) ? 0.9990 : ((activeOptimizer == 1) ? 0.99 : 0.90);

            for (int b = 0; b < frame_processing_size && isTrainingState; b++) {
                NNMatrix input = train_data.images[currentImageIdx];
                NNMatrix target = train_data.labels[currentImageIdx];
                
                NNMatrix output = nn.forward(input);
                NNMatrix diff = NNMatrix::subtract(target, output);
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) sample_loss += diff.data[r][0] * diff.data[r][0];
                accumulated_loss += sample_loss;

                nn.accumulate_gradients(input, target);

                if (nn.accumulated_samples >= target_batch_size) {
                    nn.apply_gradients();
                    double avg_update_loss = accumulated_loss / target_batch_size;
                    accumulated_loss = 0.0; 

                    if (smoothed_loss == 0.0) smoothed_loss = avg_update_loss;
                    else smoothed_loss = 0.05 * avg_update_loss + 0.95 * smoothed_loss;
                    current_batch_display_loss = smoothed_loss;

                    if (previous_loss != 999999.0) {
                        if (smoothed_loss < previous_loss) nn.learningRate *= lr_inc;
                        else nn.learningRate *= lr_dec;
                    }
                    if (nn.learningRate > 0.1) nn.learningRate = 0.1;
                    if (nn.learningRate < 0.001) nn.learningRate = 0.001;
                    previous_loss = smoothed_loss;
                }

                currentImageIdx++;
                if (currentImageIdx >= totalSamples) {
                    currentImageIdx = 0; currentEpoch++;
                    if (nn.accumulated_samples > 0) { nn.apply_gradients(); accumulated_loss = 0.0; }
                    if (currentEpoch >= totalEpochs) { isTrainingState = false; nn.save_model(model_path); isModelLoaded = true; }
                }
            }
            trainingProgress = (float)(currentEpoch * totalSamples + currentImageIdx) / (totalEpochs * totalSamples);
        }

        // --- 界面渲染逻辑 ---
        BeginDrawing();
        ClearBackground(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
        
        DrawText("MNIST Neural Network: Real-time Handwritten Digit Recognition", 45, 25, 26, DARKGRAY);

        // 🌟 重新规划空间参数
        float controlPanelX = 540.0f;
        float controlPanelWidth = 620.0f; 
        float currentY = 120.0f;
        float rowHeight = 55.0f; // 拉大行距
        float labelWidth = 120.0f;

        // 1. 左侧画板区
        Rectangle canvasArea = { 50, 110, 440, 440 };
        GuiGroupBox({ 35, 80, 470, 520 }, "Drawing Canvas");
        DrawTextureRec(canvas.texture, { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height }, { canvasArea.x, canvasArea.y }, WHITE);
        DrawRectangleLinesEx(canvasArea, 2, GRAY);
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), canvasArea)) {
            BeginTextureMode(canvas); DrawCircleV({ GetMousePosition().x - canvasArea.x, GetMousePosition().y - canvasArea.y }, 15.0f, BLACK); EndTextureMode();
        }

        // 2. 右侧控制面板 (高度增加到 590)
        GuiGroupBox({ controlPanelX - 15, 80, controlPanelWidth, 590 }, "Control Center");

        // --- A: 核心配置区 ---
        GuiLabel({ controlPanelX, currentY, labelWidth, 35 }, "Activation:");
        GuiToggleGroup({ controlPanelX + labelWidth, currentY, 150, 35 }, "SIGMOID;RELU;TANH", &activeActivation);

        currentY += rowHeight;
        GuiLabel({ controlPanelX, currentY, labelWidth, 35 }, "Optimizer:");
        GuiToggleGroup({ controlPanelX + labelWidth, currentY, 150, 35 }, "SGD;MINI-BATCH;BGD", &activeOptimizer);

        // 🌟 修复重叠：去掉 GuiValueBox 里的冗余文字 (传 NULL)，并给予充足间距
        currentY += rowHeight;
        GuiLabel({ controlPanelX, currentY, 100, 35 }, "Hidden L1:");
        if (GuiValueBox({ controlPanelX + 100, currentY, 140, 35 }, NULL, &h1Nodes, 10, 512, h1EditMode)) h1EditMode = !h1EditMode;

        GuiLabel({ controlPanelX + 280, currentY, 100, 35 }, "Hidden L2:");
        if (GuiValueBox({ controlPanelX + 380, currentY, 140, 35 }, NULL, &h2Nodes, 0, 512, h2EditMode)) h2EditMode = !h2EditMode;
        
        // 实时显示网络拓扑
        currentY += rowHeight;
        string topoStr = "Structure: 784 - " + to_string(h1Nodes);
        if (h2Nodes > 0) topoStr += " - " + to_string(h2Nodes);
        topoStr += " - 10";
        DrawText(topoStr.c_str(), controlPanelX, currentY + 10, 20, DARKGRAY);

        // --- B: 按钮区 ---
        currentY += 50.0f;
        float btnWidth = (controlPanelWidth - 45) / 2;
        if (GuiButton({ controlPanelX, currentY, btnWidth, 45 }, isTrainingState ? "TRAINING..." : "Start Training")) {
            if (!isTrainingState) {
                if (train_data.images.empty()) {
                    BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Train Data...", 450, 350, 20, DARKGRAY); EndDrawing();
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                if (!train_data.images.empty()) { 
                    isTrainingState = true; 
                    vector<int> newTopology;
                    newTopology.push_back(784);
                    newTopology.push_back(h1Nodes);
                    if (h2Nodes > 0) newTopology.push_back(h2Nodes);
                    newTopology.push_back(10);
                    
                    nn = NeuralNetwork(newTopology, 0.01, (ActivationType)activeActivation); 
                    currentEpoch = 0; currentImageIdx = 0; totalSamples = train_data.images.size();
                    previous_loss = 999999.0; smoothed_loss = 0.0; accumulated_loss = 0.0;
                }
            }
        }
        if (GuiButton({ controlPanelX + btnWidth + 15, currentY, btnWidth, 45 }, "Run Test (10k)")) {
            if (test_data.images.empty()) {
                BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Test Data...", 450, 350, 20, DARKGRAY); EndDrawing();
                test_data = DataLoader::load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
            }
            if (!test_data.images.empty()) testAccuracy = (float)nn.get_accuracy(test_data);
        }

        // --- C: 核心指标 ---
        currentY += 65.0f;
        GuiLabel({ controlPanelX, currentY, 300, 30 }, TextFormat("Test Accuracy: %.2f%%", testAccuracy * 100.0f));
        DrawCircle(controlPanelX + 220, currentY + 15, 10, (testAccuracy > 0.8) ? GREEN : (testAccuracy > 0) ? ORANGE : RED);

        GuiLabel({ controlPanelX + 300, currentY, 300, 30 }, TextFormat("Learning Rate: %.6f", nn.learningRate));

        currentY += 35.0f;
        GuiLabel({ controlPanelX, currentY, 200, 30 }, "Model Persistence:");
        if (isModelLoaded) { 
            DrawText("READY", controlPanelX + 180, currentY + 5, 20, DARKGREEN); 
            DrawCircle(controlPanelX + 165, currentY + 15, 8, GREEN);
        } else { 
            DrawText("EMPTY", controlPanelX + 180, currentY + 5, 20, MAROON); 
            DrawCircle(controlPanelX + 165, currentY + 15, 8, RED);
        }

        // --- D: 预测结果展示框 ---
        currentY += 45.0f;
        DrawRectangle(controlPanelX, currentY, controlPanelWidth - 30, 110, Fade(LIGHTGRAY, 0.4f));
        DrawRectangleLines(controlPanelX, currentY, controlPanelWidth - 30, 110, GRAY);
        if (recognizedDigit != -1) {
            DrawText(TextFormat("%d", recognizedDigit), controlPanelX + 250, currentY + 10, 90, DARKBLUE);
            DrawText("PREDICTION", controlPanelX + 10, currentY + 10, 16, GRAY);
        } else {
            DrawText("READY FOR INPUT", controlPanelX + 175, currentY + 45, 24, GRAY);
        }

        // --- E: 底部操作按钮 ---
        currentY += 130.0f;
        if (GuiButton({ controlPanelX, currentY, btnWidth, 45 }, "RECOGNIZE")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture); ImageFlipVertical(&img); Image final_img = preprocess_image(img);
                NNMatrix input(784, 1); Color* p = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) input.data[i][0] = (255.0f - p[i].r) / 255.0f;
                recognizedDigit = nn.predict(input);
                UnloadImageColors(p); UnloadImage(final_img); UnloadImage(img);
            }
        }
        if (GuiButton({ controlPanelX + btnWidth + 15, currentY, btnWidth, 45 }, "CLEAR CANVAS")) { 
            BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode(); recognizedDigit = -1; 
        }

        // 3. 训练进度条（移到底部区域，防遮挡）
        if (isTrainingState) {
            DrawRectangle(0, screenHeight - 80, screenWidth, 80, Fade(BLACK, 0.85f));
            float p = trainingProgress * 100.0f;
            GuiProgressBar({ 40, screenHeight - 40, screenWidth - 80, 20 }, NULL, TextFormat("%.1f%%", p), &p, 0, 100);
            DrawText(TextFormat("EPOCH: %d/5   |   BATCH LOSS: %.4f   |   LR: %.6f", currentEpoch + 1, current_batch_display_loss, nn.learningRate), 45, screenHeight - 70, 18, WHITE);
        }
        
        EndDrawing();
    }
    UnloadRenderTexture(canvas); CloseWindow();
    return 0;
}