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
    const int screenWidth = 1000;
    const int screenHeight = 650; // 稍微加高一点，留出底部余量
    InitWindow(screenWidth, screenHeight, "NoobNetwork - MNIST Professional Suite");
    SetTargetFPS(60);
    
    GuiSetStyle(DEFAULT, TEXT_SIZE, 18); // 调大字体，看着不累

    // --- 状态变量 ---
    NeuralNetwork nn({784, 128, 10}, 0.01);
    bool isModelLoaded = false;
    string model_path = "data/trained_model.txt";
    if (file_exists(model_path)) { nn.load_model(model_path); isModelLoaded = true; }

    RenderTexture2D canvas = LoadRenderTexture(400, 400);
    BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode();

    MNISTData train_data;
    MNISTData test_data;
    bool isTrainingState = false;
    int currentEpoch = 0, currentImageIdx = 0, totalEpochs = 5, totalSamples = 0;
    float trainingProgress = 0.0f, testAccuracy = 0.0f;
    int recognizedDigit = -1;
    int activeActivation = 0; // 0:Sigmoid, 1:ReLU, 2:Tanh

    // 贪心学习率平滑
    double previous_loss = 999999.0, smoothed_loss = 0.0;
    const double LR_INCREASE = 1.01, LR_DECREASE = 0.99;
    double current_batch_display_loss = 0.0;

    while (!WindowShouldClose()) {
        // --- 逻辑：训练引擎 ---
        if (isTrainingState && !train_data.images.empty()) {
            int batchSize = 200; // 稍微加大 Batch 提高效率
            double batch_loss_sum = 0.0;
            for (int b = 0; b < batchSize && isTrainingState; b++) {
                NNMatrix input = train_data.images[currentImageIdx];
                NNMatrix target = train_data.labels[currentImageIdx];
                NNMatrix output = nn.forward(input);
                NNMatrix diff = NNMatrix::subtract(target, output);
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) sample_loss += diff.data[r][0] * diff.data[r][0];
                batch_loss_sum += sample_loss;
                nn.train(input, target);
                currentImageIdx++;
                if (currentImageIdx >= totalSamples) {
                    currentImageIdx = 0; currentEpoch++;
                    if (currentEpoch >= totalEpochs) { isTrainingState = false; nn.save_model(model_path); isModelLoaded = true; }
                }
            }
            double batch_avg = batch_loss_sum / batchSize;
            if (smoothed_loss == 0.0) smoothed_loss = batch_avg;
            else smoothed_loss = 0.05 * batch_avg + 0.95 * smoothed_loss; // 更平滑的曲线
            current_batch_display_loss = smoothed_loss;

            if (smoothed_loss < previous_loss) nn.learningRate *= LR_INCREASE;
            else nn.learningRate *= LR_DECREASE;
            
            if (nn.learningRate > 0.1) nn.learningRate = 0.1;
            if (nn.learningRate < 0.001) nn.learningRate = 0.001;
            previous_loss = smoothed_loss;
            trainingProgress = (float)(currentEpoch * totalSamples + currentImageIdx) / (totalEpochs * totalSamples);
        }

        // --- 绘图逻辑 ---
        BeginDrawing();
        ClearBackground(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
        DrawText("Mini-Digits Neural Network (Greedy Adaptive & Test Suite)", 30, 25, 22, DARKGRAY);

        // 1. 左侧画板区 (400x400)
        Rectangle canvasArea = { 40, 100, 400, 400 };
        GuiGroupBox({ 25, 80, 430, 460 }, "Drawing Canvas");
        DrawTextureRec(canvas.texture, { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height }, { 40, 100 }, WHITE);
        DrawRectangleLinesEx(canvasArea, 1, GRAY);
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), canvasArea)) {
            BeginTextureMode(canvas);
            DrawCircleV({ GetMousePosition().x - 40, GetMousePosition().y - 100 }, 15.0f, BLACK);
            EndTextureMode();
        }

        // 2. 右侧控制面板区
        Rectangle controlBox = { 480, 80, 490, 460 };
        GuiGroupBox(controlBox, "Control Center");

        // --- 子区域 A: 激活函数配置 ---
        GuiLabel({ 500, 110, 120, 30 }, "Activation Layer:");
        // 为 ToggleGroup 提供更多宽度，防止文字挤在一起
        GuiToggleGroup({ 630, 110, 100, 30 }, "SIGMOID;RELU;TANH", &activeActivation);

        // --- 子区域 B: 训练控制 ---
        if (GuiButton({ 500, 160, 225, 45 }, isTrainingState ? "TRAINING..." : "Start Training")) {
            if (!isTrainingState) {
                if (train_data.images.empty()) {
                    BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Train Data...", 400, 300, 20, DARKGRAY); EndDrawing();
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                if (!train_data.images.empty()) { 
                    isTrainingState = true; currentEpoch = 0; currentImageIdx = 0; totalSamples = train_data.images.size();
                    nn = NeuralNetwork({784, 128, 10}, 0.01, (ActivationType)activeActivation); 
                    previous_loss = 999999.0; smoothed_loss = 0.0;
                }
            }
        }
        if (GuiButton({ 735, 160, 220, 45 }, "Run Test (10k)")) {
            if (test_data.images.empty()) {
                BeginDrawing(); ClearBackground(RAYWHITE); DrawText("Loading Test Data...", 400, 300, 20, DARKGRAY); EndDrawing();
                test_data = DataLoader::load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
            }
            if (!test_data.images.empty()) testAccuracy = (float)nn.get_accuracy(test_data);
        }

        // --- 子区域 C: 指标显示 ---
        GuiLabel({ 500, 220, 400, 30 }, TextFormat("Last Test Accuracy: %.2f%%", testAccuracy * 100.0f));
        GuiLabel({ 500, 250, 400, 30 }, TextFormat("Current LR: %.6f", nn.learningRate));
        GuiLabel({ 500, 280, 200, 30 }, "Model Persistence:");
        if (isModelLoaded) {
            DrawCircle(670, 295, 10, GREEN); 
            DrawText("READY", 690, 285, 18, DARKGREEN);
        } else {
            DrawCircle(670, 295, 10, RED);
            DrawText("EMPTY", 690, 285, 18, MAROON);
        }

        // --- 子区域 D: 结果区（带背景阴影，更醒目） ---
        DrawRectangle(500, 330, 455, 120, Fade(LIGHTGRAY, 0.5f));
        DrawRectangleLines(500, 330, 455, 120, GRAY);
        if (recognizedDigit != -1) {
            DrawText(TextFormat("%d", recognizedDigit), 705, 345, 90, DARKBLUE);
            DrawText("Prediction Result", 515, 340, 16, GRAY);
        } else {
            DrawText("WAITING FOR INPUT", 580, 375, 24, GRAY);
        }

        // --- 子区域 E: 操作按钮 ---
        if (GuiButton({ 500, 470, 225, 45 }, "Recognize Digit")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture); ImageFlipVertical(&img); Image final_img = preprocess_image(img);
                NNMatrix input(784, 1); Color* p = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) input.data[i][0] = (255.0f - p[i].r) / 255.0f;
                recognizedDigit = nn.predict(input);
                UnloadImageColors(p); UnloadImage(final_img); UnloadImage(img);
            }
        }
        if (GuiButton({ 735, 470, 220, 45 }, "Clear Canvas")) { 
            BeginTextureMode(canvas); ClearBackground(RAYWHITE); EndTextureMode(); 
            recognizedDigit = -1; 
        }

        // 3. 底部训练进度条
        if (isTrainingState) {
            DrawRectangle(0, 560, screenWidth, 90, Fade(LIGHTGRAY, 0.95f));
            float p = trainingProgress * 100.0f;
            GuiProgressBar({ 50, 600, 900, 30 }, "PROGRESS", TextFormat("%.1f%%", p), &p, 0, 100);
            DrawText(TextFormat("EPOCH: %d / 5   |   SMOOTHED LOSS: %.4f   |   LR: %.6f", currentEpoch + 1, current_batch_display_loss, nn.learningRate), 55, 575, 18, DARKGREEN);
        }
        
        EndDrawing();
    }
    UnloadRenderTexture(canvas); CloseWindow();
    return 0;
}