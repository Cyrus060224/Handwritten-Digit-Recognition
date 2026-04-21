#include <iostream>
#include <string>
#include <vector>
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "neural_network.h"
#include "data_loader.h"
#include <sys/stat.h>

using namespace std;

bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// 将用户的随意手写图像转化为标准 MNIST 格式
Image preprocess_image(Image original_img) {
    Color* pixels = LoadImageColors(original_img);
    int minX = original_img.width, minY = original_img.height;
    int maxX = 0, maxY = 0;
    bool hasInk = false;

    // 1. 扫描寻找包含数字的最小边界框 (Bounding Box)
    for (int y = 0; y < original_img.height; y++) {
        for (int x = 0; x < original_img.width; x++) {
            Color c = pixels[y * original_img.width + x];
            if (c.r < 128) { // 黑色墨水（画板是白底黑字）
                hasInk = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    UnloadImageColors(pixels);

    if (!hasInk) return GenImageColor(28, 28, WHITE);

    // 2. 裁剪出只有数字的矩形
    Rectangle bbox = { (float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1) };
    Image crop_img = ImageCopy(original_img);
    ImageCrop(&crop_img, bbox);

    // 3. 等比缩放到最大边长为 20
    int new_w, new_h;
    if (crop_img.width > crop_img.height) {
        new_w = 20;
        new_h = (int)((20.0f / crop_img.width) * crop_img.height);
    } else {
        new_h = 20;
        new_w = (int)((20.0f / crop_img.height) * crop_img.width);
    }
    if (new_w == 0) new_w = 1;
    if (new_h == 0) new_h = 1;
    ImageResize(&crop_img, new_w, new_h);

    // 4. 居中放置到一个全新的 28x28 白色画布上
    Image final_img = GenImageColor(28, 28, WHITE);
    int offsetX = (28 - new_w) / 2;
    int offsetY = (28 - new_h) / 2;
    Rectangle srcRec = { 0, 0, (float)new_w, (float)new_h };
    Rectangle dstRec = { (float)offsetX, (float)offsetY, (float)new_w, (float)new_h };
    
    ImageDraw(&final_img, crop_img, srcRec, dstRec, WHITE);
    UnloadImage(crop_img);

    return final_img;
}

int main() {
    const int screenWidth = 1000;
    const int screenHeight = 620;
    InitWindow(screenWidth, screenHeight, "Mini-Digits - Neural Network");
    SetTargetFPS(60);
    
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);

    vector<int> topology = {784, 128, 10};
    // 🌟 修改点 1：初始学习率设为 0.01，更加稳妥
    NeuralNetwork nn(topology, 0.01);
    bool isModelLoaded = false; 
    string model_path = "data/trained_model.txt";
    
    if (file_exists(model_path)) {
        nn.load_model(model_path);
        isModelLoaded = true;
    }

    RenderTexture2D canvas = LoadRenderTexture(400, 400); 
    BeginTextureMode(canvas);
    ClearBackground(RAYWHITE); 
    EndTextureMode();

    float brushSize = 15.0f;
    int recognizedDigit = -1;       
    Rectangle drawingArea = { 40, 100, 400, 400 }; 

    // --- 训练状态控制变量 ---
    bool isTrainingState = false;    
    int currentEpoch = 0;           
    int currentImageIdx = 0;        
    int totalEpochs = 5;            
    int totalSamples = 0;           
    float trainingProgress = 0.0f;  
    MNISTData train_data; 

    // --- 贪心算法 (Bold Driver) 状态变量 ---
    double previous_loss = 999999.0; 
    const double LR_INCREASE = 1.01; // 温和加速
    const double LR_DECREASE = 0.98; // 温和减速
    double current_batch_loss = 0.0; 

    while (!WindowShouldClose()) {
        Vector2 mousePos = GetMousePosition();
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(mousePos, drawingArea)) {
            BeginTextureMode(canvas);
            Vector2 drawPos = { mousePos.x - drawingArea.x, mousePos.y - drawingArea.y };
            DrawCircleV(drawPos, brushSize, BLACK);
            EndTextureMode();
        }

        // =======================================================
        // --- 核心：每帧切片训练逻辑 + 贪心算法 (Bold Driver) ---
        // =======================================================
        if (isTrainingState && !train_data.images.empty()) {
            int batchSize = 150; 
            double batch_loss_sum = 0.0; 
            
            for (int b = 0; b < batchSize && isTrainingState; b++) {
                NNMatrix input = train_data.images[currentImageIdx];
                NNMatrix target = train_data.labels[currentImageIdx];

                // 1. 计算单张样本的误差并累加
                NNMatrix output = nn.forward(input);
                NNMatrix diff = NNMatrix::subtract(target, output);
                double sample_loss = 0.0;
                for (int r = 0; r < diff.rows; r++) {
                    sample_loss += diff.data[r][0] * diff.data[r][0]; 
                }
                batch_loss_sum += sample_loss; 

                // 2. 执行训练（反向传播更新权重）
                nn.train(input, target);
                
                currentImageIdx++;

                // 检查这一轮是否跑完
                if (currentImageIdx >= totalSamples) {
                    currentImageIdx = 0;
                    currentEpoch++;
                    
                    if (currentEpoch >= totalEpochs) {
                        isTrainingState = false;
                        nn.save_model(model_path); 
                        isModelLoaded = true;
                        cout << "Training completed and model saved!" << endl;
                    }
                }
            }

            // 3. 计算这 150 张图的“平均误差”
            current_batch_loss = batch_loss_sum / batchSize; 

            // 4. 温和的贪心策略判定
            if (previous_loss != 999999.0) { 
                if (current_batch_loss < previous_loss) {
                    nn.learningRate *= LR_INCREASE; 
                } else if (current_batch_loss > previous_loss) {
                    nn.learningRate *= LR_DECREASE; 
                }
            }
            
            // 🌟 修改点 2：极其关键的上下限保护锁
            if (nn.learningRate > 0.1) nn.learningRate = 0.1;
            if (nn.learningRate < 0.001) nn.learningRate = 0.001;

            previous_loss = current_batch_loss;

            // 更新 UI 进度条
            trainingProgress = (float)(currentEpoch * totalSamples + currentImageIdx) / (totalEpochs * totalSamples);
        }

        BeginDrawing();
        ClearBackground(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));

        DrawText("Mini-Digits Neural Network System", 30, 20, 20, DARKGRAY);

        // --- Left Panel ---
        Rectangle canvasGroup = { 25, 75, 430, 440 };
        GuiGroupBox(canvasGroup, "Drawing Canvas");
        
        Rectangle sourceRec = { 0.0f, 0.0f, (float)canvas.texture.width, -(float)canvas.texture.height };
        DrawTextureRec(canvas.texture, sourceRec, { drawingArea.x, drawingArea.y }, WHITE);
        DrawRectangleLines(drawingArea.x, drawingArea.y, drawingArea.width, drawingArea.height, GRAY);

        // --- Right Top Panel ---
        Rectangle modelGroup = { 480, 80, 480, 220 };
        GuiGroupBox(modelGroup, "Model Configuration");

        GuiLabel({ 500, 110, 80, 30 }, "Status:");
        GuiLabel({ 580, 110, 250, 30 }, isModelLoaded ? "Loaded (trained_model.txt)" : "Not Ready (Needs Training)");

        GuiLabel({ 500, 160, 100, 30 }, "LR Policy:");
        GuiLabel({ 600, 160, 120, 30 }, "Greedy Adaptive"); 
        
        GuiLabel({ 730, 160, 80, 30 }, "Topology:");
        GuiLabel({ 810, 160, 150, 30 }, "784-128-10");

        if (GuiButton({ 500, 230, 440, 40 }, isTrainingState ? "TRAINING IN PROGRESS..." : "Start Training")) {
            if (!isTrainingState) {
                if (train_data.images.empty()) {
                    BeginDrawing();
                    ClearBackground(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));
                    DrawText("Loading MNIST Dataset... Please wait.", 300, 300, 20, DARKGRAY);
                    EndDrawing();

                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                
                if (!train_data.images.empty()) {
                    isTrainingState = true;
                    currentEpoch = 0;
                    currentImageIdx = 0;
                    totalSamples = train_data.images.size();
                    trainingProgress = 0.0f;
                    
                    // 🌟 修改点 3：每次重新训练都必须重置状态
                    previous_loss = 999999.0; 
                    nn.learningRate = 0.01; 
                }
            }
        }

        // --- Right Bottom Panel ---
        Rectangle paintGroup = { 480, 320, 480, 180 };
        GuiGroupBox(paintGroup, "Controls");

        GuiLabel({ 500, 350, 80, 20 }, "Brush:");
        GuiSlider({ 580, 350, 160, 20 }, "Thin", "Thick", &brushSize, 5.0f, 40.0f);

        if (GuiButton({ 500, 400, 200, 40 }, "Clear Canvas")) {
            BeginTextureMode(canvas);
            ClearBackground(RAYWHITE);
            EndTextureMode();
            recognizedDigit = -1;
        }

        if (GuiButton({ 740, 400, 200, 40 }, "Recognize")) {
            if (isModelLoaded) {
                Image img = LoadImageFromTexture(canvas.texture);
                ImageFlipVertical(&img); 
                Image final_img = preprocess_image(img);

                NNMatrix input(784, 1);
                Color* final_pixels = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) {
                    float brightness = (255.0f - final_pixels[i].r) / 255.0f; 
                    input.data[i][0] = brightness;
                }
                
                UnloadImageColors(final_pixels);
                UnloadImage(final_img);
                UnloadImage(img);

                recognizedDigit = nn.predict(input);
            }
        }

        // --- Result Display ---
        DrawRectangle(480, 520, 480, 80, LIGHTGRAY);
        DrawRectangleLines(480, 520, 480, 80, GRAY);
        
        if (recognizedDigit != -1) {
            string resText = "Prediction: " + to_string(recognizedDigit);
            DrawText(resText.c_str(), 500, 535, 40, DARKBLUE);
        } else {
            DrawText("Waiting for input...", 500, 545, 30, DARKGRAY);
        }

        // =======================================================
        // --- 底部训练进度条与贪心算法状态显示 ---
        // =======================================================
        if (isTrainingState) {
            DrawRectangle(0, 530, screenWidth, 90, Fade(LIGHTGRAY, 0.9f));
            
            float progPerc = trainingProgress * 100.0f;
            GuiProgressBar({ 50, 565, 900, 30 }, "PROGRESS", TextFormat("%.1f%%", progPerc), &progPerc, 0, 100);
            
            // 🌟 修改点 4：文本显示调整为 Batch Loss
            string statusInfo = TextFormat("Epoch: %d/%d | LR: %.6f | Batch Loss: %.4f", 
                                           currentEpoch + 1, totalEpochs, nn.learningRate, current_batch_loss);
            DrawText(statusInfo.c_str(), 50, 540, 20, DARKGREEN);
        }

        EndDrawing();
    }

    UnloadRenderTexture(canvas);
    CloseWindow();
    return 0;
}