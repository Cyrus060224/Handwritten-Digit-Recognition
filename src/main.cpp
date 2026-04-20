#include <iostream>
#include <string>
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
    
    // 使用默认 UI 样式，摒弃外部字体依赖带来的不确定性
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);

    vector<int> topology = {784, 128, 10};
    NeuralNetwork nn(topology, 0.05);
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
    bool isTraining = false;        
    Rectangle drawingArea = { 40, 100, 400, 400 }; 

    // --- 训练状态控制变量 ---
    bool isTrainingState = false;    
    int currentEpoch = 0;           
    int currentImageIdx = 0;        
    int totalEpochs = 5;            
    int totalSamples = 0;           
    float trainingProgress = 0.0f;  
    MNISTData train_data; // 把数据集变量放在外面，这样整个程序都能用

    while (!WindowShouldClose()) {
        Vector2 mousePos = GetMousePosition();
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(mousePos, drawingArea)) {
            BeginTextureMode(canvas);
            Vector2 drawPos = { mousePos.x - drawingArea.x, mousePos.y - drawingArea.y };
            DrawCircleV(drawPos, brushSize, BLACK);
            EndTextureMode();
        }

        // --- 核心：每帧切片训练逻辑，保证 UI 不卡顿 ---
        if (isTrainingState && !train_data.images.empty()) {
            int batchSize = 150; // 每一帧训练 150 张图
            
            for (int b = 0; b < batchSize && isTrainingState; b++) {
                // 喂一张图给神经网络
                nn.train(train_data.images[currentImageIdx], train_data.labels[currentImageIdx]);
                currentImageIdx++;

                // 检查这一轮的 6万张图 是否跑完
                if (currentImageIdx >= totalSamples) {
                    currentImageIdx = 0;
                    currentEpoch++;
                    
                    // 检查全部 5 轮是否跑完
                    if (currentEpoch >= totalEpochs) {
                        isTrainingState = false;
                        nn.save_model(model_path); // 自动保存
                        isModelLoaded = true;
                        cout << "Training completed and model saved!" << endl;
                    }
                }
            }
            // 实时更新进度条百分比
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
        GuiLabel({ 600, 160, 80, 30 }, "Adaptive"); 
        
        GuiLabel({ 700, 160, 80, 30 }, "Topology:");
        GuiLabel({ 780, 160, 150, 30 }, "784-128-10");

        // 修改后的按钮：点击只负责“开灯”，不跑循环
        if (GuiButton({ 500, 230, 440, 40 }, "Start Training")) {
            if (!isTrainingState) {
                // 1. 如果还没加载数据，就加载一次
                if (train_data.images.empty()) {
                    train_data = DataLoader::load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
                }
                
                // 2. 启动状态机
                if (!train_data.images.empty()) {
                    isTrainingState = true;
                    currentEpoch = 0;
                    currentImageIdx = 0;
                    totalSamples = train_data.images.size();
                    trainingProgress = 0.0f;
                    cout << "UI Training started..." << endl;
                } else {
                    cout << "Error: Cannot load dataset. Please check data/ directory!" << endl;
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
                // 1. 获取画板的原始图像
                Image img = LoadImageFromTexture(canvas.texture);
                
                // 🌟 解决 Raylib 倒影 BUG 的关键：垂直翻转图像！
                ImageFlipVertical(&img); 
                
                // 2. 调用刚才写好的预处理流水线
                Image final_img = preprocess_image(img);

                // 3. 转换为神经网络输入向量 (784x1)
                NNMatrix input(784, 1);
                Color* final_pixels = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) {
                    // 白底黑字转为黑底白字（1.0代表纯黑，0.0代表纯白）
                    float brightness = (255.0f - final_pixels[i].r) / 255.0f; 
                    input.data[i][0] = brightness;
                }
                
                // 4. 清理内存
                UnloadImageColors(final_pixels);
                UnloadImage(final_img);
                UnloadImage(img);

                // 5. 喂给模型预测
                recognizedDigit = nn.predict(input);
                
            } else {
                cout << "Please train or load a model first!" << endl;
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

        // --- 在屏幕底部绘制 UI 进度条 ---
        if (isTrainingState) {
            // 画一个半透明的底色框
            DrawRectangle(0, 530, screenWidth, 90, Fade(LIGHTGRAY, 0.8f));
            
            float progPerc = trainingProgress * 100.0f;
            // 调用 Raygui 画出动态进度条
            GuiProgressBar({ 50, 565, 900, 30 }, "PROGRESS", TextFormat("%.1f%%", progPerc), &progPerc, 0, 100);
            
            // 显示当前是在第几个 Epoch
            DrawText(TextFormat("Training Epoch: %d / %d", currentEpoch + 1, totalEpochs), 50, 540, 20, DARKGRAY);
        }

        EndDrawing();
    }

    UnloadRenderTexture(canvas);
    CloseWindow();
    return 0;
}