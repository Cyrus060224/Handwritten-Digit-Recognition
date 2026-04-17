#include <iostream>
#include <vector>
#include "raylib.h" // 引入跨平台图形库
#include "neural_network.h"

// 定义画板常数
const int SCREEN_WIDTH = 400;
const int SCREEN_HEIGHT = 400;
const int CANVAS_SIZE = 280;
const int OFFSET_X = (SCREEN_WIDTH - CANVAS_SIZE) / 2;
const int OFFSET_Y = 50;

int main() {
    // 1. 初始化窗口 (取代 EasyX 的 initgraph)
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "跨平台手写数字识别系统");
    SetTargetFPS(60);

    // TODO: 这里可以放你加载模型 (load_network) 的代码...

    // 2. 主循环 (只要窗口不关闭，就一直循环刷新)
    while (!WindowShouldClose()) {
        
        // --- 逻辑更新区 ---
        // TODO: 检测鼠标左键 (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) 并记录画笔轨迹
        // TODO: 检测键盘空格键 (IsKeyPressed(KEY_SPACE)) 触发神经网络识别

        // --- 画面渲染区 ---
        BeginDrawing();
        ClearBackground(RAYWHITE); // 清屏为白色

        // 画提示文字
        DrawText("Draw a digit (0-9)", 100, 10, 20, DARKGRAY);
        DrawText("Press SPACE to recognize", 70, 360, 20, GRAY);

        // 画一个黑色的画板框
        DrawRectangleLines(OFFSET_X, OFFSET_Y, CANVAS_SIZE, CANVAS_SIZE, BLACK);

        // TODO: 渲染用户写下的轨迹...

        EndDrawing();
    }

    // 3. 关闭窗口与清理内存
    CloseWindow();
    // TODO: 记得释放神经网络的内存 (delete net)

    return 0;
}