cmake_minimum_required(VERSION 3.10)
project(NoobNetwork)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# --- 直接引入本地已下载好的 raylib ---
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) 
# 这行代码告诉 CMake：去当前目录下的 raylib 文件夹里找配置
add_subdirectory(raylib) 
# ------------------------------------

include_directories(src)

# 你的源文件列表（已移除 activations.cpp）
set(SOURCES 
    src/main.cpp 
    src/NNMatrix.cpp 
    src/data_loader.cpp 
    src/neural_network.cpp
)

add_executable(NoobNetwork ${SOURCES})

# 链接本地 raylib
target_link_libraries(NoobNetwork PRIVATE raylib)

# 每次编译完成后，自动将根目录的 data 文件夹拷贝到 build 目录下
add_custom_command(TARGET NoobNetwork POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data
    COMMENT "Auto-copying data directory to build folder..."
)