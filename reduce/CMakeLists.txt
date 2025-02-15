# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言支持（C++ 和 CUDA）
project(reduce LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA 工具包
find_package(CUDA REQUIRED)

# 设置 CUDA 架构（根据你的 GPU 计算能力调整）
set(CUDA_ARCHITECTURES 80)  # 例如 80 对应 Ampere 架构（如 RTX 30 系列）
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80)

# 添加源文件和头文件
set(SOURCE_FILES main.cu reduce_v.cu)  # 主程序和函数实现
set(HEADER_FILES utils.h)              # 头文件

# 添加可执行文件并指定源文件和头文件
add_executable(reduce ${SOURCE_FILES} ${HEADER_FILES})

# 链接 CUDA 库
target_link_libraries(reduce ${CUDA_LIBRARIES})

# 添加当前目录为包含路径（确保 utils.h 可以被找到）
include_directories(${CMAKE_CURRENT_SOURCE_DIR})