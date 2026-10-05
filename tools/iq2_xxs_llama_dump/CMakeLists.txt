cmake_minimum_required(VERSION 3.14)
project(iq2_xxs_llama_dump C)

set(CMAKE_C_STANDARD 11)

# llama.cpp 路径
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp-0.5.0")

# 添加 llama.cpp 子目录
add_subdirectory(${LLAMA_DIR} llama_build)

# 创建可执行文件
add_executable(iq2_xxs_llama_dump iq2_xxs_llama_dump.c)

# 链接 ggml 库
target_link_libraries(iq2_xxs_llama_dump PRIVATE ggml)

# 包含路径
target_include_directories(iq2_xxs_llama_dump PRIVATE
    ${LLAMA_DIR}/ggml/src
    ${LLAMA_DIR}/ggml/src/ggml-cpu
    ${LLAMA_DIR}/ggml/include
)
