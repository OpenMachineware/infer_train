# operators/meson.build
project('infer_train_operators', 'cpp',
        version: '0.1.0',
        default_options: [
            'cpp_std=c++17',
            'optimization=3',
            'warning_level=3'
        ]
)

inc = include_directories('include')

sources = files(
    'src/math/add.cpp',
    'src/math/matmul.cpp',
    'src/math/quantized.cpp',
    'src/nn/relu.cpp',
    'src/nn/softmax.cpp',
    'src/nn/quantized.cpp',
    'src/c_interface.cpp'
)

lib = static_library('infer_train_cpp',
                     sources,
                     include_directories: inc
)

infer_train_dep = declare_dependency(
    include_directories: inc,
    link_with: lib
)

# 单元测试（用 get_option 检查是否启用）
if get_option('build_tests')
    # 如果你的测试用 Google Test，需要依赖 gtest
    # 先简单测试一下编译
    message('Tests enabled')

    # 这里可以先留空，等需要时再写
    # test_exe = executable('test_operators',
    #     files('tests/test_matmul.cpp'),
    #     dependencies: [infer_train_dep],
    #     include_directories: inc
    # )
    # test('C++ operator tests', test_exe)
endif
