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
    # 浮点数学
    'src/math/add.cpp',
    'src/math/sub.cpp',
    'src/math/mul.cpp',
    'src/math/div.cpp',
    'src/math/pow.cpp',
    'src/math/sqrt.cpp',
    'src/math/exp.cpp',
    'src/math/log.cpp',
    'src/math/abs.cpp',
    'src/math/clamp.cpp',
    'src/math/round.cpp',
    'src/math/reduce.cpp',
    'src/math/arg.cpp',
    'src/math/std_var.cpp',
    'src/math/matmul.cpp',

    # 量化数学
    'src/math/add_q.cpp',
    'src/math/sub_q.cpp',
    'src/math/mul_q.cpp',
    'src/math/div_q.cpp',
    'src/math/pow_q.cpp',
    'src/math/sqrt_q.cpp',
    'src/math/exp_q.cpp',
    'src/math/log_q.cpp',
    'src/math/abs_q.cpp',
    'src/math/clamp_q.cpp',
    'src/math/round_q.cpp',
    'src/math/reduce_q.cpp',
    'src/math/std_var_q.cpp',
    'src/math/matmul_q.cpp',
    'src/math/compare.cpp',
    'src/math/slice.cpp',
    'src/math/cat.cpp',

    # 浮点 NN
    'src/nn/activation.cpp',
    'src/nn/relu.cpp',
    'src/nn/softmax.cpp',
    'src/nn/conv2d.cpp',
    'src/nn/conv.cpp',
    'src/nn/pooling.cpp',
    'src/nn/pool.cpp',
    'src/nn/batchnorm.cpp',
    'src/nn/layernorm.cpp',
    'src/nn/linear.cpp',
    'src/nn/norm.cpp',
    'src/nn/dropout.cpp',
    'src/nn/embedding.cpp',
    'src/nn/attention.cpp',
    'src/nn/rotary_embedding.cpp',
    'src/nn/rmsnorm.cpp',
    'src/nn/swiglu.cpp',

    # 量化 NN
    'src/nn/activation_q.cpp',
    'src/nn/relu_q.cpp',
    'src/nn/softmax_q.cpp',
    'src/nn/conv2d_q.cpp',
    'src/nn/conv_q.cpp',
    'src/nn/pooling_q.cpp',
    'src/nn/pool_q.cpp',
    'src/nn/batchnorm_q.cpp',
    'src/nn/layernorm_q.cpp',
    'src/nn/linear_q.cpp',
    'src/nn/norm_q.cpp',
    'src/nn/embedding_q.cpp',
    'src/nn/attention_q.cpp',
    'src/nn/rotary_embedding_q.cpp',
    'src/nn/rmsnorm_q.cpp',

    # C API
    'src/c_api/c_api_common.cpp',
    'src/c_api/c_api_math.cpp',
    'src/c_api/c_api_math_q.cpp',
    'src/c_api/c_api_reduce.cpp',
    'src/c_api/c_api_reduce_q.cpp',
    'src/c_api/c_api_matmul.cpp',
    'src/c_api/c_api_matmul_q.cpp',
    'src/c_api/c_api_nn.cpp',
    'src/c_api/c_api_nn_q.cpp',
    'src/c_api/c_api_activation.cpp',
    'src/c_api/c_api_activation_q.cpp',
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
