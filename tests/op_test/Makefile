# 操作符测试 Makefile
# 编译llama.cpp版本的测试

LLAMA_CPP_ROOT = ../llama.cpp-1.0.0
LLAMA_BUILD = $(LLAMA_CPP_ROOT)/build

CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17

# Include paths
INCLUDES = -I$(LLAMA_CPP_ROOT) \
           -I$(LLAMA_CPP_ROOT)/ggml \
           -I$(LLAMA_CPP_ROOT)/ggml/src \
           -I$(LLAMA_CPP_ROOT)/ggml/src/ggml-cpu \
           -I$(LLAMA_CPP_ROOT)/ggml/src/ggml-cpu/arch/arm \
           -I$(LLAMA_CPP_ROOT)/ggml/src/ggml-backend

# Library paths
LDFLAGS = -L$(LLAMA_BUILD)/ggml/src \
          -L$(LLAMA_BUILD)/ggml/src/ggml-cpu \
          -L$(LLAMA_BUILD)/ggml/src/ggml-backend \
          -lggml -lggml-cpu -lggml-backend -lggml-base \
          -framework Accelerate

TARGET = op_test_cpp

all: $(TARGET)

$(TARGET): op_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

# 测试示例
test-rms: $(TARGET)
	./$(TARGET) --op rms_norm --verbose
	./$(TARGET) --op rms_norm --bench 1000

test-q8k: $(TARGET)
	./$(TARGET) --op quantize_q8_k --verbose

test-q5k: $(TARGET)
	./$(TARGET) --op vec_dot_q5_k --verbose
	./$(TARGET) --op vec_dot_q5_k --bench 1000

.PHONY: all clean test-rms test-q8k test-q5k
