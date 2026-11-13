// 算子定义 + 类型特征表

/// 量化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

/// 核心算子
#[derive(Debug, Clone)]
pub enum CoreOperator {
    /// RMS 归一化
    RmsNorm { hidden_size: usize, eps: f32 },

    /// 线性层（支持量化）
    Linear { in_features: usize, out_features: usize, quant_type: QuantType },

    /// RoPE 位置编码
    Rope { head_dim: usize, max_seq_len: usize },

    /// Flash Attention
    FlashAttention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },

    /// SwiGLU 激活
    SwiGLU { hidden_size: usize, intermediate_size: usize },

    /// MoE 路由
    MoERouter { n_experts: usize, top_k: usize },

    /// MoE 专家
    MoEExpert {
        expert_id: usize,
        in_features: usize,
        out_features: usize,
        quant_type: QuantType,
    },
}

/// 算子执行错误
#[derive(Debug)]
pub enum OpError {
    NotImplemented(String),
    InvalidInput(String),
    DeviceError(String),
}
