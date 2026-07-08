use std::collections::HashMap;


// ============================================================
// Tape 记录
// ============================================================

#[derive(Debug, Clone)]
pub enum TapeEntry {
    Add {
        input_a: u64,
        input_b: u64,
        output: u64,
    },
    Sub {
        input_a: u64,
        input_b: u64,
        output: u64,
    },
    Mul {
        input_a: u64,
        input_b: u64,
        output: u64,
    },
    Div {
        input_a: u64,
        input_b: u64,
        output: u64,
    },
    Pow {
        input: u64,
        exponent: f32,
        output: u64,
    },
    Exp {
        input: u64,
        output: u64,
    },
    Sqrt {
        input: u64,
        output: u64,
    },
    Log {
        input: u64,
        output: u64,
    },
    Neg {
        input: u64,
        output: u64,
    },
    MatMul {
        input_a: u64,
        input_b: u64,
        output: u64,
    },
    Conv2d {
        input: u64,
        weight: u64,
        bias: Option<u64>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        output: u64,
    },
    Relu {
        input: u64,
        output: u64,
    },
    Sigmoid {
        input: u64,
        output: u64,
    },
    Tanh {
        input: u64,
        output: u64,
    },
    BatchNorm {
        input: u64,
        weight: u64,
        bias: u64,
        running_mean: u64,
        running_var: u64,
        eps: f32,
        output: u64,
    },
    LayerNorm {
        input: u64,
        weight: u64,
        bias: u64,
        eps: f32,
        output: u64,
    },
    RMSNorm {
        input: u64,
        weight: u64,
        eps: f32,
        output: u64,
    },
    MaxPool {
        input: u64,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output: u64,
    },
    AvgPool {
        input: u64,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output: u64,
    },
    Reshape {
        input: u64,
        output: u64,
        new_shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<u64>,
        dim: usize,
        output: u64,
    },
    Slice {
        input: u64,
        dim: usize,
        start: i64,
        end: i64,
        step: i64,
        output: u64,
    },
    Squeeze {
        input: u64,
        dim: Option<usize>,
        output: u64,
    },
    Unsqueeze {
        input: u64,
        dim: usize,
        output: u64,
    },
    Embedding {
        indices: u64,
        weight: u64,
        output: u64,
    },
    Softmax {
        input: u64,
        dim: usize,
        output: u64,
    },
    ReduceSum {
        input: u64,
        dim: usize,
        keepdim: bool,
        output: u64,
    },
    ReduceMean {
        input: u64,
        dim: usize,
        keepdim: bool,
        output: u64,
    },
    Select {
        condition: u64,
        true_val: u64,
        false_val: u64,
        output: u64,
    },
    // 参数节点 (叶子节点)
    Parameter {
        id: u64,
    },
    // 输入节点
    Input {
        id: u64,
    },
    // 常量节点 (不可训练)
    Constant {
        id: u64,
    },
}

impl TapeEntry {
    pub fn output_id(&self) -> u64 {
        match self {
            TapeEntry::Add { output, .. } => *output,
            TapeEntry::Sub { output, .. } => *output,
            TapeEntry::Mul { output, .. } => *output,
            TapeEntry::Div { output, .. } => *output,
            TapeEntry::Pow { output, .. } => *output,
            TapeEntry::Exp { output, .. } => *output,
            TapeEntry::Sqrt { output, .. } => *output,
            TapeEntry::Log { output, .. } => *output,
            TapeEntry::Neg { output, .. } => *output,
            TapeEntry::MatMul { output, .. } => *output,
            TapeEntry::Conv2d { output, .. } => *output,
            TapeEntry::Relu { output, .. } => *output,
            TapeEntry::Sigmoid { output, .. } => *output,
            TapeEntry::Tanh { output, .. } => *output,
            TapeEntry::BatchNorm { output, .. } => *output,
            TapeEntry::LayerNorm { output, .. } => *output,
            TapeEntry::RMSNorm { output, .. } => *output,
            TapeEntry::MaxPool { output, .. } => *output,
            TapeEntry::AvgPool { output, .. } => *output,
            TapeEntry::Reshape { output, .. } => *output,
            TapeEntry::Concat { output, .. } => *output,
            TapeEntry::Slice { output, .. } => *output,
            TapeEntry::Squeeze { output, .. } => *output,
            TapeEntry::Unsqueeze { output, .. } => *output,
            TapeEntry::Embedding { output, .. } => *output,
            TapeEntry::Softmax { output, .. } => *output,
            TapeEntry::ReduceSum { output, .. } => *output,
            TapeEntry::ReduceMean { output, .. } => *output,
            TapeEntry::Select { output, .. } => *output,
            TapeEntry::Parameter { id, .. } => *id,
            TapeEntry::Input { id, .. } => *id,
            TapeEntry::Constant { id, .. } => *id,
        }
    }

    pub fn inputs(&self) -> Vec<u64> {
        match self {
            TapeEntry::Add { input_a, input_b, .. } => vec![*input_a, *input_b],
            TapeEntry::Sub { input_a, input_b, .. } => vec![*input_a, *input_b],
            TapeEntry::Mul { input_a, input_b, .. } => vec![*input_a, *input_b],
            TapeEntry::Div { input_a, input_b, .. } => vec![*input_a, *input_b],
            TapeEntry::Pow { input, .. } => vec![*input],
            TapeEntry::Exp { input, .. } => vec![*input],
            TapeEntry::Sqrt { input, .. } => vec![*input],
            TapeEntry::Log { input, .. } => vec![*input],
            TapeEntry::Neg { input, .. } => vec![*input],
            TapeEntry::MatMul { input_a, input_b, .. } => vec![*input_a, *input_b],
            TapeEntry::Conv2d { input, weight, bias, .. } => {
                let mut v = vec![*input, *weight];
                if let Some(b) = bias { v.push(*b); }
                v
            }
            TapeEntry::Relu { input, .. } => vec![*input],
            TapeEntry::Sigmoid { input, .. } => vec![*input],
            TapeEntry::Tanh { input, .. } => vec![*input],
            TapeEntry::BatchNorm { input, weight, bias, .. } => vec![*input, *weight, *bias],
            TapeEntry::LayerNorm { input, weight, bias, .. } => vec![*input, *weight, *bias],
            TapeEntry::RMSNorm { input, weight, .. } => vec![*input, *weight],
            TapeEntry::MaxPool { input, .. } => vec![*input],
            TapeEntry::AvgPool { input, .. } => vec![*input],
            TapeEntry::Reshape { input, .. } => vec![*input],
            TapeEntry::Concat { inputs, .. } => inputs.clone(),
            TapeEntry::Slice { input, .. } => vec![*input],
            TapeEntry::Squeeze { input, .. } => vec![*input],
            TapeEntry::Unsqueeze { input, .. } => vec![*input],
            TapeEntry::Embedding { indices, weight, .. } => vec![*indices, *weight],
            TapeEntry::Softmax { input, .. } => vec![*input],
            TapeEntry::ReduceSum { input, .. } => vec![*input],
            TapeEntry::ReduceMean { input, .. } => vec![*input],
            TapeEntry::Select { condition, true_val, false_val, .. } => vec![*condition, *true_val, *false_val],
            TapeEntry::Parameter { .. } => vec![],
            TapeEntry::Input { .. } => vec![],
            TapeEntry::Constant { .. } => vec![],
        }
    }
}

// ============================================================
// Tape (计算图)
// ============================================================

#[derive(Debug, Clone, Default)]
pub struct Tape {
    entries: Vec<TapeEntry>,
    // value_id -> entry_index (用于快速查找)
    value_to_entry: HashMap<u64, usize>,
    // 参数列表
    param_ids: Vec<u64>,
    // 输入列表
    input_ids: Vec<u64>,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            entries: Vec::new(),
            value_to_entry: HashMap::new(),
            param_ids: Vec::new(),
            input_ids: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: TapeEntry) -> u64 {
        let output_id = entry.output_id();
        let idx = self.entries.len();
        self.entries.push(entry);
        self.value_to_entry.insert(output_id, idx);
        output_id
    }

    pub fn register_parameter(&mut self, id: u64) {
        self.param_ids.push(id);
        let entry = TapeEntry::Parameter { id };
        let idx = self.entries.len();
        self.entries.push(entry);
        self.value_to_entry.insert(id, idx);
    }

    pub fn register_input(&mut self, id: u64) {
        self.input_ids.push(id);
        let entry = TapeEntry::Input { id };
        let idx = self.entries.len();
        self.entries.push(entry);
        self.value_to_entry.insert(id, idx);
    }

    pub fn register_constant(&mut self, id: u64) {
        let entry = TapeEntry::Constant { id };
        let idx = self.entries.len();
        self.entries.push(entry);
        self.value_to_entry.insert(id, idx);
    }

    pub fn get_entry(&self, value_id: u64) -> Option<&TapeEntry> {
        self.value_to_entry.get(&value_id).map(|&idx| &self.entries[idx])
    }

    pub fn get_entry_mut(&mut self, value_id: u64) -> Option<&mut TapeEntry> {
        self.value_to_entry.get(&value_id).map(|&idx| &mut self.entries[idx])
    }

    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    pub fn param_ids(&self) -> &[u64] {
        &self.param_ids
    }

    pub fn input_ids(&self) -> &[u64] {
        &self.input_ids
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // 获取计算图的拓扑序（反向）
    pub fn reverse_order(&self, output_id: u64) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();

        fn dfs(
            id: u64,
            tape: &Tape,
            visited: &mut std::collections::HashSet<u64>,
            result: &mut Vec<usize>,
        ) {
            if visited.contains(&id) {
                return;
            }
            visited.insert(id);

            if let Some(entry) = tape.get_entry(id) {
                for input_id in entry.inputs() {
                    dfs(input_id, tape, visited, result);
                }
                if let Some(idx) = tape.value_to_entry.get(&id) {
                    result.push(*idx);
                }
            }
        }

        dfs(output_id, self, &mut visited, &mut result);
        result.reverse();
        result
    }
}
