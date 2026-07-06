// src/ir/dag.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// ============================================================
// 数据类型
// ============================================================
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I32,
    I64,
    Bool,
}

// ============================================================
// 张量类型
// ============================================================
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorType {
    pub dtype: DataType,
    pub shape: Vec<i64>,  // -1 表示动态维度
}

// ============================================================
// 属性值
// ============================================================
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    Shape(Vec<i64>),
}

// ============================================================
// 值（数据流边）
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    pub id: u64,
    pub name: String,
    pub ty: TensorType,
    pub producer: Option<u64>,  // 哪个 Op 产生的
}

// ============================================================
// 算子节点
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    pub id: u64,
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<u64>,          // Value ID
    pub outputs: Vec<u64>,         // Value ID
    pub attrs: HashMap<String, AttrValue>,
}

// ============================================================
// DAG 计算图
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagGraph {
    pub name: String,
    pub values: HashMap<u64, Value>,
    pub ops: HashMap<u64, Op>,
    pub constants: HashMap<u64, Vec<u8>>,  // 常量张量数据
    pub inputs: Vec<u64>,                   // 输入 Value ID
    pub outputs: Vec<u64>,                  // 输出 Value ID
    pub next_id: u64,
}

impl DagGraph {
    pub fn new(name: &str) -> Self {
        DagGraph {
            name: name.to_string(),
            values: HashMap::new(),
            ops: HashMap::new(),
            constants: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add_value(&mut self, name: &str, ty: TensorType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.values.insert(id, Value {
            id,
            name: name.to_string(),
            ty,
            producer: None,
        });
        id
    }

    pub fn add_constant(&mut self, name: &str, ty: TensorType, data: Vec<u8>) -> u64 {
        let id = self.add_value(name, ty);
        self.constants.insert(id, data);
        id
    }

    pub fn add_op(&mut self, op_type: &str, inputs: Vec<u64>, outputs: Vec<u64>, attrs: HashMap<String, AttrValue>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        for &out_id in &outputs {
            if let Some(v) = self.values.get_mut(&out_id) {
                v.producer = Some(id);
            }
        }
        self.ops.insert(id, Op {
            id,
            name: format!("{}_{}", op_type, id),
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attrs,
        });
        id
    }

    pub fn set_inputs(&mut self, inputs: Vec<u64>) {
        self.inputs = inputs;
    }

    pub fn set_outputs(&mut self, outputs: Vec<u64>) {
        self.outputs = outputs;
    }

    pub fn get_value(&self, id: u64) -> Option<&Value> {
        self.values.get(&id)
    }

    pub fn get_op(&self, id: u64) -> Option<&Op> {
        self.ops.get(&id)
    }

    // ============================================================
    // 拓扑排序
    // ============================================================
    pub fn topological_sort(&self) -> Result<Vec<u64>, String> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();

        // 初始化
        for (&id, _) in &self.ops {
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_insert(Vec::new());
        }

        // 构建边
        for (op_id, op) in &self.ops {
            for &out_id in &op.outputs {
                // 找到使用这个输出的算子
                for (next_id, next_op) in &self.ops {
                    if next_op.inputs.contains(&out_id) {
                        adj.entry(*op_id).or_insert(Vec::new()).push(*next_id);
                        *in_degree.entry(*next_id).or_insert(0) += 1;
                    }
                }
            }
        }

        // Kahn 算法
        let mut queue: Vec<u64> = in_degree.iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();

        let mut result = Vec::new();
        while let Some(id) = queue.pop() {
            result.push(id);
            if let Some(neighbors) = adj.get(&id) {
                for &next in neighbors {
                    let deg = in_degree.get_mut(&next).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(next);
                    }
                }
            }
        }

        if result.len() != self.ops.len() {
            return Err("Graph has cycles".to_string());
        }

        Ok(result)
    }
}
