// src/transform/simplify.rs

use crate::ir::dag::AttrValue;
use crate::ir::dag::{DagGraph, DataType, Op};
use std::collections::HashMap;

pub struct AlgebraicSimplifyPass;

// ============================================================
// 定义返回类型
// ============================================================
enum SimplifyResult {
    None,
    Replace(u64),
    NewOps(Vec<Op>),
}

impl AlgebraicSimplifyPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut iteration = 0;
        const MAX_ITER: usize = 10;

        loop {
            let mut iter_changed = false;
            iter_changed |= Self::run_pass(graph);
            changed |= iter_changed;

            iteration += 1;
            if !iter_changed || iteration >= MAX_ITER {
                break;
            }
        }

        changed
    }

    fn run_pass(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();
        let mut new_ops = Vec::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o.clone(),
                None => continue,
            };

            // ---------- 代数化简 ----------
            let result = Self::simplify_algebraic(graph, &op);
            match result {
                SimplifyResult::Replace(repl) => {
                    replacements.insert(op.outputs[0], repl);
                    to_remove.push(op_id);
                    changed = true;
                    continue;
                }
                SimplifyResult::NewOps(ops) => {
                    // 处理多个新算子
                    let output_id = op.outputs[0];
                    let mut last_output = output_id;
                    for new_op in ops {
                        // 分配真实 ID
                        let real_id = graph.allocate_op_id();
                        let mut inserted_op = new_op.clone();
                        inserted_op.id = real_id;
                        inserted_op.name =
                            format!("{}_{}", new_op.op_type, real_id);
                        // 更新 outputs
                        if inserted_op.outputs.is_empty() {
                            inserted_op.outputs.push(graph.next_id);
                            graph.next_id += 1;
                        }
                        last_output = inserted_op.outputs[0];
                        new_ops.push(inserted_op);
                    }
                    // 最后一个算子的输出替换原来 op 的输出
                    replacements.insert(output_id, last_output);
                    to_remove.push(op_id);
                    changed = true;
                    continue;
                }
                SimplifyResult::None => {}
            }

            // ---------- 张量操作化简 ----------
            let result = Self::simplify_tensor_ops(graph, &op);
            if let Some((new_op, replacement)) = result {
                if let Some(repl) = replacement {
                    replacements.insert(op.outputs[0], repl);
                }
                if let Some(new_op) = new_op {
                    new_ops.push(new_op);
                }
                to_remove.push(op_id);
                changed = true;
                continue;
            }

            // ---------- 激活函数化简 ----------
            let result = Self::simplify_activation(graph, &op);
            if let Some(replacement) = result {
                replacements.insert(op.outputs[0], replacement);
                to_remove.push(op_id);
                changed = true;
                continue;
            }

            // ---------- 池化化简 ----------
            let result = Self::simplify_pool(graph, &op);
            if let Some(replacement) = result {
                replacements.insert(op.outputs[0], replacement);
                to_remove.push(op_id);
                changed = true;
                continue;
            }
        }

        // 应用替换
        for (old_id, new_id) in &replacements {
            Self::apply_replacement(graph, *old_id, *new_id);
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        for op in new_ops {
            graph.insert_op(op);
        }

        changed
    }

    // ============================================================
    // 代数化简
    // ============================================================

    fn simplify_algebraic(graph: &mut DagGraph, op: &Op) -> SimplifyResult {
        match op.op_type.as_str() {
            // ---------- ADD ----------
            "add" => {
                if op.inputs.len() >= 2 {
                    let in1 = op.inputs[0];
                    let in2 = op.inputs[1];

                    // ADD(x, -x) -> 0
                    if Self::is_neg_of(graph, in1, in2)
                        || Self::is_neg_of(graph, in2, in1)
                    {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // ADD(x, 0) -> x
                    if graph.is_zero_constant(in2) {
                        return SimplifyResult::Replace(in1);
                    }
                    if graph.is_zero_constant(in1) {
                        return SimplifyResult::Replace(in2);
                    }

                    // ADD(x, ADD(y, z)) -> ADD(ADD(x, y), z)  (结合律)
                    if let Some(inner_op) = Self::get_producer_op(graph, in2) {
                        if inner_op.op_type == "add" {
                            let inner_input0 = inner_op.inputs[0];
                            let inner_input1 = inner_op.inputs[1];
                            let _output0 = op.outputs[0];

                            let new_out1 = graph.next_id;
                            graph.next_id += 1;
                            let new_out2 = graph.next_id;
                            graph.next_id += 1;

                            let add1 = Op {
                                id: 0,
                                name: "add_temp1".to_string(),
                                op_type: "add".to_string(),
                                inputs: vec![in1, inner_input0],
                                outputs: vec![new_out1],
                                attrs: HashMap::new(),
                            };

                            let add2 = Op {
                                id: 0,
                                name: "add_temp2".to_string(),
                                op_type: "add".to_string(),
                                inputs: vec![new_out1, inner_input1],
                                outputs: vec![new_out2],
                                attrs: HashMap::new(),
                            };

                            return SimplifyResult::NewOps(vec![add1, add2]);
                        }
                    }

                    // ADD(x, -y) -> SUB(x, y)  (如果 y 是负常数，转为减法)
                    if let Some(neg_const_id) =
                        Self::get_neg_constant(graph, in2)
                    {
                        let sub_op =
                            Self::create_binary_op("sub", in1, neg_const_id);
                        return SimplifyResult::NewOps(vec![sub_op]);
                    }
                    if let Some(neg_const_id) =
                        Self::get_neg_constant(graph, in1)
                    {
                        let sub_op =
                            Self::create_binary_op("sub", in2, neg_const_id);
                        return SimplifyResult::NewOps(vec![sub_op]);
                    }
                }
            }

            // ---------- SUB ----------
            "sub" => {
                if op.inputs.len() >= 2 {
                    let in1 = op.inputs[0];
                    let in2 = op.inputs[1];

                    // SUB(x, x) -> 0
                    if in1 == in2 {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // SUB(x, 0) -> x
                    if graph.is_zero_constant(in2) {
                        return SimplifyResult::Replace(in1);
                    }

                    // SUB(0, x) -> NEG(x)
                    if graph.is_zero_constant(in1) {
                        let neg_op = Self::create_unary_op("neg", in2);
                        return SimplifyResult::NewOps(vec![neg_op]);
                    }
                }
            }

            // ---------- MUL ----------
            "mul" => {
                if op.inputs.len() >= 2 {
                    let in1 = op.inputs[0];
                    let in2 = op.inputs[1];

                    // MUL(x, 0) -> 0
                    if graph.is_zero_constant(in2)
                        || graph.is_zero_constant(in1)
                    {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // MUL(x, 1) -> x
                    if graph.is_one_constant(in2) {
                        return SimplifyResult::Replace(in1);
                    }
                    if graph.is_one_constant(in1) {
                        return SimplifyResult::Replace(in2);
                    }
                }
            }

            // ---------- DIV ----------
            "div" => {
                if op.inputs.len() >= 2 {
                    let in1 = op.inputs[0];
                    let in2 = op.inputs[1];

                    // DIV(0, x) -> 0
                    if graph.is_zero_constant(in1) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // DIV(x, 1) -> x
                    if graph.is_one_constant(in2) {
                        return SimplifyResult::Replace(in1);
                    }

                    // DIV(x, x) -> 1
                    if in1 == in2 {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let one_id = graph.get_or_create_one(dtype);
                        return SimplifyResult::Replace(one_id);
                    }
                }
            }

            // ---------- POW ----------
            "pow" => {
                if op.inputs.len() >= 2 {
                    let in1 = op.inputs[0];
                    let in2 = op.inputs[1];

                    // POW(x, 1) -> x
                    if graph.is_one_constant(in2) {
                        return SimplifyResult::Replace(in1);
                    }

                    // POW(1, x) -> 1
                    if graph.is_one_constant(in1) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let one_id = graph.get_or_create_one(dtype);
                        return SimplifyResult::Replace(one_id);
                    }

                    // POW(x, 0) -> 1
                    if graph.is_zero_constant(in2) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let one_id = graph.get_or_create_one(dtype);
                        return SimplifyResult::Replace(one_id);
                    }
                }
            }

            // ---------- EXP ----------
            "exp" => {
                if op.inputs.len() >= 1 && graph.is_zero_constant(op.inputs[0])
                {
                    let dtype = graph
                        .values
                        .get(&op.outputs[0])
                        .map(|v| v.ty.dtype)
                        .unwrap_or(DataType::F32);
                    let one_id = graph.get_or_create_one(dtype);
                    return SimplifyResult::Replace(one_id);
                }
            }

            // ---------- LOG ----------
            "log" => {
                if op.inputs.len() >= 1 {
                    let in1 = op.inputs[0];

                    // LOG(1) -> 0
                    if graph.is_one_constant(in1) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // LOG(e) -> 1 (e ≈ 2.71828)
                    // 可选：检测常数 e
                }
            }

            // ---------- NEG ----------
            "neg" => {
                if op.inputs.len() >= 1 {
                    if let Some(producer) =
                        Self::get_producer_op(graph, op.inputs[0])
                    {
                        // NEG(NEG(x)) -> x
                        if producer.op_type == "neg" {
                            return SimplifyResult::Replace(producer.inputs[0]);
                        }
                    }
                }
            }

            // ---------- ABS ----------
            "abs" => {
                if op.inputs.len() >= 1 && graph.is_zero_constant(op.inputs[0])
                {
                    let dtype = graph
                        .values
                        .get(&op.outputs[0])
                        .map(|v| v.ty.dtype)
                        .unwrap_or(DataType::F32);
                    let zero_id = graph.get_or_create_zero(dtype);
                    return SimplifyResult::Replace(zero_id);
                }
            }

            // ---------- SQRT ----------
            "sqrt" => {
                if op.inputs.len() >= 1 {
                    let in1 = op.inputs[0];

                    // SQRT(0) -> 0
                    if graph.is_zero_constant(in1) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let zero_id = graph.get_or_create_zero(dtype);
                        return SimplifyResult::Replace(zero_id);
                    }

                    // SQRT(1) -> 1
                    if graph.is_one_constant(in1) {
                        let dtype = graph
                            .values
                            .get(&op.outputs[0])
                            .map(|v| v.ty.dtype)
                            .unwrap_or(DataType::F32);
                        let one_id = graph.get_or_create_one(dtype);
                        return SimplifyResult::Replace(one_id);
                    }
                }
            }

            _ => {}
        }

        SimplifyResult::None
    }

    // ============================================================
    // 张量操作化简
    // ============================================================

    fn simplify_tensor_ops(
        graph: &DagGraph,
        op: &Op,
    ) -> Option<(Option<Op>, Option<u64>)> {
        match op.op_type.as_str() {
            // ---------- RESHAPE 化简 ----------
            "reshape" => {
                if op.inputs.len() >= 1 {
                    // 连续 RESHAPE 合并
                    if let Some(producer) =
                        Self::get_producer_op(graph, op.inputs[0])
                    {
                        if producer.op_type == "reshape" {
                            let new_shape = Self::get_reshape_shape(op);
                            if let Some(shape) = new_shape {
                                let new_op = Self::create_reshape_op(
                                    producer.inputs[0],
                                    &shape,
                                );
                                return Some((Some(new_op), None));
                            }
                        }
                    }

                    // RESHAPE 后形状不变 -> 消除
                    if let Some(input) = graph.values.get(&op.inputs[0]) {
                        if let Some(output) = graph.values.get(&op.outputs[0]) {
                            if input.ty.shape == output.ty.shape {
                                return Some((None, Some(op.inputs[0])));
                            }
                        }
                    }
                }
                None
            }

            // ---------- 连续 TRANSPOSE 合并 ----------
            "transpose" => {
                if op.inputs.len() >= 1 {
                    if let Some(producer) =
                        Self::get_producer_op(graph, op.inputs[0])
                    {
                        if producer.op_type == "transpose" {
                            return Some((None, Some(producer.inputs[0])));
                        }
                    }
                }
                None
            }

            // ---------- 单输入 CONCAT 消除 ----------
            "concat" => {
                if op.inputs.len() == 1 {
                    return Some((None, Some(op.inputs[0])));
                }
                None
            }

            // ---------- 单输出 SPLIT 消除 ----------
            "split" => {
                if op.inputs.len() >= 1 && op.outputs.len() == 1 {
                    return Some((None, Some(op.inputs[0])));
                }
                None
            }

            _ => None,
        }
    }

    // ============================================================
    // 激活函数化简
    // ============================================================

    fn simplify_activation(graph: &DagGraph, op: &Op) -> Option<u64> {
        match op.op_type.as_str() {
            // ---------- RELU(RELU(x)) -> RELU(x) ----------
            "relu" => {
                if op.inputs.len() >= 1 {
                    if let Some(producer) =
                        Self::get_producer_op(graph, op.inputs[0])
                    {
                        if producer.op_type == "relu" {
                            return Some(producer.inputs[0]);
                        }
                    }
                }
            }

            _ => {}
        }

        None
    }

    // ============================================================
    // 池化化简
    // ============================================================

    fn simplify_pool(_graph: &DagGraph, op: &Op) -> Option<u64> {
        match op.op_type.as_str() {
            "maxpool2d" | "avgpool2d" => {
                let kernel_size = op
                    .attrs
                    .get("kernel_size")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let stride = op
                    .attrs
                    .get("stride")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let padding = op
                    .attrs
                    .get("padding")
                    .and_then(|v| match v {
                        AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0);

                if kernel_size == 1 && stride == 1 && padding == 0 {
                    if op.inputs.len() >= 1 {
                        return Some(op.inputs[0]);
                    }
                }
            }
            _ => {}
        }

        None
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    fn is_neg_of(graph: &DagGraph, a: u64, b: u64) -> bool {
        if let Some(producer) = Self::get_producer_op(graph, b) {
            if producer.op_type == "neg" {
                return producer.inputs[0] == a;
            }
        }
        false
    }

    fn get_producer_op<'a>(
        graph: &'a DagGraph,
        value_id: u64,
    ) -> Option<&'a Op> {
        if let Some(value) = graph.values.get(&value_id) {
            if let Some(producer_id) = value.producer {
                return graph.ops.get(&producer_id);
            }
        }
        None
    }

    fn get_reshape_shape(op: &Op) -> Option<Vec<i64>> {
        op.attrs.get("shape").and_then(|v| match v {
            AttrValue::Shape(shape) => Some(shape.clone()),
            AttrValue::IntList(list) => Some(list.clone()),
            _ => None,
        })
    }

    fn create_binary_op(op_type: &str, in1: u64, in2: u64) -> Op {
        Op {
            id: 0,
            name: format!("{}_{}", op_type, 0),
            op_type: op_type.to_string(),
            inputs: vec![in1, in2],
            outputs: vec![0],
            attrs: HashMap::new(),
        }
    }

    fn create_unary_op(op_type: &str, input: u64) -> Op {
        Op {
            id: 0,
            name: format!("{}_{}", op_type, 0),
            op_type: op_type.to_string(),
            inputs: vec![input],
            outputs: vec![0],
            attrs: HashMap::new(),
        }
    }

    fn create_reshape_op(input: u64, shape: &[i64]) -> Op {
        let mut attrs = HashMap::new();
        attrs.insert("shape".to_string(), AttrValue::Shape(shape.to_vec()));
        Op {
            id: 0,
            name: format!("reshape_{}", 0),
            op_type: "reshape".to_string(),
            inputs: vec![input],
            outputs: vec![0],
            attrs,
        }
    }

    // 获取负常数
    fn get_neg_constant(graph: &mut DagGraph, value_id: u64) -> Option<u64> {
        if let Some(data) = graph.constants.get(&value_id) {
            if let Some(val) = graph.values.get(&value_id) {
                match val.ty.dtype {
                    DataType::F32 => {
                        let floats: Vec<f32> = data
                            .chunks(4)
                            .map(|c| {
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            })
                            .collect();
                        let all_neg = floats.iter().all(|&x| x < 0.0);
                        let all_zero = floats.iter().all(|&x| x == 0.0);
                        if all_neg && !all_zero {
                            let neg_data: Vec<f32> =
                                floats.iter().map(|&x| -x).collect();
                            let neg_bytes: Vec<u8> = neg_data
                                .iter()
                                .flat_map(|&v| v.to_le_bytes())
                                .collect();
                            let ty = val.ty.clone();
                            let neg_id = graph.add_constant(
                                &format!("neg_const_{}", value_id),
                                ty,
                                neg_bytes,
                            );
                            return Some(neg_id);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    fn apply_replacement(graph: &mut DagGraph, old_id: u64, new_id: u64) {
        for (_, op) in graph.ops.iter_mut() {
            for input in &mut op.inputs {
                if *input == old_id {
                    *input = new_id;
                }
            }
            for output in &mut op.outputs {
                if *output == old_id {
                    *output = new_id;
                }
            }
        }

        for output in &mut graph.outputs {
            if *output == old_id {
                *output = new_id;
            }
        }

        if let Some(value) = graph.values.get(&old_id) {
            if let Some(producer_id) = value.producer {
                if let Some(op) = graph.ops.get_mut(&producer_id) {
                    for output in &mut op.outputs {
                        if *output == old_id {
                            *output = new_id;
                        }
                    }
                }
            }
        }

        if let Some(data) = graph.constants.get(&old_id) {
            if !graph.constants.contains_key(&new_id) {
                graph.constants.insert(new_id, data.clone());
            }
        }
    }
}
