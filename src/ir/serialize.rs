// src/ir/serialize.rs

use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

use crate::ir::dag::DagGraph;
use crate::ir::cfg::CfgGraph;

// ============================================================
// 常量定义
// ============================================================
const MAGIC: [u8; 8] = [0x49, 0x54, 0x52, 0x41, 0x49, 0x4E, 0x00, 0x00];  // "ITRAIN\0\0"
const VERSION: u32 = 1;
const MIN_VERSION: u32 = 1;

// ============================================================
// 模型类型
// ============================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Inference,   // 纯推理模型
    Trainable,   // 可训练模型（带梯度信息）
}

// ============================================================
// Model Header
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub model_type: ModelType,
    pub framework: String,      // "torch", "tensorflow", "jax"
    pub model_name: String,
    pub created_at: u64,        // timestamp
    pub num_ops: u64,
    pub num_constants: u64,
}

// ============================================================
// 训练状态（用于边推边训）
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: u64,
    pub step: u64,
    pub loss: f32,
    pub optimizer_type: String,  // "sgd", "adam", "adamw"
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    // 优化器状态（每个参数的momentum, variance等）
    pub optimizer_state: Vec<u8>, // 序列化的优化器状态
}

// ============================================================
// 完整的模型文件
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub header: ModelHeader,
    pub graph: DagGraph,
    pub training_state: Option<TrainingState>,
    // 元数据
    pub metadata: std::collections::HashMap<String, String>,
}

impl ModelFile {
    pub fn new(name: &str, framework: &str, graph: DagGraph) -> Self {
        let num_ops = graph.ops.len() as u64;
        let num_constants = graph.constants.len() as u64;

        ModelFile {
            header: ModelHeader {
                magic: MAGIC,
                version: VERSION,
                model_type: ModelType::Inference,
                framework: framework.to_string(),
                model_name: name.to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                num_ops,
                num_constants,
            },
            graph,
            training_state: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// 创建可训练模型
    pub fn new_trainable(
        name: &str,
        framework: &str,
        graph: DagGraph,
        optimizer_type: &str,
        learning_rate: f32,
    ) -> Self {
        let mut model = Self::new(name, framework, graph);
        model.header.model_type = ModelType::Trainable;
        model.training_state = Some(TrainingState {
            epoch: 0,
            step: 0,
            loss: 0.0,
            optimizer_type: optimizer_type.to_string(),
            learning_rate,
            momentum: 0.9,
            weight_decay: 0.0,
            optimizer_state: Vec::new(),
        });
        model
    }

    // ============================================================
    // 保存到文件
    // ============================================================
    pub fn save(&self, path: &Path) -> Result<(), String> {
        // 更新 header 统计信息
        let mut header = self.header.clone();
        header.num_ops = self.graph.ops.len() as u64;
        header.num_constants = self.graph.constants.len() as u64;

        let model_data = ModelFile {
            header,
            graph: self.graph.clone(),
            training_state: self.training_state.clone(),
            metadata: self.metadata.clone(),
        };

        let bytes = bincode::serialize(&model_data)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;

        // 使用临时文件，避免写入失败时损坏原文件
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        fs::rename(temp_path, path)
            .map_err(|e| format!("Failed to rename file: {}", e))?;

        Ok(())
    }

    // ============================================================
    // 从文件加载
    // ============================================================
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let model: ModelFile = bincode::deserialize(&bytes)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;

        // 验证魔数
        if model.header.magic != MAGIC {
            return Err("Invalid magic number: not an ITRAIN model".to_string());
        }

        // 版本检查
        if model.header.version < MIN_VERSION || model.header.version > VERSION {
            return Err(format!(
                "Unsupported version: {} (supported: {}-{})",
                model.header.version, MIN_VERSION, VERSION
            ));
        }

        Ok(model)
    }

    // ============================================================
    // 便捷方法
    // ============================================================
    pub fn export(&self, path: &str) -> Result<(), String> {
        self.save(Path::new(path))
    }

    pub fn import(path: &str) -> Result<Self, String> {
        Self::load(Path::new(path))
    }

    /// 保存训练状态（增量保存）
    pub fn save_training_state(&mut self, path: &Path) -> Result<(), String> {
        if let Some(state) = &self.training_state {
            let state_bytes = bincode::serialize(state)
                .map_err(|e| format!("Failed to serialize training state: {}", e))?;

            let state_path = path.with_extension("train");
            fs::write(&state_path, state_bytes)
                .map_err(|e| format!("Failed to write training state: {}", e))?;

            Ok(())
        } else {
            Err("Model is not trainable".to_string())
        }
    }

    /// 加载训练状态
    pub fn load_training_state(&mut self, path: &Path) -> Result<(), String> {
        let state_path = path.with_extension("train");
        let bytes = fs::read(&state_path)
            .map_err(|e| format!("Failed to read training state: {}", e))?;

        let state: TrainingState = bincode::deserialize(&bytes)
            .map_err(|e| format!("Failed to deserialize training state: {}", e))?;

        self.training_state = Some(state);
        Ok(())
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

// ============================================================
// Python 绑定
// ============================================================

#[pyclass]
pub struct PyModelFile {
    pub(crate) inner: ModelFile,
}

#[pymethods]
impl PyModelFile {
    #[staticmethod]
    pub fn new(name: &str, framework: &str) -> Self {
        let graph = DagGraph::new(name);
        PyModelFile {
            inner: ModelFile::new(name, framework, graph),
        }
    }

    #[staticmethod]
    pub fn new_trainable(
        name: &str,
        framework: &str,
        optimizer: &str,
        lr: f32,
    ) -> Self {
        let graph = DagGraph::new(name);
        PyModelFile {
            inner: ModelFile::new_trainable(name, framework, graph, optimizer, lr),
        }
    }

    pub fn export(&self, path: &str) -> PyResult<()> {
        self.inner.export(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    #[staticmethod]
    pub fn import(path: &str) -> PyResult<Self> {
        let inner = ModelFile::import(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(PyModelFile { inner })
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = ModelFile::import(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(PyModelFile { inner })
    }

    pub fn save_training_state(&mut self, path: &str) -> PyResult<()> {
        self.inner.save_training_state(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn load_training_state(&mut self, path: &str) -> PyResult<()> {
        self.inner.load_training_state(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.inner.add_metadata(key, value);
    }

    pub fn get_metadata(&self, key: &str) -> Option<String> {
        self.inner.get_metadata(key).cloned()
    }

    pub fn is_trainable(&self) -> bool {
        self.inner.header.model_type == ModelType::Trainable
    }

    pub fn get_graph(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            // 返回 DAG 的 Python 表示
            let py_dag = PyDagGraph { inner: self.inner.graph.clone() };
            Ok(Py::new(py, py_dag)?)
        })
    }

    pub fn set_graph(&mut self, dag: PyDagGraph) {
        self.inner.graph = dag.inner;
        // 更新 header 统计
        self.inner.header.num_ops = self.inner.graph.ops.len() as u64;
        self.inner.header.num_constants = self.inner.graph.constants.len() as u64;
    }

    pub fn graph_string(&self) -> String {
        format!("{:#?}", self.inner.graph)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyModelFile(name={}, framework={}, type={:?}, ops={}, constants={})",
            self.inner.header.model_name,
            self.inner.header.framework,
            self.inner.header.model_type,
            self.inner.graph.ops.len(),
            self.inner.graph.constants.len()
        )
    }
}

// ============================================================
// DAG 的 Python 包装（用于返回）
// ============================================================

#[pyclass]
pub struct PyDagGraph {
    pub inner: DagGraph,
}

#[pymethods]
impl PyDagGraph {
    pub fn num_ops(&self) -> usize {
        self.inner.ops.len()
    }

    pub fn num_values(&self) -> usize {
        self.inner.values.len()
    }

    pub fn num_constants(&self) -> usize {
        self.inner.constants.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DagGraph(name={}, ops={}, values={}, constants={})",
            self.inner.name,
            self.inner.ops.len(),
            self.inner.values.len(),
            self.inner.constants.len()
        )
    }
}

// ============================================================
// CFG 到 ModelFile 的转换
// ============================================================

impl ModelFile {
    /// 从 CFG 创建 ModelFile
    pub fn from_cfg(
        cfg: &CfgGraph,
        name: &str,
        framework: &str,
        trainable: bool,
    ) -> Result<Self, String> {
        // 转换 CFG → DAG
        let dag = crate::transform::cfg_to_dag::CfgToDagConverter::convert(cfg)?;

        if trainable {
            Ok(Self::new_trainable(name, framework, dag, "adam", 0.001))
        } else {
            Ok(Self::new(name, framework, dag))
        }
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::ir::dag::{DataType, TensorType};

    #[test]
    fn test_model_serialization() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.itm");

        // 创建模型
        let mut graph = DagGraph::new("test_model");
        let v1 = graph.add_value("x", TensorType {
            dtype: DataType::F32,
            shape: vec![1, 3, 224, 224],
        });
        let v2 = graph.add_value("y", TensorType {
            dtype: DataType::F32,
            shape: vec![1, 1000],
        });
        graph.set_inputs(vec![v1]);
        graph.set_outputs(vec![v2]);

        let model = ModelFile::new("test_model", "torch", graph);

        // 保存
        model.save(&path).unwrap();

        // 加载
        let loaded = ModelFile::load(&path).unwrap();

        assert_eq!(loaded.header.model_name, "test_model");
        assert_eq!(loaded.header.framework, "torch");
        assert_eq!(loaded.header.version, VERSION);
    }

    #[test]
    fn test_trainable_model() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trainable.itm");
        let train_path = dir.path().join("trainable.train");

        let graph = DagGraph::new("trainable_model");
        let mut model = ModelFile::new_trainable(
            "trainable_model",
            "torch",
            graph,
            "adam",
            0.001,
        );

        // 保存
        model.save(&path).unwrap();

        // 保存训练状态
        model.save_training_state(&train_path).unwrap();

        // 更新训练状态
        if let Some(state) = &mut model.training_state {
            state.step = 100;
            state.epoch = 5;
            state.loss = 0.234;
        }

        // 重新加载
        let mut loaded = ModelFile::load(&path).unwrap();
        loaded.load_training_state(&train_path).unwrap();

        assert!(loaded.is_trainable());
        assert_eq!(loaded.training_state.as_ref().unwrap().step, 100);
    }
}
