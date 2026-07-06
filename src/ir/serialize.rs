use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;
use crate::ir::dag::DagGraph;


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

    #[staticmethod]
    pub fn save_file(model: &PyModelFile, path: &str) -> PyResult<()> {
        model.inner.export(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn graph_string(&self) -> String {
        format!("{:#?}", self.inner.graph)
    }
}

// ============================================================
// 模型文件 Header
// ============================================================
const MAGIC: [u8; 8] = [0x49, 0x54, 0x52, 0x41, 0x49, 0x4E, 0x00, 0x00];  // "ITRAIN\0\0"
const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub framework: String,      // "torch", "tensorflow", "jax"
    pub model_name: String,
}

// ============================================================
// 完整的模型文件
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub header: ModelHeader,
    pub graph: DagGraph,
    // 权重已经包含在 graph.constants 中
}

impl ModelFile {
    pub fn new(name: &str, framework: &str, graph: DagGraph) -> Self {
        ModelFile {
            header: ModelHeader {
                magic: MAGIC,
                version: VERSION,
                framework: framework.to_string(),
                model_name: name.to_string(),
            },
            graph,
        }
    }

    // ============================================================
    // 保存到文件
    // ============================================================
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = bincode::serialize(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;

        fs::write(path, bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;

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
            return Err("Invalid magic number".to_string());
        }

        if model.header.version != VERSION {
            return Err(format!(
                "Unsupported version: {} (expected {})",
                model.header.version, VERSION
            ));
        }

        Ok(model)
    }

    // ============================================================
    // 导出到文件（便捷方法）
    // ============================================================
    pub fn export(&self, path: &str) -> Result<(), String> {
        self.save(Path::new(path))
    }

    pub fn import(path: &str) -> Result<Self, String> {
        Self::load(Path::new(path))
    }
}
