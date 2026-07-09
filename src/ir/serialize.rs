use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

use crate::ir::cfg::CfgGraph;
use crate::ir::dag::DagGraph;

// ============================================================
// Constants
// ============================================================
pub const MAGIC: [u8; 8] = [0x49, 0x54, 0x52, 0x41, 0x49, 0x4E, 0x00, 0x00];
pub const VERSION: u32 = 2;

// ============================================================
// File header (fixed size for fast reading)
// ============================================================
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub flags: u32,
    pub header_offset: u64,
    pub graph_offset: u64,
    pub weights_offset: u64,
    pub training_state_offset: u64,
    pub file_size: u64,
    pub checksum: u32,
    pub _padding: [u8; 4],
}

// ============================================================
// Model type
// ============================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Inference,
    Trainable,
}

// ============================================================
// Model Header
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    pub version: u32,
    pub model_type: ModelType,
    pub framework: String,
    pub model_name: String,
    pub created_at: u64,
    pub num_ops: u64,
    pub num_constants: u64,
    pub metadata: HashMap<String, String>,
}

// ============================================================
// Training state
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: u64,
    pub step: u64,
    pub loss: f32,
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub optimizer_state: Vec<u8>,
}

// ============================================================
// Weights block (ID-indexed byte stream)
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsBlock {
    pub ids: Vec<u64>,       // List of constant IDs
    pub offsets: Vec<usize>, // Offset of each weight in data
    pub sizes: Vec<usize>,   // Size of each weight
    pub data: Vec<u8>,       // All weights stored consecutively
}

// ============================================================
// ModelFile (new design)
// ============================================================

pub struct ModelFile {
    header: ModelHeader,
    graph: DagGraph,
    weights: WeightsBlock,
    training_state: Option<TrainingState>,
    #[allow(dead_code)]
    mmap: Option<Mmap>, // Keep mapping alive if loaded via mmap
}

impl ModelFile {
    // ============================================================
    // Constructors
    // ============================================================

    pub fn new(name: &str, framework: &str, graph: DagGraph) -> Self {
        let weights = Self::extract_weights(&graph);
        ModelFile {
            header: ModelHeader {
                version: VERSION,
                model_type: ModelType::Inference,
                framework: framework.to_string(),
                model_name: name.to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                num_ops: graph.ops.len() as u64,
                num_constants: graph.constants.len() as u64,
                metadata: HashMap::new(),
            },
            graph,
            weights,
            training_state: None,
            mmap: None,
        }
    }

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
    // Weight extraction
    // ============================================================

    fn extract_weights(graph: &DagGraph) -> WeightsBlock {
        if graph.constants.is_empty() {
            return WeightsBlock {
                ids: Vec::new(),
                offsets: Vec::new(),
                sizes: Vec::new(),
                data: Vec::new(),
            };
        }

        let mut ids = Vec::new();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut data = Vec::new();

        for (&id, bytes) in &graph.constants {
            ids.push(id);
            offsets.push(data.len());
            sizes.push(bytes.len());
            data.extend_from_slice(bytes);
        }

        WeightsBlock { ids, offsets, sizes, data }
    }

    fn restore_weights(&mut self) {
        for (i, &id) in self.weights.ids.iter().enumerate() {
            let start = self.weights.offsets[i];
            let end = start + self.weights.sizes[i];
            let data = &self.weights.data[start..end];
            self.graph.constants.insert(id, data.to_vec());
        }
    }

    // ============================================================
    // Save
    // ============================================================

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let temp_path = path.with_extension("tmp");

        let mut file = File::create(&temp_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        // Serialize each block
        let header_bytes = bincode::serialize(&self.header)
            .map_err(|e| format!("Failed to serialize header: {}", e))?;
        let graph_bytes = bincode::serialize(&self.graph)
            .map_err(|e| format!("Failed to serialize graph: {}", e))?;
        let weights_bytes = bincode::serialize(&self.weights)
            .map_err(|e| format!("Failed to serialize weights: {}", e))?;
        let training_bytes = match &self.training_state {
            Some(s) => bincode::serialize(s).map_err(|e| {
                format!("Failed to serialize training state: {}", e)
            })?,
            None => Vec::new(),
        };

        // Calculate offsets
        let header_offset = std::mem::size_of::<FileHeader>() as u64;
        let graph_offset = header_offset + header_bytes.len() as u64;
        let weights_offset = graph_offset + graph_bytes.len() as u64;
        let training_offset = weights_offset + weights_bytes.len() as u64;
        let total_size = training_offset + training_bytes.len() as u64;

        // Write file header
        let file_header = FileHeader {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            header_offset,
            graph_offset,
            weights_offset,
            training_state_offset: training_offset,
            file_size: total_size,
            checksum: 0,
            _padding: [0; 4],
        };

        let header_bytes_raw = bytemuck::bytes_of(&file_header);
        file.write_all(header_bytes_raw)
            .map_err(|e| format!("Failed to write header: {}", e))?;

        // Write each block
        file.write_all(&header_bytes)
            .map_err(|e| format!("Failed to write header block: {}", e))?;
        file.write_all(&graph_bytes)
            .map_err(|e| format!("Failed to write graph block: {}", e))?;
        file.write_all(&weights_bytes)
            .map_err(|e| format!("Failed to write weights block: {}", e))?;
        if !training_bytes.is_empty() {
            file.write_all(&training_bytes).map_err(|e| {
                format!("Failed to write training state: {}", e)
            })?;
        }

        // Sync and rename
        file.sync_all().map_err(|e| format!("Failed to sync file: {}", e))?;
        std::fs::rename(temp_path, path)
            .map_err(|e| format!("Failed to rename file: {}", e))?;

        Ok(())
    }

    // ============================================================
    // Load
    // ============================================================

    pub fn load(path: &Path) -> Result<Self, String> {
        Self::load_with_mmap(path, false)
    }

    pub fn load_with_mmap(path: &Path, use_mmap: bool) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        let file_size = file
            .metadata()
            .map_err(|e| format!("Failed to get file size: {}", e))?
            .len();

        if file_size < std::mem::size_of::<FileHeader>() as u64 {
            return Err("File too small".to_string());
        }

        // Read file header
        let header_bytes_raw = if use_mmap {
            let mmap = unsafe { MmapOptions::new().map(&file) }
                .map_err(|e| format!("Failed to mmap file: {}", e))?;
            let slice = &mmap[0..std::mem::size_of::<FileHeader>()];
            let mut arr = [0u8; std::mem::size_of::<FileHeader>()];
            arr.copy_from_slice(slice);
            arr
        } else {
            let mut buffer = [0u8; std::mem::size_of::<FileHeader>()];
            let mut reader = std::io::BufReader::new(&file);
            reader
                .read_exact(&mut buffer)
                .map_err(|e| format!("Failed to read header: {}", e))?;
            buffer
        };

        let file_header: FileHeader = *bytemuck::from_bytes(&header_bytes_raw);

        // Validation
        if file_header.magic != MAGIC {
            return Err("Invalid magic number".to_string());
        }
        if file_header.version < 1 || file_header.version > VERSION {
            return Err(format!(
                "Unsupported version: {} (max: {})",
                file_header.version, VERSION
            ));
        }

        // Read each block
        let (mmap, reader) = if use_mmap {
            let mmap = unsafe { MmapOptions::new().map(&file) }
                .map_err(|e| format!("Failed to mmap file: {}", e))?;
            (Some(mmap), None)
        } else {
            (None, Some(std::io::BufReader::new(file)))
        };

        let header_bytes = Self::read_block_at(
            &mmap,
            reader.as_ref(),
            file_header.header_offset as usize,
            file_header.graph_offset as usize
                - file_header.header_offset as usize,
        )?;

        let graph_bytes = Self::read_block_at(
            &mmap,
            reader.as_ref(),
            file_header.graph_offset as usize,
            file_header.weights_offset as usize
                - file_header.graph_offset as usize,
        )?;

        let weights_bytes = Self::read_block_at(
            &mmap,
            reader.as_ref(),
            file_header.weights_offset as usize,
            file_header.training_state_offset as usize
                - file_header.weights_offset as usize,
        )?;

        let training_bytes =
            if file_header.training_state_offset < file_header.file_size {
                Some(Self::read_block_at(
                    &mmap,
                    reader.as_ref(),
                    file_header.training_state_offset as usize,
                    (file_header.file_size - file_header.training_state_offset)
                        as usize,
                )?)
            } else {
                None
            };

        // Deserialize
        let header: ModelHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| format!("Failed to deserialize header: {}", e))?;

        let graph: DagGraph = bincode::deserialize(&graph_bytes)
            .map_err(|e| format!("Failed to deserialize graph: {}", e))?;

        let weights: WeightsBlock = bincode::deserialize(&weights_bytes)
            .map_err(|e| format!("Failed to deserialize weights: {}", e))?;

        let training_state = if let Some(bytes) = training_bytes {
            if !bytes.is_empty() {
                Some(bincode::deserialize(&bytes).map_err(|e| {
                    format!("Failed to deserialize training state: {}", e)
                })?)
            } else {
                None
            }
        } else {
            None
        };

        let mut model =
            ModelFile { header, graph, weights, training_state, mmap };

        // Restore weights to graph
        model.restore_weights();

        Ok(model)
    }

    fn read_block_at(
        mmap: &Option<Mmap>,
        reader: Option<&std::io::BufReader<File>>,
        offset: usize,
        size: usize,
    ) -> Result<Vec<u8>, String> {
        if size == 0 {
            return Ok(Vec::new());
        }

        if let Some(mmap) = mmap {
            Ok(mmap[offset..offset + size].to_vec())
        } else if let Some(reader) = reader {
            let mut buffer = vec![0u8; size];
            let mut reader_ref = reader.get_ref();
            reader_ref
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| format!("Failed to seek: {}", e))?;
            reader_ref
                .read_exact(&mut buffer)
                .map_err(|e| format!("Failed to read block: {}", e))?;
            Ok(buffer)
        } else {
            Err("No reader available".to_string())
        }
    }

    // ============================================================
    // Incremental save
    // ============================================================

    pub fn save_training_state(&self, path: &Path) -> Result<(), String> {
        let state =
            self.training_state.as_ref().ok_or("Model is not trainable")?;

        let bytes = bincode::serialize(state).map_err(|e| {
            format!("Failed to serialize training state: {}", e)
        })?;

        let state_path = path.with_extension("train");
        fs::write(&state_path, bytes)
            .map_err(|e| format!("Failed to write training state: {}", e))?;
        Ok(())
    }

    pub fn load_training_state(&mut self, path: &Path) -> Result<(), String> {
        let state_path = path.with_extension("train");
        let bytes = fs::read(&state_path)
            .map_err(|e| format!("Failed to read training state: {}", e))?;

        let state: TrainingState =
            bincode::deserialize(&bytes).map_err(|e| {
                format!("Failed to deserialize training state: {}", e)
            })?;

        self.training_state = Some(state);
        Ok(())
    }

    // ============================================================
    // Accessors
    // ============================================================

    pub fn header(&self) -> &ModelHeader {
        &self.header
    }
    pub fn graph(&self) -> &DagGraph {
        &self.graph
    }
    pub fn graph_mut(&mut self) -> &mut DagGraph {
        &mut self.graph
    }
    pub fn training_state(&self) -> Option<&TrainingState> {
        self.training_state.as_ref()
    }
    pub fn training_state_mut(&mut self) -> Option<&mut TrainingState> {
        self.training_state.as_mut()
    }
    pub fn is_trainable(&self) -> bool {
        self.header.model_type == ModelType::Trainable
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.header.metadata.insert(key.to_string(), value.to_string());
    }

    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.header.metadata.get(key)
    }

    // ============================================================
    // CFG conversion
    // ============================================================

    pub fn from_cfg(
        cfg: &CfgGraph,
        name: &str,
        framework: &str,
        trainable: bool,
    ) -> Result<Self, String> {
        let dag =
            crate::transform::cfg_to_dag::CfgToDagConverter::convert(cfg)?;
        if trainable {
            Ok(Self::new_trainable(name, framework, dag, "adam", 0.001))
        } else {
            Ok(Self::new(name, framework, dag))
        }
    }

    pub fn export(&self, path: &str) -> Result<(), String> {
        self.save(Path::new(path))
    }

    pub fn import(path: &str) -> Result<Self, String> {
        Self::load(Path::new(path))
    }
}

impl Clone for ModelFile {
    fn clone(&self) -> Self {
        ModelFile {
            header: self.header.clone(),
            graph: self.graph.clone(),
            weights: self.weights.clone(),
            training_state: self.training_state.clone(),
            mmap: None, // mmap not cloned, reload if needed
        }
    }
}

// ============================================================
// Python bindings
// ============================================================

#[pyclass]
pub struct PyModelFile {
    pub(crate) inner: std::sync::Arc<std::sync::Mutex<ModelFile>>,
}

#[pymethods]
impl PyModelFile {
    #[staticmethod]
    pub fn new(name: &str, framework: &str) -> Self {
        let graph = DagGraph::new(name);
        PyModelFile {
            inner: std::sync::Arc::new(std::sync::Mutex::new(ModelFile::new(
                name, framework, graph,
            ))),
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
            inner: std::sync::Arc::new(std::sync::Mutex::new(
                ModelFile::new_trainable(name, framework, graph, optimizer, lr),
            )),
        }
    }

    pub fn export(&self, path: &str) -> PyResult<()> {
        let guard = self.inner.lock().unwrap();
        guard
            .save(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    #[staticmethod]
    pub fn import(path: &str) -> PyResult<Self> {
        let inner = ModelFile::load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        })?;
        Ok(PyModelFile {
            inner: std::sync::Arc::new(std::sync::Mutex::new(inner)),
        })
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        Self::import(path)
    }

    pub fn save_training_state(&self, path: &str) -> PyResult<()> {
        let guard = self.inner.lock().unwrap();
        guard
            .save_training_state(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn load_training_state(&self, path: &str) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard
            .load_training_state(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn set_metadata(&self, key: &str, value: &str) {
        let mut guard = self.inner.lock().unwrap();
        guard.add_metadata(key, value);
    }

    pub fn get_metadata(&self, key: &str) -> Option<String> {
        let guard = self.inner.lock().unwrap();
        guard.get_metadata(key).cloned()
    }

    pub fn is_trainable(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        guard.is_trainable()
    }

    pub fn get_graph(&self) -> PyResult<Py<PyAny>> {
        let guard = self.inner.lock().unwrap();
        Python::with_gil(|py| {
            let py_dag = PyDagGraph { inner: guard.graph().clone() };
            Ok(Py::new(py, py_dag)?.into_any())
        })
    }

    pub fn __repr__(&self) -> String {
        let guard = self.inner.lock().unwrap();
        format!(
            "PyModelFile(name={}, framework={}, type={:?}, ops={},\
             constants={})",
            guard.header().model_name,
            guard.header().framework,
            guard.header().model_type,
            guard.graph().ops.len(),
            guard.graph().constants.len()
        )
    }
}

// ============================================================
// PyDagGraph
// ============================================================

#[pyclass]
#[derive(Clone)]
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
