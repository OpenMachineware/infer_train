// src/ir.rs

pub mod dag;
pub mod executor;
pub mod serialize;

pub use dag::DagGraph;
pub use dag::DataType;
pub use dag::TensorType;
pub use dag::AttrValue;
pub use executor::Executor;
pub use serialize::ModelFile;
pub use serialize::ModelHeader;
