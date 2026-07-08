// src/ops/embedding_lookup/mod.rs

pub mod embedding;
pub mod gather_scatter;
pub mod one_hot;
pub mod topk;

pub use embedding::{embedding, embedding_backward, EmbeddingOp};
pub use gather_scatter::{
    gather, gather_backward, scatter, scatter_backward, GatherOp, ScatterOp,
};
pub use one_hot::{one_hot, OneHotOp};
pub use topk::{topk, topk_backward, TopkOp};
