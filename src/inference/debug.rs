// Debug state for layer-wise inspection

/// Debug state for a single layer
#[derive(Debug, Clone)]
pub struct LayerDebugState {
    pub layer_idx: usize,
    pub seq_pos: usize,
    /// Hidden state after embedding/before this layer
    pub hidden_before: Vec<f32>,
    /// Hidden state after this layer
    pub hidden_after: Vec<f32>,
    /// Attention norm output
    pub normed_attn: Vec<f32>,
    /// Q after projection (before norm)
    pub q: Vec<f32>,
    /// K after projection (before norm)
    pub k: Vec<f32>,
    /// V after projection
    pub v: Vec<f32>,
    /// Q after norm
    pub q_normed: Vec<f32>,
    /// K after norm
    pub k_normed: Vec<f32>,
    /// Attention output (before output projection)
    pub attn_output: Vec<f32>,
    /// Attention output after output projection
    pub attn_out: Vec<f32>,
    /// FFN norm output
    pub normed_ffn: Vec<f32>,
    /// Gate projection output (before SiLU)
    pub gate: Vec<f32>,
    /// Up projection output
    pub up: Vec<f32>,
    /// Gate * Up (SwiGLU intermediate)
    pub gate_times_up: Vec<f32>,
    /// FFN output
    pub ffn_out: Vec<f32>,
}
