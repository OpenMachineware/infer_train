pub mod cfg_cse;
pub mod cfg_dce;
pub mod cfg_inline;
pub mod cfg_to_dag;
pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod fusion;
pub mod shape_inference;
pub mod simplify;
pub mod verify;

pub use cfg_cse::CfgCSEPass;
pub use cfg_dce::CfgDCEPass;
pub use cfg_inline::CfgInlinePass;
pub use cfg_to_dag::CfgToDagConverter;
pub use constant_fold::ConstantFoldingPass;
pub use cse::CommonSubexpressionEliminationPass;
pub use dce::DCEPass;
pub use fusion::FusionPass;
pub use shape_inference::ShapeInferencePass;
pub use simplify::AlgebraicSimplifyPass;
pub use verify::VerifyPass;

use crate::ir::cfg::CfgGraph;
use crate::ir::dag::DagGraph;

// ============================================================
// CFG 优化 Pipeline
// ============================================================
pub struct CfgOptimizer;

impl CfgOptimizer {
    pub fn optimize(cfg: &mut CfgGraph) -> Result<(), String> {
        let mut iteration = 0;
        const MAX_ITER: usize = 5;

        loop {
            let mut changed = false;

            changed |= CfgInlinePass::apply(cfg);
            changed |= CfgDCEPass::apply(cfg);
            changed |= CfgCSEPass::apply(cfg);

            iteration += 1;
            if !changed || iteration >= MAX_ITER {
                break;
            }
        }

        CfgDCEPass::apply(cfg);

        Ok(())
    }
}

// ============================================================
// DAG 优化 Pipeline
// ============================================================
pub struct DagOptimizer;

impl DagOptimizer {
    pub fn optimize(graph: &mut DagGraph) -> Result<(), String> {
        VerifyPass::verify(graph)?;

        let mut iteration = 0;
        const MAX_ITER: usize = 5;

        loop {
            let mut changed = false;

            changed |= ConstantFoldingPass::apply(graph);
            changed |= AlgebraicSimplifyPass::apply(graph);
            changed |= CommonSubexpressionEliminationPass::apply(graph);
            changed |= FusionPass::apply(graph);
            changed |= DCEPass::apply(graph);

            VerifyPass::verify(graph)?;

            iteration += 1;
            if !changed || iteration >= MAX_ITER {
                break;
            }
        }

        DCEPass::apply(graph);
        ShapeInferencePass::apply(graph)?;
        VerifyPass::verify(graph)?;

        Ok(())
    }
}

// ============================================================
// 完整优化 Pipeline（CFG + DAG）
// ============================================================
pub struct FullOptimizer;

impl FullOptimizer {
    pub fn optimize_cfg(cfg: &mut CfgGraph) -> Result<(), String> {
        CfgOptimizer::optimize(cfg)
    }

    pub fn optimize_dag(dag: &mut DagGraph) -> Result<(), String> {
        DagOptimizer::optimize(dag)
    }

    pub fn optimize_full(cfg: &mut CfgGraph) -> Result<DagGraph, String> {
        CfgOptimizer::optimize(cfg)?;
        let mut dag = CfgToDagConverter::convert(cfg)?;
        DagOptimizer::optimize(&mut dag)?;
        Ok(dag)
    }
}
