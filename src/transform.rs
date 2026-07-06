// src/transform/mod.rs

pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod fusion;
pub mod simplify;
pub mod shape_inference;
pub mod cfg_to_dag;
pub mod verify;
pub mod cfg_inline;
pub mod cfg_dce;
pub mod cfg_cse;

pub use constant_fold::ConstantFoldingPass;
pub use cse::CommonSubexpressionEliminationPass;
pub use dce::DCEPass;
pub use fusion::FusionPass;
pub use simplify::AlgebraicSimplifyPass;
pub use shape_inference::ShapeInferencePass;
pub use cfg_to_dag::CfgToDagConverter;
pub use verify::VerifyPass;
pub use cfg_inline::CfgInlinePass;
pub use cfg_dce::CfgDCEPass;
pub use cfg_cse::CfgCSEPass;

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

            // 1. 函数内联
            changed |= CfgInlinePass::apply(cfg);

            // 2. 死基本块删除
            changed |= CfgDCEPass::apply(cfg);

            // 3. 跨基本块公共子表达式消除
            changed |= CfgCSEPass::apply(cfg);

            iteration += 1;
            if !changed || iteration >= MAX_ITER {
                break;
            }
        }

        // 最后再做一次 DCE 清理
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
        // 1. 验证初始图
        VerifyPass::verify(graph)?;

        let mut iteration = 0;
        const MAX_ITER: usize = 5;

        loop {
            let mut changed = false;

            // 2. 常量折叠
            changed |= ConstantFoldingPass::apply(graph);

            // 3. 代数化简
            changed |= AlgebraicSimplifyPass::apply(graph);

            // 4. 公共子表达式消除
            changed |= CommonSubexpressionEliminationPass::apply(graph);

            // 5. 算子融合
            changed |= FusionPass::apply(graph);

            // 6. 死代码消除
            changed |= DCEPass::apply(graph);

            // 7. 验证中间状态
            VerifyPass::verify(graph)?;

            iteration += 1;
            if !changed || iteration >= MAX_ITER {
                break;
            }
        }

        // 8. 最后再做一次 DCE 清理
        DCEPass::apply(graph);

        // 9. Shape 推导（必须在所有优化之后）
        ShapeInferencePass::apply(graph)?;

        // 10. 最终验证
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
        // 1. 优化 CFG
        CfgOptimizer::optimize(cfg)?;

        // 2. CFG → DAG
        let mut dag = CfgToDagConverter::convert(cfg)?;

        // 3. 优化 DAG
        DagOptimizer::optimize(&mut dag)?;

        Ok(dag)
    }
}
