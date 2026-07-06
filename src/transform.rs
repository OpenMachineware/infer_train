// src/transform/mod.rs

pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod fusion;
pub mod simplify;
pub mod shape_inference;
pub mod verify;

pub use constant_fold::ConstantFoldingPass;
pub use cse::CommonSubexpressionEliminationPass;
pub use dce::DCEPass;
pub use fusion::FusionPass;
pub use simplify::AlgebraicSimplifyPass;
pub use shape_inference::ShapeInferencePass;
pub use verify::VerifyPass;

pub struct Optimizer;

impl Optimizer {
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
