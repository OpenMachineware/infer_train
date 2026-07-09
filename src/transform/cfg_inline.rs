use crate::ir::cfg::{CfgGraph, CfgOp};
use std::collections::HashMap;
use std::sync::OnceLock;

static FUNCTION_REGISTRY: OnceLock<HashMap<String, CfgGraph>> = OnceLock::new();

pub struct CfgInlinePass;

impl CfgInlinePass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let call_ops: Vec<u64> = cfg
            .blocks
            .values()
            .flat_map(|block| block.ops.iter())
            .filter(|op| Self::is_call_op(&op.op_type))
            .map(|op| op.id)
            .collect();

        for op_id in call_ops {
            if Self::inline_call(cfg, op_id) {
                changed = true;
            }
        }

        changed
    }

    fn is_call_op(op_type: &str) -> bool {
        matches!(op_type, "call" | "call_function" | "call_method")
    }

    fn inline_call(cfg: &mut CfgGraph, call_op_id: u64) -> bool {
        let (block_id, call_op) =
            match cfg.blocks.iter_mut().find_map(|(id, block)| {
                block
                    .ops
                    .iter()
                    .find(|op| op.id == call_op_id)
                    .map(|op| (*id, op.clone()))
            }) {
                Some(found) => found,
                None => return false,
            };

        let func_name = match call_op.attrs.get("function_name") {
            Some(crate::ir::dag::AttrValue::String(s)) => s.clone(),
            _ => return false,
        };

        let func_cfg = match Self::find_function(&func_name) {
            Some(cfg) => cfg,
            None => return false,
        };

        let block = match cfg.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => return false,
        };

        let pos = match block.ops.iter().position(|op| op.id == call_op_id) {
            Some(p) => p,
            None => return false,
        };

        // Inline function body
        let mut inline_ops = func_cfg
            .blocks
            .values()
            .flat_map(|b| b.ops.clone())
            .collect::<Vec<CfgOp>>();

        // Remap Value IDs
        let mut value_map = HashMap::new();
        for op in &mut inline_ops {
            op.id = cfg.next_id;
            cfg.next_id += 1;

            for inp in &mut op.inputs {
                if let Some(&new_id) = value_map.get(inp) {
                    *inp = new_id;
                }
            }

            for out in &mut op.outputs {
                let new_id = cfg.next_id;
                cfg.next_id += 1;
                value_map.insert(*out, new_id);
                *out = new_id;
            }
        }

        block.ops.splice(pos..pos + 1, inline_ops);
        true
    }

    fn find_function(name: &str) -> Option<CfgGraph> {
        FUNCTION_REGISTRY.get().and_then(|registry| registry.get(name).cloned())
    }

    pub fn register_function(name: &str, cfg: CfgGraph) -> Result<(), String> {
        let _registry = FUNCTION_REGISTRY.get_or_init(|| HashMap::new());
        // Note: mutable access is needed here,
        // but OnceLock's get_or_init returns &T
        // Need to use Mutex or get first then insert
        // Simplified: use Mutex for protection
        Self::register_function_mutex(name, cfg)
    }

    // Mutex version (safer)
    pub fn register_function_mutex(
        name: &str,
        cfg: CfgGraph,
    ) -> Result<(), String> {
        // If already exists, return error or overwrite
        // Simple approach: use Mutex
        use std::sync::Mutex;
        static REGISTRY_MUTEX: OnceLock<Mutex<HashMap<String, CfgGraph>>> =
            OnceLock::new();
        let mutex = REGISTRY_MUTEX.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = mutex.lock().unwrap();
        registry.insert(name.to_string(), cfg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::CfgGraph;
    use crate::ir::dag::AttrValue;

    #[test]
    fn test_inline_detection() {
        let mut cfg = CfgGraph::new("test");
        let block = cfg.add_block("entry");
        cfg.set_entry(block);

        let mut attrs = HashMap::new();
        attrs.insert(
            "function_name".to_string(),
            AttrValue::String("matmul".to_string()),
        );

        let op = CfgOp {
            id: 0,
            op_type: "call".to_string(),
            inputs: vec![1, 2],
            outputs: vec![3],
            attrs,
            name: "call_matmul".to_string(),
        };
        cfg.add_op(block, op).unwrap();

        assert!(CfgInlinePass::is_call_op("call"));
        assert!(CfgInlinePass::is_call_op("call_function"));
        assert!(!CfgInlinePass::is_call_op("add"));
    }
}
