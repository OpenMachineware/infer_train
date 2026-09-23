pub struct MetalContext;

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        Err("Metal not available on this platform".to_string())
    }
}
