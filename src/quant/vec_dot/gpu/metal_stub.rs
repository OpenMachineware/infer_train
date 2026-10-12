// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

pub struct MetalContext;

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        Err("Metal not available on this platform".to_string())
    }
}
