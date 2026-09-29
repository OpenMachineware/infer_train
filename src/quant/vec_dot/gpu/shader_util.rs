// Shader compilation utilities
// Handles include expansion for Metal templates

use std::path::PathBuf;

/// Recursively expands #include directives in Metal shader source
pub fn expand_includes(
    source: &str,
    shader_dir: &PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();

    for line in source.lines() {
        if line.starts_with("#include \"") {
            // Extract filename
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];

            // Skip if already included
            if visited.contains(&filename.to_string()) {
                result
                    .push_str(&format!("// Already included: {}\n", filename));
                continue;
            }

            visited.push(filename.to_string());

            // Read and expand the included file
            let include_path = shader_dir.join(filename);
            if let Ok(include_content) = std::fs::read_to_string(&include_path)
            {
                result.push_str(&format!("// --- {} ---\n", filename));
                // Remove #pragma once from included files
                let include_content =
                    include_content.replace("#pragma once", "// #pragma once");
                result.push_str(&expand_includes(
                    &include_content,
                    shader_dir,
                    visited,
                ));
                result.push_str(&format!("// --- end {} ---\n", filename));
            } else {
                result.push_str(&format!(
                    "// WARNING: Could not find: {}\n",
                    filename
                ));
                result.push_str(line);
                result.push('\n');
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Compile a Metal shader with includes expanded
pub fn compile_shader_with_includes(
    device: &metal::Device,
    source: &str,
    shader_dir: &PathBuf,
    compile_options: &metal::CompileOptions,
) -> Result<metal::Library, String> {
    let mut visited = Vec::new();
    let expanded = expand_includes(source, shader_dir, &mut visited);

    device
        .new_library_with_source(&expanded, compile_options)
        .map_err(|e| format!("Failed to compile shader: {}", e))
}
