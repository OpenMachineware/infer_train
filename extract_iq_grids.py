#!/usr/bin/env python3
import re

# Read the ggml-common.h file
with open('llama.cpp-0.5.0/ggml/src/ggml-common.h', 'r') as f:
    content = f.read()

# Find tables
def extract_table(content, table_name, type_name, size):
    pattern = rf'GGML_TABLE_BEGIN\({type_name},\s*{table_name},\s*{size}\)(.*?)GGML_TABLE_END\(\)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    values = match.group(1).strip()
    # Format for Metal
    return f"static const constant {type_name} {table_name}[{size}] = {{\n{values}\n}};"

# Extract the tables
tables = [
    ("iq2xxs_grid", "uint64_t", 256),
    ("iq2xs_grid", "uint64_t", 512),
    ("iq3xxs_grid", "uint32_t", 256),
    ("iq3s_grid", "uint32_t", 512),
]

for name, typ, size in tables:
    result = extract_table(content, name, typ, size)
    if result:
        print(f"// {name}")
        print(result)
        print()
