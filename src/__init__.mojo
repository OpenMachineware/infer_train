# infer_train package root.
#
# infer_train is a from-scratch training+inference LLM runtime.  This
# package is organized into two top-level subpackages:
#
#   core/    - tensors, devices, memory, quantization, operators, graph.
#   runtime/ - the interpreter that executes a graph and schedules devices.
#
# M1 milestone targets Apple Silicon; CPU code is target-triple agnostic and
# GPU code detects the accelerator at runtime via `std.sys.info`.
