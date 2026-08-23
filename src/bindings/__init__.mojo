# bindings/__init__.mojo
#
# M4: C ABI exports for the Python/PyTorch side.
#
# `infer_train_bindings.mojo` is compiled standalone into a shared library
# (see the module docstring there for the exact command); this file only
# makes `src.bindings` a proper package so the relative `..core` / `..runtime`
# imports resolve.
