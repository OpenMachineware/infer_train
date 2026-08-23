# infer_train Python package (M4)

This directory contains the `infer_train` Python package: ctypes bindings to
the Mojo engine, the `torch.compile` FX translation backend, and the model
wrapper.  See the repository `README.md` for full usage examples.

Install from the repository root:

```bash
pip install -e python/
```

The install runs `mojo build` automatically (through `pixi run mojo` when
pixi is available) and places the engine shared library inside the package
(`infer_train/_lib/`).
