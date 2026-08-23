"""Build/install script for the infer_train Python package (M4).

``pip install -e python/`` (or ``python setup.py build_mojo``) compiles the
Mojo C-API bindings into a shared library inside the package:

    python/infer_train/_lib/libinfer_train.{dylib,so,dll}

The Mojo compiler is invoked through ``pixi run mojo`` (the project's
environment) unless a ``mojo`` binary is on PATH.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

PKG_DIR = pathlib.Path(__file__).resolve().parent  # python/
REPO = PKG_DIR.parent  # repository root
SRC = REPO / "src" / "bindings" / "infer_train_bindings.mojo"
LIB_NAME = {
    "darwin": "libinfer_train.dylib",
    "linux": "libinfer_train.so",
    "win32": "libinfer_train.dll",
}.get(sys.platform, "libinfer_train.so")
OUT_DIR = PKG_DIR / "infer_train" / "_lib"


def _mojo_cmd() -> list[str]:
    mojo = shutil.which("mojo")
    if mojo:
        return [mojo]
    pixi = shutil.which("pixi")
    if pixi and (REPO / "pixi.toml").exists():
        return [pixi, "run", "mojo"]
    raise RuntimeError(
        "mojo (or pixi) not found on PATH; install Mojo 1.0+ or point "
        "MOJO at a mojo binary to build the engine library."
    )


TP_NAME = {
    "darwin": "libinfer_train_tp.dylib",
    "linux": "libinfer_train_tp.so",
    "win32": "libinfer_train_tp.dll",
}.get(sys.platform, "libinfer_train_tp.so")
TP_SRC = REPO / "tools" / "thread_pool.c"


def build_tp_library() -> pathlib.Path:
    """Compile the C thread pool (linked into the engine via -Xlinker)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / TP_NAME
    cc = shutil.which("cc")
    if not cc:
        raise RuntimeError("a C compiler (cc) is required to build the "
                           "engine's thread pool")
    cmd = [cc, "-O2", "-shared", "-o", str(out), str(TP_SRC)]
    print("[infer_train] building thread pool:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))
    return out


def build_mojo_library() -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tp = build_tp_library()
    out = OUT_DIR / LIB_NAME
    cmd = _mojo_cmd() + [
        "build",
        "-I",
        str(REPO),
        str(SRC),
        "-Xlinker",
        str(tp),
        "--emit",
        "shared-lib",
        "-o",
        str(out),
    ]
    print("[infer_train] building engine library:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))
    if not out.exists():
        raise RuntimeError(f"mojo build succeeded but {out} is missing")
    return out


class BuildPyWithMojo(build_py):
    """Standard build_py plus the Mojo shared-library build."""

    def run(self):
        build_mojo_library()
        super().run()


if __name__ == "__main__":
    # Allow `python setup.py build_mojo` as a standalone target.
    if len(sys.argv) >= 2 and sys.argv[1] == "build_mojo":
        out = build_mojo_library()
        print(f"[infer_train] engine library ready: {out}")
        sys.exit(0)

setup(
    name="infer_train",
    version="0.1.0",
    description=(
        "infer_train - high-performance LLM inference engine (Mojo) with "
        "a torch.compile backend (M4)"
    ),
    long_description=(PKG_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Jia Liu",
    python_requires=">=3.10",
    packages=["infer_train"],
    package_dir={"": "."},
    package_data={"infer_train": ["_lib/*"]},
    install_requires=["torch>=2.0"],
    cmdclass={"build_py": BuildPyWithMojo},
    zip_safe=False,
)
