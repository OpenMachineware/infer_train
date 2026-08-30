#!/usr/bin/env python3
"""Generate ``src/version.mojo`` from the ``version`` field of pixi.toml.

The generated module exposes a single comptime constant::

    comptime VERSION: String = "0.2.1"

which the CLI (``src/core/cli/infer_train_cli.mojo``) uses for
``--version`` / ``-V``.  It is regenerated automatically before every
build that compiles the CLI:

    pixi run version     # pixi task
    pixi run build       # version + CLI binary
    make cli             # make target (depends on `make version`)
    make package         # make target (depends on `make version`)

Run it manually any time the version in pixi.toml changes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = ROOT / "pixi.toml"
OUT = ROOT / "src" / "version.mojo"

TEMPLATE = """\
# src/version.mojo
#
# GENERATED FILE - do not edit by hand.  Produced by tools/gen_version.py
# from the `version` field of pixi.toml (pixi task: `pixi run version`,
# make target: `make version`).  Regenerated before every CLI build.

comptime VERSION: String = "{version}"
"""


def main() -> None:
    text = PIXI_TOML.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise SystemExit(f"error: no `version = \"...\"` found in {PIXI_TOML}")
    version = m.group(1)
    OUT.write_text(TEMPLATE.format(version=version), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (version {version})")


if __name__ == "__main__":
    main()
