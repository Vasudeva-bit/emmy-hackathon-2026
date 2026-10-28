#!/usr/bin/env python3
"""CLI entry point for Emmy Kernel IR -> NVIDIA PTX compiler."""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Self re-exec if running under an interpreter without emmy installed
try:
    import emmy
except ImportError:
    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists() and sys.executable != str(venv_python):
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    else:
        sys.stderr.write("Error: emmy package not found. Please install emmy-ml==0.3.11\n")
        sys.exit(1)

from emmy_ptx import compile_kernel


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python compile.py path/to/kernel.ir.json > out.ptx\n")
        return 1

    ir_path = sys.argv[1]
    try:
        ptx = compile_kernel(ir_path)
        sys.stdout.write(ptx)
        return 0
    except Exception as e:
        sys.stderr.write(f"Compilation failed for {ir_path}: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
