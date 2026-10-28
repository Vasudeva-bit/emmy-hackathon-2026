"""PTX Builder: Virtual register allocator, label generator, and instruction formatting."""

from __future__ import annotations
import struct
from typing import Literal


def float_to_ptx(val: float | int) -> str:
    """Encode IEEE-754 32-bit float as uppercase 8-digit hex literal: 0fXXXXXXXX."""
    fval = float(val)
    if fval == float("inf"):
        return "0f7F800000"
    if fval == float("-inf"):
        return "0fFF800000"
    bits = struct.unpack("<I", struct.pack("<f", fval))[0]
    return f"0f{bits:08X}"


class PTXBuilder:
    def __init__(self):
        self.instructions: list[str] = []
        self._pred_count = 0
        self._b32_count = 0
        self._b64_count = 0
        self._f32_count = 0
        self._label_count = 0

    def fresh_pred(self) -> str:
        r = f"%p{self._pred_count}"
        self._pred_count += 1
        return r

    def fresh_b32(self) -> str:
        r = f"%r{self._b32_count}"
        self._b32_count += 1
        return r

    def fresh_b64(self) -> str:
        r = f"%rd{self._b64_count}"
        self._b64_count += 1
        return r

    def fresh_f32(self) -> str:
        r = f"%f{self._f32_count}"
        self._f32_count += 1
        return r

    def fresh_reg(self, dtype: Literal["pred", "b32", "b64", "f32"]) -> str:
        if dtype == "pred":
            return self.fresh_pred()
        elif dtype == "b32":
            return self.fresh_b32()
        elif dtype == "b64":
            return self.fresh_b64()
        elif dtype == "f32":
            return self.fresh_f32()
        raise ValueError(f"Unknown register dtype: {dtype}")

    def fresh_label(self, prefix: str = "L") -> str:
        lbl = f"{prefix}_{self._label_count}"
        self._label_count += 1
        return lbl

    def emit(self, inst: str, indent: int = 1) -> None:
        pad = "    " * indent
        self.instructions.append(f"{pad}{inst}")

    def emit_label(self, label: str) -> None:
        self.instructions.append(f"{label}:")

    def emit_comment(self, comment: str, indent: int = 1) -> None:
        pad = "    " * indent
        self.instructions.append(f"{pad}// {comment}")

    @property
    def reg_declarations(self) -> list[str]:
        decls = []
        # Always declare at least 1 register per type to avoid empty syntax
        decls.append(f"    .reg .pred %p<{max(1, self._pred_count)}>;")
        decls.append(f"    .reg .b32  %r<{max(1, self._b32_count)}>;")
        decls.append(f"    .reg .b64  %rd<{max(1, self._b64_count)}>;")
        decls.append(f"    .reg .f32  %f<{max(1, self._f32_count)}>;")
        return decls
