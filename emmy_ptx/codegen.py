"""PTX Code Generator: Lowers Emmy Kernel IR nodes directly to NVIDIA PTX."""

from __future__ import annotations
import math
from typing import Any
from pathlib import Path

from emmy.compiler.ir.expr import Expr, Literal, Var, BinaryExpr, TernaryExpr, SimplifyCtx
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    Tile,
    Smem,
    Sync,
    WarpShuffle,
    TreeHalve,
    CpAsyncCopy,
    CpAsyncCommit,
    CpAsyncWait,
)
from emmy.compiler.ir.stmt.leaves import (
    Load,
    Write,
    Assign,
    Let,
    Accum,
    Init,
)
from emmy.compiler.ir.stmt.blocks import (
    Loop,
    StridedLoop,
    Cond,
)

from emmy_ptx.ptx_builder import PTXBuilder, float_to_ptx
from emmy_ptx.ir_inspector import KernelInspection


class PTXCodeGen:
    def __init__(self, inspection: KernelInspection):
        self.insp = inspection
        self.builder = PTXBuilder()
        self.symtab: dict[str, str] = {}
        self.buffer_ptrs: dict[str, str] = {}
        self.smem_base_regs: dict[str, str] = {}
        self.buffer_shapes = dict(inspection.buffer_shapes)

    def lower_expr_to_b32(self, expr: Expr | int | float) -> str:
        """Evaluate an integer/index expression into a 32-bit register (%r)."""
        res = self.lower_expr(expr)
        if isinstance(res, str) and (res.isdigit() or (res.startswith("-") and res[1:].isdigit())):
            reg = self.builder.fresh_b32()
            self.builder.emit(f"mov.b32 {reg}, {res};")
            return reg
        return res

    def lower_expr(self, expr: Any, target_reg: str | None = None) -> str:
        """Recursively lower an Expr to a register or literal operand."""
        res = self._lower_expr_raw(expr, target_reg=target_reg)
        if target_reg is not None and res != target_reg:
            if isinstance(res, str) and (res.startswith("%f") or res.startswith("0f")):
                self.builder.emit(f"mov.f32 {target_reg}, {res};")
            elif isinstance(res, str) and res.startswith("%rd"):
                self.builder.emit(f"mov.b64 {target_reg}, {res};")
            elif isinstance(res, str) and res.startswith("%p"):
                self.builder.emit(f"mov.pred {target_reg}, {res};")
            else:
                self.builder.emit(f"mov.b32 {target_reg}, {res};")
            return target_reg
        return res

    def _lower_expr_raw(self, expr: Any, target_reg: str | None = None) -> str:
        if isinstance(expr, (int, float)):
            if isinstance(expr, float):
                return float_to_ptx(expr)
            return str(expr)

        if isinstance(expr, Literal):
            if expr.dtype == "float":
                return float_to_ptx(expr.value)
            val = expr.value
            if hasattr(val, "as_static"):
                val = val.as_static()
            return str(int(val))

        if isinstance(expr, Var):
            if expr.name not in self.symtab:
                raise KeyError(f"Variable {expr.name!r} not defined in symtab: {list(self.symtab.keys())}")
            return self.symtab[expr.name]

        if isinstance(expr, BinaryExpr):
            # Try constant folding first
            try:
                simp = expr.simplify(SimplifyCtx.empty())
                if isinstance(simp, (Literal, Var)):
                    return self.lower_expr(simp, target_reg=target_reg)
                expr = simp
            except Exception:
                pass

            l_op = self.lower_expr(expr.left)
            r_op = self.lower_expr(expr.right)

            is_float = (
                (isinstance(l_op, str) and (l_op.startswith("%f") or l_op.startswith("0f")))
                or (isinstance(r_op, str) and (r_op.startswith("%f") or r_op.startswith("0f")))
            )

            op = expr.op
            if is_float:
                if op in ("<", "<=", ">", ">=", "==", "!="):
                    cmp_map = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
                    dst = target_reg or self.builder.fresh_pred()
                    self.builder.emit(f"setp.{cmp_map[op]}.f32 {dst}, {l_op}, {r_op};")
                    return dst
                dst = target_reg or self.builder.fresh_f32()
                if op == "+":
                    self.builder.emit(f"add.f32 {dst}, {l_op}, {r_op};")
                elif op == "-":
                    self.builder.emit(f"sub.f32 {dst}, {l_op}, {r_op};")
                elif op == "*":
                    self.builder.emit(f"mul.f32 {dst}, {l_op}, {r_op};")
                elif op == "/":
                    self.builder.emit(f"div.rn.f32 {dst}, {l_op}, {r_op};")
                else:
                    raise NotImplementedError(f"Float binary op {op!r} not supported")
                return dst
            else:
                # Integer operations
                if op in ("<", "<=", ">", ">=", "==", "!="):
                    cmp_map = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
                    dst = target_reg or self.builder.fresh_pred()
                    self.builder.emit(f"setp.{cmp_map[op]}.s32 {dst}, {l_op}, {r_op};")
                    return dst

                dst = target_reg or self.builder.fresh_b32()
                if op == "+":
                    self.builder.emit(f"add.s32 {dst}, {l_op}, {r_op};")
                elif op == "-":
                    self.builder.emit(f"sub.s32 {dst}, {l_op}, {r_op};")
                elif op == "*":
                    self.builder.emit(f"mul.lo.s32 {dst}, {l_op}, {r_op};")
                elif op == "/":
                    self.builder.emit(f"div.s32 {dst}, {l_op}, {r_op};")
                elif op == "%":
                    self.builder.emit(f"rem.s32 {dst}, {l_op}, {r_op};")
                elif op in ("&", "and"):
                    self.builder.emit(f"and.b32 {dst}, {l_op}, {r_op};")
                elif op in ("|", "or"):
                    self.builder.emit(f"or.b32 {dst}, {l_op}, {r_op};")
                elif op in ("^", "xor"):
                    self.builder.emit(f"xor.b32 {dst}, {l_op}, {r_op};")
                else:
                    raise NotImplementedError(f"Int binary op {op!r} not supported")
                return dst

        if isinstance(expr, TernaryExpr):
            # Lower condition to predicate
            if isinstance(expr.cond, BinaryExpr) and expr.cond.op in ("<", "<=", ">", ">=", "==", "!="):
                p_cond = self.lower_expr(expr.cond)
            else:
                cond_val = self.lower_expr(expr.cond)
                if cond_val.startswith("%p"):
                    p_cond = cond_val
                else:
                    p_cond = self.builder.fresh_pred()
                    self.builder.emit(f"setp.ne.s32 {p_cond}, {cond_val}, 0;")

            val_t = self.lower_expr(expr.if_true)
            val_f = self.lower_expr(expr.if_false)

            is_float = (
                (isinstance(val_t, str) and (val_t.startswith("%f") or val_t.startswith("0f")))
                or (isinstance(val_f, str) and (val_f.startswith("%f") or val_f.startswith("0f")))
            )

            if is_float:
                dst = target_reg or self.builder.fresh_f32()
                self.builder.emit(f"selp.f32 {dst}, {val_t}, {val_f}, {p_cond};")
            else:
                dst = target_reg or self.builder.fresh_b32()
                self.builder.emit(f"selp.b32 {dst}, {val_t}, {val_f}, {p_cond};")
            return dst

        raise NotImplementedError(f"Expression type {type(expr).__name__} not supported: {expr}")

    def _flatten_index_expr(self, buf_name: str, indices: tuple) -> Any:
        """Compute the 1-D row-major flattened element offset as an Expr."""
        if len(indices) == 0:
            return Literal(0, "int")

        if len(indices) == 1:
            return indices[0]

        shape = self.buffer_shapes.get(buf_name)
        if shape is None or len(shape) != len(indices):
            flat = indices[0]
            for i in indices[1:]:
                flat = BinaryExpr("+", flat, i)
            return flat

        terms = []
        for d, idx in enumerate(indices):
            stride = 1
            for k in range(d + 1, len(shape)):
                stride *= shape[k]
            if stride == 1:
                terms.append(idx)
            else:
                terms.append(BinaryExpr("*", idx, Literal(stride, "int")))

        flat = terms[0]
        for t in terms[1:]:
            flat = BinaryExpr("+", flat, t)
        return flat

    def flatten_index_to_reg(self, buf_name: str, indices: tuple) -> str:
        """Compute the 1-D row-major flattened element offset into an %r register."""
        expr = self._flatten_index_expr(buf_name, indices)
        return self.lower_expr_to_b32(expr)

    def lower_stmt(self, stmt: Any) -> None:
        """Dispatch and lower one statement."""
        st_name = type(stmt).__name__

        if st_name == "Tile":
            self._lower_tile(stmt)
        elif st_name == "StridedLoop":
            self._lower_strided_loop(stmt)
        elif st_name == "Loop":
            self._lower_loop(stmt)
        elif st_name == "Cond":
            self._lower_cond(stmt)
        elif st_name == "Load":
            self._lower_load(stmt)
        elif st_name == "Write":
            self._lower_write(stmt)
        elif st_name == "Assign":
            self._lower_assign(stmt)
        elif st_name == "Let":
            self._lower_let(stmt)
        elif st_name == "Accum":
            self._lower_accum(stmt)
        elif st_name == "Init":
            self._lower_init(stmt)
        elif st_name == "Smem":
            # Handled in preamble/setup
            pass
        elif st_name == "Sync":
            self.builder.emit("bar.sync 0;")
        elif st_name == "WarpShuffle":
            self._lower_warp_shuffle(stmt)
        elif st_name == "TreeHalve":
            self._lower_tree_halve(stmt)
        elif st_name == "CpAsyncCopy":
            self._lower_cp_async_copy(stmt)
        elif st_name in ("CpAsyncCommit", "CpAsyncWait"):
            # Synchronous copy bypass makes commit & wait no-ops
            pass
        else:
            raise NotImplementedError(f"Statement type {st_name} not supported: {stmt}")

    def _lower_tile(self, tile: Tile) -> None:
        # Calculate _gid = blockIdx.x * blockDim.x + threadIdx.x
        r_gid = self.builder.fresh_b32()
        r_bidx = self.symtab["ctaid.x"]
        r_tidx = self.symtab["tid.x"]

        if tile.aux_threads != 0:
            r_mul = self.builder.fresh_b32()
            self.builder.emit(f"mul.lo.s32 {r_mul}, {r_bidx}, {tile.block_threads};")
            r_mod = self.builder.fresh_b32()
            self.builder.emit(f"rem.u32 {r_mod}, {r_tidx}, {tile.block_threads};")
            self.builder.emit(f"add.s32 {r_gid}, {r_mul}, {r_mod};")
        else:
            r_bdim = self.symtab["ntid.x"]
            r_mul = self.builder.fresh_b32()
            self.builder.emit(f"mul.lo.s32 {r_mul}, {r_bidx}, {r_bdim};")
            self.builder.emit(f"add.s32 {r_gid}, {r_mul}, {r_tidx};")

        self.symtab["_gid"] = r_gid

        n = tile.n_elements
        guard = (tile.aux_threads != 0) or (n % tile.cells != 0)
        l_tile_end = None
        if guard:
            l_tile_end = self.builder.fresh_label("TILE_END")
            p_guard = self.builder.fresh_pred()
            self.builder.emit(f"setp.ge.s32 {p_guard}, {r_gid}, {n};")
            self.builder.emit(f"@{p_guard} bra {l_tile_end};")

        # Decode axes with raster support
        extents = [int(a.extent.as_static()) for a in tile.axes]
        names = [a.name for a in tile.axes]
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]

        raster = None
        if tile.raster_group and tile.raster_axes:
            if tile.raster_axes[0] in names and tile.raster_axes[1] in names:
                mi, ni = names.index(tile.raster_axes[0]), names.index(tile.raster_axes[1])
                if ni == mi + 1:
                    gi, oi = (mi, ni) if tile.raster_orient == "m" else (ni, mi)
                    g_eff = min(tile.raster_group, extents[gi])
                    if extents[gi] > 1 and g_eff > 1:
                        raster = (gi, oi, g_eff)

        if raster is not None:
            gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            mi, ni = min(gi, oi), max(gi, oi)
            sn = strides[ni]

            sub_expr = Var("_gid") if sn == 1 else BinaryExpr("/", Var("_gid"), Literal(sn, "int"))
            if mi != 0:
                sub_expr = BinaryExpr("%", sub_expr, Literal(ea * eb, "int"))
            r_rsub = self.lower_expr_to_b32(sub_expr)
            self.symtab["_rsub"] = r_rsub

            if ea % g == 0:
                a_expr = BinaryExpr(
                    "+",
                    BinaryExpr("*", BinaryExpr("/", Var("_rsub"), Literal(g * eb, "int")), Literal(g, "int")),
                    BinaryExpr("%", Var("_rsub"), Literal(g, "int")),
                )
                b_expr = BinaryExpr("/", BinaryExpr("%", Var("_rsub"), Literal(g * eb, "int")), Literal(g, "int"))
            else:
                # Ragged raster group
                r_rgrp = self.lower_expr_to_b32(BinaryExpr("/", Var("_rsub"), Literal(g * eb, "int")))
                self.symtab["_rgrp"] = r_rgrp
                r_mul_g = self.lower_expr_to_b32(BinaryExpr("*", Var("_rgrp"), Literal(g, "int")))
                r_rem_ea = self.lower_expr_to_b32(BinaryExpr("-", Literal(ea, "int"), Var(r_mul_g)))
                r_rsz = self.builder.fresh_b32()
                p_cmp = self.builder.fresh_pred()
                self.builder.emit(f"setp.lt.s32 {p_cmp}, {r_rem_ea}, {g};")
                self.builder.emit(f"selp.b32 {r_rsz}, {r_rem_ea}, {g}, {p_cmp};")
                self.symtab["_rsz"] = r_rsz

                a_expr = BinaryExpr(
                    "+",
                    BinaryExpr("*", Var("_rgrp"), Literal(g, "int")),
                    BinaryExpr("%", BinaryExpr("%", Var("_rsub"), Literal(g * eb, "int")), Var("_rsz")),
                )
                b_expr = BinaryExpr("/", BinaryExpr("%", Var("_rsub"), Literal(g * eb, "int")), Var("_rsz"))

        for i, a in enumerate(tile.axes):
            r_ax = self.symtab.get(a.name) or self.builder.fresh_b32()
            self.symtab[a.name] = r_ax
            if raster is not None and (i == mi or i == ni):
                expr = a_expr if i == gi else b_expr
                self.lower_expr(expr, target_reg=r_ax)
            else:
                s, e = strides[i], extents[i]
                term = Var("_gid") if s == 1 else BinaryExpr("/", Var("_gid"), Literal(s, "int"))
                expr = term if i == 0 else BinaryExpr("%", term, Literal(e, "int"))
                self.lower_expr(expr, target_reg=r_ax)

        # Lower tile body
        for stmt in tile.body:
            self.lower_stmt(stmt)

        if guard and l_tile_end is not None:
            self.builder.emit_label(l_tile_end)

    def _lower_strided_loop(self, loop: StridedLoop) -> None:
        # Pre-seed accumulators if requested
        if loop.seed:
            for s in loop.body:
                if isinstance(s, Accum) and s.name not in self.symtab:
                    r_acc = self.builder.fresh_f32()
                    self.symtab[s.name] = r_acc
                    self.builder.emit(f"mov.f32 {r_acc}, {float_to_ptx(s.op.identity)};")

        var_name = loop.axis.name
        r_var = self.symtab.get(var_name) or self.builder.fresh_b32()
        self.symtab[var_name] = r_var

        r_start = self.lower_expr(loop.start)
        self.builder.emit(f"mov.b32 {r_var}, {r_start};")

        end_expr = loop.end if loop.end is not None else Literal(int(loop.axis.extent.as_static()), "int")
        r_end = self.lower_expr(end_expr)
        step_expr = loop.step if isinstance(loop.step, Expr) else Literal(int(loop.step), "int")
        r_step = self.lower_expr(step_expr)

        l_top = self.builder.fresh_label("STRIDED_LOOP")
        l_end = self.builder.fresh_label("STRIDED_LOOP_END")
        p_exit = self.builder.fresh_pred()

        self.builder.emit_label(l_top)
        self.builder.emit(f"setp.ge.s32 {p_exit}, {r_var}, {r_end};")
        self.builder.emit(f"@{p_exit} bra {l_end};")

        for stmt in loop.body:
            self.lower_stmt(stmt)

        self.builder.emit(f"add.s32 {r_var}, {r_var}, {r_step};")
        self.builder.emit(f"bra {l_top};")
        self.builder.emit_label(l_end)

    def _lower_loop(self, loop: Loop) -> None:
        if loop.seed:
            for s in loop.body:
                if isinstance(s, Accum) and s.name not in self.symtab:
                    r_acc = self.builder.fresh_f32()
                    self.symtab[s.name] = r_acc
                    self.builder.emit(f"mov.f32 {r_acc}, {float_to_ptx(s.op.identity)};")

        var_name = loop.axis.name
        r_var = self.symtab.get(var_name) or self.builder.fresh_b32()
        self.symtab[var_name] = r_var

        self.builder.emit(f"mov.u32 {r_var}, 0;")
        extent_val = int(loop.axis.extent.as_static())

        l_top = self.builder.fresh_label("LOOP")
        l_end = self.builder.fresh_label("LOOP_END")
        p_exit = self.builder.fresh_pred()

        self.builder.emit_label(l_top)
        self.builder.emit(f"setp.ge.s32 {p_exit}, {r_var}, {extent_val};")
        self.builder.emit(f"@{p_exit} bra {l_end};")

        for stmt in loop.body:
            self.lower_stmt(stmt)

        self.builder.emit(f"add.s32 {r_var}, {r_var}, 1;")
        self.builder.emit(f"bra {l_top};")
        self.builder.emit_label(l_end)

    def _lower_cond(self, cond: Cond) -> None:
        p_cond = self.lower_expr(cond.cond)
        if not p_cond.startswith("%p"):
            p_tmp = self.builder.fresh_pred()
            self.builder.emit(f"setp.ne.s32 {p_tmp}, {p_cond}, 0;")
            p_cond = p_tmp

        if not cond.else_body:
            l_end = self.builder.fresh_label("ENDIF")
            self.builder.emit(f"@!{p_cond} bra {l_end};")
            for stmt in cond.body:
                self.lower_stmt(stmt)
            self.builder.emit_label(l_end)
        else:
            l_else = self.builder.fresh_label("ELSE")
            l_end = self.builder.fresh_label("ENDIF")
            self.builder.emit(f"@!{p_cond} bra {l_else};")
            for stmt in cond.body:
                self.lower_stmt(stmt)
            self.builder.emit(f"bra {l_end};")
            self.builder.emit_label(l_else)
            for stmt in cond.else_body:
                self.lower_stmt(stmt)
            self.builder.emit_label(l_end)

    def _lower_load(self, load: Load) -> None:
        r_idx = self.flatten_index_to_reg(load.input, load.index)
        if load.input in self.insp.smem_buffers:
            r_base = self.smem_base_regs[load.input]
            r_off = self.builder.fresh_b32()
            self.builder.emit(f"shl.b32 {r_off}, {r_idx}, 2;")
            r_addr = self.builder.fresh_b32()
            self.builder.emit(f"add.s32 {r_addr}, {r_base}, {r_off};")
            for k, nm in enumerate(load.names):
                dst = self.symtab.get(nm) or self.builder.fresh_f32()
                self.symtab[nm] = dst
                offset_str = f" + {4 * k}" if k > 0 else ""
                self.builder.emit(f"ld.shared.f32 {dst}, [{r_addr}{offset_str}];")
        else:
            rd_base = self.buffer_ptrs[load.input]
            rd_off = self.builder.fresh_b64()
            self.builder.emit(f"mul.wide.s32 {rd_off}, {r_idx}, 4;")
            rd_addr = self.builder.fresh_b64()
            self.builder.emit(f"add.s64 {rd_addr}, {rd_base}, {rd_off};")
            for k, nm in enumerate(load.names):
                dst = self.symtab.get(nm) or self.builder.fresh_f32()
                self.symtab[nm] = dst
                offset_str = f" + {4 * k}" if k > 0 else ""
                self.builder.emit(f"ld.global.f32 {dst}, [{rd_addr}{offset_str}];")

    def _lower_write(self, write: Write) -> None:
        r_idx = self.flatten_index_to_reg(write.output, write.index)
        if write.output in self.insp.smem_buffers:
            r_base = self.smem_base_regs[write.output]
            r_off = self.builder.fresh_b32()
            self.builder.emit(f"shl.b32 {r_off}, {r_idx}, 2;")
            r_addr = self.builder.fresh_b32()
            self.builder.emit(f"add.s32 {r_addr}, {r_base}, {r_off};")
            for k, val_nm in enumerate(write.values):
                val_reg = self.symtab[val_nm]
                offset_str = f" + {4 * k}" if k > 0 else ""
                self.builder.emit(f"st.shared.f32 [{r_addr}{offset_str}], {val_reg};")
        else:
            rd_base = self.buffer_ptrs[write.output]
            rd_off = self.builder.fresh_b64()
            self.builder.emit(f"mul.wide.s32 {rd_off}, {r_idx}, 4;")
            rd_addr = self.builder.fresh_b64()
            self.builder.emit(f"add.s64 {rd_addr}, {rd_base}, {rd_off};")
            for k, val_nm in enumerate(write.values):
                val_reg = self.symtab[val_nm]
                offset_str = f" + {4 * k}" if k > 0 else ""
                if write.atomic:
                    dummy = self.builder.fresh_f32()
                    self.builder.emit(f"atom.global.add.f32 {dummy}, [{rd_addr}{offset_str}], {val_reg};")
                else:
                    self.builder.emit(f"st.global.f32 [{rd_addr}{offset_str}], {val_reg};")

    def _lower_assign(self, assign: Assign) -> None:
        dst = self.symtab.get(assign.name) or self.builder.fresh_f32()
        self.symtab[assign.name] = dst
        args = [self.symtab[a] for a in assign.args]
        op = assign.op.name

        if op == "add":
            self.builder.emit(f"add.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "subtract":
            self.builder.emit(f"sub.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "multiply":
            self.builder.emit(f"mul.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "divide":
            self.builder.emit(f"div.rn.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "reciprocal":
            self.builder.emit(f"rcp.approx.ftz.f32 {dst}, {args[0]};")
        elif op == "maximum":
            self.builder.emit(f"max.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "minimum":
            self.builder.emit(f"min.f32 {dst}, {args[0]}, {args[1]};")
        elif op == "relu":
            self.builder.emit(f"max.f32 {dst}, {args[0]}, 0f00000000;")
        elif op == "sqrt":
            self.builder.emit(f"sqrt.approx.ftz.f32 {dst}, {args[0]};")
        elif op == "exp":
            tmp = self.builder.fresh_f32()
            self.builder.emit(f"mul.f32 {tmp}, {args[0]}, 0f3FB8AA3B;")
            self.builder.emit(f"ex2.approx.ftz.f32 {dst}, {tmp};")
        elif op == "copy":
            self.builder.emit(f"mov.f32 {dst}, {args[0]};")
        elif op == "negate":
            self.builder.emit(f"neg.f32 {dst}, {args[0]};")
        elif op == "abs":
            self.builder.emit(f"abs.f32 {dst}, {args[0]};")
        else:
            raise NotImplementedError(f"Assign op {op!r} not supported: {assign}")

    def _lower_let(self, let: Let) -> None:
        val_str = self.lower_expr(let.value)
        is_float = (
            (isinstance(val_str, str) and (val_str.startswith("%f") or val_str.startswith("0f")))
            or (let.dtype is not None and "float" in let.dtype.name)
        )
        if is_float:
            dst = self.symtab.get(let.name) or self.builder.fresh_f32()
            self.symtab[let.name] = dst
            self.builder.emit(f"mov.f32 {dst}, {val_str};")
        else:
            dst = self.symtab.get(let.name) or self.builder.fresh_b32()
            self.symtab[let.name] = dst
            self.builder.emit(f"mov.b32 {dst}, {val_str};")

    def _lower_accum(self, accum: Accum) -> None:
        dst = self.symtab[accum.name]
        lhs = self.symtab[accum.base] if accum.base is not None else dst
        rhs = self.symtab[accum.value]
        op = accum.op.name
        if op == "add":
            self.builder.emit(f"add.f32 {dst}, {lhs}, {rhs};")
        elif op == "maximum":
            self.builder.emit(f"max.f32 {dst}, {lhs}, {rhs};")
        elif op == "minimum":
            self.builder.emit(f"min.f32 {dst}, {lhs}, {rhs};")
        elif op == "multiply":
            self.builder.emit(f"mul.f32 {dst}, {lhs}, {rhs};")
        else:
            raise NotImplementedError(f"Accum op {op!r} not supported: {accum}")

    def _lower_init(self, init: Init) -> None:
        dst = self.symtab.get(init.name) or self.builder.fresh_f32()
        self.symtab[init.name] = dst
        self.builder.emit(f"mov.f32 {dst}, {float_to_ptx(init.identity)};")

    def _lower_warp_shuffle(self, ws: WarpShuffle) -> None:
        s = int(ws.length) // 2
        while s > 0:
            for st, sb in zip(ws.state, ws.state_b):
                r_st = self.symtab[st]
                r_sb = self.symtab.get(sb) or self.builder.fresh_f32()
                self.symtab[sb] = r_sb
                r_tmp = self.builder.fresh_b32()
                r_shfl = self.builder.fresh_b32()
                self.builder.emit(f"mov.b32 {r_tmp}, {r_st};")
                self.builder.emit(f"shfl.sync.bfly.b32 {r_shfl}, {r_tmp}, {s}, 0x1f, 0xffffffff;")
                self.builder.emit(f"mov.b32 {r_sb}, {r_shfl};")
            for stmt in ws.combine_states:
                self.lower_stmt(stmt)
            s >>= 1

    def _lower_tree_halve(self, th: TreeHalve) -> None:
        t_var = self.symtab[th.tid_var]
        h = int(th.length) // 2
        while h > 0:
            l_skip = self.builder.fresh_label("TREE_SKIP")
            p_skip = self.builder.fresh_pred()
            self.builder.emit(f"setp.ge.s32 {p_skip}, {t_var}, {h};")
            self.builder.emit(f"@{p_skip} bra {l_skip};")

            for buf, st, sb in zip(th.bufs, th.state, th.state_b):
                r_st = self.symtab.get(st) or self.builder.fresh_f32()
                self.symtab[st] = r_st
                r_sb = self.symtab.get(sb) or self.builder.fresh_f32()
                self.symtab[sb] = r_sb
                r_base = self.smem_base_regs[buf]

                # Load buf[t]
                r_off1 = self.builder.fresh_b32()
                r_addr1 = self.builder.fresh_b32()
                self.builder.emit(f"shl.b32 {r_off1}, {t_var}, 2;")
                self.builder.emit(f"add.s32 {r_addr1}, {r_base}, {r_off1};")
                self.builder.emit(f"ld.shared.f32 {r_st}, [{r_addr1}];")

                # Load buf[t + h]
                r_idx2 = self.builder.fresh_b32()
                r_off2 = self.builder.fresh_b32()
                r_addr2 = self.builder.fresh_b32()
                self.builder.emit(f"add.s32 {r_idx2}, {t_var}, {h};")
                self.builder.emit(f"shl.b32 {r_off2}, {r_idx2}, 2;")
                self.builder.emit(f"add.s32 {r_addr2}, {r_base}, {r_off2};")
                self.builder.emit(f"ld.shared.f32 {r_sb}, [{r_addr2}];")

            for stmt in th.combine_states:
                self.lower_stmt(stmt)

            for buf, st in zip(th.bufs, th.state):
                r_base = self.smem_base_regs[buf]
                r_off = self.builder.fresh_b32()
                r_addr = self.builder.fresh_b32()
                self.builder.emit(f"shl.b32 {r_off}, {t_var}, 2;")
                self.builder.emit(f"add.s32 {r_addr}, {r_base}, {r_off};")
                self.builder.emit(f"st.shared.f32 [{r_addr}], {self.symtab[st]};")

            self.builder.emit_label(l_skip)
            self.builder.emit("bar.sync 0;")
            h >>= 1

        # Broadcast reduced value from root (0) to every thread
        for buf, st in zip(th.bufs, th.state):
            r_st = self.symtab.get(st) or self.builder.fresh_f32()
            self.symtab[st] = r_st
            r_base = self.smem_base_regs[buf]
            self.builder.emit(f"ld.shared.f32 {r_st}, [{r_base}];")

    def _lower_cp_async_copy(self, cp: CpAsyncCopy) -> None:
        src_idx = self.flatten_index_to_reg(cp.src, cp.src_index)

        dst_expr = self._flatten_index_expr(cp.smem, cp.smem_index)
        if cp.lane_index is not None:
            cols = self.buffer_shapes[cp.smem][-1]
            lane_expr = BinaryExpr(
                "+",
                BinaryExpr("*", cp.lane_index[0], Literal(cols, "int")),
                cp.lane_index[1],
            )
            dst_expr = BinaryExpr("+", dst_expr, lane_expr)
        dst_idx = self.lower_expr_to_b32(dst_expr)

        rd_base = self.buffer_ptrs[cp.src]
        rd_off = self.builder.fresh_b64()
        rd_addr = self.builder.fresh_b64()
        self.builder.emit(f"mul.wide.s32 {rd_off}, {src_idx}, 4;")
        self.builder.emit(f"add.s64 {rd_addr}, {rd_base}, {rd_off};")

        r_smem_base = self.smem_base_regs[cp.smem]
        r_smem_off = self.builder.fresh_b32()
        r_smem_addr = self.builder.fresh_b32()
        self.builder.emit(f"shl.b32 {r_smem_off}, {dst_idx}, 2;")
        self.builder.emit(f"add.s32 {r_smem_addr}, {r_smem_base}, {r_smem_off};")

        num_floats = max(1, cp.nbytes // 4)
        for k in range(num_floats):
            tmp = self.builder.fresh_f32()
            offset_str = f" + {4 * k}" if k > 0 else ""
            self.builder.emit(f"ld.global.f32 {tmp}, [{rd_addr}{offset_str}];")
            self.builder.emit(f"st.shared.f32 [{r_smem_addr}{offset_str}], {tmp};")

    def generate_ptx(self) -> str:
        """Compile the inspected KernelOp into complete ASCII PTX."""
        insp = self.insp
        lines = [
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
        ]

        # Shared memory declarations
        if insp.is_dynamic_smem:
            lines.append(".extern .shared .align 16 .b8 _smem_pool[];")
            lines.append("")
        else:
            for name, s in insp.smem_buffers.items():
                size_bytes = math.prod(s.extents) * 4
                align = max(16, s.align or 0)
                lines.append(f".shared .align {align} .b8 {name}[{size_bytes}];")
            if insp.smem_buffers:
                lines.append("")

        # Kernel entry signature
        param_decls = []
        for name in insp.arg_order:
            param_decls.append(f"    .param .u64 param_{name}")
        lines.append(f".visible .entry {insp.entry_name}(")
        lines.append(",\n".join(param_decls))
        lines.append(")")
        lines.append("{")

        # Preamble: Load parameters
        for name in insp.arg_order:
            rd_raw = self.builder.fresh_b64()
            rd_ptr = self.builder.fresh_b64()
            self.builder.emit(f"ld.param.u64 {rd_raw}, [param_{name}];")
            self.builder.emit(f"cvta.to.global.u64 {rd_ptr}, {rd_raw};")
            self.buffer_ptrs[name] = rd_ptr

        # Preamble: Load builtins
        for b_name in ("tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y", "ntid.z", "nctaid.x"):
            r_b = self.builder.fresh_b32()
            self.builder.emit(f"mov.u32 {r_b}, %{b_name};")
            self.symtab[b_name] = r_b

        # Hardware lane & warp
        r_lane = self.builder.fresh_b32()
        self.builder.emit(f"and.b32 {r_lane}, {self.symtab['tid.x']}, 31;")
        self.symtab["lane"] = r_lane

        r_warp = self.builder.fresh_b32()
        self.builder.emit(f"shr.u32 {r_warp}, {self.symtab['tid.x']}, 5;")
        self.symtab["warp"] = r_warp

        # Smem buffer base registers
        if insp.is_dynamic_smem:
            r_pool = self.builder.fresh_b32()
            self.builder.emit("mov.u32 {r_pool}, _smem_pool;".format(r_pool=r_pool))
            for name, offset in insp.dynamic_smem_offsets.items():
                r_base = self.builder.fresh_b32()
                self.builder.emit(f"add.u32 {r_base}, {r_pool}, {offset};")
                self.smem_base_regs[name] = r_base
        else:
            for name in insp.smem_buffers:
                r_base = self.builder.fresh_b32()
                self.builder.emit(f"mov.u32 {r_base}, {name};")
                self.smem_base_regs[name] = r_base

        # Lower kernel body
        for stmt in insp.kernel_op.body:
            self.lower_stmt(stmt)

        self.builder.emit("ret;")

        # Assemble full body with register declarations at the top
        reg_lines = self.builder.reg_declarations
        body_lines = reg_lines + [""] + self.builder.instructions

        lines.extend(body_lines)
        lines.append("}")
        lines.append("")
        return "\n".join(lines)
