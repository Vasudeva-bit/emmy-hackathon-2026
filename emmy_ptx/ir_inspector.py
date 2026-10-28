"""IR Inspector: Parses Emmy Kernel IR, resolves parameter order, buffer shapes and smem info."""

from __future__ import annotations
import json
import math
from pathlib import Path
from dataclasses import dataclass

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.kernel.ir import KernelOp, Smem

STATIC_SMEM_CAP = 48 * 1024


@dataclass
class KernelInspection:
    graph: Graph
    kernel_node: Node
    kernel_op: KernelOp
    entry_name: str
    arg_order: list[str]
    buffer_shapes: dict[str, tuple[int, ...]]
    smem_buffers: dict[str, Smem]
    is_dynamic_smem: bool
    dynamic_smem_offsets: dict[str, int]
    dynamic_smem_total: int


def inspect_ir(ir_path: Path | str) -> KernelInspection:
    ir_path = Path(ir_path)
    raw_json = json.loads(ir_path.read_text())
    graph = Graph.from_dict(raw_json)

    # Locate KernelOp node
    kernel_nodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not kernel_nodes:
        raise ValueError(f"No KernelOp found in {ir_path}")
    kernel_node = kernel_nodes[0]
    kernel_op: KernelOp = kernel_node.op

    entry_name = kernel_op.name
    if not entry_name:
        entry_name = ir_path.stem.replace(".ir", "")

    # Resolve arg_order: meta.json first, then preserve the serialized
    # KernelOp input order. Graph.from_dict wraps inputs in a frozendict, whose
    # iteration order is not the launch ABI order (e.g. x1, x0 in matmul IRs).
    arg_order = None
    meta_path = ir_path.parent / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for k in meta.get("kernels", []):
                if k.get("entry") == entry_name:
                    candidate = list(k.get("arg_order", []))
                    if candidate:
                        arg_order = candidate
                    break
        except Exception:
            arg_order = None

    if arg_order is None:
        # KernelOp inputs are serialized in the same order used by Emmy's
        # CudaOp arg_order. Recover that insertion order from the JSON rather
        # than from Graph's frozendict, which can reorder the names.
        raw_kernel_node = next(
            (
                node
                for node in raw_json.get("nodes", {}).values()
                if node.get("op") == "KernelOp"
                and node.get("op_fields", {}).get("name") == entry_name
            ),
            None,
        )
        if raw_kernel_node is None:
            # Older IRs may leave KernelOp.name blank. Match its outputs.
            output_names = {t.name for t in kernel_node.outputs}
            raw_kernel_node = next(
                (
                    node
                    for node in raw_json.get("nodes", {}).values()
                    if node.get("op") == "KernelOp"
                    and {t.get("name") for t in node.get("outputs", [])} == output_names
                ),
                None,
            )
        if raw_kernel_node is None:
            raise ValueError(f"Could not recover serialized KernelOp inputs for {entry_name!r}")

        raw_inputs = raw_kernel_node.get("inputs")
        if not isinstance(raw_inputs, (dict, list, tuple)):
            raise ValueError(f"KernelOp inputs missing or malformed for {entry_name!r}")
        arg_order = list(raw_inputs) + [t.name for t in kernel_node.outputs]

    # Collect buffer shapes
    def _to_int(d):
        if hasattr(d, "as_static"):
            return int(d.as_static())
        return int(d)

    buffer_shapes: dict[str, tuple[int, ...]] = {}
    for n in graph.nodes.values():
        for t in n.outputs:
            buffer_shapes[t.name] = tuple(_to_int(d) for d in t.shape) if t.shape else (1,)

    # Collect Smem declarations from the body
    smem_buffers: dict[str, Smem] = {}
    for s in kernel_op.body.iter_of_type(Smem):
        smem_buffers[s.name] = s
        buffer_shapes[s.name] = tuple(_to_int(d) for d in s.extents)

    # Dynamic smem check
    def _nbytes(dtype_str: str) -> int:
        if "64" in dtype_str:
            return 8
        if "16" in dtype_str:
            return 2
        if "8" in dtype_str:
            return 1
        return 4

    total_smem_bytes = sum(math.prod(s.extents) * _nbytes(s.dtype) for s in smem_buffers.values())
    is_dynamic_smem = total_smem_bytes > STATIC_SMEM_CAP
    dynamic_smem_offsets: dict[str, int] = {}
    dynamic_smem_total = 0

    if is_dynamic_smem:
        cur = 0
        for name, s in smem_buffers.items():
            align = max(16, s.align or 0, _nbytes(s.dtype))
            cur = (cur + align - 1) & ~(align - 1)
            dynamic_smem_offsets[name] = cur
            cur += math.prod(s.extents) * _nbytes(s.dtype)
        dynamic_smem_total = cur

    return KernelInspection(
        graph=graph,
        kernel_node=kernel_node,
        kernel_op=kernel_op,
        entry_name=entry_name,
        arg_order=arg_order,
        buffer_shapes=buffer_shapes,
        smem_buffers=smem_buffers,
        is_dynamic_smem=is_dynamic_smem,
        dynamic_smem_offsets=dynamic_smem_offsets,
        dynamic_smem_total=dynamic_smem_total,
    )
