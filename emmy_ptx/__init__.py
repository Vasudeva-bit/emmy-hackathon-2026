"""Emmy PTX Compiler Package: Direct Emmy Kernel IR to NVIDIA PTX compiler."""

from emmy_ptx.ir_inspector import inspect_ir
from emmy_ptx.codegen import PTXCodeGen


def compile_kernel(ir_path: str) -> str:
    """Compile a Kernel IR JSON file into valid NVIDIA PTX string."""
    inspection = inspect_ir(ir_path)
    codegen = PTXCodeGen(inspection)
    return codegen.generate_ptx()
