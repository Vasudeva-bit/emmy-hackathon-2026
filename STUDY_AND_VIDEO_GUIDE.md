# Study and video guide: Emmy IR to PTX compiler

This guide explains the compiler in this repository, the changes made during review, the evidence available from the contest runner, and a file-by-file plan for a short presentation.

## 1. The compiler in one sentence

`compile.py` reads one Emmy Kernel IR JSON file, deserializes its `KernelOp`, lowers the IR statements and expressions into NVIDIA PTX, and writes only PTX to standard output.

```text
kernel.ir.json
      │
      ▼
compile.py → emmy_ptx.compile_kernel
      │
      ├── ir_inspector.inspect_ir: parse graph, entry name, ABI, shapes, shared memory
      │
      └── codegen.PTXCodeGen: lower IR to PTX using PTXBuilder
                  │
                  ▼
             output.ptx
```

The contest runner then assembles that PTX with `ptxas`, launches it with the provided grid, block, and arguments, and checks the result.

## 2. Repository map and execution flow

### `compile.py`: command-line boundary

Show [compile.py](/home/kilaruvd/IR/emmy-hackathon-2026/compile.py:24) first.

- `main()` reads the IR path from `sys.argv[1]`.
- It calls `compile_kernel(ir_path)`.
- Successful compilation writes PTX to stdout. Errors and tracebacks go to stderr, so they do not corrupt redirected PTX output.
- The optional self-re-exec block looks for the project `.venv` if the current Python cannot import Emmy.

Example command:

```bash
./.venv/bin/python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > /tmp/reduction.ptx
```

### `emmy_ptx/__init__.py`: connect inspection to code generation

Show [the package entry point](/home/kilaruvd/IR/emmy-hackathon-2026/emmy_ptx/__init__.py:7).

`compile_kernel()` is deliberately small: inspect the IR, construct `PTXCodeGen`, return the generated PTX. It does not call Emmy's CUDA renderer or another compiler.

### `emmy_ptx/ir_inspector.py`: recover the kernel contract

Show [`inspect_ir()`](/home/kilaruvd/IR/emmy-hackathon-2026/emmy_ptx/ir_inspector.py:29).

It performs four jobs:

1. Reads JSON and calls `Graph.from_dict()` so the body is represented by typed Emmy IR nodes.
2. Finds the `KernelOp` and takes its entry name from the IR (falling back to the IR filename if the name is empty).
3. Resolves argument order. If matching `meta.json` exists, its `arg_order` is authoritative. Otherwise, it reads the serialized JSON input-key order and appends outputs.
4. Collects tensor shapes and shared-memory declarations for row-major indexing and PTX shared-memory allocation.

**Why the fallback matters:** Emmy's deserialized `KernelOp.inputs` is a `frozendict`, so iterating it does not reliably preserve the launch ABI order. In the matmul cases the correct order is `x1, x0, output`, while naive iteration could produce `x0, x1, output`. The fallback preserves the original JSON order. We checked it against all 15 open IRs with their metadata absent.

### `emmy_ptx/codegen.py`: lower IR operations

Show [`PTXCodeGen`](/home/kilaruvd/IR/emmy-hackathon-2026/emmy_ptx/codegen.py:38), then follow these functions in roughly this order:

- `generate_ptx()` emits the PTX header, entry signature, parameter loads, thread/block built-ins, shared-memory bases, lowered body, and `ret`.
- `lower_stmt()` dispatches each IR statement to its lowering routine.
- `lower_expr()` / `_lower_expr_raw()` handle scalar expressions such as variables, literals, arithmetic, comparisons, and ternary selection.
- `_flatten_index_expr()` turns multidimensional coordinates into a row-major element offset using the buffer shape.
- `_lower_tile()` maps the linear thread/block ID to the IR tile axes and lowers the tile body.
- `_lower_strided_loop()` and `_lower_loop()` emit labels, predicates, loop updates, and back branches.
- `_lower_load()` / `_lower_write()` form global or shared-memory addresses and emit loads, stores, or atomics.
- `_lower_assign()`, `_lower_accum()`, and `_lower_init()` emit arithmetic and reduction state updates.
- `_lower_warp_shuffle()` and `_lower_tree_halve()` implement warp and cross-warp reductions.
- `_lower_cp_async_copy()` implements the synchronous shared-memory copy path used by the current matmul IRs.

### `emmy_ptx/ptx_builder.py`: PTX mechanics

Show [PTXBuilder](/home/kilaruvd/IR/emmy-hackathon-2026/emmy_ptx/ptx_builder.py:19) last among implementation files.

- `fresh_pred()`, `fresh_b32()`, `fresh_b64()`, and `fresh_f32()` allocate unique virtual registers. `ptxas` later performs physical register allocation.
- `fresh_label()` creates unique branch labels.
- `emit()` appends an instruction with indentation.
- `reg_declarations` emits register declarations based on how many virtual registers were used.
- `float_to_ptx()` encodes float constants as IEEE-754 hex immediates such as `0f3F800000` for `1.0f`.

## 3. PTX and IR concepts to understand

### IR versus PTX

The input is a structured program: a `Tile` contains statements such as `Load`, `Assign`, `Accum`, `Cond`, and loops. The code generator walks these typed nodes and writes lower-level PTX instructions. The generated PTX is the compiler's output; `.ref.cu` files are useful references for understanding intended behavior, but are not used to generate the submission.

### Grid, block, thread, and tile

- A CTA is a CUDA thread block.
- `%ctaid.x` is the block index and `%tid.x` is a thread's index inside that block.
- The compiler forms a linear ID, then decodes it into the axis values named by the `Tile`.
- A `StridedLoop` lets each thread process additional elements by advancing its loop index by a stride.
- Index expressions are in elements. Global addresses use a 64-bit base pointer plus an element offset scaled by four bytes for the current f32 kernels.
- Shared-memory addresses are byte offsets too, so an f32 element index is scaled by four before `ld.shared` or `st.shared`.

### Virtual registers and predicates

PTX registers are separated by type: `%f` holds f32 values, `%r` holds 32-bit values, `%rd` holds 64-bit addresses, and `%p` holds predicates. Predicates are produced by `setp` and can guard a branch or a single instruction, for example `@%p0 st.global.f32 ...`.

### Reduction flow

The open reductions follow this pattern:

1. Each thread starts an accumulator at the operation's identity (`0` for sum, a low sentinel for max in the current IR).
2. A strided loop loads values and folds them into the accumulator.
3. `WarpShuffle` uses an XOR butterfly. For a 32-lane warp the offsets are 16, 8, 4, 2, 1.
4. One lane per warp writes its partial result to shared memory.
5. `TreeHalve` combines the warp partials through shared memory, with block barriers between stages.
6. The result is broadcast and the designated thread writes the output.

`bar.sync` is CTA-wide. Every participating thread must reach it. A thread must not branch around a block barrier while other threads in its CTA execute it.

### Shared memory and matmul copies

Matmul IR includes `CpAsyncCopy`, `CpAsyncCommit`, and `CpAsyncWait`. The current lowering uses synchronous `ld.global` followed by `st.shared` for each copied f32 value. Commit and wait become no-ops in that path; the IR's following `Sync` provides the CTA ordering before shared data is read. This keeps the copy behavior simple and avoids PTX `cp.async` alignment and zero-fill details.

This choice prioritizes straightforward correctness. Do not claim that the generated matmul uses hardware asynchronous copies.

## 4. What was changed during review

The broader IR-to-PTX implementation was already present when I reviewed the repository. My functional changes are in two files.

### A. Argument order fallback — `ir_inspector.py`

When `meta.json` is present, the compiler uses its `arg_order`. If it is absent, the compiler now preserves the input order from the serialized JSON instead of iterating Emmy's `frozendict`. This protects kernels whose input order is significant, particularly the two matmuls.

### B. Padded tile threads around collectives — `codegen.py`

Previously, a tile bounds guard emitted a branch to the end of the tile. That is safe for ordinary pointwise work, but can be unsafe if padded threads skip a shuffle or CTA barrier. The new behavior applies only when a tile both needs a bounds guard and contains a collective:

- It creates an active-thread predicate rather than branching around the tile body.
- Loads for inactive threads are initialized to zero and predicated, avoiding invalid global-memory reads.
- Accumulator updates and global stores/atomics are predicated for active threads.
- Threads still execute the reduction shuffle and barriers.
- Padded threads write the reduction identity to the shared slots consumed by `TreeHalve`.

For tiles without collectives, the original early-exit guard remains, avoiding extra work on the ordinary pointwise path.

### C. Respect barrier attributes — `codegen.py`

`Sync` now emits a warp barrier when `warp=True`, otherwise emits `bar.sync` with the IR's barrier ID and optional participant count. `TreeHalve` also uses its barrier ID and optional count instead of hardcoding barrier 0. The open set uses the default CTA barrier, so these extra forms are implemented but are not exercised by the open kernels.

### D. Warp shuffle activemask & collective safety — `codegen.py`

`WarpShuffle` now queries `activemask.b32` per round and passes the dynamic active thread mask to `shfl.sync.bfly.b32`. If hidden tests execute sub-warp blocks (`blockDim.x < 32`) or divergent warp paths, non-participating threads will not cause the shuffle to deadlock or trap.

### E. Exhaustive AST & elementwise lowering for hidden kernels — `codegen.py`

To ensure full generalization on unseen test kernels:
- **All 8 Emmy `Expr` types** are supported: `Var`, `Literal`, `BinaryExpr`, `TernaryExpr`, `FlatIndex`, `Builtin` (mapped to thread/block built-ins), `CastExpr` (conversions via `cvt.rzi.s32.f32` and `cvt.rn.f32.s32`), and `FuncCallExpr` (native PTX approximation instructions for `exp`, `sqrt`, `rsqrt`, `fmax`, `fmin`, `fabs`, `sin`, `cos`, `tanh`, `log`, `pow`, `bitwise_count`).
- **Extended `Assign` elementwise ops**: Native PTX lowerings for `rsqrt`, `sin`, `cos`, `tanh`, `log` (using `lg2.approx` * ln(2)), `square`, `sigmoid`, `silu`, `where` (`selp.f32`), `negative`, `pad`, and generalized fallback to Emmy's `op_to_expr`.
- **Additional statements**: `Select` (lowered to chained ternary conditionals via `select_to_ternary`) and `SetMaxNReg`.

## 5. What the validation shows

Validation done during this review:

- All 15 open problems passed through `submit.py --compiler compile.py` on the contest runner after the changes.
- An isolated-IR check removed `meta.json` and verified the argument-order fallback for all 15 kernels.
- A synthetic sum reduction with 64 logical threads and 128 launched threads passed.
- A synthetic max reduction with 100 logical threads and 128 launched threads passed, checking that inactive lanes use the max identity and still reach the reduction barriers.
- `git diff --check` passed after cleanup.

These results support saying that the open suite passes and that the targeted partial-tile cases pass. They do not prove that every hidden IR will pass or guarantee a contest result.

## 6. Video walkthrough plan (about 4 minutes)

Use your own voice and explain the code in your own words. The contest asks for a 3–5 minute walkthrough; the goal is to demonstrate understanding, not to read a script.

### 0:00–0:20 — State the problem and interface

**Show:** terminal and `compile.py`.

Explain: “The input is one Emmy Kernel IR JSON file. The output is PTX on stdout, so the runner can assemble and execute it. This is a direct IR-to-PTX compiler.”

### 0:20–0:40 — Show the call path

**Show:** `compile.py:24` then `emmy_ptx/__init__.py:7`.

Explain that the CLI delegates to `compile_kernel()`, which inspects the IR and constructs the code generator.

### 0:40–1:15 — Explain inspection and argument order

**Show:** `ir_inspector.py:29–43`, then `ir_inspector.py:45–91`.

Explain how `Graph.from_dict()` gives typed IR nodes, how the kernel entry name is found, and why `meta.json` is used for the launch ABI when available. Point out the serialized-JSON fallback for input order because a `frozendict` can reorder it.

### 1:15–2:00 — Walk through lowering

**Show:** `codegen.py:763` (`generate_ptx()`), `codegen.py:264` (`lower_stmt()`), and `codegen.py:310` (`_lower_tile()`).

Explain the header and parameter loads, built-in IDs, linear tile index, axis decoding, and recursive lowering of statements. Pick one simple operation such as `Load → multiply → add → Write` to explain how IR values become PTX registers and instructions.

### 2:00–2:40 — Explain reductions and the fix

**Show:** `codegen.py:655` (`_lower_warp_shuffle()`), `codegen.py:671` (`_lower_tree_halve()`), then `codegen.py:330–348` and `codegen.py:632–648` for the padded-thread handling.

Explain the warp butterfly, shared-memory reduction, and why a padded thread must not branch past a block barrier. Describe the active predicate and identity contribution briefly. Avoid getting bogged down in every PTX instruction.

### 2:40–3:10 — Explain PTX construction and matmul copy path

**Show:** `ptx_builder.py:8` and `ptx_builder.py:19`; optionally `codegen.py:728`.

Explain float hex constants and typed virtual registers. For matmul, say the IR copy is implemented as global load plus shared store, with the later barrier ordering the data before reads.

### 3:10–3:50 — Demonstrate the result

**Show:** terminal command and concise passing output.

```bash
./.venv/bin/python submit.py r01_array_sum --compiler compile.py
```

Say that the same command was run for all 15 open problems and all passed. If you show the synthetic partial-tile check, label it accurately as a generated stress case, not an official contest problem.

### 3:50–4:00 — Close

Summarize the covered categories: pointwise, reductions, matmul, fused kernels, and attention-like kernels. Avoid claiming a win or a hidden-set pass that has not been observed.

## 7. Questions to rehearse

**Why use Emmy's `Graph.from_dict()`?**

It deserializes the JSON into typed Kernel IR nodes, so the compiler can inspect statements and expressions instead of parsing pretty-printed text.

**Why does argument order matter?**

Kernel parameters are positional. If the pointers are in a different order than the launcher's argument list, the kernel reads or writes the wrong buffers even if every pointer is valid.

**Why preserve participation at a barrier?**

A CTA barrier waits for the participating threads. If padded threads branch around it, threads that reach the barrier may wait forever. Padded lanes should remain in the collective and contribute an identity value.

**What does `ptxas` do with the `%f0`, `%r0`, and `%p0` names?**

They are virtual PTX registers. `ptxas` maps them onto hardware registers while assembling the code.

**Does the matmul use hardware `cp.async`?**

No. This implementation uses synchronous global loads and shared-memory stores for the IR copy operations, then follows the IR synchronization. The open matmul problems pass with that lowering.

**What does passing the open set prove?**

It proves correctness for those 15 checked problems under their provided launch configurations and tolerances. It is evidence of broad coverage, but it does not guarantee hidden-set correctness or winning.

## 8. Before recording/submitting

- Make sure the public repository includes the current changes in `emmy_ptx/codegen.py` and `emmy_ptx/ir_inspector.py`.
- Rehearse the flow once without recording, and use a terminal with readable font size.
- Keep the demo short: show one compile command and a few clear pass lines rather than scrolling through the full suite.
- Use your own voice and only make claims you can explain.
