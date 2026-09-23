# NInfer Op Development Rules

This document defines the repository-wide rules for admitting, specifying, owning, implementing,
qualifying, and measuring NInfer Ops. An **Op** is a semantic execution contract. A CUDA
**kernel** is one implementation, or one stage of an implementation, of an Op.

Repository-wide product scope, numerical principles, and evidence requirements remain in
[`AGENTS.md`](../../AGENTS.md). When this document and a concrete Op contract differ about that
Op's represented inputs, formula, supported domain, or observable effects, the contract in
`include/ninfer/ops/` is the concrete authority and the inconsistency must be resolved.

## 1. Scope and authorities

This file owns only rules that apply across Op families:

- Op admission and semantic boundaries;
- contract content and implementation freedom;
- Op, wrapper, launcher, kernel, core, and target ownership;
- state, workspace, naming, and dependency rules;
- Op-level qualification and performance evidence.

The remaining authorities are:

- `include/ninfer/ops/<family>.h` — the formula, represented inputs, supported domain, outputs,
  state effects, alias rules, and workspace contract of a concrete Op;
- [`tests/README.md`](../../tests/README.md) — test organization, commands, and common reporting
  mechanisms;
- [`bench/README.md`](../../bench/README.md) — benchmark executables, command lines, fixtures, and
  measurement behavior;
- family-specific maintainer references — current family design that cannot be expressed in its
  contract header.

Do not add current source inventories, target geometry tables, private route thresholds, one-time
migration steps, profiler command lines, or test-tool output schemas here. A family reference may
specialize these rules but must not redefine them. Remove a migration document after its target
state becomes the implementation and active contract.

## 2. Op admission and semantic boundary

An Op is a host-callable, semantically closed computation in the engine execution layer:

```text
(outputs, new_state) = F(inputs, weights, old_state, semantic_parameters)
```

Workspace, CUDA stream, and device facts are execution resources. They may select the
implementation but do not change the result or state transition.

A component is an Op only when all of the following hold:

1. **Logical effect.** It computes or changes a logical tensor value or an explicit local state.
2. **Semantic closure.** Its explicit inputs and metadata completely determine every observable
   output, mutation, valid region, alias restriction, and semantic cast or rounding boundary.
3. **Schedule independence.** It does not decide model call order, request policy, sequence
   frontier, transaction commit or rollback, or state lifetime.
4. **Implementation independence.** Its semantic interface does not expose grids, blocks, warps,
   tiles, launch counts, CUDA symbols, implementation filenames, or model-labelled backend paths.
5. **Independent invocation.** It has a meaningful host-callable boundary and can be qualified
   without running a complete model schedule.

One caller, one registered shape or format, one optimized token extent, one device implementation,
mutable state, or a fused formula does not disqualify an Op. Reuse count and support breadth do not
determine ownership.

Classify a proposed target-callable computation in this order:

1. If it has no logical tensor or state effect, it is infrastructure, validation, planning, or raw
   transfer rather than an Op.
2. If its complete transformation cannot be defined from explicit arguments and metadata, it
   belongs to target or product schedule.
3. If it decides call order, lifecycle, frontier, commit, rollback, or request policy, it belongs
   to Program, schedule, or runtime.
4. If it is only a partial step beneath a complete transformation, it is a private implementation
   detail of the enclosing Op.
5. Otherwise, admit it as an Op.

A useful admission test is:

> If the model name and call site disappear, can the transformation still be described and
> qualified completely?

If the answer still requires a model phase, weight role, or current request frontier, the proposed
boundary is not semantically closed.

### 2.1 Boundaries outside the Op layer

Program, target schedule, and runtime policy own model topology, call order, weight-role and state
instance selection, prompt chunking, multimodal spans, generated-token transactions,
prefix/frontier/commit/rollback policy, persistent state lifetime, CUDA Graph variants, and
publication of product statistics.

Core owns target-neutral storage and execution mechanisms: tensor and weight views, checked
layouts, arenas, physical cache containers, CUDA Graph lifetime, and raw host/device or
device/device transfers. A physical container may produce a checked view, but it does not acquire a
logical sequence cursor or transaction policy.

Artifact, frontend, product, and tooling code own framing and materialization, converter recipes,
tokenization and templates, media handling, request translation, transport, diagnostics, and
profiling control. These activities prepare, move, observe, or present data; they do not define a
device transformation.

Wrapper validation, launchers, CUDA entry points, codecs, implementation plans, partial reductions,
staging kernels, workspace layout helpers, and device primitives are parts of an Op implementation,
not independently target-callable Ops.

Every target-callable device transformation belongs to the central Op layer. A target invokes a
contract from `include/ninfer/ops/`, composes existing Ops and core mechanisms, or keeps host-side
schedule composition in the target. There is no target-private Op category.

### 2.2 Semantic extents and private routes

When a contract defines an axis as the Text/MTP token extent `T`, it admits every positive value
representable by its views and available storage unless the contract declares a semantic capacity.
Decode, small-T, and prefill may name private implementation or benchmark regimes; they are not
semantic variants or separate target-callable entries.

Other axes retain the finite geometry or capacity declared by their own contracts. A matrix column
does not become Text/MTP `T` merely because an implementation uses the same physical layout.

A wrapper may select a route or split a call into several launches for a valid extent. Private
dispatch thresholds and CUDA grid or resource limits do not narrow the supported semantic domain.

### 2.3 Stateful, fused, and stochastic Ops

A stateful Op defines both its produced value and its state transition:

```text
output    = G(old_state, input)
new_state = F(old_state, input)
```

Its contract identifies every location read or written and whether a newly written value is
visible within the call. Cursor movement, instance selection, transaction publication, commit, and
rollback remain schedule policy.

A fused computation is an Op when its complete composition and all effects form one closed
contract. Only observable intermediate casts or rounding seams are semantic. Tile decoders,
partial reductions, epilogue fragments, and warp primitives remain private implementation details.

A stochastic Op receives every semantic random input explicitly, including the state or seed,
logical position, domain separation, and draw index required by its algorithm. It promises a
backend bitstream or sampled sequence only when the contract deliberately makes that observable.

### 2.4 Execution envelopes

An Op may accept a host execution envelope when a device-resident semantic value cannot be read on
the host without synchronization but affects launch capacity or finite route selection. The
envelope is an execution promise, not a second mathematical input:

- device tensors still determine the exact output and state transition;
- the caller proves that the device value lies inside the declared interval;
- changing to another valid envelope may change capacity or implementation choice, not semantics;
- a captured launch is valid for the complete declared envelope.

Lifecycle cursors and graph-tier policy remain outside the Op. The Op owns only validation,
capacity, and finite dispatch for the explicit envelope it accepts.

### 2.5 Logical assignment and raw transfer

Classification follows the logical interface rather than the CUDA primitive used underneath. A
typed assignment or a copy with cast, transpose, concat, scatter, remap, or another logical index
mapping is the corresponding Op. An interface expressed only as addresses, byte count, and transfer
direction is a core or host transfer.

## 3. Contract headers

Repository-internal contracts live in:

```text
include/ninfer/ops/<family>.h
namespace ninfer::ops
```

They are execution-layer interfaces, not a public product ABI. A header may group closely related
operations or overloads; do not create empty families to force one function per file.

Each semantic Op or closely related overload group has one authoritative contract comment using
the applicable fields:

```cpp
/**
 * Op: <semantic operation name>
 *
 * Math / indexing:
 *   <complete formula, algorithm, probability process, or index mapping>
 *
 * Logical shapes:
 *   <symbols, semantic axes, valid ranges, and layout interpretation>
 *
 * Supported domain:
 *   <dtype, numeric format, storage layout, geometry, and semantic variants>
 *
 * Numeric:
 *   <logical decode, output representation, epsilon, masking, ties, and observable boundaries>
 *
 * Effects:
 *   <outputs, old/new state, in-place writes, valid regions, and alias rules>
 *
 * Workspace:
 *   <caller-owned transient scratch requirement or none>
 *
 * Execution:
 *   <execution envelope and stream/capture requirements or none>
 */
```

The contract must be sufficient to write an independent implementation and oracle. A name such as
“fused projection” or “attention path” is not a formula. A fused contract states the complete
composition; a stateful contract states old and new state; an indexing contract defines every axis
mapping and valid region; a stochastic contract defines its probability process and state effects.

A contract may cite the authoritative tensor-format or layout reference instead of duplicating a
registered encoding, but it still identifies the accepted format and interpretation.

The contract describes behavior shared by every implementation. It must not freeze:

- reduction association or thread execution order;
- warp, CTA, tile, or launch decomposition;
- kernel symbols or implementation filenames;
- backend approximation instructions;
- private accumulator, staging, operand, or workspace dtypes;
- private intermediate casts or rounding points; or
- bitwise equality unless an exact semantic format requires it.

Workspace sizing helpers and related overloads refer to the same contract rather than repeating
its formula.

## 4. Implementation ownership

Every Op follows one responsibility chain:

```text
semantic contract
       |
       v
wrapper: validation, workspace scope, finite dispatch
       |
       v
launcher: private CUDA launch policy
       |
       v
kernel: device implementation
```

The default horizontal layout uses `src/ops/{wrapper,launcher,kernel}`. A semantically closed family
with several routes or stages may instead own a vertical `src/ops/<category>/<family>/` subtree.
Physical layout does not change the contract boundary or dependency direction. Do not create empty
source layers merely to mirror this model.

### 4.1 Wrapper

The wrapper owns:

- semantic dtype, rank, shape, layout, alignment, scalar, configuration, and state validation;
- the call-scoped workspace scope;
- finite implementation selection from semantic variant, format, geometry, extent, state dtype,
  execution envelope, and supported device capability;
- invocation of the selected launcher or a composed fallback.

The wrapper must not dispatch on target key, artifact tensor name, converter field, model weight
role, Program phase, or arbitrary registry string. It does not own persistent state, allocate
hidden device memory, capture graphs, or choose model call order.

### 4.2 Launcher

The launcher owns private launch declarations and host definitions, grid/block/shared-memory
policy, template instantiation, and launch-error handling. Launcher headers are private:
contract headers, targets, product code, permanent tests, and public benchmarks do not include
them.

### 4.3 Kernel and common facilities

The kernel layer owns `__global__` functions and Op-local `__device__` computation. A kernel may
encode exact shape, format, SM capability, tiling, padding, and alignment assumptions. Every
assumption has a matching wrapper or launcher predicate and is never inferred from model identity.

`src/ops/common/` and category-private common code contain narrow zero-cost arithmetic, memory,
warp, and MMA facilities. They are not a second semantic catalog and are not included by targets.

A family may share private launch or computation bodies across related Ops. Each owning Op still
defines its output mapping, observable fusion boundary, validation, and dispatch; sharing does not
create a target-callable private backend.

## 5. State, workspace, naming, and dependencies

### 5.1 State and execution views

Ops receive non-owning execution views. They inspect only facts required for execution, such as
dtype and numeric format, storage layout, logical and padded shape, payload planes, quantization
geometry, and alignment.

Ops do not receive artifact object names, converter source fields, model weight roles, recipes, or
provenance. Artifact binding and target loading translate those concepts into explicit execution
views.

Program owns persistent state instances and lifetime; core owns target-neutral physical
containers; an Op owns only the documented reads, writes, outputs, and state transition of one
call.

### 5.2 Workspace

Caller-owned `WorkspaceArena&` or an explicit caller-owned scratch view is the transient workspace
boundary. An Op may suballocate within its call scope, but it must not allocate hidden device
memory, retain a scratch pointer, or own the arena.

An entry with statically zero scratch omits both the arena argument and a sizing query. A
scratch-capable entry accepts caller-owned workspace and exposes one authoritative capacity query
for its fixed host-visible profile and inclusive extent interval.

The query returns the minimum high-water capacity sufficient for every legal extent in that
interval. An exact-call query uses equal interval endpoints. Invalid profiles and intervals throw;
zero is reserved for a legal route that requires no caller-owned global scratch.

Capacity is measured from a 256-byte-aligned Op-local cursor, includes internal alignment gaps, and
excludes trailing padding. The sizing query and execution share the same route facts and allocation
recipe. A composed Op reports its complete public scratch requirement rather than exposing private
child requirements.

Ending the host workspace scope does not end device use. Kernels that consume a suballocation must
be submitted on the declared stream before the scope is released, and reuse relies on stream
ordering.

### 5.3 Naming

- Name an Op after its mathematical transformation or explicit state transition, not its first
  model, layer role, schedule phase, or CUDA strategy.
- Use `ninfer::ops` for semantic entries. Use `ninfer::ops::detail` only for private material that
  must cross translation-unit boundaries.
- Name an implementation specialization by real match facts such as format, geometry, extent,
  device capability, or algorithm.
- Reserve `kernel` for CUDA implementation concepts.
- Prefer one family name across contract, implementation, tests, and benchmarks unless several
  contracts intentionally share an implementation family.

### 5.4 Dependencies

The dependency direction is:

```text
core <- ops <- target <- runtime/engine product route
```

Artifact and loading code may materialize execution views for a target, but the Op layer does not
depend on artifact provenance or target binding concepts.

Enforce the boundary in code and build ownership:

- contract headers include only required L0, CUDA host, and peer semantic contract types;
- `src/ops/**` does not include target, Program, schedule, product, or artifact-provenance headers;
- target schedule includes contract headers, never private launcher, kernel, common, codec, or plan
  headers;
- `ninfer_ops` does not link a target;
- core and artifact do not link Ops or targets;
- explicit source lists give every implementation one build and link owner.

## 6. Qualification

Semantic Op tests live under `tests/ops/` and invoke the public contract independently of model call
order. They link the Op layer and required L0 libraries rather than a target package. Commands and
common reporting behavior belong in [`tests/README.md`](../../tests/README.md).

### 6.1 Oracle

Every floating-point Op uses one independent naive FP32/FP64 mathematical oracle over the logical
values represented by its public inputs. Test-owned fixture code independently decodes packed
values before invoking the oracle. The oracle evaluates the complete formula at high precision and
retains that result. It does not reproduce a production route's staging casts, activation
quantization, reduction tree, workspace dtype, public output rounding, or another implementation's
output.

Exact transforms and codecs use an independent exact oracle. A fused oracle evaluates the complete
fused formula instead of composing production Ops. A stateful oracle computes both output and new
state. Another GPU route, target reference, generated model output, or pairwise implementation
parity is supplementary evidence, never a second oracle.

The oracle determines correctness but does not prescribe production arithmetic. Private precision,
instruction operands, reduction association, staging, workspace representation, and kernel
decomposition remain implementation choices unless the contract makes an intermediate value
observable.

### 6.2 Conformance domain

One semantic Op or closely related overload group owns one identifiable qualification suite. Its
finite conformance matrix covers:

- every public entry and registered semantic dtype, format, layout, mode, state form, and real
  geometry affected by the contract;
- every reachable production route through the public wrapper, at its real match boundaries;
- representative inputs and semantic or route boundaries that can change results;
- every output and documented mutation, including untouched regions and preserved inputs when
  promised.

For an unbounded positive Text/MTP `T`, qualify `T=1`, each private route boundary at `b-1`, `b`,
and `b+1` where valid, and a representative interior extent for each route. Other axes use their
declared finite domains and relevant boundaries. Do not build a redundant Cartesian product when
one case establishes several dimensions.

When CUDA Graph capture/replay is part of the public execution contract, qualify the captured
public Op and its observable effects against the same oracle.

Keep schedule composition, persistent-state lifetime, and end-to-end behavior in target or product
integration tests. Private candidates, plan choices, launcher symbols, filenames, and source
organization are not qualification subjects.

### 6.3 Acceptance criteria

Exact outputs and regions compare every observable bit or byte. Each approximate output or state
uses a suite-owned named criterion for its real arithmetic or quantization profile; tests do not
define per-case tolerance overrides.

A floating-point criterion states its non-finite policy. Elementwise work uses a pointwise bound.
Reductions and matrix or attention computations use a normwise bound with a finite gross
pointwise-error cap so cancellation does not require strict elementwise agreement and isolated
corruption cannot hide in an aggregate norm.

Different production routes use different criteria only when their arithmetic or quantization
profiles differ materially. Widening a criterion requires a numerical reason and requalification
of its complete affected domain; one failing implementation is not sufficient justification.

## 7. Performance evidence

An Op microbenchmark measures the public semantic operation at an exact shape, format, layout,
extent, device/toolchain, and stated cache and timing condition. It is implementation evidence, not
a correctness oracle or proof of end-to-end improvement. Commands and executable-specific behavior
belong in [`bench/README.md`](../../bench/README.md).

A long-lived Op benchmark calls the public contract and, when applicable, its public workspace
capacity query. It does not include private implementation headers, call private launchers, expose
candidate or kernel forcing, or duplicate candidate legality and production dispatch tables.

The sole standing exception is `ninfer_gated_delta_net_bench --chunked-only --breakdown`: the
complete chunked pipeline is still measured through the public Op, while the benchmark may call
exactly its three intrinsic `prepare_wy_wu`, `state_passing`, and `output` stage launchers for
algorithm-stage attribution. It may not call the private complete-pipeline launcher or any other
private launcher.

Candidate comparison is task-local development work. A temporary sweep may call private launchers
and encode the exact overlapping candidate domains needed for a decision. Measure candidates under
the same inputs, cache, device, and timing conditions; put the selected winner or crossover in
production dispatch; verify final correctness and performance through the public Op; then delete
the losing candidates, temporary controls, and comparison-only entry points. The selected
production implementation is not temporary merely because it originated in the sweep.

An Op-scoped performance claim ends at the public Op boundary. Exact formats, layouts, shapes, and
extents can be constructed directly by the Op benchmark; they do not authorize loading a model
artifact or invoking a target, Program, Engine, or whole-round benchmark. Product-route evidence is
required only when the requested deliverable explicitly makes an end-to-end claim and includes
that product route in scope.

For a performance change:

1. establish correctness for the affected contract or route;
2. measure the relevant Op and workload extent;
3. when an explicitly scoped claim is end-to-end, measure the authorized public product route;
4. for that end-to-end work, use whole-inference profiling only when attribution is unresolved;
5. use kernel profiling only after a specific kernel-level question is identified and its answer
   can change the Op implementation decision.

Preserve only the context needed to interpret the result, as required by `AGENTS.md`.

## 8. Change checklist

For a new or changed device transformation:

1. classify the complete semantic boundary and reject schedule decisions, raw transfers,
   container lifecycle operations, and partial implementation helpers;
2. extend an existing family for a closely related overload or variant, and create a new family
   only for a distinct closed transformation;
3. define formula or indexing, logical shapes, supported domain, numeric boundaries, effects,
   aliasing, state, randomness, and workspace before selecting CUDA organization;
4. keep Text/MTP `T` semantic and positive unless the contract declares a capacity, while preserving
   the declared domains of other axes;
5. place validation and dispatch in the wrapper, launch policy in the launcher, device computation
   in kernels, and only narrow reusable primitives in common facilities;
6. keep persistent state, graph lifecycle, and schedule policy outside the Op;
7. qualify every affected public entry and reachable production profile directly against the
   independent oracle;
8. measure the Op when performance changes; measure a product route only for an explicitly scoped
   end-to-end claim;
9. integrate targets only through semantic contract headers and explicit operands;
10. give every source and symbol one clear build and link owner.

A contract change updates the authoritative comment, affected implementations, callers whose
assumptions changed, and tests protecting the changed behavior. A faster private implementation
under an unchanged contract changes private dispatch and implementation code plus the evidence
needed for that route; it does not create another semantic entry.
