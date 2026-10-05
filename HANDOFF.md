# Handoff: NInfer performance work on the RTX 4090 (Bonsai and Qwen3.8)

This document is for the Claude Code session working on the user's PC (Windows, RTX 4090).
Until now the work was split between two sessions. A cloud session, without a GPU, wrote the
code, partially compiled it, and uploaded it. This local session compiled, tested, and measured
on the GPU. From now on this session does both. It is a temporary active work plan: delete it when
everything pending is closed (AGENTS.md: "Temporary plans are useful only for active work").

The user writes in Spanish; reply in Spanish, clearly and without exaggerating results.

---

## 1. How to work (the most important part)

Read `AGENTS.md` in full before touching code. In addition, these rules come from real mistakes in this
project:

1. **Measured or estimated, always stated.** Any figure you do not measure on the GPU is an estimate and
   you must say so. The cloud session's estimates failed several times:
   - The cp.async optimization of the `rk4v4-e8` attention was going to drop from 284 to ~180 µs and dropped to 260 µs.
   - Splitting QK into warp pairs (`0354c53`) made the kernel worse, from 260 to 326 µs.
   - The A8 fusion (`c1eb910`) gave no measurable gain in the round.

   No gain is real until it is measured.
2. **Correctness before speed.** Every numerical change is validated against the independent FP64
   oracle in its test. Do not loosen a test criterion to make it pass: if a test fails, find the
   cause. Real example: the new `rk4v4-e8` test failed because of a double rounding to BF16, which was a real bug (fixed in `787766f`), not an overly strict criterion.
3. **One change at a time when measuring performance.** There is ±1 ms of noise per round between runs: the
   Windows desktop uses the same 4090 at 60 Hz, plus thermal and clock drift. To attribute a
   gain:
   - compare old and new binaries in the same session and one after the other;
   - look at the per-kernel time in nsys/ncu, not just the ms per round;
   - if the difference is smaller than the noise, say so: "within the noise."
4. **Report failures as they are.** If something is worse, write it as worse. If something is reverted, write
   why.
5. **Record every result** in `docs/maintainer/bonsai-ternary-design.md`, section 9.1 (numbered list
   "Current state and next steps"). Note the measured commit, the command, the hardware, and the
   numbers. Qwen3.8 results go in `WINDOWS_PORT.md`.
6. **Commits:** use the Conventional Commits format (`perf(...)`, `fix(...)`, `test(...)`,
   `docs(...)`). Work on the `main` branch of the repo `JGamboa/ninfer-4090-windows`.
   Do not upload a performance change without having passed its tests.

### CUDA traps that have already bitten us in this project

- **48 KiB of static shared memory per kernel.** With `-rdc=true` the compiler does not detect the excess:
  it appears only at device link time (`nvlink error: uses too much shared data
  (0xc040 bytes, 0xc000 max)`). If you add a `__shared__`, compute the total for **each**
  instantiation of the template, especially the wide variants (Br = 48).
- **`bar.sync` with the ID in a register reserves all 16 hardware barriers** and leaves 1 CTA per SM.
  This happened in `0354c53`: occupancy dropped from 15 to 8 active warps. Use immediate IDs
  (`bar.sync 1, 64;`). To verify it in the binary:
  `cuobjdump -sass <obj> | findstr "BAR.SYNC"` must not show `BAR.SYNC R..`.
- **Dynamic indices in local arrays** (`a[2*part]` with a variable `part`) send the array to local memory
  (stack). Check `-Xptxas -v` or `cuobjdump --dump-resource-usage` (STACK must be 0).
- **`__launch_bounds__(threads, minBlocks)`** limits registers. A badly set limit causes spills
  (real case: Q4 K-split with 40 registers and spills, fixed with `q4_ksplit_resident_ctas`).
- **The 4090 has 128 SMs** (`kTargetSmCount`). Check the grid sizes and how many waves
  come out; the original code was designed for the 170 SMs of the 5090.
- **Measured read ceiling: ~845 GB/s**, not the 1008 from the datasheet. Compare GB/s against 845.

---

## 2. Environment

- **Repo:** `E:\LLM\ninfer-4090-bonsai`, branch `main` (previously `feat/bonsai-ternary`, which is frozen). There is another Qwen-only checkout at
  `E:\LLM\infer-4090-winport`.
- **Build:** open a shell with MSVC (`vcvars64.bat`) and CUDA in the PATH, and run
  `cmake --build build -j`. To configure from scratch, follow `WINDOWS_PORT.md`, section "Building on
  Windows" (`-DCMAKE_CUDA_ARCHITECTURES=89`).
- **Python:** `.venv\Scripts\python.exe` (3.12; on this machine it has always been 3.12).
- **Artifacts:**
  - `E:\LLM\bonsai2_27b_vl.ninfer`: Bonsai 2 27B ternary t5, with MTP in Q8 and vision.
  - `E:\LLM\qwen3_8_27b.ninfer`: Qwen3.8 27B Q4/Q5.
  - Bonsai sources: `E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf` and
    `E:\LLM\bonsai\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`.
- **Servers (the user's .bat files):** `start-bonsai-server - ninfer.bat` (Bonsai) and
  `E:\LLM\ninfer\start-ninfer-server.bat` (Qwen3.8: DFlash2 d6, 3 lanes, 100K KV `rk4v4-e8`,
  `--no-cuda-graph`).
- **Profiles:** `profiles\nsys\`, `profiles\ncu\`, and `profiles\bench\`.
- **Hardware:** RTX 4090 (sm_89, 128 SMs, 24 GB, 72 MB of L2). The monitor is connected to the 4090 at
  60 Hz, so the Windows compositor takes time away from the GPU.

---

## 3. Branch state (newest to oldest)

| Commit | What | Status |
|---|---|---|
| `1515b53` | docs: the slowness of the pairwise split is reproduced with the GPU free | record |
| `2f4789e` | fix: immediate IDs in the warp-pair barrier (decode small-T) | **not compiled or measured** |
| `e319660` | perf: the int8/`rk4v4` prefill kernel (`prompt_i8.cuh`) rewritten with inter-tile pipelining (~1000 lines) | **not compiled or measured** (ptxas OK in the cloud) |
| `ae421af` | perf: t5 tensor-core path for T = 5..32 (multiple lanes) | **not compiled or measured** (ptxas OK in the cloud) |
| `1bde24c` | feat: `ngram-mod` simulator (`tools/spec_sim/`), `NgramDraftPool`, and plan | Python and C++ tests passed in the cloud |
| `531f09b` | feat: MTP layer in Q5/Q4/mixed, plus converter recipes | **not compiled** (Python tests passed) |
| `19d7567` | fix: the pairwise split within 48 KiB | validated: correct, but **326 µs versus 260 µs** (worse) |
| `0354c53` | perf: QK split into warp pairs, `block_table` prefetch, `pos[]` out of the loop | worse because of the register barrier; see `2f4789e` |
| `b14ed06` | perf: cp.async of the packed KV codes (decode small-T) | validated: −8.4% in the kernel |
| `787766f` | fix: inverse V rotation in FP32 before the single BF16 rounding | validated: the oracle passes |
| `8b202c4` | test: FP64 oracle for `rk4v4-e8` (`--rk4v4-e8-only`) | validated |
| `c1eb910` | perf: rmsnorm and SwiGLU fused into the A8 quantization | validated: correct, gain within the noise |

`build\` is still compiled at `19d7567`.

---

## 4. Immediate task: compile, test, and measure everything pending

Do it in this order. If something does not compile or a test fails, stop there, find the cause, and fix it (sections 5 and 6).

**0. Save the old binaries for comparisons.** Before recompiling, copy these binaries to
`build_19d7567\`: `ninfer.exe`, `ninfer-serve.exe`, `ninfer_t5_bench`, and
`ninfer_causal_softmax_attention_bench`.

**1. Full build:** `cmake --build build -j`.

**2. Tests.** All must give PASS/OK:
- `ninfer_linear_t5_test` (new cases T = 5..72) and `ninfer_attn_input_proj_test`
- `ninfer_linear_q4_a16_test` and `ninfer_linear_q5_a16_test`
- `ninfer_softmax_attention_test`, full and with `--rk4v4-e8-only`
- `ninfer_kv_cache_append_test`
- `ctest -R "prism_loading|ngram_pool"`
- `python -m pytest tests/test_spec_sim.py tests/convert/test_bonsai_recipe.py`

**3. Quick regression:**
- Bonsai quick perplexity (section 9.1, point 5). It must give ~5.8549.
- The 6 Bonsai prompts with MTP draft 2, `--lm-head-draft` and greedy, with 1 lane. There must be no
  regression.
- md5 of the Qwen3.8 text with int8 KV at short context. It must come out the same as before. With `rk4v4-e8`
  it may change, and that is expected.

**4. Decode attention at 128K (`2f4789e`).**
- ncu of the same launch as always: Bonsai 128K `rk4v4-e8`, `--launch-skip 200`.
- Expected: back to 2 CTAs per SM (~15 active warps), less waiting at barriers than `b14ed06`, and less
  than 260 µs.
- If it is still at 1 CTA per SM or slower than 260 µs, revert the pairwise split (section 5).

**5. Multiple lanes (`ae421af`).**
- `ninfer_t5_bench`, old binary versus new, at T = 6, 8, 9, 12, 16, 24, and 32 per shape. With
  T = 1..4 nothing should change.
- Run this with the new binary and with `build_19d7567\ninfer-serve.exe`:

  ```
  python tools/bench/run_serve_concurrency.py --serve build\apps\ninfer-serve.exe --artifact bonsai=E:\LLM\bonsai2_27b_vl.ninfer --mode mtp2 --sampling greedy --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 3 --kv-capacity auto --output profiles\bench\bonsai_lanes_after
  ```

  Before measuring, confirm that the script passes `--lm-head-draft`.
- Measured starting point: 143.6 tok/s and 14.49 ms per round with 1 lane; 166.9 tok/s total and
  37.09 ms per round with 3 lanes.
- Estimate (not measured): ~2× the total with 3 lanes.

**6. Prefill at long context (`e319660`).**
- Run this with the old binary and with the new one:

  ```
  ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 --kv-dtype int8 --batch 1 --tokens 1024 --context 8192,32768,65536,131072 --mapping fragmented --execution eager --cache cold --warmup 5 --repeat 21
  ```

  and the same with `--kv-dtype rk4v4-e8`.
- Run `long_niah_64k` and `long_niah_128k`, with int8 and with `rk4v4-e8`:
  `--max-context 262144 --prefill-chunk 1024 --no-thinking --max-new 128 --greedy`.
  - The answer must be exactly `ORCHID=493817; COLOR=COBALT`.
  - Prefill times before: 24.9 s (int8) and 25.7 s (`rk4v4-e8`) at 64K; 61.0 s and 65.5 s at 128K.
- Agent estimate: −13 to −23% at 128K.

**7. MTP in Q4/Q5 (`531f09b`).**
- Convert the recipes `bonsai2_27b_mtp_q5`, `bonsai2_27b_mtp_q4`, and `bonsai2_27b_mtp_q4q5`. The
  command is in `docs/maintainer/bonsai-ternary-conversion.md`.
- Compare them against Q8 on the 6 prompts: tok/s, **tokens per round**, and ms per round. If
  acceptance drops, the variant is no good even if the round is faster.
- Estimate: ≤0.5 ms per round (~4%).

**8. `ngram-mod` simulator (does not use the GPU).**
- `pip install tokenizers`
- Then:

  ```
  python -m tools.spec_sim %USERPROFILE%\.claude\projects\<project>\*.jsonl --claude-code --artifact E:\LLM\qwen3_8_27b.ninfer --ngram-n 8,12,24 --caps 15,32,64 --json profiles\spec_sim.json
  ```

- Report the accepted tokens per round and the % of rounds that would exceed 15 tokens.
- Note: those transcripts were written by another model, so they only measure how much this type of
  work repeats, not the behavior of Qwen3.8 or Bonsai.

---

## 5. Safe point to go back to

**The last fully validated state of the code is `73aee4b`.** That commit only adds
documentation; the code is the same as `b14ed06`. In that state:
- all attention tests, the `rk4v4-e8` test, the KV append test, the t5 test, and perplexity gave PASS;
- `rk4v4-e8` decode attention at 128K measured 260 µs, the best value so far;
- the Bonsai int8 text came out with the same md5 as always.

Everything after `73aee4b` failed or is unvalidated:
- `0354c53` + `19d7567` gave **worse** results (326 µs);
- `2f4789e`, `531f09b`, `1bde24c`, `ae421af`, and `e319660` are **not compiled or measured**;
- the later `docs(...)` commits only record measurements, so they are kept.

**To have good binaries while debugging:** compile `73aee4b` in a separate folder and use those
binaries in the `.bat` files if the new build fails:

```
git worktree add E:\LLM\ninfer-known-good 73aee4b
cd E:\LLM\ninfer-known-good
:: configure and build the same as in WINDOWS_PORT.md, "Building on Windows"
```

**To go back on the branch without losing the rest:** use `git revert` on the commit that fails. Do
not do `git reset --hard` or `push --force` on `main`: measurements and the
other changes would be lost. Here is how to revert each part:

| If it fails... | Revert (in this order) | It goes back to |
|---|---|---|
| Decode attention does not recover 2 CTAs per SM or is still above 260 µs | `git revert 2f4789e 19d7567 0354c53` | the `b14ed06` kernel (260 µs) |
| Multi-lane path (t5 T = 5..32) | `git revert ae421af` | the previous GEMV/GEMM paths |
| Prefill kernel | `git revert e319660` | the previous, validated prefill kernel |
| MTP Q4/Q5 layer | `git revert 531f09b` | MTP only in Q8, which already worked |
| `ngram-mod` simulator | `git revert 1bde24c` (does not touch the runtime, will almost never be needed) | – |

If `git revert` conflicts with the documentation commits, keep the documentation version and
note in 9.1 what was reverted and why.

---

## 6. What to do if something fails

- **A compilation error:** read the file and the line and fix it. If it is a link error
  (`nvlink ... shared data`), check section 1 and compute the shared memory of each
  instantiation.
- **A test of `ae421af` fails (t5 T = 5..32):**
  - The new kernel is `small_t_kernel<NTiles>` in `src/ops/linear/t5/t5_a8.cuh`. The m16n8k32 MMA
    fragment mapping, the k permutation, and the swizzle were only reviewed by hand.
  - If the error appears at all T in the range, suspect the fragment mapping. If it appears only at
    some edges, suspect the routing in `t5_project.cu`.
  - If it is not resolved quickly: `git revert ae421af` and leave everything else.
- **A test of `e319660` fails (prefill):**
  - The rewrite is large. If it is not fixed quickly, do `git revert e319660`: the previous kernel
    is validated.
  - Note: there were already two variants without oracle coverage, `rk4v4` without E8 and `rk8v4`.
- **`2f4789e` does not recover 2 CTAs per SM, or the kernel stays above 260 µs:** revert
  `2f4789e`, `19d7567`, and `0354c53`, in that order. This returns to the `b14ed06` kernel, the best measured one.
- **An MTP Q4/Q5 variant lowers acceptance:** Bonsai stays with Q8. Record the result anyway.

---

## 7. Measured reference numbers (for comparison)

- **Bonsai** (t5, MTP draft 2, `--lm-head-draft`, 60 Hz):
  - Average decode ~163–166 tok/s on the 6 prompts, ~12.3–13 ms per round, and ~1.86 tokens per round.
  - pp512 3061 tok/s and pp2048 3349 tok/s.
  - Quick perplexity 5.8549.
  - Quality eval 43/45 (Qwen3.8: 44/45).
- **Breakdown of the Bonsai round at short context** (section 9.1, point 6):
  - t5 GEMV: 7.70 ms (62%).
  - MTP layer in Q8: 1.32 ms.
  - GDN: 0.94 ms.
  - Proposal head: 0.83 ms.
  - Quantization: 0.70 ms.
  - Attention: 0.46 ms.
- **Qwen3.8:**
  - MTP3: ~124 tok/s and 25.9 ms per round (3.23 tokens per round).
  - DFlash2 d12: ~33.5 ms per round.
  - CUDA graphs are worth <5%.
  - With 3 lanes, 100K KV, and graphs, 1182 MiB of VRAM are missing.
- **`rk4v4-e8` attention at 128K:** 260 µs per call in `b14ed06` (the best measured) and 326 µs in
  `19d7567`.

---

## 8. Pending ideas, from best to worst gain/effort ratio

None has been started unless indicated. The gains are estimates.

1. **Monitor on the integrated GPU.** The compositor at 60 Hz takes ~2.2–2.8 ms from each ~15 ms round
   (section 9.1, point 2). Estimated gain: +15–18%. It is only a configuration test: connect
   the monitor to the motherboard and measure the 6 prompts.
2. **Qwen3.8 with MTP3 and CUDA graphs** instead of DFlash2 d6 with `--no-cuda-graph`. MTP3 measured 124 versus
   112 tok/s, and its graphs cost 86 MiB per lane versus 480 for DFlash2. It must be confirmed that it fits in
   VRAM with the lanes and KV the user uses.
3. **`--proposal-rows 34816`** in Bonsai for English and code: +2.1% measured, same text, but it loses
   in other languages.
4. **`ngram-mod` phase 1** (plan in `docs/maintainer/ngram-speculation-plan.md`, 4–6 days): host
   drafts per MTP round, up to 15. Decide with the simulator (step 8) before starting. Going past
   16 costs 4–10 more days and is only justified if the simulator shows it.
5. **Readapt Bonsai's MTP with self-distillation.** It is the biggest lever: +20–40% estimated.
   - Today Bonsai's MTP is Qwen3.8's, trained on the unquantized model, and it gets ~56%
     right per position.
   - Feasibility study (summary):
     - Retrain only the MTP layer (~425M parameters) against Bonsai's top-32.
     - Data: final hidden states, after `final_norm`, exported from NInfer (there is no tool today:
       `score_tokens` only returns logprobs).
     - PyTorch training on the 4090: ~16–18 GB.
     - Generate 5M tokens: ~8 h. Train: ~1 h.
     - First, a pilot of ~200K tokens. Continue only if accuracy rises to ≥65%.
     - Quality does not change, because verification is lossless.
     - The trained MTP can be reinserted with the converter (HF names `mtp.*`).
   - The user has not yet decided whether to do it.
6. **Decode attention at 128K, next step:** double buffer, or expand K directly into registers to
   free the K tile from shared memory. Gain in the round: ~2–4% at 128K, ~0 at short
   context. Low priority.

Ideas already evaluated and discarded: AirLLM (layer streaming: does not apply, the models fit), NVIDIA
Dynamo (it is for multiple GPUs), rk3v3/rk4v3 (they save the same as `rk2v4`), and the clean A/B of `c1eb910`
(the gain is smaller than the noise).

---

## 9. Reference documents

- `AGENTS.md`: repo rules, mandatory.
- `docs/maintainer/bonsai-ternary-design.md`, section 9.1: live state and Bonsai measurements.
- `WINDOWS_PORT.md`: Windows build, Qwen3.8 measurements, and VRAM budget.
- `docs/maintainer/op-development.md`: how to qualify a kernel (oracle and performance).
- `docs/maintainer/ngram-speculation-plan.md`: `ngram-mod` plan.
- `tests/README.md` and `bench/README.md`: test and benchmark commands.