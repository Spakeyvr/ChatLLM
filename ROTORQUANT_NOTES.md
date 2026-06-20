# RotorQuant Notes

Sources read:

- https://github.com/scrya-com/rotorquant
- https://github.com/abysslover/rotorquant_improved
- https://github.com/ggml-org/llama.cpp/discussions/20969
- Local source clones used for code-level math checks:
  - `/tmp/chatllm-rotorquant-research/scrya-rotorquant`
  - `/tmp/chatllm-rotorquant-research/rotorquant-improved`

## Required Data Flow

RotorQuant-family KV compression stores directions separately from magnitudes:

1. Convert the key or value vector to float32.
2. Compute and store the float32 norm.
3. Divide by the norm, with a small epsilon for zero vectors.
4. Apply a block-diagonal orthogonal rotation to the unit vector.
5. Quantize each rotated coordinate to a learned Lloyd-Max centroid; store centroid indices as uint8.
6. On read, map indices back through the centroid table.
7. Apply the inverse block rotation.
8. Multiply by the stored norm and cast back to the requested dtype.

The inverse rotation is mandatory for the value cache. The upstream RotorQuant README calls out that V-cache dequantization must explicitly apply the inverse Givens/quaternion rotation; TurboQuant's Hadamard/WHT path does not need the same explicit inverse because that transform is self-inverse in the attention-weighted-sum path. Treating RotorQuant blocks as if they self-cancel is a correctness bug.

## Variants

The family has three relevant rotation choices:

- PlanarQuant: 2D Givens rotations. Each pair uses `[cos(theta), sin(theta)]`; forward is `[c*x0 - s*x1, s*x0 + c*x1]`; inverse is `[c*y0 + s*y1, -s*y0 + c*y1]`. It is the fastest and tiles any even head dimension.
- IsoQuant: 4D quaternion rotations. Upstream recommends IsoQuant-Fast as the default. The fast mode uses one unit quaternion per 4D block: forward `q_L * v`, inverse `conj(q_L) * v`. Full mode uses `q_L * v * conj(q_R)` with inverse `conj(q_L) * v * q_R`.
- Clifford RotorQuant: Cl(3,0) rotors. 3D vector blocks are embedded in grade-1 multivectors, rotated with the sandwich product `R x reverse(R)`, extracted back to 3D vector coordinates for centroid quantization, and inverted with `reverse(R) x R`. This is the research form, not the fastest default.

Default choice: IsoQuant-Fast unless local benchmarks prove PlanarQuant or another variant is better for ChatLLM's actual MLX workload. This matches the improved repository's guidance and keeps the 4D block aligned with head dimensions 128 and 256.

## Centroid Quantizer

The scalar centroids are Lloyd-Max centroids for the coordinate distribution of a random rotation of a unit vector. The local implementation computes this distribution from the beta density over `[-1, 1]` in the shared Lloyd-Max codebook helper, with midpoint boundaries between centroids.

RotorQuant uses that codebook shape:

- `centroids`: float32, length `2^bits`
- `boundaries`: float32, length `2^bits + 1` conceptually, with stored interior boundaries for index selection
- `indices`: uint8 tensor with one value per original coordinate
- `norms`: float32 tensor with one value per vector
- rotation parameters: stored once per cache quantizer/layer, derived deterministically from the layer seed

## Deferred Quantization

Upstream reports a deferred K-cache path:

- During prefill, K remains FP16.
- During decode insertion, new K rows are quantized as they are inserted into the persistent cache.
- This improves perplexity and avoids dequantization during prefill attention.

ChatLLM's RotorQuant cache uses an `exactBufferSize` model: recent entries stay exact and older entries are flushed to compressed storage. This preserves deferred quantization explicitly rather than quantizing the whole prefill prompt up front. Decode flushes use a bounded slack window of `min(16, exactBufferSize / 8)` tokens (minimum 1) before flushing the oldest rows back to the configured exact size; this keeps the exact tail close to the target while avoiding a tiny quantization graph launch on every single decode token.

Implementation interpretation for this repo:

- Multi-token updates are prefill-like and should stay exact until flushed by the deferred policy.
- Decode single-token inserts should quantize rows that leave the exact buffer.
- Attention must combine compressed blocks with exact recent rows.

## Head-Dimension 256 Landmine

The llama.cpp discussion reports that Qwen3.5/3.6 hybrid models use `head_dim = 256`. A block factorization that pads or spills at 256 can inflate K storage and erase the memory win.

Rules for ChatLLM:

- Read key/value head dimensions at runtime from `keys.dim(3)` and `values.dim(3)`.
- IsoQuant's block size 4 tiles both 128 and 256 exactly.
- PlanarQuant's block size 2 also tiles both 128 and 256 exactly.
- Clifford Cl(3,0) block size 3 does not tile 128 or 256 and requires padding/tail handling, so it is a poor default for Qwen3.5.
- If a dimension does not tile a selected block, either choose a tiling variant for that cache instance or preserve the tail without inflating the stored index count beyond the original head dimension.

## Hybrid Qwen3.5 Landmine

The llama.cpp discussion notes that Qwen3.5 9B is a hybrid DeltaNet plus full-attention model with `full_attention_interval=4`, so only a fraction of layers maintain a real KV cache. Local ChatLLM source matches this architecture:

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` creates `MambaCache()` for `layer.isLinear`.
- Full-attention layers call `makeLayerKVCache(parameters:layerIndex:)`.
- `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` follows the same pattern.

RotorQuant must never wrap or compress recurrent linear/DeltaNet state. Only full-attention layers should receive RotorQuant caches. The memory win is bounded by the number of full-attention layers, not the full layer count.

## Partial Rotary and MRoPE

Qwen3.5 full-attention layers use partial rotary, with `partialRotaryFactor = 0.25`, plus MRoPE sections. Local source computes:

- `ropeDims = Int(Float(headDim) * args.partialRotaryFactor)` in the text model.
- VLM Qwen3.5 similarly derives `rotaryDim` from `partialRotaryFactor` and uses `mrope_section`.

KV compression sits after key/value projection and RoPE application in the cache update path. It must not assume the whole head dimension is rotary-covered, and it must not transform only the RoPE-covered prefix. The full post-RoPE key vector and the full value vector are the compression inputs.

## Practical Format

Use the upstream format requirements:

- key/value coordinate indices: uint8
- key/value norms: float32
- rotation parameters: float32 generated once per quantizer/layer
- exact/deferred rows: original dtype, usually float16 or bfloat16

For 3-bit or 4-bit modes, packed storage can still be used internally, but the logical format remains uint8 indices plus float32 norms.

## RAM Accounting

For compressed rows with FP16/BF16 dense baseline:

- Dense per KV head token: `(K dim + V dim) * 2` bytes.
- RotorQuant per KV head token: `ceil(K dim * keyBits / 8) + 4-byte K norm + ceil(V dim * valueBits / 8) + 4-byte V norm`.
- At `head_dim = 128`, IsoQuant 3-bit K / 2-bit V stores `48 + 4 + 32 + 4 = 88` bytes versus dense `512` bytes, a `82.8%` compressed-row reduction.
- At `head_dim = 256`, IsoQuant 3-bit K / 2-bit V stores `96 + 4 + 64 + 4 = 168` bytes versus dense `1024` bytes, an `83.6%` compressed-row reduction.
- Qwen3.5 4B-style hybrid full-attention layers with `4` KV heads and `8` full-attention layers store about `5.25 KiB/token` compressed rows versus `32 KiB/token` dense full-attention KV rows.

The end-to-end app memory win is bounded by hybrid architecture: recurrent DeltaNet layers do not have a real KV cache and must remain untouched. Recent exact-buffer rows also remain dense by design.

## Decode Speed Status

The current Swift/MLX implementation has fused compressed decode-attention kernels for the default IsoQuant path. `RotorQuantKVCache.attention` ingests the new K/V slice and, for supported single-token decode calls, uses:

- a one-kernel path for short compressed prefixes
- a block-parallel compressed-prefix kernel plus a reduce kernel when the compressed prefix spans multiple attention blocks

- rotates query groups into key-rotation space
- scores compressed K directly from packed centroid indices and FP32 norms
- accumulates compressed V in value-rotation space
- applies the mandatory inverse value rotation once to the attention output
- includes the exact/deferred tail in the same streaming softmax

Fallback remains for prefill, multi-token queries, unsupported variants, unsupported masks, and unsupported dimensions. The current fused specialization covers IsoQuant at `head_dim = 128` and `256`, GQA repeats `1...4`, and `.none`/`.causal` decode masks. The block-parallel compressed-prefix and reduce kernels now run per KV head and compute all GQA repeats inside that workgroup, so compressed and exact K/V rows are not re-decoded or re-rotated once per query head. The default attention block size is now `128` tokens after spot tuning; the media profile keeps a smaller `64` token block.

The decode path is not wrapped in `MLX.compile()`. `GenerateParameters` generation calls the model directly each token, Qwen3.5 calls `attentionWithCacheUpdate`, and `RotorQuantKVCache.attention` invokes the custom kernel directly. Live compressed/exact lengths and the active compressed block count are passed as scalar dynamic kernel inputs, not per-token shape-changing templates. The block-parallel partial tensors are sized from compressed storage capacity rather than active block count, so their shapes change only when backing cache storage grows. Kernel templates still include storage capacities, packed widths, dtype, head counts, and bit widths, so specialization can change at compressed-capacity growth boundaries, but not every token or every attention-block boundary.

Pre-block-parallel context sweep evidence, ISO k3/v2, exact buffer 128, 192 PPL tokens, 128 generated tokens:

| Model | Prompt | Cache | Decode tok/s | KV memory | PPL |
| --- | ---: | --- | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | dense | 56.5 | 20.9 MiB | 3.7107 |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | legacy MLX quantized KV 4-bit | 52.4 | 12.8 MiB | 3.6923 |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | RotorQuant ISO k3/v2 | 37.7 | 12.7 MiB | 3.7105 |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | dense | 55.4 | 54.6 MiB | 3.7107 |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | legacy MLX quantized KV 4-bit | 46.0 | 22.3 MiB | 3.6923 |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | RotorQuant ISO k3/v2 | 24.1 | 18.3 MiB | 3.7105 |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | dense | 38.2 | 189.6 MiB | 3.7107 |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | legacy MLX quantized KV 4-bit | 28.1 | 60.3 MiB | 3.6923 |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | RotorQuant ISO k3/v2 | 7.3 | 40.4 MiB | 3.7105 |
| `mlx-community/Qwen3.5-2B-4bit` | 16384 | dense | 50.5 | 369.4 MiB | 3.7107 |
| `mlx-community/Qwen3.5-2B-4bit` | 16384 | legacy MLX quantized KV 4-bit | 45.8 | 110.8 MiB | 3.6923 |
| `mlx-community/Qwen3.5-2B-4bit` | 16384 | RotorQuant ISO k3/v2 | 3.8 | 69.9 MiB | 3.7105 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 512 | dense | 40.5 | 55.0 MiB | 3.9246 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 512 | legacy MLX quantized KV 4-bit | 29.0 | 33.5 MiB | 3.9687 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 512 | RotorQuant ISO k3/v2 | 21.9 | 33.4 MiB | 3.9146 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | dense | 29.8 | 145.0 MiB | 3.9246 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | legacy MLX quantized KV 4-bit | 26.5 | 58.8 MiB | 3.9687 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | RotorQuant ISO k3/v2 | 13.7 | 48.1 MiB | 3.9146 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 8192 | dense | 39.1 | 505.0 MiB | 3.9246 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 8192 | legacy MLX quantized KV 4-bit | 37.1 | 160.1 MiB | 3.9687 |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 8192 | RotorQuant ISO k3/v2 | 6.1 | 107.2 MiB | 3.9146 |

The mixed 16k run was not completed because the Mac hit a system watchdog panic during the benchmark session. The panic log named `AppleARMWatchdogTimer` and `watchdogd`, not RotorQuantEval, MLX, Metal, or an explicit GPU driver, so it is recorded as an unsafe missing data point rather than a RotorQuant correctness failure.

Interpretation:

- RotorQuant preserves the hybrid boundary: only full-attention caches are compressed; Mamba/DeltaNet caches remain untouched.
- RotorQuant substantially reduces KV memory at long context, e.g. mixed 8k uses `107.2 MiB` versus dense `505.0 MiB`.
- RotorQuant PPL is within the dense baseline delta and beats legacy on the mixed 3/6 sample.
- The old single-kernel compressed-prefix path regressed badly with context length, which pointed to serial per-row unpack/rotation overhead inside the fused kernel.

Post-block-parallel spot checks, ISO k3/v2, exact buffer 128, 192 PPL tokens, 128 generated tokens:

| Model | Prompt | Cache | Decode tok/s | KV memory | PPL | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | dense | 55.5 | 20.9 MiB | 3.7107 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | legacy MLX quantized KV 4-bit | 51.1 | 12.8 MiB | 3.6923 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | RotorQuant ISO k3/v2 | 36.7 | 12.7 MiB | 3.7105 | block-parallel available, short-prefix dominated |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | dense | 58.8 | 54.6 MiB | 3.7107 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | legacy MLX quantized KV 4-bit | 54.3 | 22.3 MiB | 3.6923 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | RotorQuant ISO k3/v2 | 38.4 | 18.3 MiB | 3.7105 | default block size 128 |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | dense | 51.7 | 189.6 MiB | 3.7107 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | legacy MLX quantized KV 4-bit | 47.0 | 60.3 MiB | 3.6923 | same run as below |
| `mlx-community/Qwen3.5-2B-4bit` | 8192 | RotorQuant ISO k3/v2 | 35.6 | 40.4 MiB | 3.7105 | block-parallel, measured with block size 256 before default retune |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | dense | 42.9 | 145.0 MiB | 3.9246 | same run as below |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | legacy MLX quantized KV 4-bit | 39.6 | 58.8 MiB | 3.9687 | same run as below |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | RotorQuant ISO k3/v2 | 28.4 | 48.1 MiB | 3.9146 | GQA-shared block+reduce kernels, block size 128 |

Shape-stabilized block partial spot checks, ISO k3/v2, exact buffer 128, block size 128, 192 PPL tokens, 64 generated tokens:

| Model | Prompt | Decode tok/s | KV memory | PPL | Cache summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 35.9 | 18.3 MiB | 3.7105 | `rotor=6 mamba=18` |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 26.0 | 48.1 MiB | 3.9146 | `rotor=8 mamba=24` |

Batched exact-flush same-run checks, ISO k3/v2, exact buffer 128, block size 128, 192 PPL tokens, 64 generated tokens:

| Model | Prompt | Cache | Decode tok/s | KV memory | PPL | Ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | dense | 51.8 | 54.6 MiB | 3.7107 | 1.00x |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | legacy MLX quantized KV 4-bit | 44.1 | 22.3 MiB | 3.6923 | 0.85x dense |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | RotorQuant ISO k3/v2 | 55.2 | 18.3 MiB | 3.6990 | 1.07x dense / 1.25x legacy |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | dense | 38.6 | 145.0 MiB | 3.9246 | 1.00x |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | legacy MLX quantized KV 4-bit | 35.2 | 58.8 MiB | 3.9687 | 0.91x dense |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | RotorQuant ISO k3/v2 | 33.1 | 48.1 MiB | 3.9280 | 0.86x dense / 0.94x legacy |

Current 2B same-run check after float16 block partial values, ISO k3/v2, exact buffer 128, block size 128, 192 PPL tokens, 128 generated tokens:

| Model | Prompt | Cache | Decode tok/s | KV memory | PPL | Ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | dense | 55.8 | 54.6 MiB | 3.7107 | 1.00x |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | legacy MLX quantized KV 4-bit | 51.7 | 22.3 MiB | 3.6923 | 0.93x dense |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | RotorQuant ISO k3/v2 | 49.8 | 18.3 MiB | 3.6990 | 0.89x dense / 0.96x legacy |

Mixed 4B order-bias checks at the requested 2048 prompt / 128 generated-token length, ISO k3/v2, exact buffer 128, block size 128, 192 PPL tokens:

| Scenario order | Dense tok/s | Legacy tok/s | RotorQuant tok/s | Rotor ratio | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense, legacy, rotor | 35.0 | 34.7 | 29.5 | 0.84x dense / 0.85x legacy | Rotor last |
| rotor, dense, legacy | 27.2 | 27.6 | 28.4 | 1.04x dense / 1.03x legacy | Rotor first |
| rotor, dense, legacy | 42.8 | 39.6 | 39.3 | 0.92x dense / 0.99x legacy | post-rollback check; Rotor PPL `3.9280`, dense `3.9246`, legacy `3.9687` |
| rotor, dense, legacy | 40.2 | 34.9 | 37.3 | 0.93x dense / 1.07x legacy | float16 block partial values; Rotor PPL `3.9280`, dense `3.9246`, legacy `3.9687` |

The current Mac shows enough order/thermal variance that this is not proof of the final no-regression gate. The stable findings are that RotorQuant PPL remains close to dense (`3.9280` versus dense `3.9246` and legacy `3.9687`), KV memory stays `48.1 MiB` versus dense `145.0 MiB` and legacy `58.8 MiB`, and only the `8` full-attention caches are RotorQuant while `24` DeltaNet/Mamba caches remain untouched. A wider 32-token exact-flush slack was tested and rejected for 128-token generation because RotorQuant dropped to `25.0 tok/s` in dense-first order; the default remains the conservative 16-token bounded slack.

Current status:

- The block-parallel path removes the catastrophic long-context slowdown seen in the first sweep, e.g. 2B 8k improved from `7.3 tok/s` to `35.6 tok/s`.
- Batched exact flushing removes most of the remaining 2B 2048 decode gap on the 64-token probe: RotorQuant is `1.07x` dense and `1.25x` legacy while using `33.45%` of dense KV memory and `81.97%` of legacy KV memory.
- RotorQuant still does not have clean proof for the original "no decode regression versus dense/legacy" performance gate on the primary mixed 4B 2048 check. The 128-token generated runs are order-sensitive; a clean final audit needs a cooler or randomized benchmark protocol.
- The app exposes RotorQuant as an experimental MLX-only option with a tappable `(E)` badge and beta warning.
- The app runtime no longer falls back to legacy MLX quantized KV when RotorQuant is disabled or unsupported. In-app choices are RotorQuant or dense/rotating KV; legacy quantized KV remains only as an explicit benchmark/evaluator baseline.

Exact-tail tuning evidence:

| Model | Prompt | Exact Buffer | Decode tok/s | KV memory | PPL | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | 32 | 34.5 | 11.8 MiB | 3.9734 | rejected: PPL regression |
| `mlx-community/Qwen3.5-2B-4bit` | 512 | 64 | 35.0 | 12.1 MiB | 3.6620 | no useful speed gain |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 64 | 50.3 | 17.6 MiB | 3.6716 | improves 2B PPL/memory after partial handoff |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 128 | 49.8 | 18.3 MiB | 3.6990 | current text default |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 256 | 50.9 | 19.5 MiB | 3.7107 | rejected: PPL regresses to dense |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 64 | 36.1 | 46.5 MiB | 3.9248 | improves PPL/memory but slows primary decode |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 128 | 37.3 | 48.1 MiB | 3.9280 | current speed default |

Shrinking `exactBufferSize` is useful for 2B quality/memory and may be a low-memory profile, but it slows the primary mixed 4B speed path. The default remains `128` for text and `32` only for the existing media-tuned profile.

Attention-block tuning after GQA-shared block/reduce kernels:

| Model | Prompt | Block Tokens | Decode tok/s | KV memory | PPL | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 64 | 27.6 | 48.1 MiB | 3.9146 | rejected: more reduce traffic |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 128 | 28.4 | 48.1 MiB | 3.9146 | current text default |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 256 | 28.6 | 48.1 MiB | 3.9146 | tiny mixed-model gain |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 512 | 28.6 | 48.1 MiB | 3.9146 | tied with 256 |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 256 | 35.6 | 18.3 MiB | 3.7105 | rejected: below prior 128 result |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | 256 | 37.5 | 48.1 MiB | 3.9280 | after float16 partials; tied with 128 |
| `mlx-community/Qwen3.5-2B-4bit` | 2048 | 256 | 51.0 | 18.3 MiB | 3.6990 | after float16 partials; tied with 128 |

The default remains `attentionBlockTokens = 128`: after float16 partial values, `256` is only a tiny/noisy gain on 4B and effectively tied on 2B, so changing the app default is not justified.

Required speed fix:

- Reduce remaining compressed-block/reduction overhead while preserving RotorQuant's normalized rotated-centroid format.
- A direct k3/v2 packed-index micro-specialization was tested on 2B 2048 and did not improve decode (`35.8-36.6 tok/s` versus the prior `38.4 tok/s` reference). It was rechecked on mixed 4B 2048/128 and showed no positive evidence (`34.2 tok/s`, PPL `3.9280`, KV `48.1 MiB`), so the patch was reverted. The next speed work should target higher-level duplicated work, reduction cost, or a separate hot decode layout rather than only bit extraction.
- Moving the reduce kernel to the same KV-head/GQA-repeat structure preserved correctness but did not materially move the mixed 4B 2048 decode result (`28.4 tok/s`), so the exact-tail reducer is not the main remaining bottleneck.
- An attempted exact-tail optimization that accumulated exact V rows in original value space and only inverse-rotated the compressed accumulator was rejected. It preserved focused kernel parity tests but regressed the mixed 4B 2048/128 real-model PPL to `7.0626` versus dense `3.9246`; restoring the existing exact V rotate-plus-inverse path returned RotorQuant PPL to `3.9280`. Do not remove exact-tail value rotation again without a stronger model-level correctness test.
- Pre-rotating exact-tail K/V into float32 side buffers was also tested and rejected. It kept PPL acceptable (`3.9302`) but moved exact-tail rotation cost into decode insertion, dropping mixed 4B 2048/128 RotorQuant decode to `29.7 tok/s` versus same-run dense `37.7` and legacy `34.4`; after rollback, a RotorQuant-only sanity check returned to PPL `3.9280`, `36.4 tok/s`, and `48.1 MiB` KV. The retained part is only the reload-order fix that rebuilds layer quantizers after `metaState` is applied, so non-default saved layer seeds are honored when `loadPromptCache` sets `state` before `metaState`.
- Float16 block partial values were kept: the compressed-block kernel now writes value partials in the query dtype while `partial_max` and `partial_sum` remain float32. This halves the block/reduce value handoff traffic without changing packed RotorQuant storage or the mandatory final inverse. Focused fused tests still pass, and the mixed 4B 2048/128 same-run check stayed at RotorQuant PPL `3.9280` while reaching `37.3 tok/s`, versus dense `40.2` and legacy `34.9`.
- Forcing the single fused decode kernel by setting `attentionBlockTokens = 4096` was tested on mixed 4B 2048/128 and rejected (`17.9 tok/s`, PPL `3.9280`, KV `48.1 MiB`), so the block-parallel path is not the source of the short-context speed gap at 2k.
- Moving exact-tail attention into additional block partials was tested and rejected. Storing exact partial values in query dtype produced mixed 4B `37.6 tok/s` but worsened 2B PPL to `3.7057`; storing exact partials separately as float32 reduced the quality hit but still showed no clear speed win (`36.9 tok/s`, PPL `3.9378` on mixed 4B; `47.3 tok/s`, PPL `3.7015` on 2B, measured in concurrent quick evals). The code was reverted to the reducer-side exact-tail path.
- `loadPromptCache` now validates RotorQuant metadata values and state shape before assigning into `RotorQuantKVCache` setters. Malformed RotorQuant prompt-cache files throw `KVCacheError` instead of crashing through `preconditionFailure`; existing `"TurboQuantKVCache"` metadata still migrates into a simple dense cache for read compatibility.
- Keep V inverse rotation mandatory; do not replace it with Hadamard-style self-canceling assumptions.
- Retain dense fallback for prefill, multi-token queries, unsupported variants, unsupported masks, and validation comparisons.

## Current Repository Scope

This checkout contains Swift app code, Apple Foundation Models integration, and a vendored MLX Swift LM package. There is no active llama.cpp/GGML backend in this repo. The implemented RotorQuant path therefore targets the MLX backend and keeps AFM untouched.

The llama.cpp discussion is still used as design input for the head-dimension 256, hybrid-layer, and deferred-quantization landmines.
