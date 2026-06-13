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

ChatLLM's RotorQuant cache uses an `exactBufferSize` model: recent entries stay exact and older entries are flushed to compressed storage. This preserves deferred quantization explicitly rather than quantizing the whole prefill prompt up front.

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

Fallback remains for prefill, multi-token queries, unsupported variants, unsupported masks, and unsupported dimensions. The current fused specialization covers IsoQuant at `head_dim = 128` and `256`, GQA repeats `1...4`, and `.none`/`.causal` decode masks. The default attention block size is now `128` tokens after spot tuning; the media profile keeps a smaller `64` token block.

The decode path is not wrapped in `MLX.compile()`. `GenerateParameters` generation calls the model directly each token, Qwen3.5 calls `attentionWithCacheUpdate`, and `RotorQuantKVCache.attention` invokes the custom kernel directly. Live compressed/exact lengths are passed as scalar dynamic kernel inputs, not shape-changing templates. Kernel templates still include storage capacities, packed widths, dtype, head counts, and bit widths, so specialization can change at compressed-capacity growth boundaries, but not every token.

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
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | dense | 28.7 | 145.0 MiB | 3.9246 | same run as below |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | legacy MLX quantized KV 4-bit | 16.3 | 58.8 MiB | 3.9687 | noisy single run |
| `Spakie/Qwen3.5-4B-MLX-mixed36` | 2048 | RotorQuant ISO k3/v2 | 20.0 | 48.1 MiB | 3.9146 | block size 256 spot check |

Current status:

- The block-parallel path removes the catastrophic long-context slowdown seen in the first sweep, e.g. 2B 8k improved from `7.3 tok/s` to `35.6 tok/s`.
- RotorQuant still does not meet the original "no decode regression versus dense/legacy" performance gate. At 2B 2048 with default block size 128, it is `0.65x` dense and `0.71x` legacy while using less KV memory than both.
- The app exposes RotorQuant as an experimental MLX-only option with a tappable `(E)` badge and beta warning.

Required speed fix:

- Reduce per-row unpack/rotation overhead in the block-parallel kernels while preserving RotorQuant's normalized rotated-centroid format.
- Investigate byte/lane-friendly packed-index layout for k3/v2, or a separate hot decode layout, because current cross-byte bit extraction is still expensive.
- Keep V inverse rotation mandatory; do not replace it with Hadamard-style self-canceling assumptions.
- Retain dense fallback for prefill, multi-token queries, unsupported variants, unsupported masks, and validation comparisons.

## Current Repository Scope

This checkout contains Swift app code, Apple Foundation Models integration, and a vendored MLX Swift LM package. There is no active llama.cpp/GGML backend in this repo. The implemented RotorQuant path therefore targets the MLX backend and keeps AFM untouched.

The llama.cpp discussion is still used as design input for the head-dimension 256, hybrid-layer, and deferred-quantization landmines.
