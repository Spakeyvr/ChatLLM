# RotorQuant Integration Map

This map covers the current checkout at `/Users/nevio/Desktop/Projects/ChatLLM`.

## Backends Present

- MLX Swift LM: present under `Vendor/mlx-swift-lm`.
- Apple Foundation Models: present in `On-Device_LLM_Chat/LLMGenerator.swift` and related `FoundationModels` chat/title/tool code.
- llama.cpp/GGML: not present in this checkout. There is no `ggml`, `llama.cpp`, C/C++ backend bridge, CMake target, or GGUF runtime surface to wire.

RotorQuant is implemented for the MLX backend. AFM remains outside the KV-cache compression path because it manages its own model runtime and cache behavior.

## Apple Foundation Models Boundary

AFM is isolated behind:

- `On-Device_LLM_Chat/LLMGenerator.swift`
- `OnDeviceLLMGenerator`
- `FoundationModels.LanguageModelSession`

This path does not use `GenerateParameters`, `KVCache`, `KVCacheCompressionMode`, MLX cache classes, or vendored MLX attention helpers. RotorQuant is not threaded through AFM session creation.

## User-Facing MLX Switches

RotorQuant settings live at:

- `On-Device_LLM_Chat/SettingsSheet.swift`
  - `mlxRotorQuantInfoMessage`
  - `mlxRotorQuantExperimentalTitle`
  - `mlxRotorQuantExperimentalMessage`
  - `mlxRotorQuantAccessibilityHint`
  - toggle label `RotorQuant (MLX Only)`
  - tappable `(E)` badge that presents the early beta warning
  - `settings.mlxEnableRotorQuant`
- `On-Device_LLM_Chat/AppSettings.swift`
  - `mlxEnableRotorQuant`
- `On-Device_LLM_Chat/ModelBackendBridge.swift`
  - `UserDefaults.mlxEnableRotorQuant`

Fresh installs default to RotorQuant enabled for persistent MLX chats. The older generic `mlxEnableKVCacheQuantization` preference no longer controls RotorQuant defaults.

## App-Level Compression Configuration

`On-Device_LLM_Chat/MLXModelManager.swift` defines:

- `CacheCompressionMode.none`
- `CacheCompressionMode.rotorQuant`
- `MLXCachePolicy.persistentRotorQuant`

Default persistent MLX generation emits `.rotorQuant(RotorQuantConfiguration(... variant: .iso))` when the RotorQuant preference is enabled. If RotorQuant is disabled or unsupported, the app uses dense persistent KV rather than falling back to legacy quantized KV. The old TurboQuant runtime mode and configuration API have been removed from the app path.

App telemetry and cache-policy labels now use RotorQuant-specific names such as `persistent-rotorquant`; stale app-level `quantizedKVStart` tuning/metadata was removed because RotorQuant is selected through `cacheCompression` at construction time, while the vendored legacy quantized mode remains only for explicit evaluator/benchmark comparison.

Low-memory or bounded-cache decisions still use rotating caches rather than persistent compression. That means RotorQuant is default for non-AFM MLX persistent caches, while low-memory eviction remains bounded sliding-window behavior.

## Vendored Generation API

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` exposes:

- `RotorQuantVariant` with `.iso`, `.planar`, `.clifford`
- `RotorQuantConfiguration`
- `KVCacheCompressionMode.rotorQuant(RotorQuantConfiguration)`

`RotorQuantConfiguration.configurationForLayer(_:)` derives deterministic per-layer seeds.

## Cache Construction

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` constructs caches through:

- `makePromptCache(model:parameters:)`
- `makeLayerKVCache(parameters:layerIndex:)`

Construction order:

1. `maxKVSize` returns `RotatingKVCache`.
2. `.rotorQuant` returns `RotorQuantKVCache(configurationForLayer(layerIndex))`.
3. Otherwise, `KVCacheSimple` is used.

`maybeApplyKVCacheCompression` treats RotorQuant as a construction-time strategy, not a dynamic post-hoc conversion.

## Attention Dispatch

Current attention update path:

- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift`
  - `attentionWithCacheUpdate(...)`

Dispatch order:

1. `AttentionCapableKVCache.attention(...)`
2. `QuantizedKVCacheProtocol.updateQuantized(...)`
3. regular `KVCache.update(...)` plus `MLXFast.scaledDotProductAttention`

`RotorQuantKVCache` conforms to `AttentionCapableKVCache`. It ingests the new K/V slice and uses fused `MLXFast.metalKernel` calls for supported IsoQuant single-token decode calls. Short compressed prefixes use a one-kernel path. Longer compressed prefixes use a block-parallel compressed-prefix kernel plus a reduce kernel that combines compressed block partials with the exact/deferred tail. The block and reduce kernels are KV-head parallel and compute all GQA repeats inside the same workgroup, avoiding repeated K/V decode or exact-tail rotation for the same KV head. The fused paths score compressed K directly from packed centroid indices and FP32 norms, accumulate compressed V in value-rotation space, apply the mandatory inverse value rotation to the final output, and include the exact/deferred tail in the same streaming softmax. Dense fallback remains for prefill, multi-token queries, unsupported variants, unsupported masks, and validation comparisons.

The fused kernel is launched directly from `RotorQuantKVCache.attention`; this attention path is not inside `MLX.compile()`. Live compressed/exact lengths and the active compressed block count are scalar dynamic kernel inputs. They are not template arguments and do not change shape every token. Block-parallel partial outputs are sized from compressed storage capacity, so specialization can change when compressed storage reallocates but not at every attention-block boundary.

## RotorQuant Storage Path

Implementation:

- `RotorQuantRotationFactory`
- `RotorQuantMSEQuantizer`
- `RotorQuantKVCache`
- `KVCacheSimple.toRotorQuantized(...)`

Shared helpers:

- `SeededGaussianRandom`
- `CentroidQuantizationMath`
- `QuantizationParameterStore`
- `PackedCentroidIndices`

Data flow:

1. Convert vectors to float32.
2. Store float32 vector norms.
3. Normalize by safe norm.
4. Apply block-diagonal rotation.
5. Quantize rotated coordinates to Lloyd-Max centroid indices.
6. Pack logical uint8 indices to the configured bit width.
7. On read, unpack indices, map centroids, apply inverse rotation, restore norms, and cast to the original dtype.

Supported variants:

- IsoQuant: 4D quaternion blocks, default when the head dimension tiles by 4.
- PlanarQuant: 2D Givens blocks, fallback for even dimensions that do not tile by 4.
- Clifford Cl(3,0): 3D rotor sandwich, available when requested; it stores extracted 3D vector coordinates after the sandwich rotation and is not the Qwen3.5 default.

For `head_dim = 256`, IsoQuant tiles exactly: 256 coordinates become 256 logical indices, packed as 96 bytes at 3 bits for K and 64 bytes at 2 bits for V. No padded 128-only path inflates storage.

Compressed-row RAM math:

- `head_dim = 128`: `88` bytes per KV head token versus `512` dense FP16/BF16 bytes, `82.8%` lower.
- `head_dim = 256`: `168` bytes per KV head token versus `1024` dense FP16/BF16 bytes, `83.6%` lower.
- Qwen3.5 with 4 KV heads and 8 full-attention layers: about `5.25 KiB/token` compressed rows versus `32 KiB/token` dense full-attention KV rows. Linear/DeltaNet state is intentionally excluded.

## Deferred Quantization

`RotorQuantKVCache.ingest(keys:values:)` treats multi-token updates as prefill and appends them to exact storage. Single-token decode inserts keep a bounded exact-tail slack of `min(16, exactBufferSize / 8)` tokens (minimum 1), then flush the oldest rows back to `exactBufferSize` in a small batch. This preserves deferred quantization while avoiding a tiny quantization graph launch on every decode token after the exact tail fills.

This gives:

- prefill K/V stay exact while attention is computed
- decode inserts quantize rows as they leave the exact buffer
- compressed and exact rows are combined in cache order

## Serialization and Migration

Prompt cache persistence:

- `savePromptCache(url:cache:metadata:)` saves `RotorQuantKVCache` class names for new RotorQuant caches.
- `loadPromptCache(url:)` loads `RotorQuantKVCache` directly.
- `loadPromptCache(url:)` also recognizes historical `"TurboQuantKVCache"` cache-class metadata and migrates that serialized state into `KVCacheSimple`.
- `loadPromptCache(url:)` validates RotorQuant metadata values and state shape before assigning into cache setters, so malformed RotorQuant prompt-cache files throw `KVCacheError` instead of crashing through `preconditionFailure`.

RotorQuant serialized state:

- compressed key indices
- key norms
- compressed value indices
- value norms
- optional exact keys
- optional exact values
- metadata: step, offset, bits, seed, block size, requested/effective variant, dimensions, dtype, compressed/exact counts, exact buffer size

Historical compressed cache state is not format-identical to RotorQuant. The local migration guarantee is read compatibility without crashing; migrated legacy state resumes as a regular uncompressed KV cache, and new saves use RotorQuant.

## Hybrid Layer Boundaries

Qwen3.5 text:

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
  - linear layers return `MambaCache()`
  - full-attention layers call `makeLayerKVCache(parameters:layerIndex:)`

Qwen3.5 VLM:

- `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift`
  - linear layers return `MambaCache()`
  - full-attention layers call `makeLayerKVCache(parameters:layerIndex:)`

RotorQuant only appears where full-attention layers request a real KV cache. DeltaNet/recurrent linear layers remain `MambaCache` and are not compressed.

`testRotorQuantQwen35HybridCacheOnlyCompressesFullAttentionLayers` now compares dense and RotorQuant cache construction layer-by-layer: full-attention layers switch from `KVCacheSimple` to `RotorQuantKVCache`, while every linear/DeltaNet layer remains `MambaCache` with matching empty state and metadata.

## Partial Rotary and MRoPE Sites

Qwen3.5 text:

- `partialRotaryFactor`
- `ropeParameters["mrope_section"]`
- `ropeDims = Int(Float(headDim) * args.partialRotaryFactor)`

Qwen3.5 VLM:

- `partialRotaryFactor`
- `mrope_section`
- `rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)`

Compression receives the projected post-RoPE key vectors in the cache update path. RotorQuant compresses the full key/value head vector and does not assume full RoPE coverage or only process the rotary prefix.

## Test Surfaces

Vendored package tests in `Vendor/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift` cover:

- RotorQuant round-trip reconstruction at head dimensions 128 and 256
- 256-dim IsoQuant packed width, with no K-buffer inflation
- deferred quantization: prefill exact, decode flushes to compressed
- GQA attention path at head dimension 256
- fused IsoQuant decode parity with fallback, including exact-tail and block-parallel paths
- Qwen3.5 hybrid cache creation: full-attention layers use RotorQuant and linear layers remain `MambaCache`
- historical compressed prompt-cache migration into `KVCacheSimple`

App tests in `On-Device_LLM_ChatTests/On_Dvice_LLM_ChatTests.swift` cover:

- RotorQuant settings copy
- experimental `(E)` beta-warning copy
- default-on preference
- structured generation configuration emits RotorQuant with IsoQuant defaults
- media profile uses smaller exact/dequant block settings

Benchmark helpers in `Vendor/mlx-swift-lm/Tests/Benchmarks/RotorQuantBenchmarks.swift` print RAM, prefill, and decode measurements when invoked from a benchmark run. They are registered as Swift Testing tests and are skipped unless the existing `RUN_BENCHMARKS` opt-in environment variable is set.

The executable evaluator in `Vendor/mlx-swift-lm/Tools/RotorQuantEval/main.swift` runs dense, legacy MLX quantized KV, and RotorQuant scenarios against a real Hub model. `ROTORQUANT_EVAL_SCENARIOS` can restrict runs to comma-separated scenario keys such as `dense`, `legacy`, or `rotor` for faster tuning loops, and the evaluator now preserves the order given in that variable so order-bias checks are possible. Current measured runs use the project model links:

- `mlx-community/Qwen3.5-2B-4bit`
- `Spakie/Qwen3.5-4B-MLX-4bit-hybrid`

With ISO k3/v2, exact buffer 128, 192 PPL tokens, and 128 generated tokens, the current fused path keeps PPL within the dense baseline delta and saves KV memory, but still decodes slower than the dense path:

- 2B, 512 prompt: RotorQuant `36.7 tok/s`, dense `55.5 tok/s`, legacy `51.1 tok/s`; RotorQuant KV `12.7 MiB` versus dense `20.9 MiB`.
- 2B, 2048 prompt with default block size 128: RotorQuant `38.4 tok/s`, dense `58.8 tok/s`, legacy `54.3 tok/s`; RotorQuant KV `18.3 MiB` versus dense `54.6 MiB`.
- 2B, 8192 prompt with block-parallel path: RotorQuant `35.6 tok/s`, dense `51.7 tok/s`, legacy `47.0 tok/s`; RotorQuant KV `40.4 MiB` versus dense `189.6 MiB`.
- mixed 3/6 4B, 2048 prompt same-run check: RotorQuant `28.4 tok/s`, dense `42.9 tok/s`, legacy `39.6 tok/s`; RotorQuant KV `48.1 MiB` versus dense `145.0 MiB` and legacy `58.8 MiB`.

The decode path is not inside `MLX.compile()`. Qwen3.5 reaches RotorQuant through `attentionWithCacheUpdate` and `RotorQuantKVCache.attention`, which invokes custom Metal kernels directly. Compressed/exact token counts and active compressed block count are scalar kernel inputs. The block-parallel partial tensors are now sized by compressed storage capacity rather than active block count, so their shapes change only when the cache backing storage grows instead of every attention-block boundary.

After the shape-stabilization patch, rotor-only 2048/64 generated-token checks still preserved PPL and memory but did not close the speed gap: 2B was `35.9 tok/s`, `18.3 MiB`, PPL `3.7105`, with `rotor=6 mamba=18`; mixed 4B was `26.0 tok/s`, `48.1 MiB`, PPL `3.9146`, with `rotor=8 mamba=24`.

Batched exact flushing then removed most of the tiny decode-insert overhead. In a same-run 2048/64 check, 2B RotorQuant reached `55.2 tok/s` versus dense `51.8` and legacy `44.1`, with KV `18.3 MiB` and PPL `3.6990`. The mixed 4B primary case improved but still missed the no-regression gate in dense-first order: RotorQuant was `33.1 tok/s` versus dense `38.6` and legacy `35.2`, with KV `48.1 MiB`, PPL `3.9280`, and cache summary `rotor=8 mamba=24`.

At the requested 2048/128 generation length, mixed 4B results are order-sensitive on the current Mac and are not strong enough for final completion. Dense-first order produced dense `35.0 tok/s`, legacy `34.7`, RotorQuant `29.5`. Rotor-first order produced RotorQuant `28.4`, dense `27.2`, legacy `27.6`; a later post-rollback rotor-first check produced RotorQuant `39.3`, dense `42.8`, legacy `39.6`. The same PPL/KV pattern held: RotorQuant PPL `3.9280`, dense `3.9246`, legacy `3.9687`, RotorQuant KV `48.1 MiB`. A wider 32-token flush slack was tested and rejected for 128-token generation because RotorQuant dropped to `25.0 tok/s` in the dense-first run; the conservative 16-token bounded slack remains.

Exact-buffer spot checks keep the text default at `128`. A 32-token exact tail regressed 2B PPL to `3.9734`; after the float16 block-partial handoff, a 64-token exact tail improved 2B PPL/memory (`50.3 tok/s`, `17.6 MiB`, PPL `3.6716`) but slowed the primary mixed 4B path (`36.1 tok/s`, `46.5 MiB`, PPL `3.9248`) versus exact 128 (`37.3 tok/s`, `48.1 MiB`, PPL `3.9280`). A 256-token exact tail was rejected because 2B PPL regressed to the dense value (`3.7107`) while memory rose to `19.5 MiB`.

Attention-block tuning after GQA-sharing and float16 partials also keeps the text default at `128`: mixed 4B 2048/128 was `37.3 tok/s` at block 128 and `37.5` at block 256, while 2B was `50.7` at block 128 and `51.0` at block 256. The tiny/noisy gain does not justify changing the app default. A direct k3/v2 packed-index micro-specialization also failed to improve the 2B 2048 decode check and was rechecked on mixed 4B 2048/128 with no positive evidence (`34.2 tok/s`, PPL `3.9280`, KV `48.1 MiB`), so it was not retained. This means the block-parallel kernel removed the severe long-context collapse from the first sweep and GQA-sharing helped the mixed 4B shape, but the original dense no-regression speed gate is still unmet. Current evidence points at compressed-block work, reduction traffic, and possible hot decode layout changes as the next targets while preserving RotorQuant's normalized rotated-centroid design.

Forcing the single fused decode kernel at mixed 4B 2048/128 by setting `attentionBlockTokens = 4096` was also worse (`17.9 tok/s`, PPL `3.9280`, KV `48.1 MiB`), so the block-parallel path is not being applied too early at the primary 2k context length.

Moving exact-tail attention into extra block partials was also rejected. Query-dtype exact partials gave a small 4B speed signal but worsened 2B PPL; separate float32 exact partials did not keep a clear speed win and produced mixed 4B PPL `3.9378` in the quick check. The retained path keeps exact-tail work in the reducer to preserve the earlier PPL envelope.

One exact-tail shortcut was explicitly rejected: accumulating exact V rows in original value space while inverse-rotating only the compressed accumulator regressed mixed 4B PPL to `7.0626`. Restoring the existing exact V rotate-plus-inverse path returned PPL to `3.9280`, so exact-tail value rotation remains part of the current fused design until a stronger model-level correctness test proves a replacement.

A second exact-tail shortcut was also rejected: pre-rotating exact K/V into float32 side buffers preserved PPL but slowed mixed 4B 2048/128 RotorQuant decode to `29.7 tok/s` versus dense `37.7` and legacy `34.4`, because the rotation cost moved into decode insertion. After rollback, a RotorQuant-only sanity check returned to PPL `3.9280`, `36.4 tok/s`, `48.1 MiB` KV, and `rotor=8 mamba=24`. The remaining code change from that investigation is a reload-order fix: when prompt-cache loading sets `state` before `metaState`, `RotorQuantKVCache.metaState` now rebuilds quantizers after applying saved bits/seed/variant so non-default layer seeds are honored.

The retained speed improvement after that is the block/reduce handoff format: compressed block partial values are now written in the query dtype while block maxima and sums stay float32. This preserves RotorQuant's packed storage and final inverse-rotation math, but halves the partial-value traffic between the compressed-block kernel and reducer. On the mixed 4B 2048/128 same-run check, RotorQuant stayed at PPL `3.9280` and decoded at `37.3 tok/s` versus dense `40.2` and legacy `34.9`, with KV memory unchanged at `48.1 MiB`. The 2B 2048/128 same-run check decoded at `49.8 tok/s` versus dense `55.8` and legacy `51.7`, with RotorQuant KV `18.3 MiB` versus dense `54.6 MiB` and legacy `22.3 MiB`; PPL was `3.6990` versus dense `3.7107` and legacy `3.6923`.
