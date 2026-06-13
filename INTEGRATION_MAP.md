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

Fresh installs default to RotorQuant enabled for persistent MLX chats. The older generic `mlxEnableKVCacheQuantization` preference is read only when the new RotorQuant key is absent.

## App-Level Compression Configuration

`On-Device_LLM_Chat/MLXModelManager.swift` defines:

- `CacheCompressionMode.none`
- `CacheCompressionMode.legacyQuantized`
- `CacheCompressionMode.rotorQuant`

Default persistent MLX generation emits `.rotorQuant(RotorQuantConfiguration(... variant: .iso))` when the RotorQuant preference is enabled. The old TurboQuant runtime mode and configuration API have been removed.

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

`RotorQuantKVCache` conforms to `AttentionCapableKVCache`. It ingests the new K/V slice and uses fused `MLXFast.metalKernel` calls for supported IsoQuant single-token decode calls. Short compressed prefixes use a one-kernel path. Longer compressed prefixes use a block-parallel compressed-prefix kernel plus a reduce kernel that combines compressed block partials with the exact/deferred tail. The fused paths score compressed K directly from packed centroid indices and FP32 norms, accumulate compressed V in value-rotation space, apply the mandatory inverse value rotation to the final output, and include the exact/deferred tail in the same streaming softmax. Dense fallback remains for prefill, multi-token queries, unsupported variants, unsupported masks, and validation comparisons.

The fused kernel is launched directly from `RotorQuantKVCache.attention`; this attention path is not inside `MLX.compile()`. Live compressed/exact lengths are scalar dynamic kernel inputs. They are not template arguments and do not change shape every token. Storage capacities are template arguments, so specialization can change when compressed storage reallocates.

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

`RotorQuantKVCache.ingest(keys:values:)` treats multi-token updates as prefill and appends them to exact storage. Single-token decode inserts flush only the oldest exact rows needed to preserve `exactBufferSize`.

This gives:

- prefill K/V stay exact while attention is computed
- decode inserts quantize rows as they leave the exact buffer
- compressed and exact rows are combined in cache order

## Serialization and Migration

Prompt cache persistence:

- `savePromptCache(url:cache:metadata:)` saves `RotorQuantKVCache` class names for new RotorQuant caches.
- `loadPromptCache(url:)` loads `RotorQuantKVCache` directly.
- `loadPromptCache(url:)` also recognizes historical `"TurboQuantKVCache"` cache-class metadata and migrates that serialized state into `KVCacheSimple`.

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

Benchmark helpers in `Vendor/mlx-swift-lm/Tests/Benchmarks/RotorQuantBenchmarks.swift` print RAM, prefill, and decode measurements when invoked from a benchmark run.

The executable evaluator in `Vendor/mlx-swift-lm/Tools/RotorQuantEval/main.swift` runs dense, legacy MLX quantized KV, and RotorQuant scenarios against a real Hub model. Current measured runs use the project model links:

- `mlx-community/Qwen3.5-2B-4bit`
- `Spakie/Qwen3.5-4B-MLX-mixed36`

With ISO k3/v2, exact buffer 128, 192 PPL tokens, and 128 generated tokens, the current fused path keeps PPL within the dense baseline delta and saves KV memory, but still decodes slower than the dense path:

- 2B, 512 prompt: RotorQuant `36.7 tok/s`, dense `55.5 tok/s`, legacy `51.1 tok/s`; RotorQuant KV `12.7 MiB` versus dense `20.9 MiB`.
- 2B, 2048 prompt with default block size 128: RotorQuant `38.4 tok/s`, dense `58.8 tok/s`, legacy `54.3 tok/s`; RotorQuant KV `18.3 MiB` versus dense `54.6 MiB`.
- 2B, 8192 prompt with block-parallel path: RotorQuant `35.6 tok/s`, dense `51.7 tok/s`, legacy `47.0 tok/s`; RotorQuant KV `40.4 MiB` versus dense `189.6 MiB`.
- mixed 3/6 4B, 2048 prompt spot check: RotorQuant `20.0 tok/s`, dense `28.7 tok/s`, legacy `16.3 tok/s`; RotorQuant KV `48.1 MiB` versus dense `145.0 MiB`.

This means the block-parallel kernel removed the severe long-context collapse from the first sweep, but the original no-regression speed gate is still unmet. Current evidence points at packed-index unpack, rotation work, and reduction overhead as the next targets while preserving RotorQuant's normalized rotated-centroid design.
