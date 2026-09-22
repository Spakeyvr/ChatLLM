# ChatLLM MLX Swift fork

This directory vendors `mlx-swift` 0.30.6 (upstream revision
`6ba4827fb82c97d012eec9ab4b2de21f85c3b33d`) so ChatLLM can carry narrowly
scoped CoreSimulator Metal compatibility fixes.

The fork preserves the physical-device MLX path. Simulator-specific changes:

- derive the host Apple GPU architecture before MLX initializes;
- use standalone Metal buffers because CoreSimulator does not implement MLX's
  `MTLHeap` allocation path;
- size generic dispatches to the pipeline's reported threadgroup limit;
- use unfused GPU attention where the fused kernel requires 1024 threads; and
- precompile the float16 Qwen embedding lookup used after simulator weight
  normalization;
- disable NAX kernels, whose MetalPerformancePrimitives headers are unavailable
  to CoreSimulator's runtime Metal compiler.

ChatLLM's simulator loader widens only BF16 floating parameters to FP32 on CPU
before inference. The quantized integer weight storage remains packed. This
avoids fatal BF16 runtime-kernel compilation in CoreSimulator; physical devices
retain native BF16 and NAX paths.

Xcode 27 compatibility patches:

- async device/stream/state helpers use `nonisolated(nonsending)` closures for
  the non-deprecated `TaskLocal.withValue` overload;
- optional module/parameter wrappers explicitly preserve nested Optional values;
- Metal template recursion uses C++14 specialization, and constant attention
  branches use valid C++14 `if` statements;
- shared Metal constants are annotated as potentially unused in individual JIT
  compilation units; generated copies are kept in sync;
- fmt defers its standard string-view constraint to conversion, avoiding an
  unsupported `std::char_traits` instantiation for its internal UTF-8 enum;
- only Cmlx opts out of Xcode's additional `-Wshorten-64-to-32` diagnostic. MLX's
  upstream dimension API mixes 32-bit dimensions and 64-bit element counts.
  This preserves upstream arithmetic; it does not repair or audit every narrowing
  conversion. Other diagnostics, including deprecations, remain enabled.

The sibling `mlx-swift-lm` patches use the current attention-mask API, remove an
unused vision binding, and decode tool JSON through the existing Sendable
`JSONValue` representation instead of passing unconstrained `Any` across actors.

Regression coverage lives in `ChatLLMTests+LFM.swift`; the opt-in real model tests
and their simulator setup are documented in the root README. Reassess these
patches against upstream before updating either dependency.
