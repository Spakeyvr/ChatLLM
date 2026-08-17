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
  normalization.
