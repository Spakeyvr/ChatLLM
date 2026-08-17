// Simulator-safe precompiled kernel used by Qwen token embedding lookups.

#include <metal_stdlib>

#include "bf16.h"

using namespace metal;

[[kernel]] void gather_frontfloat16_int32_int_2(
    const device half* src,
    const device int32_t* indices,
    device half* out,
    const constant int64_t& stride,
    const constant int& size,
    uint2 index [[thread_position_in_grid]]) {
  int idx = indices[index.y];
  if (idx < 0) {
    idx += size;
  }

  int srcIndex = int(stride) * idx;
  int outIndex = int(stride) * int(index.y);
  int sliceIndex = 2 * int(index.x);
  for (int i = 0; i < 2 && sliceIndex < stride; ++i, ++sliceIndex) {
    out[outIndex + sliceIndex] = src[srcIndex + sliceIndex];
  }
}
