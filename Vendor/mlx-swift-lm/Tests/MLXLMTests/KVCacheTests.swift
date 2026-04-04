import Foundation
import MLX
import MLXLMCommon
import Testing

@Test(
    .serialized,
    arguments: [
        ({ KVCacheSimple() }),
        ({ RotatingKVCache(maxSize: 32) }),
        ({ QuantizedKVCache() }),
        ({ TurboQuantKVCache() }),
        ({ ChunkedKVCache(chunkSize: 16) }),
        ({ ArraysCache(size: 2) }),
        ({ MambaCache() }),
    ])
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

@Test
func testTurboQuantAttentionPath() {
    let cache = TurboQuantKVCache(bits: 3)
    let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float16)
    let keys = MLXArray.ones([1, 4, 2, 64], dtype: .float16)
    let values = MLXArray.ones([1, 4, 2, 64], dtype: .float16)

    let output = attentionWithCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: 1.0 / sqrt(64.0),
        mask: .causal
    )

    #expect(cache.offset == 2)
    #expect(cache.state.count == 2)
    #expect(output.shape == [1, 4, 1, 64])
}
