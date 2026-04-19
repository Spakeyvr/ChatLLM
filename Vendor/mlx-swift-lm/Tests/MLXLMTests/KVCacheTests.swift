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
    let cache = TurboQuantKVCache(
        configuration: TurboQuantConfiguration(
            keyTotalBits: 3,
            valueBits: 2,
            seed: 42,
            exactBufferSize: 0,
            attentionBlockTokens: 64
        )
    )
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
    #expect(cache.state.count == 6)
    #expect(output.shape == [1, 4, 1, 64])
}

@Test
func testTurboQuantMaskSentinel() {
    let cache = QuantizedKVCache(groupSize: 2, bits: 8)
    let queries = MLXArray([Float(10), 0]).reshaped([1, 1, 1, 2]).asType(.float16)
    let keys = MLXArray([Float(10), 0, 0, 10]).reshaped([1, 1, 2, 2]).asType(.float16)
    let values = MLXArray([Float(1), 2, 100, 200]).reshaped([1, 1, 2, 2]).asType(.float16)
    let maskArray = MLXArray([true, false]).reshaped([1, 1, 1, 2])
    let (quantizedKeys, quantizedValues) = cache.updateQuantized(keys: keys, values: values)

    let quantizedOutput = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        scale: 1.0,
        mask: .array(maskArray),
        groupSize: cache.groupSize,
        bits: cache.bits,
        mode: cache.mode
    )
    let referenceOutput = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: 1.0,
        mask: .array(maskArray)
    )

    let elementCount = quantizedOutput.shape.reduce(1, *)
    let quantizedValuesFlat = quantizedOutput.reshaped([elementCount]).asArray(Float.self)
    let referenceValuesFlat = referenceOutput.reshaped([elementCount]).asArray(Float.self)
    let maxDiff = zip(quantizedValuesFlat, referenceValuesFlat)
        .map { abs($0 - $1) }
        .max() ?? .infinity

    #expect(maxDiff < 0.05)
}

@Test
func testTurboQuantCompressedStateRemainsPacked() {
    let cache = TurboQuantKVCache(
        configuration: TurboQuantConfiguration(
            keyTotalBits: 3,
            valueBits: 2,
            seed: 42,
            exactBufferSize: 0,
            attentionBlockTokens: 64
        )
    )
    let keys = MLXArray.ones([1, 2, 4, 64], dtype: .float16)
    let values = MLXArray.ones([1, 2, 4, 64], dtype: .float16)

    _ = cache.update(keys: keys, values: values)

    let serialized = cache.state
    let runtime = cache.innerState()

    #expect(serialized.count == 6)
    #expect(runtime.count == 6)
    #expect(serialized[0].shape == [1, 2, 4, 16])
    #expect(serialized[2].shape == [1, 2, 4, 8])
    #expect(serialized[4].shape == [1, 2, 4, 16])
    #expect(runtime[0].shape == serialized[0].shape)
    #expect(runtime[2].shape == serialized[2].shape)
    #expect(runtime[4].shape == serialized[4].shape)
}

@Test
func testTurboQuantMetaStatePersistsStructuredConfiguration() {
    let cache = TurboQuantKVCache(
        configuration: TurboQuantConfiguration(
            keyTotalBits: 3,
            valueBits: 2,
            seed: 77,
            exactBufferSize: 16,
            attentionBlockTokens: 32,
            qjlProjectionDimension: 40
        )
    )

    #expect(cache.metaState.count == 13)
    #expect(cache.metaState[2] == "3")
    #expect(cache.metaState[3] == "2")
    #expect(cache.metaState[5] == "32")
    #expect(cache.metaState[6] == "40")
}

@Test
func testTurboQuantSupportsNonPowerOfTwoHeadDimensions() {
    let cache = TurboQuantKVCache(
        configuration: TurboQuantConfiguration(
            keyTotalBits: 3,
            valueBits: 2,
            seed: 11,
            exactBufferSize: 0,
            attentionBlockTokens: 32
        )
    )
    let queries = MLXArray.ones([1, 2, 1, 80], dtype: .float16)
    let keys = MLXArray.ones([1, 2, 3, 80], dtype: .float16)
    let values = MLXArray.ones([1, 2, 3, 80], dtype: .float16)

    let output = attentionWithCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: 1.0 / sqrt(80.0),
        mask: .causal
    )

    #expect(output.shape == [1, 2, 1, 80])
    #expect(cache.offset == 3)
}
