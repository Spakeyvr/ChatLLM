import Foundation
import MLXLLM
import MLX
@testable import MLXLMCommon
import Testing

typealias KVCacheCreator = @Sendable () -> any KVCache

/// MLX kernels dispatch to a shared Metal command queue, which is not safe to
/// drive from Swift Testing's default parallel execution -- concurrent cases
/// trip `A command encoder is already encoding to this command buffer`.
@Suite(.serialized)
struct KVCacheTests {
    @Test(
        .serialized,
        arguments: [
            ({ KVCacheSimple() } as KVCacheCreator),
            ({ RotatingKVCache(maxSize: 32) } as KVCacheCreator),
            ({ QuantizedKVCache() } as KVCacheCreator),
            ({ RotorQuantKVCache() } as KVCacheCreator),
            ({ ChunkedKVCache(chunkSize: 16) } as KVCacheCreator),
            ({ ArraysCache(size: 2) } as KVCacheCreator),
            ({ MambaCache() } as KVCacheCreator),
        ])
    func testCacheSerialization(creator: KVCacheCreator) async throws {
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

    private func patternedArray(shape: [Int], scale: Float = 0.01) -> MLXArray {
        let count = shape.reduce(1, *)
        let values = (0 ..< count).map { index in
            sin(Float(index) * 0.17) * scale + cos(Float(index) * 0.07) * scale * 0.5
        }
        return MLXArray(values, shape).asType(.float16)
    }

    private func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let lhsValues = lhs.asType(.float32).reshaped([lhs.shape.reduce(1, *)]).asArray(Float.self)
        let rhsValues = rhs.asType(.float32).reshaped([rhs.shape.reduce(1, *)]).asArray(Float.self)
        return zip(lhsValues, rhsValues).map { abs($0 - $1) }.max() ?? .infinity
    }

    @Test(arguments: [128, 256])
    func testRotorQuantRoundTripReconstruction(headDimension: Int) {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 8,
                valueBits: 8,
                seed: 42,
                exactBufferSize: 0,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let keys = patternedArray(shape: [1, 2, 1, headDimension])
        let values = patternedArray(shape: [1, 2, 1, headDimension], scale: 0.015)

        let (restoredKeys, restoredValues) = cache.update(keys: keys, values: values)

        #expect(restoredKeys.shape == keys.shape)
        #expect(restoredValues.shape == values.shape)
        #expect(maxAbsoluteDifference(restoredKeys, keys) < 0.01)
        #expect(maxAbsoluteDifference(restoredValues, values) < 0.01)
    }

    @Test
    func testRotorQuantIsoTilingDoesNotInflateHeadDimension256() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 7,
                exactBufferSize: 0,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let keys = patternedArray(shape: [1, 4, 1, 256])
        let values = patternedArray(shape: [1, 4, 1, 256])

        _ = cache.update(keys: keys, values: values)
        let state = cache.state

        #expect(state.count == 4)
        #expect(state[0].shape == [1, 4, 1, 96])
        #expect(state[2].shape == [1, 4, 1, 64])
        #expect(cache.metaState[7] == "iso")
    }

    @Test
    func testRotorQuantCliffordStoresVectorCoordinatesNotFullMultivectors() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 7,
                exactBufferSize: 0,
                attentionBlockTokens: 64,
                variant: .clifford
            )
        )
        let keys = patternedArray(shape: [1, 1, 1, 6])
        let values = patternedArray(shape: [1, 1, 1, 6])

        _ = cache.update(keys: keys, values: values)
        let state = cache.state

        #expect(state.count == 4)
        #expect(state[0].shape == [1, 1, 1, 3])
        #expect(state[2].shape == [1, 1, 1, 2])
        #expect(cache.metaState[7] == "clifford")
    }

    @Test
    func testRotorQuantDeferredQuantizationKeepsPrefillExactThenCompressesOnDecode() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 42,
                exactBufferSize: 2,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let prefillKeys = patternedArray(shape: [1, 1, 4, 128])
        let prefillValues = patternedArray(shape: [1, 1, 4, 128])
        let decodeKeys = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)
        let decodeValues = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)

        _ = cache.update(keys: prefillKeys, values: prefillValues)
        #expect(cache.state.count == 2)
        #expect(cache.offset == 4)

        _ = cache.update(keys: decodeKeys, values: decodeValues)
        #expect(cache.state.count == 6)
        #expect(cache.metaState[11] == "3")
        #expect(cache.metaState[12] == "2")
        #expect(cache.offset == 5)

        let runtimeArrays = cache.innerState()
        #expect(runtimeArrays.count == 6)
        #expect(runtimeArrays[4].dim(2) == 3)
        #expect(runtimeArrays[5].dim(2) == 3)
    }

    @Test
    func testRotorQuantReleasesDensePrefillCapacityAfterDeferredFlush() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 42,
                exactBufferSize: 16,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let prefillKeys = patternedArray(shape: [1, 1, 256, 128])
        let prefillValues = patternedArray(shape: [1, 1, 256, 128])
        let decodeKeys = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)
        let decodeValues = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)

        _ = cache.update(keys: prefillKeys, values: prefillValues)
        #expect(cache.innerState()[0].dim(2) == 256)

        _ = cache.update(keys: decodeKeys, values: decodeValues)
        let runtimeArrays = cache.innerState()

        #expect(runtimeArrays.count == 6)
        #expect(runtimeArrays[4].dim(2) == 18)
        #expect(runtimeArrays[5].dim(2) == 18)
        #expect(cache.metaState[12] == "16")
    }

    @Test
    func testRotorQuantDeferredQuantizationBatchesTinyDecodeFlushes() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 42,
                exactBufferSize: 16,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let prefillKeys = patternedArray(shape: [1, 1, 16, 128])
        let prefillValues = patternedArray(shape: [1, 1, 16, 128])
        let decodeKeys = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)
        let decodeValues = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)

        _ = cache.update(keys: prefillKeys, values: prefillValues)
        _ = cache.update(keys: decodeKeys, values: decodeValues)
        #expect(cache.state.count == 2)
        #expect(cache.metaState[11] == "0")
        #expect(cache.metaState[12] == "17")

        _ = cache.update(keys: decodeKeys, values: decodeValues)
        #expect(cache.state.count == 2)
        #expect(cache.metaState[11] == "0")
        #expect(cache.metaState[12] == "18")

        _ = cache.update(keys: decodeKeys, values: decodeValues)
        #expect(cache.state.count == 6)
        #expect(cache.metaState[11] == "3")
        #expect(cache.metaState[12] == "16")
        #expect(cache.offset == 19)
    }

    @Test
    func testRotorQuantCompressedCapacityAvoidsDoublingSlack() {
        #expect(
            RotorQuantKVCache.nextStorageCapacity(current: 4_608, required: 4_609, step: 256)
                == 6_912
        )
        #expect(
            RotorQuantKVCache.nextStorageCapacity(current: 0, required: 64, step: 256)
                == 512
        )
        #expect(
            RotorQuantKVCache.nextStorageCapacity(current: 512, required: 800, step: 256)
                == 1_024
        )
    }

    @Test
    func testRotorQuantInitialExactStorageIncludesFlushSlack() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 42,
                exactBufferSize: 16,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let keys = patternedArray(shape: [1, 1, 16, 128])
        let values = patternedArray(shape: [1, 1, 16, 128])

        _ = cache.update(keys: keys, values: values)

        let runtimeArrays = cache.innerState()
        #expect(runtimeArrays.count == 2)
        #expect(runtimeArrays[0].dim(2) == 18)
        #expect(runtimeArrays[1].dim(2) == 18)
    }

    @Test
    func testRotorQuantTrimDropsNewestTokensAcrossCompressedAndExactRows() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 8,
                valueBits: 8,
                seed: 42,
                exactBufferSize: 2,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let prefillKeys = patternedArray(shape: [1, 1, 6, 128])
        let prefillValues = patternedArray(shape: [1, 1, 6, 128], scale: 0.015)
        let decodeKeys = patternedArray(shape: [1, 1, 1, 128], scale: 0.02)
        let decodeValues = patternedArray(shape: [1, 1, 1, 128], scale: 0.022)
        let appendedKeys = patternedArray(shape: [1, 1, 1, 128], scale: 0.024)
        let appendedValues = patternedArray(shape: [1, 1, 1, 128], scale: 0.026)

        _ = cache.update(keys: prefillKeys, values: prefillValues)
        _ = cache.update(keys: decodeKeys, values: decodeValues)
        #expect(cache.offset == 7)
        #expect(cache.metaState[11] == "5")
        #expect(cache.metaState[12] == "2")

        #expect(cache.trim(3) == 3)
        #expect(cache.offset == 4)

        let (restoredKeys, restoredValues) = cache.update(
            keys: appendedKeys,
            values: appendedValues
        )

        #expect(restoredKeys.shape == [1, 1, 5, 128])
        #expect(
            maxAbsoluteDifference(
                restoredKeys[.ellipsis, ..<4, 0...],
                prefillKeys[.ellipsis, ..<4, 0...]
            ) < 0.01
        )
        #expect(
            maxAbsoluteDifference(
                restoredValues[.ellipsis, ..<4, 0...],
                prefillValues[.ellipsis, ..<4, 0...]
            ) < 0.01
        )
        #expect(maxAbsoluteDifference(restoredKeys[.ellipsis, 4..<5, 0...], appendedKeys) < 0.01)
    }

    @Test
    func testTrimPromptCacheTrimsEveryLayer() {
        let cache: [KVCache] = [KVCacheSimple(), KVCacheSimple(), RotorQuantKVCache()]
        let keys = patternedArray(shape: [1, 1, 8, 128])
        let values = patternedArray(shape: [1, 1, 8, 128])
        for item in cache {
            _ = item.update(keys: keys, values: values)
        }

        #expect(trimPromptCache(cache, numTokens: 3) == 3)
        for item in cache {
            #expect(item.offset == 5)
        }
    }

    @Test
    func testRotorQuantAttentionPathSupportsGQAHeadDimension256() {
        let cache = RotorQuantKVCache(
            configuration: RotorQuantConfiguration(
                keyBits: 3,
                valueBits: 2,
                seed: 42,
                exactBufferSize: 0,
                attentionBlockTokens: 64,
                variant: .iso
            )
        )
        let queries = patternedArray(shape: [1, 4, 1, 256])
        let keys = patternedArray(shape: [1, 2, 1, 256])
        let values = patternedArray(shape: [1, 2, 1, 256])

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: 1.0 / sqrt(256.0),
            mask: .causal
        )

        #expect(output.shape == [1, 4, 1, 256])
        #expect(cache.offset == 1)
    }

    @Test
    func testRotorQuantFusedIsoDecodeMatchesFallbackHeadDimension256() {
        let configuration = RotorQuantConfiguration(
            keyBits: 3,
            valueBits: 2,
            seed: 42,
            exactBufferSize: 0,
            attentionBlockTokens: 64,
            variant: .iso
        )
        let fusedCache = RotorQuantKVCache(configuration: configuration)
        let fallbackCache = RotorQuantKVCache(configuration: configuration)
        let queries = patternedArray(shape: [1, 4, 1, 256], scale: 0.013)
        let keys = patternedArray(shape: [1, 2, 1, 256], scale: 0.017)
        let values = patternedArray(shape: [1, 2, 1, 256], scale: 0.019)

        let fused = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: fusedCache,
            scale: 1.0 / sqrt(256.0),
            mask: .none
        )
        let fallback = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: fallbackCache,
            scale: 1.0 / sqrt(256.0),
            mask: .array(MLXArray.zeros([1, 4, 1, 1], dtype: .float16))
        )

        #expect(fused.shape == [1, 4, 1, 256])
        #expect(maxAbsoluteDifference(fused, fallback) < 0.002)
    }

    @Test
    func testRotorQuantFusedIsoDecodeMatchesFallbackWithExactTail() {
        let configuration = RotorQuantConfiguration(
            keyBits: 3,
            valueBits: 2,
            seed: 42,
            exactBufferSize: 2,
            attentionBlockTokens: 64,
            variant: .iso
        )
        let fusedCache = RotorQuantKVCache(configuration: configuration)
        let fallbackCache = RotorQuantKVCache(configuration: configuration)
        let prefillKeys = patternedArray(shape: [1, 2, 4, 256], scale: 0.011)
        let prefillValues = patternedArray(shape: [1, 2, 4, 256], scale: 0.014)
        let queries = patternedArray(shape: [1, 4, 1, 256], scale: 0.013)
        let decodeKeys = patternedArray(shape: [1, 2, 1, 256], scale: 0.017)
        let decodeValues = patternedArray(shape: [1, 2, 1, 256], scale: 0.019)

        _ = fusedCache.update(keys: prefillKeys, values: prefillValues)
        _ = fallbackCache.update(keys: prefillKeys, values: prefillValues)

        let fused = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fusedCache,
            scale: 1.0 / sqrt(256.0),
            mask: .none
        )
        let fallback = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fallbackCache,
            scale: 1.0 / sqrt(256.0),
            mask: .array(MLXArray.zeros([1, 4, 1, 5], dtype: .float16))
        )

        #expect(fused.shape == [1, 4, 1, 256])
        #expect(maxAbsoluteDifference(fused, fallback) < 0.002)
    }

    @Test
    func testRotorQuantReloadHonorsMetaStateBeforeFusedDecode() {
        let configuration = RotorQuantConfiguration(
            keyBits: 3,
            valueBits: 2,
            seed: 99,
            exactBufferSize: 2,
            attentionBlockTokens: 64,
            variant: .iso
        )
        let sourceCache = RotorQuantKVCache(configuration: configuration)
        let fusedCache = RotorQuantKVCache()
        let fallbackCache = RotorQuantKVCache()
        let prefillKeys = patternedArray(shape: [1, 2, 4, 256], scale: 0.011)
        let prefillValues = patternedArray(shape: [1, 2, 4, 256], scale: 0.014)
        let seedKeys = patternedArray(shape: [1, 2, 1, 256], scale: 0.015)
        let seedValues = patternedArray(shape: [1, 2, 1, 256], scale: 0.016)
        let queries = patternedArray(shape: [1, 4, 1, 256], scale: 0.013)
        let decodeKeys = patternedArray(shape: [1, 2, 1, 256], scale: 0.017)
        let decodeValues = patternedArray(shape: [1, 2, 1, 256], scale: 0.019)

        _ = sourceCache.update(keys: prefillKeys, values: prefillValues)
        _ = sourceCache.update(keys: seedKeys, values: seedValues)
        let savedState = sourceCache.state
        let savedMetaState = sourceCache.metaState

        #expect(savedState.count == 6)
        fusedCache.state = savedState
        fusedCache.metaState = savedMetaState
        fallbackCache.state = savedState
        fallbackCache.metaState = savedMetaState

        let fused = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fusedCache,
            scale: 1.0 / sqrt(256.0),
            mask: .none
        )
        let fallback = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fallbackCache,
            scale: 1.0 / sqrt(256.0),
            mask: .array(MLXArray.zeros([1, 4, 1, 6], dtype: .float16))
        )

        #expect(fused.shape == [1, 4, 1, 256])
        #expect(maxAbsoluteDifference(fused, fallback) < 0.002)
    }

    @Test
    func testRotorQuantBlockParallelIsoDecodeMatchesFallbackWithExactTail() {
        let configuration = RotorQuantConfiguration(
            keyBits: 3,
            valueBits: 2,
            seed: 42,
            exactBufferSize: 2,
            attentionBlockTokens: 2,
            variant: .iso
        )
        let fusedCache = RotorQuantKVCache(configuration: configuration)
        let fallbackCache = RotorQuantKVCache(configuration: configuration)
        let prefillKeys = patternedArray(shape: [1, 2, 39, 256], scale: 0.011)
        let prefillValues = patternedArray(shape: [1, 2, 39, 256], scale: 0.014)
        let queries = patternedArray(shape: [1, 4, 1, 256], scale: 0.013)
        let decodeKeys = patternedArray(shape: [1, 2, 1, 256], scale: 0.017)
        let decodeValues = patternedArray(shape: [1, 2, 1, 256], scale: 0.019)

        _ = fusedCache.update(keys: prefillKeys, values: prefillValues)
        _ = fallbackCache.update(keys: prefillKeys, values: prefillValues)

        let fused = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fusedCache,
            scale: 1.0 / sqrt(256.0),
            mask: .none
        )
        let fallback = attentionWithCacheUpdate(
            queries: queries,
            keys: decodeKeys,
            values: decodeValues,
            cache: fallbackCache,
            scale: 1.0 / sqrt(256.0),
            mask: .array(MLXArray.zeros([1, 4, 1, 40], dtype: .float16))
        )

        #expect(fused.shape == [1, 4, 1, 256])
        #expect(maxAbsoluteDifference(fused, fallback) < 0.002)
    }

    @Test
    func testRotorQuantQwen35HybridCacheOnlyCompressesFullAttentionLayers() throws {
        let json = """
            {
              "model_type": "qwen3_5",
              "hidden_size": 1024,
              "num_hidden_layers": 8,
              "intermediate_size": 256,
              "num_attention_heads": 4,
              "num_key_value_heads": 4,
              "head_dim": 256,
              "linear_num_value_heads": 4,
              "linear_num_key_heads": 4,
              "linear_key_head_dim": 16,
              "linear_value_head_dim": 16,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 128,
              "full_attention_interval": 4,
              "partial_rotary_factor": 0.25
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)
        let model = Qwen35TextModel(config)
        let denseCache = model.newCache(parameters: GenerateParameters())
        let rotorCache = model.newCache(
            parameters: GenerateParameters(
                cacheCompression: .rotorQuant(
                    RotorQuantConfiguration(
                        keyBits: 3,
                        valueBits: 2,
                        seed: 42,
                        exactBufferSize: 128,
                        attentionBlockTokens: 256,
                        variant: .iso
                    )
                )
            )
        )

        let rotorCount = rotorCache.filter { $0 is RotorQuantKVCache }.count
        let mambaCount = rotorCache.filter { $0 is MambaCache }.count

        #expect(denseCache.count == 8)
        #expect(rotorCache.count == 8)
        #expect(rotorCount == 2)
        #expect(mambaCount == 6)
        for layerIndex in 0 ..< 8 {
            if (layerIndex + 1) % 4 == 0 {
                #expect(denseCache[layerIndex] is KVCacheSimple)
                #expect(rotorCache[layerIndex] is RotorQuantKVCache)
            } else {
                #expect(denseCache[layerIndex] is MambaCache)
                #expect(rotorCache[layerIndex] is MambaCache)
                #expect(denseCache[layerIndex].state.count == rotorCache[layerIndex].state.count)
                #expect(denseCache[layerIndex].metaState == rotorCache[layerIndex].metaState)
            }
        }
    }

    @Test
    func testLegacyQuantizedCompressionHandlesHybridCacheFirstLayer() {
        var cache: [KVCache] = [MambaCache(), KVCacheSimple()]
        let keys = patternedArray(shape: [1, 1, 1, 64])
        let values = patternedArray(shape: [1, 1, 1, 64])
        _ = cache[1].update(keys: keys, values: values)

        maybeApplyKVCacheCompression(
            cache: &cache,
            compression: .quantized(bits: 4, groupSize: 64, startStep: 0)
        )

        #expect(cache[0] is MambaCache)
        #expect(cache[1] is QuantizedKVCache)
    }

    @Test
    func testQuantizedKVCacheMaskSentinel() {
        let cache = QuantizedKVCache(groupSize: 32, bits: 8)
        let queryValues = [Float(10)] + Array(repeating: Float(0), count: 31)
        let firstKey = [Float(10)] + Array(repeating: Float(0), count: 31)
        let secondKey = [Float(0), Float(10)] + Array(repeating: Float(0), count: 30)
        let firstValue = (0 ..< 32).map { Float($0 + 1) / 32 }
        let secondValue = (0 ..< 32).map { Float($0 + 101) / 32 }
        let queries = MLXArray(queryValues).reshaped([1, 1, 1, 32]).asType(.float16)
        let keys = MLXArray(firstKey + secondKey).reshaped([1, 1, 2, 32]).asType(.float16)
        let values = MLXArray(firstValue + secondValue).reshaped([1, 1, 2, 32]).asType(.float16)
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
    func testLegacyTurboQuantPromptCacheMigratesToSimpleCache() throws {
        let keyPackedWidth = (128 * 2 + 7) / 8
        let valuePackedWidth = (128 * 2 + 7) / 8
        let compressedKeys = MLXArray.zeros([1, 1, 1, keyPackedWidth], dtype: .uint8)
        let keyNorms = MLXArray.ones([1, 1, 1], dtype: .float32)
        let residualSigns = MLXArray.zeros([1, 1, 1, 16], dtype: .uint8)
        let residualNorms = MLXArray.zeros([1, 1, 1], dtype: .float32)
        let compressedValues = MLXArray.zeros([1, 1, 1, valuePackedWidth], dtype: .uint8)
        let valueNorms = MLXArray.ones([1, 1, 1], dtype: .float32)
        let exactKeys = patternedArray(shape: [1, 1, 1, 128])
        let exactValues = patternedArray(shape: [1, 1, 1, 128])
        let state = [
            compressedKeys, keyNorms, residualSigns, residualNorms,
            compressedValues, valueNorms, exactKeys, exactValues,
        ]
        let metaState = [
            "256", "2", "3", "2", "42", "64", "0",
            "128", "128", "float16", "1", "1", "128",
        ]

        let migrated = try migrateLegacyCompressedPromptCache(state: state, metaState: metaState)

        #expect(migrated.offset == 2)
        #expect(migrated.state.count == 2)
        #expect(migrated.state[0].shape == [1, 1, 2, 128])
        #expect(migrated.state[1].shape == [1, 1, 2, 128])
    }

    @Test
    func testMalformedRotorQuantPromptCacheThrowsInsteadOfCrashing() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        let malformedState = MLXArray.zeros([1, 1, 1], dtype: .float32)
        let malformedMetaState = [
            "256", "1", "3", "2", "42", "128", "iso",
            "iso", "256", "256", "float16", "0", "1",
        ]
        var metadata: [String: String] = ["2.0": "RotorQuantKVCache"]
        for (index, value) in malformedMetaState.enumerated() {
            metadata["0.0.\(index)"] = value
        }

        try save(arrays: ["0.0": malformedState], metadata: metadata, url: url)

        #expect(throws: KVCacheError.self) {
            _ = try loadPromptCache(url: url)
        }
    }

    @Test
    func testMalformedRotorQuantPromptCacheMetadataThrowsInsteadOfCrashing() throws {
        let invalidBitsURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        let invalidCountURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        let exactKeys = MLXArray.zeros([1, 1, 1, 128], dtype: .float16)
        let exactValues = MLXArray.zeros([1, 1, 1, 128], dtype: .float16)
        let validMetaState = [
            "256", "1", "3", "2", "42", "128", "iso",
            "iso", "128", "128", "float16", "0", "1", "128",
        ]

        func saveRotorCache(url: URL, metaState: [String]) throws {
            var metadata: [String: String] = ["2.0": "RotorQuantKVCache"]
            for (index, value) in metaState.enumerated() {
                metadata["0.0.\(index)"] = value
            }
            try save(
                arrays: [
                    "0.0": exactKeys,
                    "0.1": exactValues,
                ],
                metadata: metadata,
                url: url
            )
        }

        var invalidBitsMetaState = validMetaState
        invalidBitsMetaState[2] = "9"
        try saveRotorCache(url: invalidBitsURL, metaState: invalidBitsMetaState)

        var invalidCountMetaState = validMetaState
        invalidCountMetaState[1] = "2"
        invalidCountMetaState[12] = "2"
        try saveRotorCache(url: invalidCountURL, metaState: invalidCountMetaState)

        #expect(throws: KVCacheError.self) {
            _ = try loadPromptCache(url: invalidBitsURL)
        }
        #expect(throws: KVCacheError.self) {
            _ = try loadPromptCache(url: invalidCountURL)
        }
    }
}
