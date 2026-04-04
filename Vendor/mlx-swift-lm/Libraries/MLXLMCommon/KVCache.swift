// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = QuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
public protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create an attention mask for this cache
    ///
    /// This method encapsulates cache-specific mask creation logic. Implementations should handle offset capping, window size logic,
    /// and optimization decisions (symbolic vs array masks).
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMM(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
public protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// Quantization mode
    var mode: QuantizationMode { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
}

/// Protocol for cache implementations that provide their own attention path.
///
/// TurboQuant keys are not fully reconstructible from their compressed form, so
/// the cache needs to participate in attention score computation directly.
public protocol AttentionCapableKVCache: KVCache {
    func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    /// Default implementation for caches without special mask requirements
    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For single token, no mask needed
        if n == 1 {
            return .none
        }

        // For multi-token sequences
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    return mask
}

/// Create an attention mask matching mlx-lm's create_attention_mask helper.
///
/// This returns `.causal` when a symbolic mask is sufficient, avoiding
/// materializing a full mask array.
public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

@available(
    *, deprecated,
    message: "Use createAttentionMask(h:cache:windowSize:returnArray:) with a single cache instead"
)
public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize - 1, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

/// Create an attention mask with explicit window size parameter.
///
/// - Parameters:
///   - h: The input array (used to determine sequence length)
///   - cache: Optional single KV cache
///   - windowSize: Optional sliding window size (if provided, creates windowed attention)
///   - returnArray: Force return of array mask instead of symbolic "causal"
/// - Returns: Attention mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's makeMask if available
    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    // Fallback for no cache
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

/// Standard KV cache implementation based on Python's KVCache
/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to quantized cache for maximum efficiency
    ///
    /// Use `updateQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset

        if let keys = self.keys, let values = self.values {
            // Quantize the current keys and values
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]

            let quantizedKeys = quantized(currentKeys, groupSize: groupSize, bits: bits)
            let quantizedValues = quantized(currentValues, groupSize: groupSize, bits: bits)

            // Set the quantized state
            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }
        }

        return quantizedCache
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

private final class TurboQuantMSEQuantizer {
    let dimension: Int
    let bits: Int
    let signs: MLXArray
    let centroids: MLXArray
    let boundaries: [Float]
    let normalizedHadamard: MLXArray
    let rotationScale: Float

    init(dimension: Int, bits: Int, seed: UInt64) {
        precondition(
            dimension > 0 && (dimension & (dimension - 1)) == 0,
            "TurboQuant requires a power-of-two head dimension"
        )
        precondition((1 ... 4).contains(bits), "TurboQuant MSE stage supports 1-4 bits.")
        self.dimension = dimension
        self.bits = bits
        self.signs = Self.makeSigns(dimension: dimension, seed: seed)
        self.centroids = MLXArray(Self.gaussianCodebook(bits: bits)).asType(.float32)
        self.boundaries = Self.gaussianBoundaries(bits: bits)
        self.normalizedHadamard = Self.makeNormalizedHadamardMatrix(dimension: dimension)
        self.rotationScale = sqrt(Float(dimension))
    }

    func quantize(_ vectors: MLXArray) -> (indices: MLXArray, norms: MLXArray, reconstructed: MLXArray) {
        let floatVectors = vectors.asType(.float32)
        let norms = sqrt(sum(floatVectors * floatVectors, axis: -1, keepDims: true))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let rotated = matmul((floatVectors / safeNorms) * signs, normalizedHadamard)
        let scaled = rotated * rotationScale
        var indices = MLXArray.zeros(scaled.shape, dtype: .uint8)
        for boundary in boundaries {
            indices = indices + greater(scaled, MLXArray(boundary)).asType(.uint8)
        }
        let squeezedNorms = norms.squeezed(axis: -1)
        let reconstructed = dequantize(indices: indices, norms: squeezedNorms, dtype: .float32)
        return (indices, squeezedNorms, reconstructed)
    }

    func dequantize(indices: MLXArray, norms: MLXArray, dtype: DType) -> MLXArray {
        let shape = indices.shape
        let flatCount = shape.reduce(1, *)
        let flatIndices = indices.asType(.int32).reshaped([flatCount])
        let rotated = centroids[flatIndices].reshaped(shape) / rotationScale
        let restoredUnit = matmul(rotated, normalizedHadamard) * signs
        let restored = restoredUnit * expandedDimensions(norms.asType(.float32), axis: -1)
        return restored.asType(dtype)
    }

    private static func gaussianCodebook(bits: Int) -> [Float] {
        switch bits {
        case 1:
            return [-0.7979, 0.7979]
        case 2:
            return [-1.5104, -0.4528, 0.4528, 1.5104]
        case 3:
            return [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]
        case 4:
            return [
                -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
            ]
        default:
            fatalError("TurboQuant supports 1-4 bits. Received \(bits).")
        }
    }

    private static func gaussianBoundaries(bits: Int) -> [Float] {
        let codebook = gaussianCodebook(bits: bits)
        return zip(codebook, codebook.dropFirst()).map { ($0 + $1) / 2.0 }
    }

    private static func makeSigns(dimension: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let mask = MLXRandom.bernoulli(0.5, [dimension], key: key)
        return MLX.where(mask, MLXArray(1.0), MLXArray(-1.0)).asType(.float32)
    }

    private static func makeNormalizedHadamardMatrix(dimension: Int) -> MLXArray {
        var matrix: [[Float]] = [[1.0]]
        while matrix.count < dimension {
            let top = matrix.map { row in row + row }
            let bottom = matrix.map { row in row + row.map { -$0 } }
            matrix = top + bottom
        }
        let scale: Float = 1.0 / sqrt(Float(dimension))
        let flattened = matrix.flatMap { row in row.map { $0 * scale } }
        return MLXArray(flattened, [dimension, dimension]).asType(.float32)
    }
}

private final class TurboQuantResidualSketch {
    let dimension: Int
    let sketchDimension: Int
    let projection: MLXArray
    let estimatorScale: Float

    init(dimension: Int, sketchDimension: Int? = nil, seed: UInt64) {
        self.dimension = dimension
        self.sketchDimension = sketchDimension ?? dimension
        let key = MLXRandom.key(seed)
        self.projection = MLXRandom.normal(
            [dimension, self.sketchDimension],
            type: Float.self,
            key: key
        ).asType(.float32)
        self.estimatorScale = sqrt(Float.pi / 2.0) / Float(self.sketchDimension)
    }

    func quantize(_ residual: MLXArray) -> (signs: MLXArray, norms: MLXArray) {
        let floatResidual = residual.asType(.float32)
        let norms = sqrt(sum(floatResidual * floatResidual, axis: -1, keepDims: true))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let projected = matmul(floatResidual / safeNorms, projection)
        let signs = greaterEqual(projected, MLXArray(Float(0.0))).asType(.uint8)
        return (signs, norms.squeezed(axis: -1))
    }

    func score(
        queries: MLXArray,
        signs: MLXArray,
        norms: MLXArray
    ) -> MLXArray {
        let projectedQueries = matmul(queries.asType(.float32), projection)
        let signedResidual = signs.asType(.float32) * 2.0 - 1.0
        let rawScores = matmul(projectedQueries, signedResidual.swappedAxes(-1, -2))
        let broadcastNorms =
            if norms.shape.count == rawScores.shape.count - 1 {
                expandedDimensions(norms.asType(.float32), axis: -2)
            } else {
                norms.asType(.float32)
            }
        return rawScores * broadcastNorms * estimatorScale
    }
}

private enum TurboQuantBitPacker {
    static func packedByteCount(valueCount: Int, bitWidth: Int) -> Int {
        precondition(valueCount >= 0)
        precondition((1 ... 8).contains(bitWidth))
        return (valueCount * bitWidth + 7) / 8
    }

    static func pack(_ values: MLXArray, bitWidth: Int) -> MLXArray {
        precondition((1 ... 8).contains(bitWidth))
        let valueCount = values.dim(-1)
        let prefixShape = Array(values.shape.dropLast())
        let packedCount = packedByteCount(valueCount: valueCount, bitWidth: bitWidth)

        let uintValues = values.asType(.uint8)
        let bitPlanes = (0 ..< bitWidth).map { bit in
            bitwiseAnd(rightShift(uintValues, UInt8(bit)), UInt8(1)).asType(.uint8)
        }
        var bitStream = stacked(bitPlanes, axis: -1).reshaped(prefixShape + [valueCount * bitWidth])

        let padBits = packedCount * 8 - valueCount * bitWidth
        if padBits > 0 {
            let padding = MLXArray.zeros(prefixShape + [padBits], dtype: .uint8)
            bitStream = concatenated([bitStream, padding], axis: bitStream.shape.count - 1)
        }

        let byteWeights = MLXArray([UInt32(1), 2, 4, 8, 16, 32, 64, 128]).asType(.uint32)
        let packed = sum(
            bitStream.reshaped(prefixShape + [packedCount, 8]).asType(.uint32) * byteWeights,
            axis: -1
        )
        return packed.asType(.uint8)
    }

    static func unpack(_ packed: MLXArray, bitWidth: Int, valueCount: Int) -> MLXArray {
        precondition((1 ... 8).contains(bitWidth))
        let prefixShape = Array(packed.shape.dropLast())
        let packedCount = packed.dim(-1)
        let shiftValues = MLXArray((0 ..< 8).map(UInt8.init)).asType(.uint8)
        let expanded = packed.asType(.uint8).expandedDimensions(axis: -1)
        var bitStream = bitwiseAnd(rightShift(expanded, shiftValues), UInt8(1)).asType(.uint8)
            .reshaped(prefixShape + [packedCount * 8])

        let usedBits = valueCount * bitWidth
        if packedCount * 8 > usedBits {
            bitStream = bitStream[.ellipsis, ..<usedBits]
        }

        let valueWeights = MLXArray((0 ..< bitWidth).map { UInt32(1 << $0) }).asType(.uint32)
        let unpacked = sum(
            bitStream.reshaped(prefixShape + [valueCount, bitWidth]).asType(.uint32) * valueWeights,
            axis: -1
        )
        return unpacked.asType(.uint8)
    }
}

public final class TurboQuantKVCache: BaseKVCache, AttentionCapableKVCache {
    private var keyBaseIndices: MLXArray?
    private var keyBaseNorms: MLXArray?
    private var keyResidualSigns: MLXArray?
    private var keyResidualNorms: MLXArray?
    private var valueIndices: MLXArray?
    private var valueNorms: MLXArray?
    private var exactKeys: MLXArray?
    private var exactValues: MLXArray?
    private var compressedCount: Int = 0
    private var step: Int
    public private(set) var bits: Int
    public private(set) var seed: UInt64
    private var keyQuantizer: TurboQuantMSEQuantizer?
    private var valueQuantizer: TurboQuantMSEQuantizer?
    private var residualSketch: TurboQuantResidualSketch?
    private var keyDimension: Int?
    private var valueDimension: Int?
    private var originalDType: DType = .float16

    private var keyBits: Int { bits - 1 }
    private var keyPackedWidth: Int {
        guard let keyDimension else { return 0 }
        return TurboQuantBitPacker.packedByteCount(valueCount: keyDimension, bitWidth: keyBits)
    }
    private var residualPackedWidth: Int {
        let residualDimension = residualSketch?.sketchDimension ?? keyDimension ?? 0
        return TurboQuantBitPacker.packedByteCount(valueCount: residualDimension, bitWidth: 1)
    }
    private var valuePackedWidth: Int {
        guard let valueDimension else { return 0 }
        return TurboQuantBitPacker.packedByteCount(valueCount: valueDimension, bitWidth: bits)
    }
    private let attentionBlockTokens = 128
    private var exactBufferSize: Int
    private var exactCount: Int { exactKeys?.dim(2) ?? 0 }

    public init(bits: Int = 3, seed: UInt64 = 42, step: Int = 256, exactBufferSize: Int = 128) {
        precondition((2 ... 4).contains(bits), "TurboQuant supports 2-4 total bits.")
        self.bits = bits
        self.seed = seed
        self.step = step
        self.exactBufferSize = exactBufferSize
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [
            keyBaseIndices, keyBaseNorms, keyResidualSigns, keyResidualNorms,
            valueIndices, valueNorms, exactKeys, exactValues,
        ].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        ingest(keys: keys, values: values)
        return currentFallbackState()
    }

    public func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        ingest(keys: keys, values: values)

        guard let keyQuantizer,
              let valueQuantizer,
              let residualSketch
        else {
            fatalError("TurboQuantKVCache is not initialized")
        }

        let total = offset
        let (batch, queryHeads, queryLength, queryDimension) = (
            queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
        )
        let kvHeads =
            keyBaseIndices?.dim(1)
            ?? exactKeys?.dim(1)
            ?? keys.dim(1)
        let repeats = queryHeads / kvHeads
        let maskedValue = MLXArray(-Float.greatestFiniteMagnitude)

        var scaledQueries = queries.asType(.float32) * scale
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped([batch, kvHeads, repeats, queryLength, queryDimension])
        }

        let queryStart = total - queryLength

        func applyBlockMask(
            _ scores: MLXArray,
            start: Int,
            end: Int
        ) -> MLXArray {
            switch mask {
            case .causal:
                let qIndices = MLXArray(queryStart ..< total)
                let kIndices = MLXArray(start ..< end)
                let causalMask = greaterEqual(
                    expandedDimensions(qIndices, axis: -1),
                    expandedDimensions(kIndices, axis: -2)
                )
                return MLX.where(causalMask, scores, maskedValue)
            case .array(let maskArray):
                let slicedMask = maskArray[.ellipsis, start..<end]
                if slicedMask.dtype == .bool {
                    return MLX.where(slicedMask, scores, maskedValue)
                } else {
                    return scores + slicedMask
                }
            case .arrays(let maskArrays):
                if let maskArray = maskArrays.first {
                    let slicedMask = maskArray[.ellipsis, start..<end]
                    if slicedMask.dtype == .bool {
                        return MLX.where(slicedMask, scores, maskedValue)
                    } else {
                        return scores + slicedMask
                    }
                }
                return scores
            case .none:
                return scores
            }
        }

        var runningMax: MLXArray?
        var runningNormalizer: MLXArray?
        var runningOutput: MLXArray?

        func mergeBlock(_ blockScores: MLXArray, preparedValues: MLXArray) {
            let blockMax = max(blockScores, axis: -1, keepDims: true)

            if let currentRunningMax = runningMax,
               let currentRunningNormalizer = runningNormalizer,
               let currentRunningOutput = runningOutput
            {
                let combinedMax = maximum(currentRunningMax, blockMax)
                let previousScale = exp(currentRunningMax - combinedMax)
                let blockWeights = exp(blockScores - combinedMax)
                let combinedNormalizer =
                    currentRunningNormalizer * previousScale
                    + sum(blockWeights, axis: -1, keepDims: true)
                let combinedOutput =
                    currentRunningOutput * previousScale
                    + matmul(blockWeights, preparedValues)

                runningMax = combinedMax
                runningNormalizer = combinedNormalizer
                runningOutput = combinedOutput
            } else {
                let blockWeights = exp(blockScores - blockMax)
                runningMax = blockMax
                runningNormalizer = sum(blockWeights, axis: -1, keepDims: true)
                runningOutput = matmul(blockWeights, preparedValues)
            }
        }

        if compressedCount > 0 {
            guard let keyBaseIndices,
                  let keyBaseNorms,
                  let keyResidualSigns,
                  let keyResidualNorms,
                  let valueIndices,
                  let valueNorms
            else {
                fatalError("TurboQuantKVCache compressed state is missing")
            }

            for start in stride(from: 0, to: compressedCount, by: attentionBlockTokens) {
                let end = min(start + attentionBlockTokens, compressedCount)
                let unpackedKeyIndices = TurboQuantBitPacker.unpack(
                    keyBaseIndices[.ellipsis, start..<end, 0...],
                    bitWidth: keyBits,
                    valueCount: keyDimension!
                )
                let approxKeys = keyQuantizer.dequantize(
                    indices: unpackedKeyIndices,
                    norms: keyBaseNorms[.ellipsis, start..<end],
                    dtype: .float32
                )
                let residualSigns = TurboQuantBitPacker.unpack(
                    keyResidualSigns[.ellipsis, start..<end, 0...],
                    bitWidth: 1,
                    valueCount: residualSketch.sketchDimension
                )
                let residualNorms = keyResidualNorms[.ellipsis, start..<end]
                let unpackedValueIndices = TurboQuantBitPacker.unpack(
                    valueIndices[.ellipsis, start..<end, 0...],
                    bitWidth: bits,
                    valueCount: valueDimension!
                )
                let approxValues = valueQuantizer.dequantize(
                    indices: unpackedValueIndices,
                    norms: valueNorms[.ellipsis, start..<end],
                    dtype: .float32
                )

                var preparedKeys = approxKeys
                var preparedResidualSigns = residualSigns
                var preparedResidualNorms = residualNorms
                var preparedValues = approxValues

                if repeats > 1 {
                    preparedKeys = expandedDimensions(preparedKeys, axis: -3)
                    preparedResidualSigns = expandedDimensions(preparedResidualSigns, axis: -3)
                    preparedResidualNorms = expandedDimensions(preparedResidualNorms, axis: -2)
                    preparedValues = expandedDimensions(preparedValues, axis: -3)
                }

                var blockScores = matmul(scaledQueries, preparedKeys.swappedAxes(-1, -2))
                blockScores = blockScores + residualSketch.score(
                    queries: scaledQueries,
                    signs: preparedResidualSigns,
                    norms: preparedResidualNorms
                )
                blockScores = applyBlockMask(blockScores, start: start, end: end)
                mergeBlock(blockScores, preparedValues: preparedValues)
            }
        }

        if let exactKeys, let exactValues, exactCount > 0 {
            var preparedKeys = exactKeys.asType(.float32)
            var preparedValues = exactValues.asType(.float32)

            if repeats > 1 {
                preparedKeys = expandedDimensions(preparedKeys, axis: -3)
                preparedValues = expandedDimensions(preparedValues, axis: -3)
            }

            var exactScores = matmul(scaledQueries, preparedKeys.swappedAxes(-1, -2))
            exactScores = applyBlockMask(
                exactScores,
                start: compressedCount,
                end: compressedCount + exactCount
            )
            mergeBlock(exactScores, preparedValues: preparedValues)
        }

        guard let runningNormalizer, let runningOutput else {
            fatalError("TurboQuantKVCache attention produced no blocks")
        }

        var output = runningOutput / maximum(runningNormalizer, MLXArray(Float(1e-8)))
        if repeats > 1 {
            output = output.reshaped([batch, queryHeads, queryLength, output.dim(-1)])
        }

        return output.asType(originalDType)
    }

    public override var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []

            if compressedCount > 0 {
                guard let keyBaseIndices,
                      let keyBaseNorms,
                      let keyResidualSigns,
                      let keyResidualNorms,
                      let valueIndices,
                      let valueNorms
                else {
                    fatalError("TurboQuantKVCache compressed state is missing")
                }
                arrays.append(contentsOf: [
                    keyBaseIndices[.ellipsis, ..<compressedCount, 0...],
                    keyBaseNorms[.ellipsis, ..<compressedCount],
                    keyResidualSigns[.ellipsis, ..<compressedCount, 0...],
                    keyResidualNorms[.ellipsis, ..<compressedCount],
                    valueIndices[.ellipsis, ..<compressedCount, 0...],
                    valueNorms[.ellipsis, ..<compressedCount],
                ])
            }

            if let exactKeys, let exactValues, exactCount > 0 {
                arrays.append(exactKeys[.ellipsis, ..<exactCount, 0...])
                arrays.append(exactValues[.ellipsis, ..<exactCount, 0...])
            }

            return arrays
        }
        set {
            keyBaseIndices = nil
            keyBaseNorms = nil
            keyResidualSigns = nil
            keyResidualNorms = nil
            valueIndices = nil
            valueNorms = nil
            exactKeys = nil
            exactValues = nil
            compressedCount = 0

            switch newValue.count {
            case 0:
                break
            case 2:
                exactKeys = newValue[0]
                exactValues = newValue[1]
            case 6:
                keyBaseIndices = newValue[0]
                keyBaseNorms = newValue[1]
                keyResidualSigns = newValue[2]
                keyResidualNorms = newValue[3]
                valueIndices = newValue[4]
                valueNorms = newValue[5]
                compressedCount = keyBaseIndices?.dim(2) ?? 0
            case 8:
                keyBaseIndices = newValue[0]
                keyBaseNorms = newValue[1]
                keyResidualSigns = newValue[2]
                keyResidualNorms = newValue[3]
                valueIndices = newValue[4]
                valueNorms = newValue[5]
                compressedCount = keyBaseIndices?.dim(2) ?? 0
                exactKeys = newValue[6]
                exactValues = newValue[7]
            default:
                fatalError("TurboQuantKVCache state must have 0, 2, 6, or 8 arrays")
            }
            offset = compressedCount + exactCount
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(step),
                String(offset),
                String(bits),
                String(seed),
                String(keyDimension ?? 0),
                String(valueDimension ?? 0),
                Self.dtypeName(originalDType),
                String(compressedCount),
                String(exactCount),
                String(exactBufferSize),
            ]
        }
        set {
            guard newValue.count == 7 || newValue.count == 10 else {
                fatalError("TurboQuantKVCache metaState must have 7 or 10 values")
            }
            step = Int(newValue[0]) ?? step
            offset = Int(newValue[1]) ?? 0
            bits = Int(newValue[2]) ?? bits
            seed = UInt64(newValue[3]) ?? seed
            keyDimension = (Int(newValue[4]) ?? 0) > 0 ? Int(newValue[4]) : nil
            valueDimension = (Int(newValue[5]) ?? 0) > 0 ? Int(newValue[5]) : nil
            originalDType = Self.dtype(from: newValue[6])
            if newValue.count == 10 {
                compressedCount = Int(newValue[7]) ?? 0
                let serializedExactCount = Int(newValue[8]) ?? 0
                exactBufferSize = Int(newValue[9]) ?? exactBufferSize
                offset = compressedCount + serializedExactCount
            } else {
                compressedCount = offset
            }
            if let keyDimension, let valueDimension {
                ensureQuantizers(keyDimension: keyDimension, valueDimension: valueDimension)
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }

        var remaining = n
        let trimmedCompressed = min(compressedCount, remaining)
        if trimmedCompressed > 0 {
            keyBaseIndices = keyBaseIndices?[.ellipsis, trimmedCompressed..<compressedCount, 0...]
            keyBaseNorms = keyBaseNorms?[.ellipsis, trimmedCompressed..<compressedCount]
            keyResidualSigns = keyResidualSigns?[.ellipsis, trimmedCompressed..<compressedCount, 0...]
            keyResidualNorms = keyResidualNorms?[.ellipsis, trimmedCompressed..<compressedCount]
            valueIndices = valueIndices?[.ellipsis, trimmedCompressed..<compressedCount, 0...]
            valueNorms = valueNorms?[.ellipsis, trimmedCompressed..<compressedCount]
            compressedCount -= trimmedCompressed
            remaining -= trimmedCompressed
        }

        var trimmedExact = 0
        if remaining > 0, let exactKeys, let exactValues {
            let currentExactCount = exactKeys.dim(2)
            trimmedExact = min(remaining, currentExactCount)
            if trimmedExact >= currentExactCount {
                self.exactKeys = nil
                self.exactValues = nil
            } else {
                self.exactKeys = exactKeys[.ellipsis, trimmedExact..<currentExactCount, 0...]
                self.exactValues = exactValues[.ellipsis, trimmedExact..<currentExactCount, 0...]
            }
        }

        offset = compressedCount + exactCount
        return trimmedCompressed + trimmedExact
    }

    private func ingest(keys: MLXArray, values: MLXArray) {
        originalDType = keys.dtype
        ensureQuantizers(keyDimension: keys.dim(3), valueDimension: values.dim(3))

        let combinedKeys =
            if let exactKeys, exactCount > 0 {
                concatenated([exactKeys[.ellipsis, ..<exactCount, 0...], keys], axis: 2)
            } else {
                keys
            }
        let combinedValues =
            if let exactValues, exactCount > 0 {
                concatenated([exactValues[.ellipsis, ..<exactCount, 0...], values], axis: 2)
            } else {
                values
            }

        let combinedCount = combinedKeys.dim(2)
        let flushCount = max(0, combinedCount - exactBufferSize)

        if flushCount > 0 {
            appendCompressed(
                keys: combinedKeys[.ellipsis, ..<flushCount, 0...],
                values: combinedValues[.ellipsis, ..<flushCount, 0...]
            )
        }

        let keepCount = combinedCount - flushCount
        if keepCount > 0 {
            exactKeys = combinedKeys[.ellipsis, flushCount..<combinedCount, 0...]
            exactValues = combinedValues[.ellipsis, flushCount..<combinedCount, 0...]
        } else {
            exactKeys = nil
            exactValues = nil
        }

        offset = compressedCount + keepCount
    }

    private func appendCompressed(keys: MLXArray, values: MLXArray) {
        guard keys.dim(2) > 0 else { return }

        let previous = compressedCount
        ensureStorage(batch: keys.dim(0), kvHeads: keys.dim(1), tokenCount: keys.dim(2))

        let quantizedKeys = keyQuantizer!.quantize(keys)
        let keyResidual = keys.asType(.float32) - quantizedKeys.reconstructed
        let residualQuantization = residualSketch!.quantize(keyResidual)
        let quantizedValues = valueQuantizer!.quantize(values)
        let packedKeyIndices = TurboQuantBitPacker.pack(quantizedKeys.indices, bitWidth: keyBits)
        let packedResidualSigns = TurboQuantBitPacker.pack(residualQuantization.signs, bitWidth: 1)
        let packedValueIndices = TurboQuantBitPacker.pack(quantizedValues.indices, bitWidth: bits)

        compressedCount += keys.dim(2)

        keyBaseIndices?[.ellipsis, previous ..< compressedCount, 0...] = packedKeyIndices
        keyBaseNorms?[.ellipsis, previous ..< compressedCount] = quantizedKeys.norms
        keyResidualSigns?[.ellipsis, previous ..< compressedCount, 0...] = packedResidualSigns
        keyResidualNorms?[.ellipsis, previous ..< compressedCount] = residualQuantization.norms
        valueIndices?[.ellipsis, previous ..< compressedCount, 0...] = packedValueIndices
        valueNorms?[.ellipsis, previous ..< compressedCount] = quantizedValues.norms
    }

    private func ensureStorage(batch: Int, kvHeads: Int, tokenCount: Int) {
        let previous = compressedCount
        let needsAllocation =
            if let keyBaseIndices, (previous + tokenCount) > keyBaseIndices.dim(2) {
                true
            } else {
                keyBaseIndices == nil
            }

        guard needsAllocation else { return }

        let steps = ((step + tokenCount - 1) / step) * step
        let newKeyIndexStorage = MLXArray.zeros([batch, kvHeads, steps, keyPackedWidth], dtype: .uint8)
        let newKeyNormStorage = MLXArray.zeros([batch, kvHeads, steps], dtype: .float32)
        let newKeyResidualSignStorage = MLXArray.zeros(
            [batch, kvHeads, steps, residualPackedWidth],
            dtype: .uint8
        )
        let newKeyResidualNormStorage = MLXArray.zeros([batch, kvHeads, steps], dtype: .float32)
        let newValueIndexStorage = MLXArray.zeros([batch, kvHeads, steps, valuePackedWidth], dtype: .uint8)
        let newValueNormStorage = MLXArray.zeros([batch, kvHeads, steps], dtype: .float32)

        if var currentKeyBaseIndices = self.keyBaseIndices,
           var currentKeyBaseNorms = self.keyBaseNorms,
           var currentKeyResidualSigns = self.keyResidualSigns,
           var currentKeyResidualNorms = self.keyResidualNorms,
           var currentValueIndices = self.valueIndices,
           var currentValueNorms = self.valueNorms
        {
            if previous % step != 0 {
                currentKeyBaseIndices = currentKeyBaseIndices[.ellipsis, ..<previous, 0...]
                currentKeyBaseNorms = currentKeyBaseNorms[.ellipsis, ..<previous]
                currentKeyResidualSigns = currentKeyResidualSigns[.ellipsis, ..<previous, 0...]
                currentKeyResidualNorms = currentKeyResidualNorms[.ellipsis, ..<previous]
                currentValueIndices = currentValueIndices[.ellipsis, ..<previous, 0...]
                currentValueNorms = currentValueNorms[.ellipsis, ..<previous]
            }
            self.keyBaseIndices = concatenated([currentKeyBaseIndices, newKeyIndexStorage], axis: 2)
            self.keyBaseNorms = concatenated([currentKeyBaseNorms, newKeyNormStorage], axis: 2)
            self.keyResidualSigns = concatenated([currentKeyResidualSigns, newKeyResidualSignStorage], axis: 2)
            self.keyResidualNorms = concatenated([currentKeyResidualNorms, newKeyResidualNormStorage], axis: 2)
            self.valueIndices = concatenated([currentValueIndices, newValueIndexStorage], axis: 2)
            self.valueNorms = concatenated([currentValueNorms, newValueNormStorage], axis: 2)
        } else {
            self.keyBaseIndices = newKeyIndexStorage
            self.keyBaseNorms = newKeyNormStorage
            self.keyResidualSigns = newKeyResidualSignStorage
            self.keyResidualNorms = newKeyResidualNormStorage
            self.valueIndices = newValueIndexStorage
            self.valueNorms = newValueNormStorage
        }
    }

    private func ensureQuantizers(keyDimension: Int, valueDimension: Int) {
        if self.keyDimension != keyDimension || keyQuantizer == nil || residualSketch == nil {
            self.keyDimension = keyDimension
            self.keyQuantizer = TurboQuantMSEQuantizer(
                dimension: keyDimension,
                bits: keyBits,
                seed: seed
            )
            self.residualSketch = TurboQuantResidualSketch(
                dimension: keyDimension,
                sketchDimension: keyDimension,
                seed: seed &+ 0x9E37_79B9_7F4A_7C15
            )
        }
        if self.valueDimension != valueDimension || valueQuantizer == nil {
            self.valueDimension = valueDimension
            self.valueQuantizer = TurboQuantMSEQuantizer(
                dimension: valueDimension,
                bits: bits,
                seed: seed &+ 1
            )
        }
    }

    private func currentFallbackState() -> (MLXArray, MLXArray) {
        guard let keyQuantizer, let valueQuantizer else {
            fatalError("TurboQuantKVCache is not initialized")
        }

        var allKeys: [MLXArray] = []
        var allValues: [MLXArray] = []

        if compressedCount > 0 {
            guard let keyBaseIndices,
                  let keyBaseNorms,
                  let valueIndices,
                  let valueNorms
            else {
                fatalError("TurboQuantKVCache compressed state is missing")
            }
            let currentKeyIndices = TurboQuantBitPacker.unpack(
                keyBaseIndices[.ellipsis, ..<compressedCount, 0...],
                bitWidth: keyBits,
                valueCount: keyDimension!
            )
            let currentKeyNorms = keyBaseNorms[.ellipsis, ..<compressedCount]
            let currentValueIndices = TurboQuantBitPacker.unpack(
                valueIndices[.ellipsis, ..<compressedCount, 0...],
                bitWidth: bits,
                valueCount: valueDimension!
            )
            let currentValueNorms = valueNorms[.ellipsis, ..<compressedCount]
            allKeys.append(
                keyQuantizer.dequantize(indices: currentKeyIndices, norms: currentKeyNorms, dtype: originalDType)
            )
            allValues.append(
                valueQuantizer.dequantize(indices: currentValueIndices, norms: currentValueNorms, dtype: originalDType)
            )
        }

        if let exactKeys, let exactValues, exactCount > 0 {
            allKeys.append(exactKeys[.ellipsis, ..<exactCount, 0...].asType(originalDType))
            allValues.append(exactValues[.ellipsis, ..<exactCount, 0...].asType(originalDType))
        }

        guard let firstKeys = allKeys.first, let firstValues = allValues.first else {
            fatalError("TurboQuantKVCache is empty")
        }

        if allKeys.count == 1 {
            return (firstKeys, firstValues)
        }

        return (
            concatenated(allKeys, axis: 2),
            concatenated(allValues, axis: 2)
        )
    }

    private static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16:
            return "bfloat16"
        case .float32:
            return "float32"
        default:
            return "float16"
        }
    }

    private static func dtype(from name: String) -> DType {
        switch name {
        case "bfloat16":
            return .bfloat16
        case "float32":
            return .float32
        default:
            return .float16
        }
    }
}

extension KVCacheSimple {
    public func toTurboQuantized(bits: Int = 3, seed: UInt64 = 42) -> TurboQuantKVCache {
        let turboQuantCache = TurboQuantKVCache(bits: bits, seed: seed)

        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            _ = turboQuantCache.update(keys: currentKeys, values: currentValues)
        }

        return turboQuantCache
    }
}

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0

    public override var maxSize: Int? { maxCacheSize }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            // Allow temporary cache growth during multi-token processing (e.g., prompt prefill).
            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context
            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // May not have hit the max size yet, so potentially keep growing the cache
        if self.keys == nil
            || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < maxCacheSize)
        {
            let newSize = min(step, maxCacheSize - prev)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
            idx = keep
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice
        if offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result =
            if keys.dim(2) == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            return [String(keep), String(maxCacheSize), String(step), String(offset), String(idx)]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("RotatingKVCache metaState must have exactly 5 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
        }
    }

    public override var isTrimmable: Bool {
        return offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    /// Optimized mask creation for rotating cache with offset capping
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            // Multi-token case
            let actualWindowSize = windowSize ?? maxCacheSize
            let cappedOffset = min(maxCacheSize - 1, offset)

            // Decide if we need an array mask
            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single token case (n == 1)
            guard let windowSize = windowSize else {
                return .none
            }

            // May need a mask when window_size < max_size and cache has wrapped
            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }

                let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)

                // Roll the mask to account for rotation
                let rolledMask = roll(mask, shift: currentIdx + 1)

                return .array(rolledMask)
            }
            return .none
        }
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx)"
    }

    /// Convert to quantized cache
    /// Note: This is complex due to the rotating nature and temporal ordering
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        // For now, throw an error like the Python version does
        // A full implementation would need to handle the temporal ordering correctly
        fatalError(
            "RotatingKVCache quantization not yet implemented - temporal ordering makes this complex"
        )

        // Future implementation would need to:
        // 1. Put keys/values in temporal order using temporalOrder()
        // 2. Quantize the temporally ordered arrays
        // 3. Store metadata about rotation state
        // 4. Implement corresponding dequantization with rotation restoration
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    /// Array of keys and values -- this will have either 6 elements or 4 elements (if biases are nil).
    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 4:
                // nil biases case
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }

            self.offset = Int(newValue[1]) ?? 0
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

/// Chunked KV cache for processing large contexts in chunks
public class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
    private var cache: [MLXArray?]
    private var leftPadding: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    /// In-place filter to keep just the given indices in the cache
    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = nil
    }

    /// In-place extend this cache with the other cache
    public func extend(other: ArraysCache) {
        cache = zip(cache, other.cache).map { (c, o) in
            if let c = c, let o = o {
                return MLX.concatenated([c, o])
            }
            return c ?? o
        }
        leftPadding = nil
    }

    /// Create attention mask based on left padding
    public func makeMask(N: Int) -> MLXArray? {
        if cache[0] == nil, let leftPadding = leftPadding {
            return MLXArray(0 ..< N) .>= leftPadding[0..., .newAxis]
        } else {
            return nil
        }
    }
}

/// Simple cache for Mamba-style state space models
public class MambaCache: ArraysCache {
    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }
}

// MARK: - Error Types

struct KVCacheError: Error {
    let message: String
}

// MARK: - Utility Functions

/// Save a pre-computed prompt cache to a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    // Use Python-compatible class names for cross-platform compatibility
    let cacheClasses = cache.map { cache -> String in
        switch cache {
        case is ChunkedKVCache:
            return "ChunkedKVCache"  // Must precede KVCacheSimple because of inheritance
        case is KVCacheSimple:
            return "KVCache"  // Python uses "KVCache" for the basic cache
        case is RotatingKVCache:
            return "RotatingKVCache"
        case is QuantizedKVCache:
            return "QuantizedKVCache"
        case is TurboQuantKVCache:
            return "TurboQuantKVCache"
        case is MambaCache:
            return "MambaCache"  // Must precede ArraysCache because of inheritance
        case is ArraysCache:
            return "ArraysCache"
        case is CacheList:
            return "CacheList"
        default:
            return "KVCache"  // Default fallback
        }
    }

    // Flatten cache data using tree_flatten compatible structure: "i.j" format
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // Create cache_metadata structure compatible with Python: [cache_info, metadata, cache_classes]
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata)
    for (key, value) in metadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    try save(arrays: flattenedData, metadata: flattenedMetadata, url: url)
}

/// Load a prompt cache from a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)

    // Unflatten arrays using tree_unflatten compatible logic
    let cacheData = unflattenArrays(arrays)

    // Unflatten metadata using tree_unflatten compatible logic
    let unflattenedMetadata = unflattenMetadata(metadata)

    // Extract cache_info, user_metadata, and cache_classes from unflattened structure
    // Structure: [cache_info, user_metadata, cache_classes]
    guard unflattenedMetadata.count >= 3 else {
        throw KVCacheError(message: "Invalid cache metadata format")
    }

    let cacheInfo = unflattenedMetadata[0] as? [[String]] ?? []
    let userMetadata = unflattenedMetadata[1] as? [String: String] ?? [:]
    let cacheClasses = unflattenedMetadata[2] as? [String] ?? []

    guard cacheData.count == cacheInfo.count && cacheData.count == cacheClasses.count else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    // Reconstruct cache instances
    var caches: [KVCache] = []
    for i in 0 ..< cacheData.count {
        let className = cacheClasses[i]

        var cache: KVCache
        switch className {
        case "KVCache", "KVCacheSimple":  // Handle both Python and Swift names
            cache = KVCacheSimple()
        case "RotatingKVCache":
            // Parse metaState first to get maxSize, then create cache
            let info = i < cacheInfo.count ? cacheInfo[i] : []
            guard info.count >= 5 else {
                throw KVCacheError(message: "Invalid RotatingKVCache metaState - expected 5 values")
            }
            if info[1] == "None" {
                throw KVCacheError(
                    message:
                        "RotatingKVCache with maxSize=None is not supported. This cache was created with invalid parameters."
                )
            }
            guard let maxSize = Int(info[1]) else {
                throw KVCacheError(
                    message: "Failed to parse RotatingKVCache maxSize from: \(info[1])")
            }
            cache = RotatingKVCache(maxSize: maxSize)  // Create with parsed maxSize
        case "QuantizedKVCache":
            cache = QuantizedKVCache()
        case "TurboQuantKVCache":
            cache = TurboQuantKVCache()
        case "ChunkedKVCache":
            cache = ChunkedKVCache()
        case "MambaCache":
            cache = MambaCache()
        case "ArraysCache":
            // Size doesn't matter here as it's only needed to initialize the `cache` container inside
            // The container will be set as a `state` with correct size before returning a cache
            cache = ArraysCache(size: 0)
        case "CacheList":
            // Note: CacheList requires special handling as it contains sub-caches
            // For now, create an empty CacheList - this may not work correctly
            // for complex cache hierarchies loaded from Python
            cache = CacheList()
            print("Warning: CacheList loading may not preserve sub-cache structure correctly")
        default:
            throw KVCacheError(message: "Unknown cache class: \(className)")
        }

        cache.state = cacheData[i]
        if i < cacheInfo.count {
            cache.metaState = cacheInfo[i]
        }
        caches.append(cache)
    }

    return (caches, userMetadata)
}

/// Unflatten arrays from tree_flatten format (e.g., "0.1", "1.0") to nested structure
private func unflattenArrays(_ flatArrays: [String: MLXArray]) -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        if components.count >= 2,
            let i = Int(components[0]),
            let j = Int(components[1])
        {
            if arrayMap[i] == nil {
                arrayMap[i] = [:]
            }
            arrayMap[i]![j] = array
        }
    }

    // Convert to ordered array structure
    var result: [[MLXArray]] = []
    let maxI = arrayMap.keys.max() ?? -1

    for i in 0 ... maxI {
        if let innerMap = arrayMap[i] {
            let maxJ = innerMap.keys.max() ?? -1
            var innerArray: [MLXArray] = []
            for j in 0 ... maxJ {
                if let array = innerMap[j] {
                    innerArray.append(array)
                }
            }
            result.append(innerArray)
        } else {
            result.append([])
        }
    }

    return result
}

/// Unflatten metadata from tree_flatten format to nested structure
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) -> [KVCache] {
    if let maxKVSize = maxKVSize {
        return (0 ..< numLayers).map { _ in
            RotatingKVCache(maxSize: maxKVSize, keep: 4)
        }
    } else {
        return (0 ..< numLayers).map { _ in KVCacheSimple() }
    }
}

/// Create a per-layer cache instance for the given generation parameters.
///
/// TurboQuant must start at cache construction time to reduce the peak
/// memory footprint during prompt prefill. If we wait to swap in a compressed
/// cache after a generation step, the large temporary KV allocation has
/// already happened and long-context crashes occur at the same boundary.
public func makeLayerKVCache(
    parameters: GenerateParameters?,
    layerIndex: Int = 0
) -> KVCache {
    if let maxKVSize = parameters?.maxKVSize {
        return RotatingKVCache(maxSize: maxKVSize, keep: 4)
    }

    if case .turboQuant(let bits, _, let seed)? = parameters?.resolvedCacheCompression {
        return TurboQuantKVCache(bits: bits, seed: seed &+ UInt64(layerIndex))
    }

    return KVCacheSimple()
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Type Aliases

/// Standard KV cache - alias to KVCacheSimple for compatibility
public typealias StandardKVCache = KVCacheSimple

// MARK: - Quantized Attention Operations

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Apply mask
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, MLXArray(Float.leastNormalMagnitude))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    let attentionWeights = softmax(scores, axis: -1)

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits,
        mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output
}

// MARK: - Dynamic Cache Quantization

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Converts regular caches to quantized caches when:
/// - kvBits is specified
/// - The cache is not already quantized
/// - The cache offset is greater than quantizedKVStart
///
/// - Parameters:
///   - cache: Array of KV caches to potentially quantize
///   - kvBits: Number of bits for quantization (nil = no quantization)
///   - kvGroupSize: Group size for quantization
///   - quantizedKVStart: Token count threshold to begin quantizing
public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0
) {
    guard let kvBits = kvBits,
        !cache.isEmpty,
        !(cache[0] is QuantizedKVCache),
        cache[0].offset > quantizedKVStart
    else {
        return
    }

    for i in 0 ..< cache.count {
        // Handle cache types that support quantization
        if let simpleCache = cache[i] as? KVCacheSimple {
            cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
        }
        // TODO: RotatingKVCache.toQuantized() is not implemented yet, like in Python.
        // When implemented, add: else if let rotatingCache = cache[i] as? RotatingKVCache { ... }
        // MambaCache and CacheList don't use traditional KV quantization
    }
}

public func maybeTurboQuantizeKVCache(
    cache: inout [KVCache],
    bits: Int,
    startStep: Int = 0,
    seed: UInt64 = 42
) {
    guard !cache.isEmpty,
          !(cache[0] is TurboQuantKVCache),
          cache[0].offset > startStep
    else {
        return
    }

    for index in 0 ..< cache.count {
        if let simpleCache = cache[index] as? KVCacheSimple {
            cache[index] = simpleCache.toTurboQuantized(
                bits: bits,
                seed: seed &+ UInt64(index)
            )
        }
    }
}

public func maybeApplyKVCacheCompression(
    cache: inout [KVCache],
    compression: KVCacheCompressionMode?
) {
    guard let compression else { return }

    switch compression {
    case .quantized(let bits, let groupSize, let startStep):
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: bits,
            kvGroupSize: groupSize,
            quantizedKVStart: startStep
        )
    case .turboQuant(let bits, let startStep, let seed):
        maybeTurboQuantizeKVCache(
            cache: &cache,
            bits: bits,
            startStep: startStep,
            seed: seed
        )
    }
}
