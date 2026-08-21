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
/// Compressed caches can use this hook to combine approximate and exact cache
/// blocks without materializing the entire sequence at once.
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

private struct CentroidCodebook: Sendable {
    let centroids: [Float]
    let boundaries: [Float]
}

private struct OrthogonalTransform: Sendable {
    let matrix: [[Float]]
    let transpose: [[Float]]
}

private struct SeededGaussianRandom: Sendable {
    private var state: UInt64
    private var spare: Double?

    init(seed: UInt64) {
        self.state = seed &+ 0x9E37_79B9_7F4A_7C15
        self.spare = nil
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private mutating func nextUnit() -> Double {
        let raw = nextUInt64() >> 11
        return Double(raw) / Double(1 << 53)
    }

    mutating func nextGaussian() -> Float {
        if let spare {
            self.spare = nil
            return Float(spare)
        }

        var u1 = nextUnit()
        if u1 < 1e-12 {
            u1 = 1e-12
        }
        let u2 = nextUnit()
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Double.pi * u2
        spare = radius * sin(theta)
        return Float(radius * cos(theta))
    }
}

private enum CentroidQuantizationMath {
    static func orthogonalTransform(dimension: Int, seed: UInt64) -> OrthogonalTransform {
        precondition(dimension > 0, "Orthogonal transform requires a positive dimension")

        var generator = SeededGaussianRandom(seed: seed)
        var matrix = Array(
            repeating: Array(repeating: Float(0), count: dimension),
            count: dimension
        )
        for row in 0 ..< dimension {
            for column in 0 ..< dimension {
                matrix[row][column] = generator.nextGaussian()
            }
        }

        var qColumns = Array(
            repeating: Array(repeating: Float(0), count: dimension),
            count: dimension
        )

        for column in 0 ..< dimension {
            var vector = (0 ..< dimension).map { matrix[$0][column] }
            for previous in 0 ..< column {
                let projection = dot(vector, qColumns[previous])
                for index in 0 ..< dimension {
                    vector[index] -= projection * qColumns[previous][index]
                }
            }

            var norm = sqrt(max(dot(vector, vector), Float(1e-12)))
            if norm < 1e-6 {
                vector = Array(repeating: Float(0), count: dimension)
                vector[column] = 1
                for previous in 0 ..< column {
                    let projection = dot(vector, qColumns[previous])
                    for index in 0 ..< dimension {
                        vector[index] -= projection * qColumns[previous][index]
                    }
                }
                norm = sqrt(max(dot(vector, vector), Float(1e-12)))
            }

            for index in 0 ..< dimension {
                qColumns[column][index] = vector[index] / norm
            }
        }

        var rotation = Array(
            repeating: Array(repeating: Float(0), count: dimension),
            count: dimension
        )
        var transpose = Array(
            repeating: Array(repeating: Float(0), count: dimension),
            count: dimension
        )
        for row in 0 ..< dimension {
            for column in 0 ..< dimension {
                rotation[row][column] = qColumns[column][row]
                transpose[column][row] = rotation[row][column]
            }
        }
        return OrthogonalTransform(matrix: rotation, transpose: transpose)
    }

    static func codebook(dimension: Int, bits: Int) -> CentroidCodebook {
        precondition(dimension >= 3, "Centroid quantization requires dimension >= 3")
        precondition((1 ... 8).contains(bits), "MSE centroid stage supports 1-8 bits.")

        let clusterCount = 1 << bits
        let gridCount = 16_385
        let epsilon = Float(1e-4)
        let lowerBound = -1 + epsilon
        let upperBound = 1 - epsilon
        let dx = (upperBound - lowerBound) / Float(gridCount - 1)
        let halfDimension = Double(dimension) / 2.0
        let halfPreviousDimension = Double(dimension - 1) / 2.0
        let logConstant =
            Foundation.lgamma(halfDimension)
            - 0.5 * Foundation.log(Double.pi)
            - Foundation.lgamma(halfPreviousDimension)
        let constant = Float(Foundation.exp(logConstant))
        let exponent = Float(dimension - 3) / 2

        var grid = Array(repeating: Float(0), count: gridCount)
        var cdf = Array(repeating: Float(0), count: gridCount)
        var prefixWeight = Array(repeating: Float(0), count: gridCount + 1)
        var prefixWeightedX = Array(repeating: Float(0), count: gridCount + 1)
        var prefixWeightedX2 = Array(repeating: Float(0), count: gridCount + 1)
        var running = Float(0)

        for index in 0 ..< gridCount {
            let x = lowerBound + Float(index) * dx
            let value = max(1 - x * x, Float(1e-12))
            let density = constant * pow(value, exponent)
            let weight = density * dx
            grid[index] = x
            running += weight
            cdf[index] = running
            prefixWeight[index + 1] = prefixWeight[index] + weight
            prefixWeightedX[index + 1] = prefixWeightedX[index] + x * weight
            prefixWeightedX2[index + 1] = prefixWeightedX2[index] + x * x * weight
        }

        let cdfScale = max(running, Float(1e-12))
        for index in 0 ..< gridCount {
            cdf[index] /= cdfScale
        }

        var centroids = Array(repeating: Float(0), count: clusterCount)
        for cluster in 0 ..< clusterCount {
            let target = (Float(cluster) + 0.5) / Float(clusterCount)
            let chosenIndex = cdf.firstIndex(where: { $0 >= target }) ?? (gridCount - 1)
            centroids[cluster] = grid[chosenIndex]
        }

        let maxIterations = 200
        let tolerance = Float(1e-6)
        var previousCost = Float.infinity

        func firstGridIndex(greaterThan boundary: Float) -> Int {
            var low = 0
            var high = gridCount
            while low < high {
                let mid = (low + high) / 2
                if grid[mid] > boundary {
                    high = mid
                } else {
                    low = mid + 1
                }
            }
            return low
        }

        func rangeSums(lowerExclusive: Float, upperInclusive: Float)
            -> (weight: Float, weightedX: Float, weightedX2: Float)
        {
            let start = firstGridIndex(greaterThan: lowerExclusive)
            let end = firstGridIndex(greaterThan: upperInclusive)
            guard start < end else { return (0, 0, 0) }
            return (
                prefixWeight[end] - prefixWeight[start],
                prefixWeightedX[end] - prefixWeightedX[start],
                prefixWeightedX2[end] - prefixWeightedX2[start]
            )
        }

        for _ in 0 ..< maxIterations {
            let boundaries = fullBoundaries(for: centroids)
            var nextCentroids = Array(repeating: Float(0), count: clusterCount)
            var cost = Float(0)

            for cluster in 0 ..< clusterCount {
                let lower = boundaries[cluster]
                let upper = boundaries[cluster + 1]
                let sums = rangeSums(lowerExclusive: lower, upperInclusive: upper)

                let centroid =
                    sums.weight > 1e-12 ? sums.weightedX / sums.weight : (lower + upper) * 0.5
                nextCentroids[cluster] = centroid
            }

            let nextBoundaries = fullBoundaries(for: nextCentroids)
            for cluster in 0 ..< clusterCount {
                let lower = nextBoundaries[cluster]
                let upper = nextBoundaries[cluster + 1]
                let centroid = nextCentroids[cluster]
                let sums = rangeSums(lowerExclusive: lower, upperInclusive: upper)
                cost += sums.weightedX2 - 2 * centroid * sums.weightedX + centroid * centroid * sums.weight
            }

            centroids = nextCentroids
            if abs(previousCost - cost) < tolerance {
                break
            }
            previousCost = cost
        }

        return CentroidCodebook(
            centroids: centroids,
            boundaries: fullBoundaries(for: centroids)
        )
    }

    private static func fullBoundaries(for centroids: [Float]) -> [Float] {
        var boundaries = Array(repeating: Float(0), count: centroids.count + 1)
        boundaries[0] = -1
        boundaries[boundaries.count - 1] = 1
        for index in 0 ..< (centroids.count - 1) {
            boundaries[index + 1] = (centroids[index] + centroids[index + 1]) * 0.5
        }
        return boundaries
    }

    private static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float(0)) { partial, pair in
            partial + pair.0 * pair.1
        }
    }
}

private enum QuantizationParameterStore {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var orthogonalTransforms: [String: OrthogonalTransform] = [:]
    nonisolated(unsafe) private static var codebooks: [String: CentroidCodebook] = [:]

    static func orthogonalTransform(dimension: Int, seed: UInt64) -> OrthogonalTransform {
        let key = "\(dimension)-\(seed)"
        lock.lock()
        if let cached = orthogonalTransforms[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let transform = CentroidQuantizationMath.orthogonalTransform(dimension: dimension, seed: seed)

        lock.lock()
        orthogonalTransforms[key] = transform
        lock.unlock()
        return transform
    }

    static func codebook(dimension: Int, bits: Int) -> CentroidCodebook {
        let key = "\(dimension)-\(bits)"
        lock.lock()
        if let cached = codebooks[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let codebook = CentroidQuantizationMath.codebook(dimension: dimension, bits: bits)

        lock.lock()
        codebooks[key] = codebook
        lock.unlock()
        return codebook
    }
}

private final class FullMatrixCentroidQuantizer {
    let dimension: Int
    let centroids: MLXArray
    let inverseRotation: MLXArray

    init(dimension: Int, bits: Int, seed: UInt64) {
        precondition(dimension > 0, "Centroid migration requires a positive head dimension")
        precondition((1 ... 8).contains(bits), "Centroid migration supports 1-8 bits.")
        self.dimension = dimension
        let codebook = QuantizationParameterStore.codebook(dimension: max(dimension, 3), bits: bits)
        self.centroids = MLXArray(codebook.centroids).asType(.float32)
        let transform = QuantizationParameterStore.orthogonalTransform(dimension: dimension, seed: seed)
        self.inverseRotation = MLXArray(transform.transpose.flatMap { $0 }, [dimension, dimension])
            .asType(.float32)
    }

    func dequantize(indices: MLXArray, norms: MLXArray, dtype: DType) -> MLXArray {
        let shape = indices.shape
        let flatCount = shape.reduce(1, *)
        let flatIndices = indices.reshaped([flatCount])
        let rotated = centroids[flatIndices].reshaped(shape)
        let restoredUnit = matmul(rotated, inverseRotation)
        let restored = restoredUnit * expandedDimensions(norms.asType(.float32), axis: -1)
        return restored.asType(dtype)
    }

}

private enum PackedCentroidIndices {
    nonisolated(unsafe) private static let shiftValues = MLXArray((0 ..< 8).map(UInt8.init)).asType(.uint8)
    nonisolated(unsafe) private static let byteWeights =
        MLXArray([UInt32(1), 2, 4, 8, 16, 32, 64, 128]).asType(.uint32)
    nonisolated(unsafe) private static let unpackWeightsByBitWidth: [MLXArray] = (0 ... 8).map { bitWidth in
        let values = (0 ..< max(bitWidth, 1)).map { UInt32(1 << $0) }
        return MLXArray(values).asType(.uint32)
    }

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
        let expanded = packed.asType(.uint8).expandedDimensions(axis: -1)
        var bitStream = bitwiseAnd(rightShift(expanded, shiftValues), UInt8(1)).asType(.uint8)
            .reshaped(prefixShape + [packedCount * 8])

        let usedBits = valueCount * bitWidth
        if packedCount * 8 > usedBits {
            bitStream = bitStream[.ellipsis, ..<usedBits]
        }

        let valueWeights = unpackWeightsByBitWidth[bitWidth]
        let unpacked = sum(
            bitStream.reshaped(prefixShape + [valueCount, bitWidth]).asType(.uint32) * valueWeights,
            axis: -1
        )
        return unpacked.asType(.uint8)
    }
}

private func castIfNeeded(_ array: MLXArray, to dtype: DType) -> MLXArray {
    array.dtype == dtype ? array : array.asType(dtype)
}

private func serializedDType(_ name: String) -> DType {
    switch name {
    case "bfloat16":
        return .bfloat16
    case "float32":
        return .float32
    default:
        return .float16
    }
}

func migrateLegacyCompressedPromptCache(
    state: [MLXArray],
    metaState: [String]
) throws -> KVCacheSimple {
    guard [7, 10, 13].contains(metaState.count) else {
        throw KVCacheError(message: "Invalid legacy compressed cache metadata")
    }

    let legacyKeyTotalBits: Int
    let valueBits: Int
    let seed: UInt64
    let keyDimension: Int?
    let valueDimension: Int?
    let dtype: DType
    let compressedCount: Int
    let exactCount: Int

    if metaState.count == 13 {
        legacyKeyTotalBits = Int(metaState[2]) ?? 3
        valueBits = Int(metaState[3]) ?? 2
        seed = UInt64(metaState[4]) ?? 42
        keyDimension = (Int(metaState[7]) ?? 0) > 0 ? Int(metaState[7]) : nil
        valueDimension = (Int(metaState[8]) ?? 0) > 0 ? Int(metaState[8]) : nil
        dtype = serializedDType(metaState[9])
        compressedCount = Int(metaState[10]) ?? 0
        exactCount = Int(metaState[11]) ?? 0
    } else if metaState.count == 10 {
        legacyKeyTotalBits = Int(metaState[2]) ?? 3
        valueBits = 2
        seed = UInt64(metaState[3]) ?? 42
        keyDimension = (Int(metaState[4]) ?? 0) > 0 ? Int(metaState[4]) : nil
        valueDimension = (Int(metaState[5]) ?? 0) > 0 ? Int(metaState[5]) : nil
        dtype = serializedDType(metaState[6])
        compressedCount = Int(metaState[7]) ?? 0
        exactCount = Int(metaState[8]) ?? 0
    } else {
        legacyKeyTotalBits = Int(metaState[2]) ?? 3
        valueBits = 2
        seed = UInt64(metaState[3]) ?? 42
        keyDimension = (Int(metaState[4]) ?? 0) > 0 ? Int(metaState[4]) : nil
        valueDimension = (Int(metaState[5]) ?? 0) > 0 ? Int(metaState[5]) : nil
        dtype = serializedDType(metaState[6])
        compressedCount = Int(metaState[1]) ?? 0
        exactCount = 0
    }

    var keyBlocks: [MLXArray] = []
    var valueBlocks: [MLXArray] = []

    if state.count == 2 {
        keyBlocks.append(state[0].asType(dtype))
        valueBlocks.append(state[1].asType(dtype))
    } else if [6, 8].contains(state.count), compressedCount > 0 {
        guard let keyDimension, let valueDimension else {
            throw KVCacheError(message: "Legacy compressed cache is missing head dimensions")
        }
        let keyBits = max(1, legacyKeyTotalBits - 1)
        let keyQuantizer = FullMatrixCentroidQuantizer(
            dimension: keyDimension,
            bits: keyBits,
            seed: seed
        )
        let valueQuantizer = FullMatrixCentroidQuantizer(
            dimension: valueDimension,
            bits: valueBits,
            seed: seed &+ 1
        )
        let keyIndices = PackedCentroidIndices.unpack(
            state[0][.ellipsis, ..<compressedCount, 0...],
            bitWidth: keyBits,
            valueCount: keyDimension
        )
        let valueIndices = PackedCentroidIndices.unpack(
            state[4][.ellipsis, ..<compressedCount, 0...],
            bitWidth: valueBits,
            valueCount: valueDimension
        )
        keyBlocks.append(
            keyQuantizer.dequantize(
                indices: keyIndices,
                norms: state[1][.ellipsis, ..<compressedCount],
                dtype: dtype
            )
        )
        valueBlocks.append(
            valueQuantizer.dequantize(
                indices: valueIndices,
                norms: state[5][.ellipsis, ..<compressedCount],
                dtype: dtype
            )
        )
        if state.count == 8, exactCount > 0 {
            keyBlocks.append(state[6][.ellipsis, ..<exactCount, 0...].asType(dtype))
            valueBlocks.append(state[7][.ellipsis, ..<exactCount, 0...].asType(dtype))
        }
    } else if state.isEmpty {
        return KVCacheSimple()
    } else {
        throw KVCacheError(message: "Invalid legacy compressed cache state")
    }

    guard let firstKeys = keyBlocks.first, let firstValues = valueBlocks.first else {
        return KVCacheSimple()
    }

    let migrated = KVCacheSimple()
    if keyBlocks.count == 1 {
        migrated.state = [firstKeys, firstValues]
    } else {
        migrated.state = [
            concatenated(keyBlocks, axis: 2),
            concatenated(valueBlocks, axis: 2),
        ]
    }
    return migrated
}

private struct RotorQuantRotationParameters: Sendable {
    let variant: RotorQuantVariant
    let blockSize: Int
    let paddedDimension: Int
    let parameters: [Float]
}

private enum RotorQuantRotationFactory {
    static func effectiveVariant(_ requested: RotorQuantVariant, dimension: Int) -> RotorQuantVariant {
        switch requested {
        case .iso where dimension % 4 == 0:
            return .iso
        case .planar where dimension % 2 == 0:
            return .planar
        case .clifford where dimension % 3 == 0:
            return .clifford
        default:
            if dimension % 4 == 0 {
                return .iso
            }
            if dimension % 2 == 0 {
                return .planar
            }
            return requested
        }
    }

    static func make(dimension: Int, requestedVariant: RotorQuantVariant, seed: UInt64)
        -> RotorQuantRotationParameters
    {
        let variant = effectiveVariant(requestedVariant, dimension: dimension)
        var generator = SeededGaussianRandom(seed: seed)

        switch variant {
        case .iso:
            let blockSize = 4
            let groups = (dimension + blockSize - 1) / blockSize
            var parameters: [Float] = []
            parameters.reserveCapacity(groups * 4)
            for _ in 0 ..< groups {
                let q = normalized((0 ..< 4).map { _ in generator.nextGaussian() })
                parameters.append(contentsOf: q)
            }
            return RotorQuantRotationParameters(
                variant: .iso,
                blockSize: blockSize,
                paddedDimension: groups * blockSize,
                parameters: parameters
            )
        case .planar:
            let blockSize = 2
            let groups = (dimension + blockSize - 1) / blockSize
            var parameters: [Float] = []
            parameters.reserveCapacity(groups * 2)
            for _ in 0 ..< groups {
                let pair = normalized([generator.nextGaussian(), generator.nextGaussian()])
                parameters.append(contentsOf: pair)
            }
            return RotorQuantRotationParameters(
                variant: .planar,
                blockSize: blockSize,
                paddedDimension: groups * blockSize,
                parameters: parameters
            )
        case .clifford:
            let blockSize = 3
            let groups = (dimension + blockSize - 1) / blockSize
            var parameters: [Float] = []
            parameters.reserveCapacity(groups * 4)
            for _ in 0 ..< groups {
                let rotor = normalized((0 ..< 4).map { _ in generator.nextGaussian() })
                parameters.append(contentsOf: rotor)
            }
            return RotorQuantRotationParameters(
                variant: .clifford,
                blockSize: blockSize,
                paddedDimension: groups * blockSize,
                parameters: parameters
            )
        }
    }

    private static func normalized(_ values: [Float]) -> [Float] {
        let norm = sqrt(max(values.reduce(Float(0)) { $0 + $1 * $1 }, Float(1e-12)))
        return values.map { $0 / norm }
    }
}

private final class RotorQuantMSEQuantizer {
    let dimension: Int
    let bits: Int
    let variant: RotorQuantVariant
    let blockSize: Int
    let paddedDimension: Int
    let storageDimension: Int
    let centroids: MLXArray
    let boundariesArray: MLXArray
    let rotationParameters: MLXArray

    init(dimension: Int, bits: Int, variant requestedVariant: RotorQuantVariant, seed: UInt64) {
        precondition(dimension > 0, "RotorQuant requires a positive head dimension")
        precondition((1 ... 8).contains(bits), "RotorQuant supports 1-8 centroid bits.")
        self.dimension = dimension
        self.bits = bits

        let rotations = RotorQuantRotationFactory.make(
            dimension: dimension,
            requestedVariant: requestedVariant,
            seed: seed
        )
        self.variant = rotations.variant
        self.blockSize = rotations.blockSize
        self.paddedDimension =
            rotations.variant == .clifford
            ? ((dimension + rotations.blockSize - 1) / rotations.blockSize) * rotations.blockSize
            : rotations.paddedDimension
        self.storageDimension = self.paddedDimension

        let codebook = QuantizationParameterStore.codebook(dimension: max(dimension, 3), bits: bits)
        self.centroids = MLXArray(codebook.centroids).asType(.float32)
        self.boundariesArray = MLXArray(Array(codebook.boundaries.dropFirst().dropLast())).asType(.float32)

        let parameterWidth = rotations.variant == .planar ? 2 : 4
        self.rotationParameters = MLXArray(
            rotations.parameters,
            [rotations.parameters.count / parameterWidth, parameterWidth]
        ).asType(.float32)
    }

    func quantize(_ vectors: MLXArray) -> (indices: MLXArray, norms: MLXArray) {
        let floatVectors = vectors.asType(.float32)
        let norms = sqrt(sum(floatVectors * floatVectors, axis: -1, keepDims: true))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let rotated = rotate(floatVectors / safeNorms)
        let indices = sum(
            greater(
                rotated.expandedDimensions(axis: -1),
                boundariesArray
            ).asType(.uint8),
            axis: -1
        ).asType(.uint8)
        return (indices, norms.squeezed(axis: -1))
    }

    func dequantize(indices: MLXArray, norms: MLXArray, dtype: DType) -> MLXArray {
        let shape = indices.shape
        let flatCount = shape.reduce(1, *)
        let flatIndices = indices.reshaped([flatCount])
        let rotated = centroids[flatIndices].reshaped(shape)
        let restoredUnit = inverseRotate(rotated)
        let restored = restoredUnit * expandedDimensions(norms.asType(.float32), axis: -1)
        return restored.asType(dtype)
    }

    private func padToBlockDimension(_ vectors: MLXArray) -> MLXArray {
        guard paddedDimension > dimension else { return vectors }
        let paddingShape = Array(vectors.shape.dropLast()) + [paddedDimension - dimension]
        let padding = MLXArray.zeros(paddingShape, dtype: vectors.dtype)
        return concatenated([vectors, padding], axis: -1)
    }

    private func extractOriginalDimension(_ vectors: MLXArray) -> MLXArray {
        vectors[.ellipsis, ..<dimension]
    }

    private func rotate(_ vectors: MLXArray) -> MLXArray {
        switch variant {
        case .iso:
            let padded = padToBlockDimension(vectors)
            let grouped = padded.reshaped(Array(padded.shape.dropLast()) + [paddedDimension / 4, 4])
            let rotated = quatMultiply(rotationParameters, grouped)
            return rotated.reshaped(Array(vectors.shape.dropLast()) + [storageDimension])
        case .planar:
            let padded = padToBlockDimension(vectors)
            let grouped = padded.reshaped(Array(padded.shape.dropLast()) + [paddedDimension / 2, 2])
            let rotated = planarRotate(rotationParameters, grouped)
            return rotated.reshaped(Array(vectors.shape.dropLast()) + [storageDimension])
        case .clifford:
            let padded = padToBlockDimension(vectors)
            let grouped = padded.reshaped(Array(padded.shape.dropLast()) + [paddedDimension / 3, 3])
            let multivectors = embedCliffordVectors(grouped)
            let rotated = cliffordSandwich(rotationParameters, multivectors)
            return extractCliffordVectors(rotated).reshaped(
                Array(vectors.shape.dropLast()) + [storageDimension]
            )
        }
    }

    private func inverseRotate(_ rotated: MLXArray) -> MLXArray {
        switch variant {
        case .iso:
            let grouped = rotated.reshaped(Array(rotated.shape.dropLast()) + [paddedDimension / 4, 4])
            let restored = quatMultiply(quatConjugate(rotationParameters), grouped)
                .reshaped(Array(rotated.shape.dropLast()) + [paddedDimension])
            return extractOriginalDimension(restored)
        case .planar:
            let grouped = rotated.reshaped(Array(rotated.shape.dropLast()) + [paddedDimension / 2, 2])
            let restored = planarInverseRotate(rotationParameters, grouped)
                .reshaped(Array(rotated.shape.dropLast()) + [paddedDimension])
            return extractOriginalDimension(restored)
        case .clifford:
            let grouped = rotated.reshaped(Array(rotated.shape.dropLast()) + [paddedDimension / 3, 3])
            let multivectors = embedCliffordVectors(grouped)
            let restored = cliffordSandwich(cliffordReverseRotorParameters(rotationParameters), multivectors)
            let extracted = extractCliffordVectors(restored).reshaped(
                Array(rotated.shape.dropLast()) + [paddedDimension]
            )
            return extractOriginalDimension(extracted)
        }
    }

    private func planarRotate(_ parameters: MLXArray, _ vectors: MLXArray) -> MLXArray {
        let cs = parameters.split(parts: 2, axis: -1)
        let xy = vectors.split(parts: 2, axis: -1)
        return concatenated([
            cs[0] * xy[0] - cs[1] * xy[1],
            cs[1] * xy[0] + cs[0] * xy[1],
        ], axis: -1)
    }

    private func planarInverseRotate(_ parameters: MLXArray, _ vectors: MLXArray) -> MLXArray {
        let cs = parameters.split(parts: 2, axis: -1)
        let xy = vectors.split(parts: 2, axis: -1)
        return concatenated([
            cs[0] * xy[0] + cs[1] * xy[1],
            -cs[1] * xy[0] + cs[0] * xy[1],
        ], axis: -1)
    }

    private func quatConjugate(_ q: MLXArray) -> MLXArray {
        let parts = q.split(parts: 4, axis: -1)
        return concatenated([parts[0], -parts[1], -parts[2], -parts[3]], axis: -1)
    }

    private func quatMultiply(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
        let a = lhs.split(parts: 4, axis: -1)
        let b = rhs.split(parts: 4, axis: -1)
        let rw = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        let rx = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        let ry = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        let rz = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return concatenated([rw, rx, ry, rz], axis: -1)
    }

    private func embedCliffordVectors(_ vectors: MLXArray) -> MLXArray {
        let parts = vectors.split(parts: 3, axis: -1)
        let zero = MLXArray.zeros(parts[0].shape, dtype: vectors.dtype)
        return concatenated([
            zero, parts[0], parts[1], parts[2], zero, zero, zero, zero,
        ], axis: -1)
    }

    private func extractCliffordVectors(_ multivectors: MLXArray) -> MLXArray {
        let parts = multivectors.split(parts: 8, axis: -1)
        return concatenated([parts[1], parts[2], parts[3]], axis: -1)
    }

    private func cliffordReverseRotorParameters(_ parameters: MLXArray) -> MLXArray {
        let parts = parameters.split(parts: 4, axis: -1)
        return concatenated([parts[0], -parts[1], -parts[2], -parts[3]], axis: -1)
    }

    private func cliffordReverseMultivector(_ multivectors: MLXArray) -> MLXArray {
        let parts = multivectors.split(parts: 8, axis: -1)
        return concatenated([
            parts[0], parts[1], parts[2], parts[3],
            -parts[4], -parts[5], -parts[6], -parts[7],
        ], axis: -1)
    }

    private func cliffordSandwich(_ rotorParameters: MLXArray, _ multivectors: MLXArray) -> MLXArray {
        let zero = MLXArray.zeros([rotorParameters.dim(0), 1], dtype: rotorParameters.dtype)
        let rotorParts = rotorParameters.split(parts: 4, axis: -1)
        let rotor = concatenated([
            rotorParts[0], zero, zero, zero,
            rotorParts[1], rotorParts[2], rotorParts[3], zero,
        ], axis: -1)
        return geometricProduct(geometricProduct(rotor, multivectors), cliffordReverseMultivector(rotor))
    }

    private func geometricProduct(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
        let a = lhs.split(parts: 8, axis: -1)
        let b = rhs.split(parts: 8, axis: -1)
        let r0 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
            - a[4] * b[4] - a[5] * b[5] - a[6] * b[6] - a[7] * b[7]
        let r1 = a[0] * b[1] + a[1] * b[0] - a[2] * b[4] + a[4] * b[2]
            - a[3] * b[5] + a[5] * b[3] - a[6] * b[7] - a[7] * b[6]
        let r2 = a[0] * b[2] + a[2] * b[0] + a[1] * b[4] - a[4] * b[1]
            - a[3] * b[6] + a[6] * b[3] + a[5] * b[7] + a[7] * b[5]
        let r3 = a[0] * b[3] + a[3] * b[0] + a[1] * b[5] - a[5] * b[1]
            + a[2] * b[6] - a[6] * b[2] - a[4] * b[7] - a[7] * b[4]
        let r12 = a[0] * b[4] + a[4] * b[0] + a[1] * b[2] - a[2] * b[1]
            - a[5] * b[6] + a[6] * b[5] + a[3] * b[7] + a[7] * b[3]
        let r13 = a[0] * b[5] + a[5] * b[0] + a[1] * b[3] - a[3] * b[1]
            + a[4] * b[6] - a[6] * b[4] - a[2] * b[7] - a[7] * b[2]
        let r23 = a[0] * b[6] + a[6] * b[0] + a[2] * b[3] - a[3] * b[2]
            - a[4] * b[5] + a[5] * b[4] + a[1] * b[7] + a[7] * b[1]
        let r123 = a[0] * b[7] + a[7] * b[0] + a[1] * b[6] - a[6] * b[1]
            - a[2] * b[5] - a[5] * b[2] + a[3] * b[4] + a[4] * b[3]
        return concatenated([r0, r1, r2, r3, r12, r13, r23, r123], axis: -1)
    }
}

private func makeRotorQuantIsoDecodeAttentionKernel() -> MLXFast.MLXFastKernel {
    let header = """
        uint unpack_rotor_index(const device uint8_t* row, uint value_index, uint bit_width) {
            uint bit_offset = value_index * bit_width;
            uint byte_offset = bit_offset >> 3;
            uint shift = bit_offset & 7;
            uint word = row[byte_offset];
            if (shift + bit_width > 8) {
                word |= uint(row[byte_offset + 1]) << 8;
            }
            uint mask = (1u << bit_width) - 1u;
            return (word >> shift) & mask;
        }

        float4 rotor_quat_multiply(float4 a, float4 b) {
            return float4(
                a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
            );
        }

        template <typename Norms, typename Centroids>
        float4 rotor_load_quantized_iso_group(
            const device uint8_t* row,
            Norms norms,
            Centroids centroids,
            uint token,
            uint group,
            uint packed_width,
            uint bit_width
        ) {
            const device uint8_t* packed = row + token * packed_width;
            uint base_index = group << 2;
            return float4(
                centroids[unpack_rotor_index(packed, base_index, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 1, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 2, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 3, bit_width)]
            ) * norms[token];
        }

        float4 rotor_rotate_iso_group(
            float4 value,
            const device float* rotations,
            uint group
        ) {
            float4 q = float4(
                rotations[group * 4],
                rotations[group * 4 + 1],
                rotations[group * 4 + 2],
                rotations[group * 4 + 3]
            );
            return rotor_quat_multiply(q, value);
        }

        float4 rotor_inverse_rotate_iso_group(
            float4 rotated,
            const device float* rotations,
            uint group
        ) {
            float4 q = float4(
                rotations[group * 4],
                -rotations[group * 4 + 1],
                -rotations[group * 4 + 2],
                -rotations[group * 4 + 3]
            );
            return rotor_quat_multiply(q, rotated);
        }
        """

    let source = """
        uint lane = thread_position_in_grid.x;
        uint kv_head_linear = thread_position_in_grid.y;
        uint batch = kv_head_linear / KV_HEADS;
        uint kv_head = kv_head_linear - batch * KV_HEADS;

        const device uint8_t* key_rows = key_indices + ((batch * KV_HEADS + kv_head) * KEY_CAPACITY * KEY_PACKED_WIDTH);
        auto key_norm_rows = key_norms + ((batch * KV_HEADS + kv_head) * KEY_CAPACITY);
        const device uint8_t* value_rows = value_indices + ((batch * KV_HEADS + kv_head) * VALUE_CAPACITY * VALUE_PACKED_WIDTH);
        auto value_norm_rows = value_norms + ((batch * KV_HEADS + kv_head) * VALUE_CAPACITY);
        const device T* exact_key_rows = exact_keys + ((batch * KV_HEADS + kv_head) * EXACT_CAPACITY * D);
        const device T* exact_value_rows = exact_values + ((batch * KV_HEADS + kv_head) * EXACT_CAPACITY * D);
        uint compressed_count = uint(compressed_count_scalar);
        uint exact_count = uint(exact_count_scalar);

        constexpr uint GROUPS_PER_LANE = D / 128;
        constexpr uint WORK_ITEMS = HEAD_REPEATS * GROUPS_PER_LANE;
        float4 q_rot[WORK_ITEMS];
        float4 accum_rot[WORK_ITEMS];
        float max_score[HEAD_REPEATS];
        float sum_exp[HEAD_REPEATS];
        uint groups[GROUPS_PER_LANE];

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            max_score[repeat] = -INFINITY;
            sum_exp[repeat] = 0.0f;
        }

        for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
            uint group = lane + i * 32;
            uint base = group << 2;
            groups[i] = group;
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                uint query_head = kv_head * HEAD_REPEATS + repeat;
                const device T* query_row = queries + ((batch * QUERY_HEADS + query_head) * D);
                float4 query_group = float4(
                    static_cast<float>(query_row[base]),
                    static_cast<float>(query_row[base + 1]),
                    static_cast<float>(query_row[base + 2]),
                    static_cast<float>(query_row[base + 3])
                );
                uint index = repeat * GROUPS_PER_LANE + i;
                q_rot[index] = rotor_rotate_iso_group(query_group, key_rotations, group);
                accum_rot[index] = float4(0.0f);
            }
        }

        for (uint token = 0; token < compressed_count; ++token) {
            float partial_score[HEAD_REPEATS];
            float4 value_groups[GROUPS_PER_LANE];
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                partial_score[repeat] = 0.0f;
            }

            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                float4 key_group = rotor_load_quantized_iso_group(
                    key_rows,
                    key_norm_rows,
                    key_centroids,
                    token,
                    group,
                    KEY_PACKED_WIDTH,
                    KEY_BITS
                );
                value_groups[i] = rotor_load_quantized_iso_group(
                    value_rows,
                    value_norm_rows,
                    value_centroids,
                    token,
                    group,
                    VALUE_PACKED_WIDTH,
                    VALUE_BITS
                );
                for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                    partial_score[repeat] += dot(q_rot[repeat * GROUPS_PER_LANE + i], key_group);
                }
            }

            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                float score = simd_sum(partial_score[repeat]) * scale;
                float next_max = max(max_score[repeat], score);
                float old_weight = fast::exp(max_score[repeat] - next_max);
                float new_weight = fast::exp(score - next_max);

                for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                    uint index = repeat * GROUPS_PER_LANE + i;
                    accum_rot[index] = accum_rot[index] * old_weight + value_groups[i] * new_weight;
                }
                sum_exp[repeat] = sum_exp[repeat] * old_weight + new_weight;
                max_score[repeat] = next_max;
            }
        }

        for (uint token = 0; token < exact_count; ++token) {
            const device T* key_row = exact_key_rows + token * D;
            float partial_score[HEAD_REPEATS];
            float4 value_groups[GROUPS_PER_LANE];
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                partial_score[repeat] = 0.0f;
            }

            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                uint base = group << 2;
                const device T* key_group_row = key_row + base;
                const device T* value_group_row = exact_value_rows + token * D + base;
                float4 key_group = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(key_group_row[0]),
                        static_cast<float>(key_group_row[1]),
                        static_cast<float>(key_group_row[2]),
                        static_cast<float>(key_group_row[3])
                    ),
                    key_rotations,
                    group
                );
                value_groups[i] = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(value_group_row[0]),
                        static_cast<float>(value_group_row[1]),
                        static_cast<float>(value_group_row[2]),
                        static_cast<float>(value_group_row[3])
                    ),
                    value_rotations,
                    group
                );
                for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                    partial_score[repeat] += dot(q_rot[repeat * GROUPS_PER_LANE + i], key_group);
                }
            }

            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                float score = simd_sum(partial_score[repeat]) * scale;
                float next_max = max(max_score[repeat], score);
                float old_weight = fast::exp(max_score[repeat] - next_max);
                float new_weight = fast::exp(score - next_max);

                for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                    uint index = repeat * GROUPS_PER_LANE + i;
                    accum_rot[index] = accum_rot[index] * old_weight + value_groups[i] * new_weight;
                }
                sum_exp[repeat] = sum_exp[repeat] * old_weight + new_weight;
                max_score[repeat] = next_max;
            }
        }

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            uint query_head = kv_head * HEAD_REPEATS + repeat;
            device T* out_row = output + ((batch * QUERY_HEADS + query_head) * D);
            float inv_sum = 1.0f / max(sum_exp[repeat], 1e-20f);
            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                uint index = repeat * GROUPS_PER_LANE + i;
                float4 restored = rotor_inverse_rotate_iso_group(accum_rot[index] * inv_sum, value_rotations, group);
                uint base = group << 2;
                out_row[base] = static_cast<T>(restored.x);
                out_row[base + 1] = static_cast<T>(restored.y);
                out_row[base + 2] = static_cast<T>(restored.z);
                out_row[base + 3] = static_cast<T>(restored.w);
            }
        }
        """

    return MLXFast.metalKernel(
        name: "rotorquant_iso_decode_attention",
        inputNames: [
            "queries",
            "key_indices",
            "key_norms",
            "value_indices",
            "value_norms",
            "exact_keys",
            "exact_values",
            "key_centroids",
            "value_centroids",
            "key_rotations",
            "value_rotations",
            "compressed_count_scalar",
            "exact_count_scalar",
            "scale",
        ],
        outputNames: ["output"],
        source: source,
        header: header
    )
}

private func makeRotorQuantIsoCompressedBlockKernel() -> MLXFast.MLXFastKernel {
    let header = """
        uint unpack_rotor_index(const device uint8_t* row, uint value_index, uint bit_width) {
            uint bit_offset = value_index * bit_width;
            uint byte_offset = bit_offset >> 3;
            uint shift = bit_offset & 7;
            uint word = row[byte_offset];
            if (shift + bit_width > 8) {
                word |= uint(row[byte_offset + 1]) << 8;
            }
            uint mask = (1u << bit_width) - 1u;
            return (word >> shift) & mask;
        }

        float4 rotor_quat_multiply(float4 a, float4 b) {
            return float4(
                a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
            );
        }

        template <typename Norms, typename Centroids>
        float4 rotor_load_quantized_iso_group(
            const device uint8_t* row,
            Norms norms,
            Centroids centroids,
            uint token,
            uint group,
            uint packed_width,
            uint bit_width
        ) {
            const device uint8_t* packed = row + token * packed_width;
            uint base_index = group << 2;
            return float4(
                centroids[unpack_rotor_index(packed, base_index, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 1, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 2, bit_width)],
                centroids[unpack_rotor_index(packed, base_index + 3, bit_width)]
            ) * norms[token];
        }

        float4 rotor_rotate_iso_group(
            float4 value,
            const device float* rotations,
            uint group
        ) {
            float4 q = float4(
                rotations[group * 4],
                rotations[group * 4 + 1],
                rotations[group * 4 + 2],
                rotations[group * 4 + 3]
            );
            return rotor_quat_multiply(q, value);
        }
        """

    let source = """
        uint lane = thread_position_in_grid.x;
        uint linear = thread_position_in_grid.y;
        uint active_block_count = uint(active_block_count_scalar);
        uint block = linear % active_block_count;
        uint kv_linear = linear / active_block_count;
        uint batch = kv_linear / KV_HEADS;
        uint kv_head = kv_linear - batch * KV_HEADS;
        uint token_start = block * BLOCK_TOKENS;
        uint compressed_count = uint(compressed_count_scalar);
        uint token_end = min(token_start + uint(BLOCK_TOKENS), compressed_count);

        const device uint8_t* key_rows = key_indices + ((batch * KV_HEADS + kv_head) * KEY_CAPACITY * KEY_PACKED_WIDTH);
        auto key_norm_rows = key_norms + ((batch * KV_HEADS + kv_head) * KEY_CAPACITY);
        const device uint8_t* value_rows = value_indices + ((batch * KV_HEADS + kv_head) * VALUE_CAPACITY * VALUE_PACKED_WIDTH);
        auto value_norm_rows = value_norms + ((batch * KV_HEADS + kv_head) * VALUE_CAPACITY);

        constexpr uint GROUPS_PER_LANE = D / 128;
        constexpr uint WORK_ITEMS = HEAD_REPEATS * GROUPS_PER_LANE;
        float4 q_rot[WORK_ITEMS];
        float4 accum_rot[WORK_ITEMS];
        float max_score[HEAD_REPEATS];
        float sum_exp[HEAD_REPEATS];
        uint groups[GROUPS_PER_LANE];

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            max_score[repeat] = -INFINITY;
            sum_exp[repeat] = 0.0f;
        }

        for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
            uint group = lane + i * 32;
            uint base = group << 2;
            groups[i] = group;
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                uint query_head = kv_head * HEAD_REPEATS + repeat;
                const device T* query_row = queries + ((batch * QUERY_HEADS + query_head) * D);
                uint index = repeat * GROUPS_PER_LANE + i;
                q_rot[index] = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(query_row[base]),
                        static_cast<float>(query_row[base + 1]),
                        static_cast<float>(query_row[base + 2]),
                        static_cast<float>(query_row[base + 3])
                    ),
                    key_rotations,
                    group
                );
                accum_rot[index] = float4(0.0f);
            }
        }

        for (uint token = token_start; token < token_end; ++token) {
            float partial_score[HEAD_REPEATS];
            float4 value_groups[GROUPS_PER_LANE];
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                partial_score[repeat] = 0.0f;
            }

            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                float4 key_group = rotor_load_quantized_iso_group(
                    key_rows,
                    key_norm_rows,
                    key_centroids,
                    token,
                    group,
                    KEY_PACKED_WIDTH,
                    KEY_BITS
                );
                value_groups[i] = rotor_load_quantized_iso_group(
                    value_rows,
                    value_norm_rows,
                    value_centroids,
                    token,
                    group,
                    VALUE_PACKED_WIDTH,
                    VALUE_BITS
                );
                for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                    partial_score[repeat] += dot(q_rot[repeat * GROUPS_PER_LANE + i], key_group);
                }
            }

            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                float score = simd_sum(partial_score[repeat]) * scale;
                float next_max = max(max_score[repeat], score);
                float old_weight = fast::exp(max_score[repeat] - next_max);
                float new_weight = fast::exp(score - next_max);

                for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                    uint index = repeat * GROUPS_PER_LANE + i;
                    accum_rot[index] = accum_rot[index] * old_weight + value_groups[i] * new_weight;
                }
                sum_exp[repeat] = sum_exp[repeat] * old_weight + new_weight;
                max_score[repeat] = next_max;
            }
        }

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            uint query_head = kv_head * HEAD_REPEATS + repeat;
            uint stats_index = (batch * QUERY_HEADS + query_head) * BLOCK_CAPACITY + block;
            device T* partial_row = partial_values + (stats_index * D);
            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint base = groups[i] << 2;
                uint index = repeat * GROUPS_PER_LANE + i;
                partial_row[base] = static_cast<T>(accum_rot[index].x);
                partial_row[base + 1] = static_cast<T>(accum_rot[index].y);
                partial_row[base + 2] = static_cast<T>(accum_rot[index].z);
                partial_row[base + 3] = static_cast<T>(accum_rot[index].w);
            }
            partial_max[stats_index] = max_score[repeat];
            partial_sum[stats_index] = sum_exp[repeat];
        }
        """

    return MLXFast.metalKernel(
        name: "rotorquant_iso_compressed_blocks",
        inputNames: [
            "queries",
            "key_indices",
            "key_norms",
            "value_indices",
            "value_norms",
            "key_centroids",
            "value_centroids",
            "key_rotations",
            "compressed_count_scalar",
            "active_block_count_scalar",
            "scale",
        ],
        outputNames: ["partial_values", "partial_max", "partial_sum"],
        source: source,
        header: header
    )
}

private func makeRotorQuantIsoBlockReduceKernel() -> MLXFast.MLXFastKernel {
    let header = """
        float4 rotor_quat_multiply(float4 a, float4 b) {
            return float4(
                a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
            );
        }

        float4 rotor_rotate_iso_group(
            float4 value,
            const device float* rotations,
            uint group
        ) {
            float4 q = float4(
                rotations[group * 4],
                rotations[group * 4 + 1],
                rotations[group * 4 + 2],
                rotations[group * 4 + 3]
            );
            return rotor_quat_multiply(q, value);
        }

        float4 rotor_inverse_rotate_iso_group(
            float4 rotated,
            const device float* rotations,
            uint group
        ) {
            float4 q = float4(
                rotations[group * 4],
                -rotations[group * 4 + 1],
                -rotations[group * 4 + 2],
                -rotations[group * 4 + 3]
            );
            return rotor_quat_multiply(q, rotated);
        }
        """

    let source = """
        uint lane = thread_position_in_grid.x;
        uint kv_head_linear = thread_position_in_grid.y;
        uint batch = kv_head_linear / KV_HEADS;
        uint kv_head = kv_head_linear - batch * KV_HEADS;
        uint active_block_count = uint(active_block_count_scalar);
        uint exact_count = uint(exact_count_scalar);

        const device T* exact_key_rows = exact_keys + ((batch * KV_HEADS + kv_head) * EXACT_CAPACITY * D);
        const device T* exact_value_rows = exact_values + ((batch * KV_HEADS + kv_head) * EXACT_CAPACITY * D);

        constexpr uint GROUPS_PER_LANE = D / 128;
        constexpr uint WORK_ITEMS = HEAD_REPEATS * GROUPS_PER_LANE;
        float4 q_rot[WORK_ITEMS];
        float4 accum_rot[WORK_ITEMS];
        float max_score[HEAD_REPEATS];
        float sum_exp[HEAD_REPEATS];
        uint groups[GROUPS_PER_LANE];

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            max_score[repeat] = -INFINITY;
            sum_exp[repeat] = 0.0f;
        }

        for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
            uint group = lane + i * 32;
            uint base = group << 2;
            groups[i] = group;
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                uint query_head = kv_head * HEAD_REPEATS + repeat;
                const device T* query_row = queries + ((batch * QUERY_HEADS + query_head) * D);
                uint index = repeat * GROUPS_PER_LANE + i;
                q_rot[index] = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(query_row[base]),
                        static_cast<float>(query_row[base + 1]),
                        static_cast<float>(query_row[base + 2]),
                        static_cast<float>(query_row[base + 3])
                    ),
                    key_rotations,
                    group
                );
                accum_rot[index] = float4(0.0f);
            }
        }

        for (uint block = 0; block < active_block_count; ++block) {
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                uint query_head = kv_head * HEAD_REPEATS + repeat;
                uint stats_index = (batch * QUERY_HEADS + query_head) * BLOCK_CAPACITY + block;
                float block_sum = partial_sum[stats_index];
                if (block_sum <= 0.0f) {
                    continue;
                }
                float block_max = partial_max[stats_index];
                float next_max = max(max_score[repeat], block_max);
                float old_weight = fast::exp(max_score[repeat] - next_max);
                float new_weight = fast::exp(block_max - next_max);
                const device T* partial_row = partial_values + (stats_index * D);

                for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                    uint base = groups[i] << 2;
                    uint index = repeat * GROUPS_PER_LANE + i;
                    float4 block_value = float4(
                        static_cast<float>(partial_row[base]),
                        static_cast<float>(partial_row[base + 1]),
                        static_cast<float>(partial_row[base + 2]),
                        static_cast<float>(partial_row[base + 3])
                    );
                    accum_rot[index] = accum_rot[index] * old_weight + block_value * new_weight;
                }
                sum_exp[repeat] = sum_exp[repeat] * old_weight + block_sum * new_weight;
                max_score[repeat] = next_max;
            }
        }

        for (uint token = 0; token < exact_count; ++token) {
            const device T* key_row = exact_key_rows + token * D;
            float partial_score[HEAD_REPEATS];
            float4 value_groups[GROUPS_PER_LANE];
            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                partial_score[repeat] = 0.0f;
            }

            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                uint base = group << 2;
                const device T* key_group_row = key_row + base;
                const device T* value_group_row = exact_value_rows + token * D + base;
                float4 key_group = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(key_group_row[0]),
                        static_cast<float>(key_group_row[1]),
                        static_cast<float>(key_group_row[2]),
                        static_cast<float>(key_group_row[3])
                    ),
                    key_rotations,
                    group
                );
                value_groups[i] = rotor_rotate_iso_group(
                    float4(
                        static_cast<float>(value_group_row[0]),
                        static_cast<float>(value_group_row[1]),
                        static_cast<float>(value_group_row[2]),
                        static_cast<float>(value_group_row[3])
                    ),
                    value_rotations,
                    group
                );
                for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                    partial_score[repeat] += dot(q_rot[repeat * GROUPS_PER_LANE + i], key_group);
                }
            }

            for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
                float score = simd_sum(partial_score[repeat]) * scale;
                float next_max = max(max_score[repeat], score);
                float old_weight = fast::exp(max_score[repeat] - next_max);
                float new_weight = fast::exp(score - next_max);

                for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                    uint index = repeat * GROUPS_PER_LANE + i;
                    accum_rot[index] = accum_rot[index] * old_weight + value_groups[i] * new_weight;
                }
                sum_exp[repeat] = sum_exp[repeat] * old_weight + new_weight;
                max_score[repeat] = next_max;
            }
        }

        for (uint repeat = 0; repeat < HEAD_REPEATS; ++repeat) {
            uint query_head = kv_head * HEAD_REPEATS + repeat;
            device T* out_row = output + ((batch * QUERY_HEADS + query_head) * D);
            float inv_sum = 1.0f / max(sum_exp[repeat], 1e-20f);
            for (uint i = 0; i < GROUPS_PER_LANE; ++i) {
                uint group = groups[i];
                uint base = group << 2;
                uint index = repeat * GROUPS_PER_LANE + i;
                float4 restored = rotor_inverse_rotate_iso_group(accum_rot[index] * inv_sum, value_rotations, group);
                out_row[base] = static_cast<T>(restored.x);
                out_row[base + 1] = static_cast<T>(restored.y);
                out_row[base + 2] = static_cast<T>(restored.z);
                out_row[base + 3] = static_cast<T>(restored.w);
            }
        }
        """

    return MLXFast.metalKernel(
        name: "rotorquant_iso_block_reduce",
        inputNames: [
            "partial_values",
            "partial_max",
            "partial_sum",
            "queries",
            "exact_keys",
            "exact_values",
            "key_rotations",
            "value_rotations",
            "active_block_count_scalar",
            "exact_count_scalar",
            "scale",
        ],
        outputNames: ["output"],
        source: source,
        header: header
    )
}

private final class RotorQuantDecodeAttentionKernelManager: Sendable {
    static let shared = RotorQuantDecodeAttentionKernelManager()

    let isoDecodeAttentionKernel: MLXFast.MLXFastKernel
    let isoCompressedBlockKernel: MLXFast.MLXFastKernel
    let isoBlockReduceKernel: MLXFast.MLXFastKernel

    private init() {
        isoDecodeAttentionKernel = makeRotorQuantIsoDecodeAttentionKernel()
        isoCompressedBlockKernel = makeRotorQuantIsoCompressedBlockKernel()
        isoBlockReduceKernel = makeRotorQuantIsoBlockReduceKernel()
    }
}

public final class RotorQuantKVCache: BaseKVCache, AttentionCapableKVCache {
    private var keyIndices: MLXArray?
    private var keyNorms: MLXArray?
    private var valueIndices: MLXArray?
    private var valueNorms: MLXArray?
    private var exactKeys: MLXArray?
    private var exactValues: MLXArray?
    private var compressedCount = 0
    private var compressedCapacity = 0
    private var exactCount = 0
    private var exactCapacity = 0
    private var step: Int
    public private(set) var configuration: RotorQuantConfiguration
    private var keyQuantizer: RotorQuantMSEQuantizer?
    private var valueQuantizer: RotorQuantMSEQuantizer?
    private var keyDimension: Int?
    private var valueDimension: Int?
    private var originalDType: DType = .float16

    public var keyBits: Int { configuration.keyBits }
    public var valueBits: Int { configuration.valueBits }
    public var effectiveVariant: RotorQuantVariant { keyQuantizer?.variant ?? configuration.variant }
    private var attentionBlockTokens: Int { max(configuration.attentionBlockTokens, 1) }
    private var exactFlushSlack: Int {
        guard exactBufferSize > 0 else { return 0 }
        return min(16, max(1, exactBufferSize / 8))
    }
    private var exactBufferSize: Int {
        get { configuration.exactBufferSize }
        set { configuration.exactBufferSize = newValue }
    }
    private var keyPackedWidth: Int {
        guard let keyQuantizer else { return 0 }
        return PackedCentroidIndices.packedByteCount(
            valueCount: keyQuantizer.storageDimension,
            bitWidth: configuration.keyBits
        )
    }
    private var valuePackedWidth: Int {
        guard let valueQuantizer else { return 0 }
        return PackedCentroidIndices.packedByteCount(
            valueCount: valueQuantizer.storageDimension,
            bitWidth: configuration.valueBits
        )
    }

    public init(configuration: RotorQuantConfiguration = RotorQuantConfiguration(), step: Int = 256) {
        precondition((1 ... 8).contains(configuration.keyBits), "RotorQuant supports 1-8 key bits.")
        precondition((1 ... 8).contains(configuration.valueBits), "RotorQuant supports 1-8 value bits.")
        self.configuration = configuration
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [keyIndices, keyNorms, valueIndices, valueNorms, exactKeys, exactValues].compactMap { $0 }
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
        if let fused = fusedIsoDecodeAttention(queries: queries, scale: scale, mask: mask) {
            return fused
        }
        let (cachedKeys, cachedValues) = currentFallbackState()
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }

    private func fusedIsoDecodeAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        switch mask {
        case .none, .causal:
            break
        case .array, .arrays:
            return nil
        }
        guard compressedCount > 0,
              queries.shape.count == 4,
              queries.dim(2) == 1,
              let keyQuantizer,
              let valueQuantizer,
              keyQuantizer.variant == .iso,
              valueQuantizer.variant == .iso,
              keyQuantizer.dimension == valueQuantizer.dimension,
              keyQuantizer.dimension == queries.dim(3),
              [128, 256].contains(keyQuantizer.dimension),
              queries.dim(1) % max(1, keyIndices?.dim(1) ?? 1) == 0,
              let keyIndices,
              let keyNorms,
              let valueIndices,
              let valueNorms,
              let exactKeys,
              let exactValues
        else {
            return nil
        }

        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let kvHeads = keyIndices.dim(1)
        guard kvHeads > 0,
              queryHeads % kvHeads == 0,
              (1 ... 4).contains(queryHeads / kvHeads),
              keyIndices.dim(0) == batch,
              valueIndices.dim(0) == batch,
              exactKeys.dim(0) == batch,
              exactValues.dim(0) == batch,
              exactKeys.dim(1) == kvHeads,
              exactValues.dim(1) == kvHeads
        else {
            return nil
        }

        let blockTokens = max(16, attentionBlockTokens)
        let blockCount = (compressedCount + blockTokens - 1) / blockTokens
        let blockCapacity = (max(compressedCount, keyIndices.dim(2)) + blockTokens - 1) / blockTokens
        if blockCount > 1 {
            return blockParallelIsoDecodeAttention(
                queries: queries,
                scale: scale,
                batch: batch,
                queryHeads: queryHeads,
                kvHeads: kvHeads,
                headRepeats: queryHeads / kvHeads,
                blockTokens: blockTokens,
                blockCount: blockCount,
                blockCapacity: blockCapacity,
                keyIndices: keyIndices,
                keyNorms: keyNorms,
                valueIndices: valueIndices,
                valueNorms: valueNorms,
                exactKeys: exactKeys,
                exactValues: exactValues,
                keyQuantizer: keyQuantizer,
                valueQuantizer: valueQuantizer
            )
        }

        return RotorQuantDecodeAttentionKernelManager.shared.isoDecodeAttentionKernel(
            [
                queries,
                keyIndices,
                keyNorms,
                valueIndices,
                valueNorms,
                exactKeys,
                exactValues,
                keyQuantizer.centroids,
                valueQuantizer.centroids,
                keyQuantizer.rotationParameters,
                valueQuantizer.rotationParameters,
                Int32(compressedCount),
                Int32(exactCount),
                scale,
            ],
            template: [
                ("T", queries.dtype),
                ("D", keyQuantizer.dimension),
                ("QUERY_HEADS", queryHeads),
                ("KV_HEADS", kvHeads),
                ("HEAD_REPEATS", queryHeads / kvHeads),
                ("KEY_CAPACITY", keyIndices.dim(2)),
                ("VALUE_CAPACITY", valueIndices.dim(2)),
                ("EXACT_CAPACITY", exactKeys.dim(2)),
                ("KEY_PACKED_WIDTH", keyIndices.dim(3)),
                ("VALUE_PACKED_WIDTH", valueIndices.dim(3)),
                ("KEY_BITS", configuration.keyBits),
                ("VALUE_BITS", configuration.valueBits),
            ],
            grid: (32, batch * kvHeads, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, queryHeads, 1, keyQuantizer.dimension]],
            outputDTypes: [queries.dtype]
        )[0]
    }

    private func blockParallelIsoDecodeAttention(
        queries: MLXArray,
        scale: Float,
        batch: Int,
        queryHeads: Int,
        kvHeads: Int,
        headRepeats: Int,
        blockTokens: Int,
        blockCount: Int,
        blockCapacity: Int,
        keyIndices: MLXArray,
        keyNorms: MLXArray,
        valueIndices: MLXArray,
        valueNorms: MLXArray,
        exactKeys: MLXArray,
        exactValues: MLXArray,
        keyQuantizer: RotorQuantMSEQuantizer,
        valueQuantizer: RotorQuantMSEQuantizer
    ) -> MLXArray {
        let manager = RotorQuantDecodeAttentionKernelManager.shared
        let partials = manager.isoCompressedBlockKernel(
            [
                queries,
                keyIndices,
                keyNorms,
                valueIndices,
                valueNorms,
                keyQuantizer.centroids,
                valueQuantizer.centroids,
                keyQuantizer.rotationParameters,
                Int32(compressedCount),
                Int32(blockCount),
                scale,
            ],
            template: [
                ("T", queries.dtype),
                ("D", keyQuantizer.dimension),
                ("QUERY_HEADS", queryHeads),
                ("KV_HEADS", kvHeads),
                ("HEAD_REPEATS", headRepeats),
                ("BLOCK_TOKENS", blockTokens),
                ("BLOCK_CAPACITY", blockCapacity),
                ("KEY_CAPACITY", keyIndices.dim(2)),
                ("VALUE_CAPACITY", valueIndices.dim(2)),
                ("KEY_PACKED_WIDTH", keyIndices.dim(3)),
                ("VALUE_PACKED_WIDTH", valueIndices.dim(3)),
                ("KEY_BITS", configuration.keyBits),
                ("VALUE_BITS", configuration.valueBits),
            ],
            grid: (32, batch * kvHeads * blockCount, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [
                [batch, queryHeads, blockCapacity, keyQuantizer.dimension],
                [batch, queryHeads, blockCapacity],
                [batch, queryHeads, blockCapacity],
            ],
            outputDTypes: [queries.dtype, .float32, .float32]
        )

        return manager.isoBlockReduceKernel(
            [
                partials[0],
                partials[1],
                partials[2],
                queries,
                exactKeys,
                exactValues,
                keyQuantizer.rotationParameters,
                valueQuantizer.rotationParameters,
                Int32(blockCount),
                Int32(exactCount),
                scale,
            ],
            template: [
                ("T", queries.dtype),
                ("D", keyQuantizer.dimension),
                ("QUERY_HEADS", queryHeads),
                ("KV_HEADS", kvHeads),
                ("HEAD_REPEATS", headRepeats),
                ("BLOCK_CAPACITY", blockCapacity),
                ("EXACT_CAPACITY", exactKeys.dim(2)),
            ],
            grid: (32, batch * kvHeads, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, queryHeads, 1, keyQuantizer.dimension]],
            outputDTypes: [queries.dtype]
        )[0]
    }

    public override var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []
            if compressedCount > 0 {
                guard let keyIndices, let keyNorms, let valueIndices, let valueNorms else {
                    preconditionFailure("RotorQuant compressed state is incomplete")
                }
                arrays.append(contentsOf: [
                    keyIndices[.ellipsis, ..<compressedCount, 0...],
                    keyNorms[.ellipsis, ..<compressedCount],
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
            keyIndices = nil
            keyNorms = nil
            valueIndices = nil
            valueNorms = nil
            exactKeys = nil
            exactValues = nil
            compressedCount = 0
            compressedCapacity = 0
            exactCount = 0
            exactCapacity = 0

            switch newValue.count {
            case 0:
                break
            case 2:
                exactCount = newValue[0].dim(2)
                ensureQuantizers(keyDimension: newValue[0].dim(3), valueDimension: newValue[1].dim(3))
                ensureExactStorage(
                    batch: newValue[0].dim(0),
                    kvHeads: newValue[0].dim(1),
                    keyDimension: newValue[0].dim(3),
                    valueDimension: newValue[1].dim(3),
                    dtype: newValue[0].dtype
                )
                exactKeys?[.ellipsis, ..<exactCount, 0...] = newValue[0]
                exactValues?[.ellipsis, ..<exactCount, 0...] = newValue[1]
            case 4:
                keyIndices = newValue[0]
                keyNorms = newValue[1]
                valueIndices = newValue[2]
                valueNorms = newValue[3]
                compressedCount = newValue[0].dim(2)
                compressedCapacity = keyIndices?.dim(2) ?? 0
            case 6:
                keyIndices = newValue[0]
                keyNorms = newValue[1]
                valueIndices = newValue[2]
                valueNorms = newValue[3]
                compressedCount = newValue[0].dim(2)
                compressedCapacity = keyIndices?.dim(2) ?? 0
                exactCount = newValue[4].dim(2)
                ensureQuantizers(keyDimension: newValue[4].dim(3), valueDimension: newValue[5].dim(3))
                ensureExactStorage(
                    batch: newValue[4].dim(0),
                    kvHeads: newValue[4].dim(1),
                    keyDimension: newValue[4].dim(3),
                    valueDimension: newValue[5].dim(3),
                    dtype: newValue[4].dtype
                )
                exactKeys?[.ellipsis, ..<exactCount, 0...] = newValue[4]
                exactValues?[.ellipsis, ..<exactCount, 0...] = newValue[5]
            default:
                preconditionFailure("RotorQuantKVCache state must have 0, 2, 4, or 6 arrays")
            }
            offset = compressedCount + exactCount
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(step),
                String(offset),
                String(configuration.keyBits),
                String(configuration.valueBits),
                String(configuration.seed),
                String(configuration.attentionBlockTokens),
                configuration.variant.rawValue,
                effectiveVariant.rawValue,
                String(keyDimension ?? 0),
                String(valueDimension ?? 0),
                Self.dtypeName(originalDType),
                String(compressedCount),
                String(exactCount),
                String(exactBufferSize),
            ]
        }
        set {
            guard newValue.count == 14 else {
                preconditionFailure("RotorQuantKVCache metaState must have exactly 14 values")
            }
            step = Int(newValue[0]) ?? step
            offset = Int(newValue[1]) ?? 0
            configuration.keyBits = Int(newValue[2]) ?? configuration.keyBits
            configuration.valueBits = Int(newValue[3]) ?? configuration.valueBits
            configuration.seed = UInt64(newValue[4]) ?? configuration.seed
            configuration.attentionBlockTokens = Int(newValue[5]) ?? configuration.attentionBlockTokens
            configuration.variant = RotorQuantVariant(rawValue: newValue[6]) ?? configuration.variant
            keyDimension = (Int(newValue[8]) ?? 0) > 0 ? Int(newValue[8]) : nil
            valueDimension = (Int(newValue[9]) ?? 0) > 0 ? Int(newValue[9]) : nil
            originalDType = Self.dtype(from: newValue[10])
            compressedCount = Int(newValue[11]) ?? 0
            exactCount = Int(newValue[12]) ?? 0
            exactBufferSize = Int(newValue[13]) ?? exactBufferSize
            offset = compressedCount + exactCount
            if let keyDimension, let valueDimension {
                keyQuantizer = nil
                valueQuantizer = nil
                ensureQuantizers(keyDimension: keyDimension, valueDimension: valueDimension)
            }
        }
    }

    public override var isTrimmable: Bool { true }

    /// Drops the newest `n` tokens, matching `KVCacheSimple.trim`. The newest rows live in the
    /// exact tail, so that buffer releases before any compressed row does.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let trimmedExact = min(exactCount, n)
        exactCount -= trimmedExact
        let trimmedCompressed = min(compressedCount, n - trimmedExact)
        compressedCount -= trimmedCompressed
        offset = compressedCount + exactCount
        return trimmedExact + trimmedCompressed
    }

    private func ingest(keys: MLXArray, values: MLXArray) {
        originalDType = keys.dtype
        ensureQuantizers(keyDimension: keys.dim(3), valueDimension: values.dim(3))
        ensureExactStorage(
            batch: keys.dim(0),
            kvHeads: keys.dim(1),
            keyDimension: keys.dim(3),
            valueDimension: values.dim(3),
            dtype: keys.dtype
        )

        let tokenCount = keys.dim(2)
        if tokenCount > 1 {
            appendExact(keys: keys, values: values)
            offset = compressedCount + exactCount
            return
        }

        if exactBufferSize == 0 {
            flushExactToCompressed()
            appendCompressedChunked(keys: keys, values: values)
            offset = compressedCount
            return
        }

        if exactCount + tokenCount <= exactBufferSize + exactFlushSlack {
            appendExact(keys: keys, values: values)
        } else {
            flushOldestExact(tokenCount: exactCount + tokenCount - exactBufferSize)
            appendExact(keys: keys, values: values)
        }
        offset = compressedCount + exactCount
    }

    private func appendExact(keys: MLXArray, values: MLXArray) {
        growExactStorageIfNeeded(
            batch: keys.dim(0),
            kvHeads: keys.dim(1),
            keyDimension: keys.dim(3),
            valueDimension: values.dim(3),
            dtype: keys.dtype,
            additionalTokens: keys.dim(2)
        )
        exactKeys?[.ellipsis, exactCount..<(exactCount + keys.dim(2)), 0...] = keys
        exactValues?[.ellipsis, exactCount..<(exactCount + values.dim(2)), 0...] = values
        exactCount += keys.dim(2)
    }

    private func flushExactToCompressed() {
        guard let exactKeys, let exactValues, exactCount > 0 else { return }
        appendCompressedChunked(
            keys: exactKeys[.ellipsis, ..<exactCount, 0...],
            values: exactValues[.ellipsis, ..<exactCount, 0...]
        )
        self.exactCount = 0
        compactExactStorageIfNeeded()
    }

    private func flushOldestExact(tokenCount: Int) {
        guard tokenCount > 0, let exactKeys, let exactValues, exactCount > 0 else { return }
        let flushCount = min(tokenCount, exactCount)
        appendCompressedChunked(
            keys: exactKeys[.ellipsis, ..<flushCount, 0...],
            values: exactValues[.ellipsis, ..<flushCount, 0...]
        )
        let retained = exactCount - flushCount
        if retained > 0 {
            self.exactKeys?[.ellipsis, ..<retained, 0...] =
                exactKeys[.ellipsis, flushCount..<exactCount, 0...]
            self.exactValues?[.ellipsis, ..<retained, 0...] =
                exactValues[.ellipsis, flushCount..<exactCount, 0...]
        }
        self.exactCount = retained
        compactExactStorageIfNeeded()
    }

    private func appendCompressed(keys: MLXArray, values: MLXArray) {
        guard keys.dim(2) > 0 else { return }
        let previous = compressedCount
        ensureStorage(batch: keys.dim(0), kvHeads: keys.dim(1), tokenCount: keys.dim(2))

        let quantizedKeys = keyQuantizer!.quantize(keys)
        let quantizedValues = valueQuantizer!.quantize(values)
        let packedKeyIndices = PackedCentroidIndices.pack(quantizedKeys.indices, bitWidth: configuration.keyBits)
        let packedValueIndices = PackedCentroidIndices.pack(quantizedValues.indices, bitWidth: configuration.valueBits)

        compressedCount += keys.dim(2)
        keyIndices?[.ellipsis, previous..<compressedCount, 0...] = packedKeyIndices
        keyNorms?[.ellipsis, previous..<compressedCount] = quantizedKeys.norms
        valueIndices?[.ellipsis, previous..<compressedCount, 0...] = packedValueIndices
        valueNorms?[.ellipsis, previous..<compressedCount] = quantizedValues.norms
    }

    private func appendCompressedChunked(keys: MLXArray, values: MLXArray) {
        guard keys.dim(2) > 0 else { return }
        let chunkTokens = max(16, min(64, step / 4))
        if keys.dim(2) <= chunkTokens {
            appendCompressed(keys: keys, values: values)
            return
        }
        for start in stride(from: 0, to: keys.dim(2), by: chunkTokens) {
            let end = min(start + chunkTokens, keys.dim(2))
            appendCompressed(
                keys: keys[.ellipsis, start..<end, 0...],
                values: values[.ellipsis, start..<end, 0...]
            )
        }
    }

    private func ensureStorage(batch: Int, kvHeads: Int, tokenCount: Int) {
        let requiredCapacity = compressedCount + tokenCount
        guard keyIndices == nil || requiredCapacity > compressedCapacity else { return }

        let newCapacity = Self.nextStorageCapacity(
            current: compressedCapacity,
            required: requiredCapacity,
            step: step
        )

        let newKeyIndices = MLXArray.zeros([batch, kvHeads, newCapacity, keyPackedWidth], dtype: .uint8)
        let newKeyNorms = MLXArray.zeros([batch, kvHeads, newCapacity], dtype: .float32)
        let newValueIndices = MLXArray.zeros([batch, kvHeads, newCapacity, valuePackedWidth], dtype: .uint8)
        let newValueNorms = MLXArray.zeros([batch, kvHeads, newCapacity], dtype: .float32)

        if let keyIndices, let keyNorms, let valueIndices, let valueNorms, compressedCount > 0 {
            newKeyIndices[.ellipsis, ..<compressedCount, 0...] =
                keyIndices[.ellipsis, ..<compressedCount, 0...]
            newKeyNorms[.ellipsis, ..<compressedCount] =
                keyNorms[.ellipsis, ..<compressedCount]
            newValueIndices[.ellipsis, ..<compressedCount, 0...] =
                valueIndices[.ellipsis, ..<compressedCount, 0...]
            newValueNorms[.ellipsis, ..<compressedCount] =
                valueNorms[.ellipsis, ..<compressedCount]
        }

        self.keyIndices = newKeyIndices
        self.keyNorms = newKeyNorms
        self.valueIndices = newValueIndices
        self.valueNorms = newValueNorms
        compressedCapacity = newCapacity
    }

    /// Grow compressed storage by 1.5x, rounded to the existing allocation step.
    /// Doubling retained almost 2K unused rows at common 6K-token context limits;
    /// this curve keeps reallocations bounded without carrying that large slack.
    static func nextStorageCapacity(current: Int, required: Int, step: Int) -> Int {
        let allocationStep = max(step, 1)
        let baseline = max(current, allocationStep)
        let growthTarget = max(baseline + baseline / 2, required)
        return ((growthTarget + allocationStep - 1) / allocationStep) * allocationStep
    }

    private func ensureQuantizers(keyDimension: Int, valueDimension: Int) {
        if self.keyDimension != keyDimension || keyQuantizer == nil {
            self.keyDimension = keyDimension
            self.keyQuantizer = RotorQuantMSEQuantizer(
                dimension: keyDimension,
                bits: configuration.keyBits,
                variant: configuration.variant,
                seed: configuration.seed
            )
        }
        if self.valueDimension != valueDimension || valueQuantizer == nil {
            self.valueDimension = valueDimension
            self.valueQuantizer = RotorQuantMSEQuantizer(
                dimension: valueDimension,
                bits: configuration.valueBits,
                variant: configuration.variant,
                seed: configuration.seed &+ 1
            )
        }
    }

    private func ensureExactStorage(
        batch: Int,
        kvHeads: Int,
        keyDimension: Int,
        valueDimension: Int,
        dtype: DType
    ) {
        // Include the intentional decode flush slack in the initial allocation.
        // Without this, a full exact tail grows one token at a time until the
        // first flush, briefly overlapping each old and replacement allocation.
        let capacity = max(exactBufferSize + exactFlushSlack, exactCapacity, exactCount, 1)
        let desiredKeyShape = [batch, kvHeads, capacity, keyDimension]
        let desiredValueShape = [batch, kvHeads, capacity, valueDimension]
        let needsAllocation =
            exactKeys == nil
            || exactValues == nil
            || exactKeys?.shape != desiredKeyShape
            || exactValues?.shape != desiredValueShape
            || exactKeys?.dtype != dtype
            || exactValues?.dtype != dtype
        guard needsAllocation else { return }

        let newExactKeys = MLXArray.zeros(desiredKeyShape, dtype: dtype)
        let newExactValues = MLXArray.zeros(desiredValueShape, dtype: dtype)
        if let exactKeys, let exactValues, exactCount > 0 {
            let retained = min(exactCount, capacity, exactKeys.dim(2))
            newExactKeys[.ellipsis, ..<retained, 0...] =
                castIfNeeded(exactKeys[.ellipsis, ..<retained, 0...], to: dtype)
            newExactValues[.ellipsis, ..<retained, 0...] =
                castIfNeeded(exactValues[.ellipsis, ..<retained, 0...], to: dtype)
            self.exactCount = retained
        }
        self.exactKeys = newExactKeys
        self.exactValues = newExactValues
        self.exactCapacity = capacity
    }

    private func growExactStorageIfNeeded(
        batch: Int,
        kvHeads: Int,
        keyDimension: Int,
        valueDimension: Int,
        dtype: DType,
        additionalTokens: Int
    ) {
        let required = exactCount + additionalTokens
        if exactKeys == nil || required > (exactKeys?.dim(2) ?? 0) {
            exactCapacity = max(exactCapacity, required)
        }
        ensureExactStorage(
            batch: batch,
            kvHeads: kvHeads,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            dtype: dtype
        )
    }

    /// Releases the dense prefill backing allocation after older rows have moved into
    /// compressed storage. Keeping `exactCapacity` at its prefill high-water mark retains
    /// the entire dense prompt alongside the compressed cache and defeats the RAM saving.
    private func compactExactStorageIfNeeded() {
        guard let exactKeys, let exactValues, let keyDimension, let valueDimension else { return }
        // Preserve the existing flush slack so decode does not reallocate this
        // buffer one token at a time after compaction.
        let targetCapacity = max(exactBufferSize + exactFlushSlack, exactCount, 1)
        guard exactKeys.dim(2) > targetCapacity || exactValues.dim(2) > targetCapacity else {
            exactCapacity = targetCapacity
            return
        }

        exactCapacity = targetCapacity
        ensureExactStorage(
            batch: exactKeys.dim(0),
            kvHeads: exactKeys.dim(1),
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            dtype: exactKeys.dtype
        )
    }

    private func currentFallbackState() -> (MLXArray, MLXArray) {
        var allKeys: [MLXArray] = []
        var allValues: [MLXArray] = []

        if compressedCount > 0 {
            guard let keyQuantizer,
                  let valueQuantizer,
                  let keyIndices,
                  let keyNorms,
                  let valueIndices,
                  let valueNorms
            else {
                preconditionFailure("RotorQuant compressed state is incomplete")
            }
            let unpackedKeyIndices = PackedCentroidIndices.unpack(
                keyIndices[.ellipsis, ..<compressedCount, 0...],
                bitWidth: configuration.keyBits,
                valueCount: keyQuantizer.storageDimension
            )
            let unpackedValueIndices = PackedCentroidIndices.unpack(
                valueIndices[.ellipsis, ..<compressedCount, 0...],
                bitWidth: configuration.valueBits,
                valueCount: valueQuantizer.storageDimension
            )
            allKeys.append(
                keyQuantizer.dequantize(
                    indices: unpackedKeyIndices,
                    norms: keyNorms[.ellipsis, ..<compressedCount],
                    dtype: originalDType
                )
            )
            allValues.append(
                valueQuantizer.dequantize(
                    indices: unpackedValueIndices,
                    norms: valueNorms[.ellipsis, ..<compressedCount],
                    dtype: originalDType
                )
            )
        }

        if let exactKeys, let exactValues, exactCount > 0 {
            allKeys.append(castIfNeeded(exactKeys[.ellipsis, ..<exactCount, 0...], to: originalDType))
            allValues.append(castIfNeeded(exactValues[.ellipsis, ..<exactCount, 0...], to: originalDType))
        }

        guard let firstKeys = allKeys.first, let firstValues = allValues.first else {
            preconditionFailure("RotorQuantKVCache is empty")
        }

        if allKeys.count == 1 {
            return (firstKeys, firstValues)
        }
        return (concatenated(allKeys, axis: 2), concatenated(allValues, axis: 2))
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
    public func toRotorQuantized(
        configuration: RotorQuantConfiguration = RotorQuantConfiguration()
    ) -> RotorQuantKVCache {
        let rotorQuantCache = RotorQuantKVCache(configuration: configuration)
        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            _ = rotorQuantCache.update(keys: currentKeys, values: currentValues)
        }
        return rotorQuantCache
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
    private var step: Int
    public private(set) var groupSize: Int
    public private(set) var bits: Int
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
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: values.dtype)
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

            self.step = Int(newValue[0]) ?? step
            self.offset = Int(newValue[1]) ?? 0
            self.groupSize = Int(newValue[2]) ?? groupSize
            self.bits = Int(newValue[3]) ?? bits
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
        case is RotorQuantKVCache:
            return "RotorQuantKVCache"
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
        case "RotorQuantKVCache":
            try validateRotorQuantPromptCache(
                state: cacheData[i],
                metaState: i < cacheInfo.count ? cacheInfo[i] : []
            )
            cache = RotorQuantKVCache()
        case "TurboQuantKVCache":
            caches.append(
                try migrateLegacyCompressedPromptCache(
                    state: cacheData[i],
                    metaState: i < cacheInfo.count ? cacheInfo[i] : []
                )
            )
            continue
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

private func validateRotorQuantPromptCache(state: [MLXArray], metaState: [String]) throws {
    guard metaState.count == 14 else {
        throw KVCacheError(message: "Invalid RotorQuantKVCache metaState - expected 14 values")
    }

    guard [0, 2, 4, 6].contains(state.count) else {
        throw KVCacheError(message: "Invalid RotorQuantKVCache state - expected 0, 2, 4, or 6 arrays")
    }

    func parseInt(_ index: Int, _ name: String) throws -> Int {
        guard let value = Int(metaState[index]) else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache \(name)")
        }
        return value
    }

    func parseUInt64(_ index: Int, _ name: String) throws -> UInt64 {
        guard let value = UInt64(metaState[index]) else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache \(name)")
        }
        return value
    }

    func requireRank(_ array: MLXArray, _ rank: Int, _ name: String) throws {
        guard array.shape.count == rank else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache \(name) rank - expected \(rank)")
        }
    }

    func requireSamePrefix(_ lhs: MLXArray, _ rhs: MLXArray, dimensions: Int, _ message: String) throws {
        guard lhs.shape.prefix(dimensions).elementsEqual(rhs.shape.prefix(dimensions)) else {
            throw KVCacheError(message: message)
        }
    }

    func paddedDimension(_ dimension: Int, variant: RotorQuantVariant) -> Int {
        let blockSize: Int
        switch variant {
        case .iso:
            blockSize = 4
        case .planar:
            blockSize = 2
        case .clifford:
            blockSize = 3
        }
        return ((dimension + blockSize - 1) / blockSize) * blockSize
    }

    func packedByteCount(dimension: Int, bits: Int, variant: RotorQuantVariant) -> Int {
        (paddedDimension(dimension, variant: variant) * bits + 7) / 8
    }

    let step = try parseInt(0, "step")
    let offset = try parseInt(1, "offset")
    let keyBits = try parseInt(2, "key bits")
    let valueBits = try parseInt(3, "value bits")
    _ = try parseUInt64(4, "seed")
    let attentionBlockTokens = try parseInt(5, "attention block tokens")
    guard let requestedVariant = RotorQuantVariant(rawValue: metaState[6]),
          let effectiveVariant = RotorQuantVariant(rawValue: metaState[7])
    else {
        throw KVCacheError(message: "Invalid RotorQuantKVCache variant")
    }
    let keyDimension = try parseInt(8, "key dimension")
    let valueDimension = try parseInt(9, "value dimension")
    let compressedCount = try parseInt(11, "compressed count")
    let exactCount = try parseInt(12, "exact count")
    let exactBufferSize = try parseInt(13, "exact buffer size")

    guard step > 0,
          (1 ... 8).contains(keyBits),
          (1 ... 8).contains(valueBits),
          attentionBlockTokens > 0,
          keyDimension >= 0,
          valueDimension >= 0,
          compressedCount >= 0,
          exactCount >= 0,
          exactBufferSize >= 0,
          offset == compressedCount + exactCount
    else {
        throw KVCacheError(message: "Invalid RotorQuantKVCache metaState values")
    }

    guard ["float16", "bfloat16", "float32"].contains(metaState[10]) else {
        throw KVCacheError(message: "Invalid RotorQuantKVCache dtype")
    }

    if state.isEmpty {
        guard keyDimension == 0,
              valueDimension == 0,
              compressedCount == 0,
              exactCount == 0
        else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache empty state metadata")
        }
    } else {
        guard keyDimension > 0, valueDimension > 0 else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache dimensions")
        }
    }

    if !state.isEmpty {
        guard RotorQuantRotationFactory.effectiveVariant(requestedVariant, dimension: keyDimension)
            == effectiveVariant
        else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache effective variant")
        }
    }

    if state.count >= 4 {
        try requireRank(state[0], 4, "key indices")
        try requireRank(state[1], 3, "key norms")
        try requireRank(state[2], 4, "value indices")
        try requireRank(state[3], 3, "value norms")
        try requireSamePrefix(
            state[0],
            state[1],
            dimensions: 3,
            "Invalid RotorQuantKVCache key index/norm shape mismatch"
        )
        try requireSamePrefix(
            state[2],
            state[3],
            dimensions: 3,
            "Invalid RotorQuantKVCache value index/norm shape mismatch"
        )
        try requireSamePrefix(
            state[0],
            state[2],
            dimensions: 3,
            "Invalid RotorQuantKVCache compressed key/value shape mismatch"
        )
        guard state[0].dim(2) == compressedCount,
              state[2].dim(2) == compressedCount,
              state[0].dim(3)
                == packedByteCount(dimension: keyDimension, bits: keyBits, variant: effectiveVariant),
              state[2].dim(3)
                == packedByteCount(dimension: valueDimension, bits: valueBits, variant: effectiveVariant)
        else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache compressed metadata")
        }
    } else if compressedCount != 0 {
        throw KVCacheError(message: "Invalid RotorQuantKVCache compressed count without state")
    }

    if state.count == 2 {
        try requireRank(state[0], 4, "exact keys")
        try requireRank(state[1], 4, "exact values")
        try requireSamePrefix(
            state[0],
            state[1],
            dimensions: 3,
            "Invalid RotorQuantKVCache exact key/value shape mismatch"
        )
        guard compressedCount == 0,
              state[0].dim(2) == exactCount,
              state[0].dim(3) == keyDimension,
              state[1].dim(3) == valueDimension
        else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache exact metadata")
        }
    } else if state.count == 6 {
        try requireRank(state[4], 4, "exact keys")
        try requireRank(state[5], 4, "exact values")
        try requireSamePrefix(
            state[4],
            state[5],
            dimensions: 3,
            "Invalid RotorQuantKVCache exact key/value shape mismatch"
        )
        try requireSamePrefix(
            state[0],
            state[4],
            dimensions: 2,
            "Invalid RotorQuantKVCache compressed/exact batch or KV-head mismatch"
        )
        guard state[4].dim(2) == exactCount,
              state[4].dim(3) == keyDimension,
              state[5].dim(3) == valueDimension
        else {
            throw KVCacheError(message: "Invalid RotorQuantKVCache exact metadata")
        }
    } else if exactCount != 0 {
        throw KVCacheError(message: "Invalid RotorQuantKVCache exact count without state")
    }
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
/// RotorQuant must start at cache construction time to reduce the peak
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

    if case .rotorQuant(let configuration)? = parameters?.resolvedCacheCompression {
        return RotorQuantKVCache(configuration: configuration.configurationForLayer(layerIndex))
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
    // Every layer has to be trimmed, not just the first: leaving the rest at
    // their old offsets desynchronizes the cache. Each layer clamps to its own
    // offset, so the reported count is the first layer's, matching mlx-lm.
    var trimmed = 0
    for (index, layer) in cache.enumerated() {
        let layerTrimmed = layer.trim(numTokens)
        if index == 0 {
            trimmed = layerTrimmed
        }
    }
    return trimmed
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
        scores = MLX.where(causalMask, scores, MLXArray(-Float.greatestFiniteMagnitude))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        for maskArray in maskArrays {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
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
        cache.contains(where: { item in
            guard item is KVCacheSimple, !(item is QuantizedKVCache) else { return false }
            return item.offset > quantizedKVStart
        })
    else {
        return
    }

    for i in 0 ..< cache.count {
        // Handle cache types that support quantization
        if let simpleCache = cache[i] as? KVCacheSimple,
           simpleCache.offset > quantizedKVStart {
            cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
        }
        // RotatingKVCache quantization is intentionally not supported here, matching Python.
        // MambaCache and CacheList don't use traditional KV quantization
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
    case .rotorQuant:
        // RotorQuant is a cache-construction-time strategy.
        // Delayed conversion after prefill is intentionally unsupported.
        break
    }
}
