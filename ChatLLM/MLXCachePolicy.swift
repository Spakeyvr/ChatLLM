//
//  MLXCachePolicy.swift
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLX
import MLXLMCommon

nonisolated internal enum MLXCachePolicy: Equatable, Sendable {
    case persistentSimple
    case persistentRotorQuant
    case boundedRotating(maxKVSize: Int)

    var diagnosticLabel: String {
        switch self {
        case .persistentSimple:
            return "persistent-simple"
        case .persistentRotorQuant:
            return "persistent-rotorquant"
        case .boundedRotating:
            return "bounded-rotating"
        }
    }

    var maxKVSize: Int? {
        switch self {
        case .persistentSimple, .persistentRotorQuant:
            return nil
        case .boundedRotating(let maxKVSize):
            return maxKVSize
        }
    }

    var usesRotorQuantCompression: Bool {
        switch self {
        case .persistentRotorQuant:
            return true
        case .persistentSimple, .boundedRotating:
            return false
        }
    }
}

nonisolated internal struct MLXKVBenchmarkMetadata: Sendable, Equatable {
    let cachePolicy: MLXCachePolicy
    let effectiveMaxKVSize: Int?
    let prefillStepSize: Int
}

nonisolated internal struct MLXPerformanceSample: Sendable {
    let conversationID: UUID
    let modelID: String
    let promptTokenCount: Int
    let outputTokenCount: Int
    let toolInvocationCount: Int
    let timeToFirstToken: TimeInterval?
    let totalLatency: TimeInterval
    let promptTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let stopReason: GenerateStopReason?
    let memoryBefore: Memory.Snapshot
    let memoryAfter: Memory.Snapshot
    let peakActiveBytes: Int
    let kvBenchmarkMetadata: MLXKVBenchmarkMetadata
}
