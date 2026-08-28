//
//  MLXModelManager+Memory
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLX
import OSLog

extension MLXModelManager {
    nonisolated private static let aggressiveMemoryCacheLimitBytes = 1 * 1024 * 1024
    func handleMemoryWarning() {
        logger.notice("MLX received memory warning; clearing caches and invalidating sessions")
        hasSeenMemoryWarningSinceCurrentLoad = true
        deferredTuningModelID = nil
        tuningTask?.cancel()
        tuningTask = nil
        Task {
            await inferenceWorker.invalidateAll(reason: "memory_warning")
        }
        aggressivelyFreeMemory(reason: "memory.warning")
    }
    private func resetMemoryCaches() {
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
        URLCache.shared.removeAllCachedResponses()
    }

    func applySteadyStateMemoryCachePolicy() {
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
    }

    func prepareMemoryForPrewarm() {
        aggressivelyFreeMemory(reason: "prewarm.prepare")
    }

    func cleanupMemoryAfterPrewarm() {
        applySteadyStateMemoryCachePolicy()
    }

    func prepareMemoryForGeneration() {
        applySteadyStateMemoryCachePolicy()
    }

    func cleanupMemoryAfterGeneration() {
        applySteadyStateMemoryCachePolicy()
    }

    func cleanupMemoryAfterGenerationError() {
        aggressivelyFreeMemory(reason: "generation.error")
        applySteadyStateMemoryCachePolicy()
    }

    func cleanupMemoryAfterLoadInterruption() {
        aggressivelyFreeMemory(reason: "load.interrupted")
        applySteadyStateMemoryCachePolicy()
    }

    func prepareMemoryForModelLoadTransition() {
        aggressivelyFreeMemory(reason: "model.load.transition")
    }

    func tearDownCurrentModel(
        reason: String,
        clearInferenceWorker: Bool = true
    ) {
        if container != nil || currentModel != nil || deferredPrewarmModelID != nil || deferredTuningModelID != nil || prewarmInFlightModelID != nil {
            logger.notice("MLX model teardown: reason=\(reason, privacy: .public)")
        }
        container = nil
        currentModel = nil
        let workerLoadID = installedWorkerLoadID
        installedWorkerLoadID = nil
        promptTokenCountCache.removeAll(keepingCapacity: false)
        promptTokenCountCacheOrder.removeAll(keepingCapacity: false)
        deferredPrewarmModelID = nil
        deferredTuningModelID = nil
        prewarmInFlightModelID = nil
        tuningTask?.cancel()
        tuningTask = nil
        hasSeenMemoryWarningSinceCurrentLoad = false
        if clearInferenceWorker, let workerLoadID {
            Task {
                _ = await inferenceWorker.clearLoadedModel(ifLoadID: workerLoadID)
            }
        }
        cleanupMemoryAfterUnload()
    }

    func cleanupMemoryAfterUnload() {
        aggressivelyFreeMemory(reason: "model.unload")
        applySteadyStateMemoryCachePolicy()
    }

    func aggressivelyFreeMemory(reason: String) {
        logger.notice("MLX memory cleanup: reason=\(reason, privacy: .public)")
        Memory.clearCache()
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
        URLCache.shared.removeAllCachedResponses()
    }
}
