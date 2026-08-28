//
//  MLXModelManager+Tuning
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLX
import MLXLMCommon
import OSLog
import UIKit

extension MLXModelManager {
    // MARK: - Prefill Tuning

    nonisolated static let defaultPrefillStepSize = 512
    nonisolated static let minimumAdaptivePrefillStepSize = 128

    /// Scales the tuned prefill chunk size down when the actual prompt is small,
    /// reducing peak memory without changing generation quality. Never exceeds
    /// the tuned value.
    nonisolated static func adaptivePrefillStepSize(
        tuned: Int,
        messages: [Chat.Message]
    ) -> Int {
        let approxTokens = messages.reduce(0) { partial, message in
            partial + max(1, message.content.count / 4)
        }
        if approxTokens <= 0 {
            return max(minimumAdaptivePrefillStepSize, min(tuned, 256))
        }
        var bucket = minimumAdaptivePrefillStepSize
        while bucket < approxTokens && bucket < tuned {
            bucket *= 2
        }
        return max(minimumAdaptivePrefillStepSize, min(tuned, bucket))
    }
    nonisolated private static let tuningStartupDelayNanoseconds: UInt64 = 1_000_000_000
    nonisolated private static func prefillTuningUserDefaultsKey(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> String {
        let deviceClass = deviceSupportProfile.isPhone ? "phone" : "other"
        let bytesPerGiB: UInt64 = 1_073_741_824
        let memoryTier = Int(deviceSupportProfile.physicalMemoryBytes / bytesPerGiB)
        return "mlxPrefillStepSize.\(modelID).\(deviceClass).\(memoryTier)"
    }

    nonisolated static func persistedPrefillStepSize(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> Int? {
        let key = prefillTuningUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        guard UserDefaults.standard.object(forKey: key) != nil else { return nil }
        let storedValue = UserDefaults.standard.integer(forKey: key)
        guard [256, 512, 1024].contains(storedValue) else { return nil }
        return storedValue
    }

    nonisolated private static func storePersistedPrefillStepSize(
        _ prefillStepSize: Int,
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) {
        let key = prefillTuningUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        UserDefaults.standard.set(prefillStepSize, forKey: key)
    }

    nonisolated static func recommendedWiredMemoryCapBytes(
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> Int? {
        GPU.maxRecommendedWorkingSetBytes() ?? Int(deviceSupportProfile.physicalMemoryBytes)
    }
    nonisolated internal struct TuningBenchmarkResult: Sendable {
        let candidate: Int
        let promptTokensPerSecond: Double
        let decodeTokensPerSecond: Double
        let totalLatency: TimeInterval
        let totalBytes: Int
        let measurement: MLXLMCommon.WiredMemoryMeasurement
    }

    nonisolated internal static func selectFastestSafeCandidate(
        _ candidates: [TuningBenchmarkResult],
        wiredMemoryCap: Int?,
        preferLargerCandidateOnTie: Bool
    ) -> TuningBenchmarkResult {
        let safeCandidates = candidates.filter { candidate in
            guard let wiredMemoryCap else { return true }
            return candidate.totalBytes <= Int(Double(wiredMemoryCap) * 0.85)
        }
        let performanceCandidates = safeCandidates.isEmpty ? candidates : safeCandidates
        let bestDecodeSpeed = performanceCandidates.map(\.decodeTokensPerSecond).max() ?? 0
        let decodeFloor = bestDecodeSpeed > 0 ? bestDecodeSpeed * 0.95 : 0
        let decodeSafeCandidates = performanceCandidates.filter { candidate in
            bestDecodeSpeed == 0 || candidate.decodeTokensPerSecond >= decodeFloor
        }
        let rankedCandidates = decodeSafeCandidates.isEmpty ? performanceCandidates : decodeSafeCandidates

        return rankedCandidates.max { lhs, rhs in
            if lhs.promptTokensPerSecond != rhs.promptTokensPerSecond {
                return lhs.promptTokensPerSecond < rhs.promptTokensPerSecond
            }
            if lhs.totalLatency != rhs.totalLatency {
                return lhs.totalLatency > rhs.totalLatency
            }
            if lhs.decodeTokensPerSecond != rhs.decodeTokensPerSecond {
                return lhs.decodeTokensPerSecond < rhs.decodeTokensPerSecond
            }
            return preferLargerCandidateOnTie
                ? lhs.candidate < rhs.candidate
                : lhs.candidate > rhs.candidate
        } ?? candidates[0]
    }
    private func makeBenchmarkUserInput(
        container: ModelContainer,
        targetTokenCount: Int = 2_048
    ) async -> UserInput {
        var prompt = ""
        var tokenCount = 0
        while tokenCount < targetTokenCount {
            // Approach the target in bounded increments. Doubling the whole
            // prompt could make the tuning workload nearly twice as large as
            // requested on tokenizers with roughly linear segmentation.
            let remaining = targetTokenCount - tokenCount
            prompt += String(repeating: " hello", count: min(128, remaining))
            tokenCount = await container.encode(prompt).count
        }
        return UserInput(
            chat: [.user(prompt)],
            processing: UserInput.Processing()
        )
    }

    private func benchmarkGeneration(
        container: ModelContainer,
        userInput: UserInput,
        parameters: GenerateParameters
    ) async throws -> (promptTokensPerSecond: Double, decodeTokensPerSecond: Double, totalLatency: TimeInterval) {
        try Task.checkCancellation()
        let prepared = try await container.prepare(input: userInput)
        let startedAt = Date()
        let stream = try await container.generate(input: prepared, parameters: parameters)
        var completionInfo: GenerateCompletionInfo?
        for await generation in stream {
            try Task.checkCancellation()
            if case .info(let info) = generation {
                completionInfo = info
            }
        }
        let totalLatency = Date().timeIntervalSince(startedAt)
        guard let completionInfo else {
            throw GenerationError.invalidChatHistory
        }
        return (
            promptTokensPerSecond: completionInfo.promptTokensPerSecond,
            decodeTokensPerSecond: completionInfo.tokensPerSecond,
            totalLatency: totalLatency
        )
    }

    private func measureTuningCandidate(
        container: ModelContainer,
        userInput: UserInput,
        parameters: GenerateParameters,
        candidate: Int
    ) async throws -> TuningBenchmarkResult {
        let performanceResult = try await benchmarkGeneration(
            container: container,
            userInput: userInput,
            parameters: parameters
        )
        try Task.checkCancellation()
        aggressivelyFreeMemory(reason: "tuning.candidate.between")
        let memoryMeasurement: MLXLMCommon.WiredMemoryMeasurement = try await container.perform { context in
            try await WiredMemoryUtils.tune(
                userInput: userInput,
                context: context,
                parameters: parameters
            )
        }

        return TuningBenchmarkResult(
            candidate: candidate,
            promptTokensPerSecond: performanceResult.promptTokensPerSecond,
            decodeTokensPerSecond: performanceResult.decodeTokensPerSecond,
            totalLatency: performanceResult.totalLatency,
            totalBytes: memoryMeasurement.totalBytes,
            measurement: memoryMeasurement
        )
    }
    private func defaultGenerationConfigurationForCurrentDevice(
        model: MLXModelInfo? = nil
    ) -> GenerationConfiguration {
        Self.generationConfiguration(
            isEnabled: UserDefaults.standard.mlxEnableRotorQuant,
            preferRotorQuant: shouldPreferRotorQuant,
            hasTools: false,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: prefersBoundedKVCacheAcrossTurns,
            configuredMaxOutputTokens: UserDefaults.standard.mlxMaxOutputTokensLimit,
            configuredContextWindow: configuredContextWindowLimit(for: model)
        )
    }

    func schedulePrefillTuningIfNeeded(
        model: MLXModelInfo,
        initialPrefillStepSize: Int
    ) {
        guard tuningNeedsRetry(for: model) else {
            deferredTuningModelID = nil
            return
        }

        deferredTuningModelID = model.id
        logger.notice(
            "MLX tuning deferred until after first successful generation: id=\(model.id, privacy: .public) prefill=\(initialPrefillStepSize, privacy: .public)"
        )
    }
    func startDeferredTuningIfNeeded(
        container: ModelContainer,
        model: MLXModelInfo
    ) async {
        guard deferredTuningModelID == model.id else { return }
        guard currentModel?.id == model.id else { return }
        guard isApplicationActiveForGPUWork else {
            logger.notice(
                "MLX keeping deferred tuning paused until app is active: id=\(model.id, privacy: .public)"
            )
            return
        }
        guard !hasSeenMemoryWarningSinceCurrentLoad else {
            logger.notice(
                "MLX skipping deferred tuning after load-time memory pressure: id=\(model.id, privacy: .public)"
            )
            deferredTuningModelID = nil
            return
        }
        guard tuningTask == nil else { return }

        deferredTuningModelID = nil

        let initialPrefillStepSize = await inferenceWorker.prefillStepSize(for: model.id)
        let persistedPrefill = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let defaultGenerationConfiguration = defaultGenerationConfigurationForCurrentDevice(model: model)
        let prefillCandidates = persistedPrefill.map { [$0] } ?? Array(Set([256, initialPrefillStepSize, 1024])).sorted()
        let wiredMemoryCap = Self.recommendedWiredMemoryCapBytes(
            deviceSupportProfile: deviceSupportProfile
        )

        tuningTask = Task(priority: .utility) { [weak self] in
            guard let self else { return }
            defer {
                Task { @MainActor [weak self] in
                    self?.tuningTask = nil
                }
            }

            try? await Task.sleep(nanoseconds: Self.tuningStartupDelayNanoseconds)
            guard !Task.isCancelled else { return }

            let canUseGPU = await MainActor.run { [weak self] in
                self?.isApplicationActiveForGPUWork == true
            }
            guard canUseGPU else {
                await MainActor.run {
                    if self.tuningNeedsRetry(for: model) {
                        self.deferredTuningModelID = model.id
                    }
                    self.logger.notice(
                        "MLX deferred tuning skipped after app left foreground: id=\(model.id, privacy: .public)"
                    )
                }
                return
            }

            await MainActor.run {
                self.aggressivelyFreeMemory(reason: "tuning.start")
            }

            let benchmarkInput = await self.makeBenchmarkUserInput(container: container)
            var prefillResults: [TuningBenchmarkResult] = []

            for candidate in prefillCandidates {
                guard !Task.isCancelled else { return }
                let parameters = GenerateParameters(
                    maxTokens: 16,
                    maxKVSize: defaultGenerationConfiguration.maxKVSize,
                    prefillStepSize: candidate
                )
                do {
                    let result = try await self.measureTuningCandidate(
                        container: container,
                        userInput: benchmarkInput,
                        parameters: parameters,
                        candidate: candidate
                    )
                    prefillResults.append(result)
                } catch {
                    let logger = Logger(
                        subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
                        category: "MLXModelManager"
                    )
                    logger.error(
                        "MLX prefill tuning candidate failed: model=\(model.id, privacy: .public) prefill=\(candidate, privacy: .public) error=\(error.localizedDescription, privacy: .public)"
                    )
                    await MainActor.run {
                        self.aggressivelyFreeMemory(reason: "tuning.prefill.failure")
                    }
                }
            }

            guard !Task.isCancelled else { return }
            guard !prefillResults.isEmpty else { return }

            let selectedPrefill = Self.selectFastestSafeCandidate(
                prefillResults,
                wiredMemoryCap: wiredMemoryCap,
                preferLargerCandidateOnTie: true
            )

            let selectedMeasurement = selectedPrefill.measurement

            Self.storePersistedPrefillStepSize(
                selectedPrefill.candidate,
                for: model.id,
                deviceSupportProfile: self.deviceSupportProfile
            )
            await self.inferenceWorker.updateLoadedModelTuning(
                modelID: model.id,
                prefillStepSize: selectedPrefill.candidate,
                activeBytesEstimate: selectedMeasurement.kvBytes + selectedMeasurement.workspaceBytes,
                measurement: selectedMeasurement
            )
            await MainActor.run {
                self.logger.notice(
                    "MLX tuning selected: model=\(model.id, privacy: .public) prefill=\(selectedPrefill.candidate, privacy: .public) active_bytes=\(selectedMeasurement.kvBytes + selectedMeasurement.workspaceBytes, privacy: .public)"
                )
                self.aggressivelyFreeMemory(reason: "tuning.complete")
            }
        }
    }
    private var isApplicationActiveForGPUWork: Bool {
        UIApplication.shared.applicationState == .active
    }

    private func tuningNeedsRetry(for model: MLXModelInfo) -> Bool {
        let persistedPrefill = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let needsPrefillTuning = persistedPrefill == nil
        return needsPrefillTuning
    }

    func handleApplicationInactivity(reason: String) {
        guard tuningTask != nil || deferredTuningModelID != nil else { return }

        if let model = currentModel, tuningNeedsRetry(for: model), !hasSeenMemoryWarningSinceCurrentLoad {
            deferredTuningModelID = model.id
        } else {
            deferredTuningModelID = nil
        }

        tuningTask?.cancel()
        tuningTask = nil
        logger.notice("MLX paused deferred tuning: reason=\(reason, privacy: .public)")
    }

    func cancelDeferredTuningBeforeGeneration(for model: MLXModelInfo) async {
        guard let activeTuningTask = tuningTask else { return }

        activeTuningTask.cancel()
        await activeTuningTask.value
        tuningTask = nil

        if currentModel?.id == model.id,
           tuningNeedsRetry(for: model),
           !hasSeenMemoryWarningSinceCurrentLoad {
            deferredTuningModelID = model.id
        }
        logger.notice(
            "MLX deferred tuning stopped before user generation: id=\(model.id, privacy: .public)"
        )
    }
}
