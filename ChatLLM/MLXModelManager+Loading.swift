//
//  MLXModelManager+Loading
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import OSLog

extension MLXModelManager {
    func invalidateConversationSession(_ conversationID: UUID, reason: String) {
        Task {
            await inferenceWorker.invalidateConversation(conversationID, reason: reason)
        }
    }
    // MARK: - Loading

    func startLoading(modelID: String? = nil, source: String = "unknown") {
        let preferredModel = modelID.flatMap { id in
            availableModels.first(where: { $0.id == id })
        }
        guard let model = preferredModel ?? availableModels.first else {
            return
        }
        guard model.isAvailable else {
            loadError = compatibilityError(for: model) ??
                "\(model.displayName) is not downloaded. Download it before loading."
            return
        }

        if currentModel?.id == model.id, container != nil, !isLoading {
            logger.notice("MLX load request ignored: model already loaded id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        if pendingModelToLoad?.id == model.id, isLoading {
            logger.notice("MLX load request ignored: model already loading id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        let loadID = UUID()
        hasSeenMemoryWarningSinceCurrentLoad = false
        logger.notice("MLX load requested: id=\(model.id, privacy: .public) source=\(source, privacy: .public)")

        if let architecture = packageArchitecture(for: model) {
            logger.notice(
                "MLX package metadata: id=\(model.id, privacy: .public) architecture=\(architecture, privacy: .public) package=\(model.loadPolicy.packageDescription, privacy: .public)"
            )
        }

        let replacingWorkerLoadID = installedWorkerLoadID
        cancelCurrentLoad(reason: "superseded by \(model.id)", tearDownModel: false)
        tearDownCurrentModel(reason: "preparing to load \(model.id)", clearInferenceWorker: false)
        activeLoadID = loadID
        load(
            model,
            loadID: loadID,
            replacingWorkerLoadID: replacingWorkerLoadID,
            source: source
        )
    }

    func model(withID id: String) -> MLXModelInfo? {
        availableModels.first(where: { $0.id == id })
    }
    func cancelCurrentLoad(reason: String = "cancelled", tearDownModel: Bool = true) {
        loadTask?.cancel()
        loadTask = nil
        logger.notice("MLX load cancelled: reason=\(reason, privacy: .public)")
        if isLoading {
            isLoading = false
            pendingModelToLoad = nil
        }
        if activeLoadID != nil {
            activeLoadID = nil
        }
        if tearDownModel {
            tearDownCurrentModel(reason: "cancelCurrentLoad(\(reason))")
        }
    }

    func cancelAndLoad(_ model: MLXModelInfo) {
        startLoading(modelID: model.id, source: "manager.cancelAndLoad")
    }

    private func load(
        _ model: MLXModelInfo,
        loadID: UUID,
        replacingWorkerLoadID: UUID?,
        source: String
    ) {
        guard model.isAvailable, let modelURL = modelDirectoryURL(for: model) else {
            activeLoadID = nil
            isLoading = false
            pendingModelToLoad = nil
            if let compatibilityError = compatibilityError(for: model) {
                loadError = compatibilityError
            } else {
                loadError = "Model '\(model.localDirName)' not found in Documents/Models/. Use the Download button to fetch it."
            }
            return
        }
        pendingModelToLoad = model
        isLoading = true
        loadError = nil
        let recommendedPrefillStepSize = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        ) ?? Self.defaultPrefillStepSize
        let wiredMemoryCap = Self.recommendedWiredMemoryCapBytes(
            deviceSupportProfile: deviceSupportProfile
        )

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                if let replacingWorkerLoadID {
                    _ = await self.inferenceWorker.clearLoadedModel(
                        ifLoadID: replacingWorkerLoadID
                    )
                }
                await MainActor.run {
                    self.prepareMemoryForModelLoadTransition()
                }
                await Task.yield()
                self.logger.notice(
                    "MLX container load start: id=\(model.id, privacy: .public) source=\(source, privacy: .public) path=\(modelURL.lastPathComponent, privacy: .public)"
                )
                let loaded = try await loadModelContainer(directory: modelURL)
                try Task.checkCancellation()
                guard self.activeLoadID == loadID else {
                    self.logger.notice("MLX container load discarded before install: stale load id=\(model.id, privacy: .public)")
                    return
                }
                await self.applyPreferredToolCallFormatIfNeeded(to: loaded, for: model)
                let weightBytes = await Self.estimateWeightBytes(for: loaded)
                let wiredMemoryPolicy = MLXLMCommon.WiredSumPolicy(cap: wiredMemoryCap)
                let reservationTicket = await Self.startReservationTicketIfNeeded(
                    weightBytes: weightBytes,
                    policy: wiredMemoryPolicy
                )
                await self.inferenceWorker.setLoadedModel(
                    MLXLoadedModelState(
                        loadID: loadID,
                        container: loaded,
                        model: model,
                        wiredMemoryPolicy: wiredMemoryPolicy,
                        reservationTicket: reservationTicket,
                        weightBytes: weightBytes,
                        activeBytesEstimate: 0,
                        prefillStepSize: recommendedPrefillStepSize,
                        measurement: nil
                    )
                )
                guard !Task.isCancelled else {
                    _ = await self.inferenceWorker.clearLoadedModel(ifLoadID: loadID)
                    await MainActor.run {
                        if self.activeLoadID == loadID {
                            self.isLoading = false
                            self.pendingModelToLoad = nil
                            self.loadTask = nil
                            self.activeLoadID = nil
                            self.cleanupMemoryAfterLoadInterruption()
                        }
                    }
                    return
                }
                guard self.activeLoadID == loadID else {
                    self.logger.notice("MLX container load discarded: stale load id=\(model.id, privacy: .public)")
                    _ = await self.inferenceWorker.clearLoadedModel(ifLoadID: loadID)
                    await MainActor.run {
                        self.cleanupMemoryAfterLoadInterruption()
                    }
                    return
                }
                await MainActor.run {
                    self.container = loaded
                    self.currentModel = model
                    self.installedWorkerLoadID = loadID
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.activeLoadID = nil
                    self.applySteadyStateMemoryCachePolicy()
                    self.loadTask = nil
                }
                self.logger.notice("MLX model loaded: \(model.displayName, privacy: .public)")
                self.logger.notice("MLX container load finished: id=\(model.id, privacy: .public)")
                MLXMemoryDiagnostics.log(self.logger, "load.finished")
                await MainActor.run {
                    self.logToolTemplateSupport(for: model)
                    self.schedulePrefillTuningIfNeeded(
                        model: model,
                        initialPrefillStepSize: recommendedPrefillStepSize
                    )
                }
                if model.loadPolicy.defersAutomaticPrewarm {
                    self.deferredPrewarmModelID = model.id
                    self.logger.notice("MLX prewarm deferred until first generation: id=\(model.id, privacy: .public)")
                } else {
                    await self.prewarmModelShaders(for: model, reason: "post-load")
                }
            } catch is CancellationError {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                    self.activeLoadID = nil
                    self.cleanupMemoryAfterLoadInterruption()
                }
                self.logger.notice("MLX container load cancelled during execution: id=\(model.id, privacy: .public)")
            } catch {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                    self.activeLoadID = nil
                    self.loadError = "Failed to load model: \(error.localizedDescription)"
                }
                self.tearDownCurrentModel(reason: "load failure for \(model.id)")
                self.logger.error("MLX model load error: \((error as NSError).localizedDescription, privacy: .public)")
                self.logger.error("MLX container load failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
            }
        }
    }
    nonisolated private static func estimateWeightBytes(for container: ModelContainer) async -> Int {
        await container.perform { context in
            context.model.parameters().flattened().reduce(0) { partialResult, parameter in
                partialResult + parameter.1.nbytes
            }
        }
    }

    nonisolated private static func startReservationTicketIfNeeded(
        weightBytes: Int,
        policy: MLXLMCommon.WiredSumPolicy
    ) async -> MLX.WiredMemoryTicket? {
        guard weightBytes > 0 else { return nil }
        let ticket = MLX.WiredMemoryTicket(size: weightBytes, policy: policy, kind: .reservation)
        _ = await ticket.start()
        return ticket
    }
    func loadModel(_ model: MLXModelInfo) async {
        startLoading(modelID: model.id, source: "manager.loadModel")
    }

    func unloadAllModels() {
        cancelCurrentLoad(reason: "unloadAllModels", tearDownModel: false)
        tearDownCurrentModel(reason: "unloadAllModels")
        logger.notice("Unloaded all MLX models")
    }
    // MARK: - Shader Pre-warming

    /// Runs a minimal text forward pass immediately after model load to pre-compile all LM Metal
    /// shaders (SSM, full-attention, MoE layers). Without this the compiler spike hits on the
    /// first real inference, when the device is also holding a user image in memory.
    /// Skipped on the simulator — the Metal compute stack there can't handle a full LM forward
    /// pass and triggers EXC_BAD_ACCESS.
    func prewarmModelShaders(for model: MLXModelInfo, reason: String) async {
        #if targetEnvironment(simulator)
        logger.notice("Skipping LM shader pre-warm on simulator")
        return
        #else
        guard let container else { return }
        guard currentModel?.id == model.id else { return }
        if deferredPrewarmModelID != model.id && prewarmInFlightModelID == nil && reason != "post-load" {
            return
        }
        if prewarmInFlightModelID == model.id {
            return
        }
        prewarmInFlightModelID = model.id
        logger.notice("MLX prewarm start: id=\(model.id, privacy: .public) reason=\(reason, privacy: .public)")
        MLXMemoryDiagnostics.log(logger, "prewarm.start")
        logger.notice("Pre-warming LM Metal shaders")
        // Free every cached Metal buffer before shader compilation so the compilation
        // spike has the maximum possible headroom on top of the ~3 GB model weights.
        prepareMemoryForPrewarm()
        do {
            let warmupMsg = Chat.Message.user("Hi")
            let userInput = UserInput(chat: [warmupMsg], processing: UserInput.Processing())
            let input = try await container.prepare(input: userInput)
            let params = GenerateParameters(maxTokens: 1, temperature: 1.0, topP: 1.0)
            let stream = try await container.generate(input: input, parameters: params)
            for await _ in stream {
                if Task.isCancelled {
                    throw CancellationError()
                }
            }
            cleanupMemoryAfterPrewarm()
            deferredPrewarmModelID = nil
            prewarmInFlightModelID = nil
            logger.notice("LM Metal shaders pre-warmed")
            logger.notice("MLX prewarm finished: id=\(model.id, privacy: .public)")
            MLXMemoryDiagnostics.log(logger, "prewarm.finished")
        } catch is CancellationError {
            cleanupMemoryAfterPrewarm()
            prewarmInFlightModelID = nil
            logger.notice("MLX prewarm cancelled: id=\(model.id, privacy: .public)")
        } catch {
            cleanupMemoryAfterPrewarm()
            prewarmInFlightModelID = nil
            logger.warning("Shader pre-warm failed (non-fatal): \(error.localizedDescription, privacy: .public)")
            logger.error("MLX prewarm failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
        }
        #endif
    }
}
