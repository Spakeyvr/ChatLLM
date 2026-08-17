//
//  MLXInferenceWorker.swift
//  On-Device_LLM_Chat
//
//  Actor-isolated inference executor for MLX models. Owns the loaded container,
//  persistent chat sessions, and the generation loop including tool-call
//  dispatch. `MLXModelManager` drives it by submitting `MLXInferenceRequest`
//  values and receiving `MLXInferenceResponse`.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers
import OSLog
import UIKit
import os

/// Formats the process's remaining jetsam budget alongside MLX allocator
/// stats. `os_proc_available_memory` reports how many bytes the app can
/// still allocate before iOS terminates it, which is the number that
/// actually matters when chasing out-of-memory kills.
nonisolated enum MLXMemoryDiagnostics {
    static func status() -> String {
        // One snapshot rather than three separate reads, so active/cache/peak
        // describe the same instant. `os_proc_available_memory` reports 0 when
        // the process has no jetsam limit (Simulator, Mac Catalyst), so the
        // headroom field is only meaningful on device.
        let snapshot = Memory.snapshot()
        let headroom = os_proc_available_memory()
        let headroomMB = Double(headroom) / 1_048_576
        let activeMB = Double(snapshot.activeMemory) / 1_048_576
        let cacheMB = Double(snapshot.cacheMemory) / 1_048_576
        let peakMB = Double(snapshot.peakMemory) / 1_048_576
        let headroomField = headroom > 0
            ? String(format: "headroom=%.0fMB", headroomMB)
            : "headroom=n/a"
        return String(
            format: "%@ mlx_active=%.0fMB mlx_cache=%.0fMB mlx_peak=%.0fMB",
            headroomField, activeMB, cacheMB, peakMB
        )
    }

    /// Single place the `MLX mem[...]` log line is built, so its format, level
    /// and privacy stay consistent across the ~11 checkpoints that emit it.
    static func log(_ logger: Logger, _ tag: String, _ detail: String = "") {
        let suffix = detail.isEmpty ? "" : " \(detail)"
        logger.notice(
            "MLX mem[\(tag, privacy: .public)]\(suffix, privacy: .public): \(status(), privacy: .public)"
        )
    }
}

nonisolated struct MLXVisibleMessageSignature: Equatable, Sendable {
    let role: Chat.Message.Role
    let content: String
    let imageIdentifiers: [String]
    let videoIdentifiers: [String]

    init(message: Chat.Message) {
        self.role = message.role
        self.content = message.content
        self.imageIdentifiers = message.images.map(Self.imageIdentifier(for:))
        self.videoIdentifiers = message.videos.map(Self.videoIdentifier(for:))
    }

    nonisolated private static func imageIdentifier(for image: UserInput.Image) -> String {
        switch image {
        case .url(let url):
            return "url:\(url.path)"
        case .ciImage(let image):
            return "ci:\(Int(image.extent.width.rounded()))x\(Int(image.extent.height.rounded()))"
        case .array(let array):
            return "array:\(array.shape.map(String.init).joined(separator: "x"))"
        }
    }

    nonisolated private static func videoIdentifier(for video: UserInput.Video) -> String {
        switch video {
        case .url(let url):
            return "url:\(url.path)"
        case .avAsset(let asset):
            return "asset:\(asset.description)"
        case .frames(let frames):
            return "frames:\(frames.count)"
        }
    }
}

nonisolated struct MLXSessionKey: Hashable, Sendable {
    let conversationID: UUID
    let modelID: String
    let enableThinking: Bool
    let toolsEnabled: Bool
    let includesMedia: Bool
    let instructionFingerprint: Int
}

nonisolated struct MLXLoadedModelState: @unchecked Sendable {
    let loadID: UUID
    let container: ModelContainer
    let model: MLXModelManager.MLXModelInfo
    let wiredMemoryPolicy: MLXLMCommon.WiredSumPolicy
    var reservationTicket: MLX.WiredMemoryTicket?
    var weightBytes: Int
    var activeBytesEstimate: Int
    var prefillStepSize: Int
    var measurement: MLXLMCommon.WiredMemoryMeasurement?
}

nonisolated struct MLXInferenceRequest: @unchecked Sendable {
    let conversationID: UUID
    let sessionKey: MLXSessionKey
    let model: MLXModelManager.MLXModelInfo
    let messages: [Chat.Message]
    let enableThinking: Bool
    let additionalContext: [String: any Sendable]?
    let processing: UserInput.Processing
    let tools: [MLXToolSpec]
    let params: GenerateParameters
    let suppressWrappedXMLToolMarkup: Bool
}

nonisolated struct MLXInferenceResponse: Sendable {
    let toolInvocationCount: Int
    let performanceSample: MLXPerformanceSample
}

nonisolated enum MLXPersistentSessionReleaseScope: Equatable, Sendable {
    case none
    case conversation
    case all
}

actor MLXInferenceWorker {
    private struct SessionCacheConfiguration: Equatable, Sendable {
        let maxKVSize: Int?
        let cacheCompression: KVCacheCompressionMode?
    }

    private struct SessionState {
        let key: MLXSessionKey
        let session: ChatSession
        let cacheConfiguration: SessionCacheConfiguration
        var visibleHistory: [MLXVisibleMessageSignature]
        var latestPerformance: MLXPerformanceSample?
        var lastAccessed: Date
    }

    private actor ToolInvocationState {
        private var count = 0

        func increment() -> Int {
            count += 1
            return count
        }

        func currentCount() -> Int {
            count
        }
    }

    private let logger: Logger
    private var loadedModel: MLXLoadedModelState?
    private var sessions: [MLXSessionKey: SessionState] = [:]
    private let maxPersistentSessions = 2

    init(subsystem: String, deviceSupportProfile: MLXDeviceSupportProfile) {
        self.logger = Logger(subsystem: subsystem, category: "MLXInferenceWorker")
        _ = deviceSupportProfile
    }

    func setLoadedModel(_ state: MLXLoadedModelState) async {
        let previousReservationTicket = loadedModel?.reservationTicket
        loadedModel = state
        sessions.removeAll()
        if let previousReservationTicket {
            _ = await previousReservationTicket.end()
        }
    }

    func updateLoadedModelTuning(
        modelID: String,
        prefillStepSize: Int,
        activeBytesEstimate: Int,
        measurement: MLXLMCommon.WiredMemoryMeasurement?
    ) {
        guard var loadedModel, loadedModel.model.id == modelID else { return }
        loadedModel.prefillStepSize = prefillStepSize
        loadedModel.activeBytesEstimate = max(0, activeBytesEstimate)
        loadedModel.measurement = measurement
        self.loadedModel = loadedModel
    }

    func prefillStepSize(for modelID: String) -> Int {
        guard let loadedModel, loadedModel.model.id == modelID else {
            return MLXModelManager.defaultPrefillStepSize
        }
        return loadedModel.prefillStepSize
    }

    @discardableResult
    func clearLoadedModel(ifLoadID expectedLoadID: UUID? = nil) async -> Bool {
        if let expectedLoadID, loadedModel?.loadID != expectedLoadID {
            return false
        }
        let reservationTicket = loadedModel?.reservationTicket
        loadedModel = nil
        sessions.removeAll()
        if let reservationTicket {
            _ = await reservationTicket.end()
        }
        return true
    }

    func invalidateConversation(_ conversationID: UUID, reason: String) {
        let removedCount = sessions.keys.filter { $0.conversationID == conversationID }.count
        guard removedCount > 0 else { return }
        sessions = sessions.filter { $0.key.conversationID != conversationID }
        logger.notice(
            "MLX session invalidated: conversation=\(conversationID.uuidString, privacy: .public) removed=\(removedCount, privacy: .public) reason=\(reason, privacy: .public)"
        )
    }

    func invalidateAll(reason: String) {
        guard !sessions.isEmpty else { return }
        sessions.removeAll()
        logger.notice("MLX invalidated all sessions: reason=\(reason, privacy: .public)")
    }

    func generate(
        request: MLXInferenceRequest,
        toolDispatch: (@Sendable (MLXToolCall) async throws -> String)?,
        onToken: @escaping @Sendable (String) -> Void,
        onToolCall: @escaping @Sendable (MLXToolCall) -> Void
    ) async throws -> MLXInferenceResponse {
        guard let loadedModel, loadedModel.model.id == request.model.id else {
            throw MLXModelManager.GenerationError.modelNotLoaded
        }
        guard let latestUserMessage = request.messages.last else {
            throw MLXModelManager.GenerationError.invalidChatHistory
        }

        let prefixSignatures = request.messages.dropLast().map { MLXVisibleMessageSignature(message: $0) }
        let requestedCacheConfiguration = SessionCacheConfiguration(
            maxKVSize: request.params.maxKVSize,
            cacheCompression: request.params.resolvedCacheCompression
        )

        let usesExplicitMessageHistory =
            !request.tools.isEmpty &&
            MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: loadedModel.model)

        let persistentSessionReleaseScope = Self.persistentSessionReleaseScope(
            usesExplicitMessageHistory: usesExplicitMessageHistory,
            maxKVSize: request.params.maxKVSize
        )
        try await releasePersistentSessions(
            scope: persistentSessionReleaseScope,
            conversationID: request.conversationID,
            expectedLoadID: loadedModel.loadID
        )

        let reusableSession: ChatSession?
        if usesExplicitMessageHistory {
            reusableSession = nil
            logger.notice(
                "MLX using explicit message-history tool loop: conversation=\(request.conversationID.uuidString, privacy: .public)"
            )
        } else if request.sessionKey.includesMedia {
            reusableSession = nil
            logger.notice(
                "MLX not reusing persistent session for media turn: conversation=\(request.conversationID.uuidString, privacy: .public)"
            )
        } else if var existing = sessions[request.sessionKey],
                  existing.key == request.sessionKey,
                  existing.cacheConfiguration == requestedCacheConfiguration,
                  existing.visibleHistory == prefixSignatures {
            existing.lastAccessed = Date()
            sessions[request.sessionKey] = existing
            reusableSession = existing.session
            logger.notice("MLX reusing persistent chat session: conversation=\(request.conversationID.uuidString, privacy: .public)")
        } else {
            if let existing = sessions[request.sessionKey],
               existing.key == request.sessionKey,
               existing.visibleHistory == prefixSignatures,
               existing.cacheConfiguration != requestedCacheConfiguration {
                logger.notice(
                    "MLX recreating persistent chat session due to cache configuration change: conversation=\(request.conversationID.uuidString, privacy: .public)"
                )
            }
            reusableSession = ChatSession(
                loadedModel.container,
                history: Array(request.messages.dropLast()),
                generateParameters: request.params,
                processing: request.processing,
                additionalContext: request.additionalContext,
                tools: request.tools,
                toolDispatch: nil
            )
            logger.notice("MLX created fresh chat session: conversation=\(request.conversationID.uuidString, privacy: .public)")
        }

        reusableSession?.generateParameters = request.params
        reusableSession?.processing = request.processing
        reusableSession?.additionalContext = request.additionalContext
        reusableSession?.toolDispatch = nil

        Memory.peakMemory = 0
        let memoryBefore = Memory.snapshot()
        let startedAt = Date()
        var firstTokenAt: Date?
        var assistantVisibleText = ""
        var stopReason: GenerateStopReason?
        var cumulativePromptTokenCount = 0
        var cumulativeOutputTokenCount = 0
        var cumulativePromptTime: TimeInterval = 0
        var cumulativeGenerationTime: TimeInterval = 0
        var activePromptContent = latestUserMessage.content
        var activePromptRole = latestUserMessage.role
        var activePromptImages = latestUserMessage.images
        var activePromptVideos = latestUserMessage.videos
        var activeTools = request.tools
        // Seeds every path that builds its own `ChatSession` below, not just the
        // explicit tool loop. Media turns also skip session reuse, and leaving
        // this empty for them dropped the system prompt and the entire prior
        // conversation from the fresh session.
        var explicitMessageHistory = Array(request.messages.dropLast())
        let toolInvocationState = ToolInvocationState()
        let effectiveCachePolicy = MLXModelManager.cachePolicy(for: request.params)
        let kvBenchmarkMetadata = MLXKVBenchmarkMetadata(
            cachePolicy: effectiveCachePolicy,
            effectiveMaxKVSize: request.params.maxKVSize,
            prefillStepSize: request.params.prefillStepSize
        )

        let activeTicket = makeActiveInferenceTicket(from: loadedModel)
        MLXMemoryDiagnostics.log(self.logger, "generate.entry")
        let iterateStream = {
            var streamIteration = 0
            while true {
                streamIteration += 1
                MLXMemoryDiagnostics.log(
                    self.logger, "prefill.start", "iteration=\(streamIteration)")
                let session: ChatSession
                if let reusableSession {
                    session = reusableSession
                    session.tools = activeTools
                } else {
                    session = ChatSession(
                        loadedModel.container,
                        history: explicitMessageHistory,
                        generateParameters: request.params,
                        processing: request.processing,
                        additionalContext: request.additionalContext,
                        tools: activeTools,
                        toolDispatch: nil
                    )
                }
                let outputFilter = request.suppressWrappedXMLToolMarkup
                    ? WrappedXMLToolCallStreamFilter()
                    : nil
                var completionInfo: GenerateCompletionInfo?
                var emittedToolCall: MLXToolCall?
                var chunkEventCount = 0

                for try await generation in session.streamDetails(
                    to: activePromptContent,
                    role: activePromptRole,
                    images: activePromptImages,
                    videos: activePromptVideos
                ) {
                    if Task.isCancelled {
                        throw CancellationError()
                    }
                    switch generation {
                    case .chunk(let text):
                        if firstTokenAt == nil {
                            firstTokenAt = Date()
                            MLXMemoryDiagnostics.log(self.logger, "first-token")
                        }
                        chunkEventCount += 1
                        if chunkEventCount % 128 == 0 {
                            MLXMemoryDiagnostics.log(
                                self.logger, "decode", "chunks=\(chunkEventCount)")
                        }
                        if let outputFilter {
                            if let visibleChunk = await outputFilter.consume(text), !visibleChunk.isEmpty {
                                assistantVisibleText += visibleChunk
                                onToken(visibleChunk)
                            }
                        } else if !text.isEmpty {
                            assistantVisibleText += text
                            onToken(text)
                        }
                    case .toolCall(let toolCall):
                        emittedToolCall = toolCall
                        MLXMemoryDiagnostics.log(self.logger, "tool-call.emitted")
                        onToolCall(toolCall)
                        if let outputFilter {
                            await outputFilter.didDispatchToolCall()
                        }
                        break
                    case .info(let info):
                        completionInfo = info
                    }

                    if emittedToolCall != nil {
                        break
                    }
                }

                if let outputFilter,
                   let trailingText = await outputFilter.finish(),
                   !trailingText.isEmpty {
                    if firstTokenAt == nil {
                        firstTokenAt = Date()
                    }
                    assistantVisibleText += trailingText
                    onToken(trailingText)
                }

                if let completionInfo {
                    cumulativePromptTokenCount += completionInfo.promptTokenCount
                    cumulativeOutputTokenCount += completionInfo.generationTokenCount
                    cumulativePromptTime += completionInfo.promptTime
                    cumulativeGenerationTime += completionInfo.generateTime
                    stopReason = completionInfo.stopReason
                }

                guard let emittedToolCall else {
                    if reusableSession == nil {
                        await session.synchronize()
                        await session.clear()
                        Memory.clearCache()
                        MLXMemoryDiagnostics.log(
                            self.logger, "disposable-session.cleared",
                            "iteration=\(streamIteration) reason=response-complete")
                    }
                    break
                }

                await session.synchronize()
                if reusableSession == nil {
                    await session.clear()
                    Memory.clearCache()
                    MLXMemoryDiagnostics.log(
                        self.logger, "disposable-session.cleared",
                        "iteration=\(streamIteration) reason=tool-call")
                }

                let invocationCount = await toolInvocationState.increment()
                let toolResult: String
                if invocationCount > MLXModelManager.maxToolInvocationsPerResponse {
                    self.logger.error(
                        "MLX forcing tool shutdown after excessive tool calls: count=\(invocationCount, privacy: .public) max=\(MLXModelManager.maxToolInvocationsPerResponse, privacy: .public)"
                    )
                    toolResult = MLXModelManager.excessiveToolCallToolResponse(
                        maximum: MLXModelManager.maxToolInvocationsPerResponse
                    )
                    activeTools = []
                } else {
                    guard let toolDispatch else {
                        throw MLXModelManager.GenerationError.missingToolDispatch
                    }

                    self.logger.notice("MLX dispatching tool call: name=\(emittedToolCall.function.name, privacy: .public)")
                    toolResult = try await toolDispatch(emittedToolCall)
                    MLXMemoryDiagnostics.log(
                        self.logger, "tool.dispatched", "result_chars=\(toolResult.count)")
                    if MLXModelManager.shouldDisableTools(after: toolResult) {
                        self.logger.notice("MLX disabling tools for remainder of response after search limit was reached")
                        activeTools = []
                    } else if AppWebSearchToolBridge.isInternalToolErrorResponse(toolResult) {
                        self.logger.notice("MLX received recoverable internal tool error; leaving tools enabled so the model can retry with corrected arguments")
                        activeTools = request.tools
                    }
                }

                // Unconditional: this is only ever read when `reusableSession` is
                // nil and the loop rebuilds a session per iteration, which now
                // includes media turns. Skipping it there dropped the current
                // question from every post-tool-call iteration. When a session is
                // being reused the value is never read, so appending is inert.
                explicitMessageHistory.append(
                    Chat.Message(
                        role: activePromptRole,
                        content: activePromptContent,
                        images: activePromptImages,
                        videos: activePromptVideos
                    )
                )

                activePromptContent = MLXModelManager.toolResponsePromptContent(
                    for: loadedModel.model,
                    toolResult: toolResult
                )
                activePromptRole = MLXModelManager.toolResponsePromptRole(for: loadedModel.model)
                activePromptImages = []
                activePromptVideos = []
            }
        }

        do {
            if let activeTicket {
                try await activeTicket.withWiredLimit {
                    try await iterateStream()
                }
            } else {
                try await iterateStream()
            }
        } catch {
            sessions.removeValue(forKey: request.sessionKey)
            if reusableSession == nil {
                Memory.clearCache()
            }
            throw error
        }

        let finishedAt = Date()
        MLXMemoryDiagnostics.log(logger, "generate.end")
        let memoryAfter = Memory.snapshot()
        let toolInvocationCount = await toolInvocationState.currentCount()
        let performanceSample = MLXPerformanceSample(
            conversationID: request.conversationID,
            modelID: request.model.id,
            promptTokenCount: cumulativePromptTokenCount,
            outputTokenCount: max(
                cumulativeOutputTokenCount,
                max(0, Int(ceil(Double(assistantVisibleText.count) / 4.0)))
            ),
            toolInvocationCount: toolInvocationCount,
            timeToFirstToken: firstTokenAt.map { $0.timeIntervalSince(startedAt) },
            totalLatency: finishedAt.timeIntervalSince(startedAt),
            promptTokensPerSecond: cumulativePromptTime > 0
                ? Double(cumulativePromptTokenCount) / cumulativePromptTime
                : nil,
            decodeTokensPerSecond: cumulativeGenerationTime > 0
                ? Double(cumulativeOutputTokenCount) / cumulativeGenerationTime
                : nil,
            stopReason: stopReason,
            memoryBefore: memoryBefore,
            memoryAfter: memoryAfter,
            peakActiveBytes: Memory.peakMemory,
            kvBenchmarkMetadata: kvBenchmarkMetadata
        )

        var visibleHistory = request.messages.map { MLXVisibleMessageSignature(message: $0) }
        visibleHistory.append(
            MLXVisibleMessageSignature(
                message: .assistant(assistantVisibleText)
            )
        )
        if let reusableSession, !request.sessionKey.includesMedia {
            sessions[request.sessionKey] = SessionState(
                key: request.sessionKey,
                session: reusableSession,
                cacheConfiguration: requestedCacheConfiguration,
                visibleHistory: visibleHistory,
                latestPerformance: performanceSample,
                lastAccessed: finishedAt
            )
            evictPersistentSessionsIfNeeded()
        }

        logger.notice(
            "MLX performance sample: conversation=\(request.conversationID.uuidString, privacy: .public) prompt_tokens=\(performanceSample.promptTokenCount, privacy: .public) output_tokens=\(performanceSample.outputTokenCount, privacy: .public) ttft=\(String(format: "%.3f", performanceSample.timeToFirstToken ?? 0), privacy: .public)s latency=\(String(format: "%.3f", performanceSample.totalLatency), privacy: .public)s tok_s=\(String(format: "%.2f", performanceSample.decodeTokensPerSecond ?? 0), privacy: .public) cache_policy=\(effectiveCachePolicy.diagnosticLabel, privacy: .public) max_kv=\(performanceSample.kvBenchmarkMetadata.effectiveMaxKVSize ?? -1, privacy: .public)"
        )

        self.loadedModel = loadedModel
        return MLXInferenceResponse(
            toolInvocationCount: toolInvocationCount,
            performanceSample: performanceSample
        )
    }

    private func makeActiveInferenceTicket(from loadedModel: MLXLoadedModelState) -> MLX.WiredMemoryTicket? {
        let size = max(0, loadedModel.activeBytesEstimate)
        guard size > 0 else { return nil }
        return MLX.WiredMemoryTicket(
            size: size,
            policy: loadedModel.wiredMemoryPolicy,
            kind: .active
        )
    }

    nonisolated static func persistentSessionReleaseScope(
        usesExplicitMessageHistory: Bool,
        maxKVSize: Int?
    ) -> MLXPersistentSessionReleaseScope {
        guard usesExplicitMessageHistory else { return .none }
        return maxKVSize == nil ? .conversation : .all
    }

    private func releasePersistentSessions(
        scope: MLXPersistentSessionReleaseScope,
        conversationID: UUID,
        expectedLoadID: UUID
    ) async throws {
        let keysToRemove: [MLXSessionKey]
        switch scope {
        case .none:
            return
        case .conversation:
            keysToRemove = sessions.keys.filter { $0.conversationID == conversationID }
        case .all:
            keysToRemove = Array(sessions.keys)
        }

        let discardedSessions = keysToRemove.compactMap { key -> ChatSession? in
            sessions.removeValue(forKey: key)?.session
        }
        for session in discardedSessions {
            await session.clear()
        }

        guard loadedModel?.loadID == expectedLoadID else {
            throw MLXModelManager.GenerationError.modelNotLoaded
        }

        // MLX retains freed buffers in its allocator cache. Releasing them here
        // prevents a fresh full-history tool prefill from overlapping old KV
        // allocations on memory-constrained devices.
        Memory.clearCache()
        logger.notice(
            "MLX released persistent sessions before explicit tool loop: scope=\(String(describing: scope), privacy: .public) count=\(discardedSessions.count, privacy: .public) conversation=\(conversationID.uuidString, privacy: .public)"
        )
    }

    private func evictPersistentSessionsIfNeeded() {
        guard sessions.count > maxPersistentSessions else { return }
        let overflow = sessions.count - maxPersistentSessions
        let keysToRemove = sessions
            .sorted { $0.value.lastAccessed < $1.value.lastAccessed }
            .prefix(overflow)
            .map(\.key)
        for key in keysToRemove {
            sessions.removeValue(forKey: key)
        }
        logger.notice("MLX evicted \(keysToRemove.count, privacy: .public) persistent session(s) to cap KV memory")
    }
}
