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

    func setLoadedModel(_ state: MLXLoadedModelState) {
        loadedModel = state
        sessions.removeAll()
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

    func clearLoadedModel() async {
        if let reservationTicket = loadedModel?.reservationTicket {
            _ = await reservationTicket.end()
        }
        loadedModel = nil
        sessions.removeAll()
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
        var explicitMessageHistory = usesExplicitMessageHistory
            ? Array(request.messages.dropLast())
            : []
        let toolInvocationState = ToolInvocationState()
        let effectiveCachePolicy = MLXModelManager.cachePolicy(for: request.params)
        let kvBenchmarkMetadata = MLXKVBenchmarkMetadata(
            cachePolicy: effectiveCachePolicy,
            effectiveMaxKVSize: request.params.maxKVSize,
            prefillStepSize: request.params.prefillStepSize
        )

        let activeTicket = makeActiveInferenceTicket(from: loadedModel)
        let iterateStream = {
            while true {
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
                    break
                }

                await session.synchronize()

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
                    if MLXModelManager.shouldDisableTools(after: toolResult) {
                        self.logger.notice("MLX disabling tools for remainder of response after search limit was reached")
                        activeTools = []
                    } else if AppWebSearchToolBridge.isInternalToolErrorResponse(toolResult) {
                        self.logger.notice("MLX received recoverable internal tool error; leaving tools enabled so the model can retry with corrected arguments")
                        activeTools = request.tools
                    }
                }

                if usesExplicitMessageHistory {
                    explicitMessageHistory.append(
                        Chat.Message(
                            role: activePromptRole,
                            content: activePromptContent,
                            images: activePromptImages,
                            videos: activePromptVideos
                        )
                    )
                }

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
            throw error
        }

        let finishedAt = Date()
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
