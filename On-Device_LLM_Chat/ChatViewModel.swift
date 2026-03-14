//
//  ChatViewModel.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import SwiftData
import Combine
@preconcurrency import Vision
import UIKit
import FoundationModels
import Security  // For Keychain
import os.log
@preconcurrency import MLXLMCommon

// Resolve ambiguities between MLXLMCommon and SwiftData/FoundationModels
typealias ModelContext = SwiftData.ModelContext

@MainActor
final class ChatViewModel: ObservableObject {

    // MARK: - Logging
    let logger = Logger(subsystem: "com.yourapp.chatllm", category: "ChatViewModel")

    @Published var isGenerating = false
    @Published var streamingMessageID: UUID?
    @Published var tavilyKeyMissing: Bool = false

    // Regeneration lock to prevent concurrent regenerations
    var isRegenerating = false

    internal let generator: LLMGenerator
    internal let context: ModelContext

    // Tavily search service (user-configurable)
    var searchService: TavilySearchService?
    private var keyChangeCancellable: AnyCancellable?
    private var modelPipelineResetCancellable: AnyCancellable?

    // Thread-safe order calculation to prevent race conditions
    internal var nextOrder: Int {
        let orders = conversation.messages.map { $0.order }
        return (orders.max() ?? -1) + 1
    }

    private(set) var conversation: Conversation

    // Track the active streaming task so we can cancel it
    internal var currentStreamTask: Task<Void, Never>?
    private var lastStreamOutcome: StreamOutcome = .succeeded

    // Serializes concurrent Foundation Models sessions via continuation queue.
    private let fmGate = FoundationModelsGate()

    // MARK: - Timeouts
    // Give the first token longer (model load/warm-up), then enforce a tighter per-chunk timeout.
    private let firstTokenTimeout: Duration = .seconds(30)
    private let chunkTimeout: Duration = .seconds(20)
    private let reasoningChunkTimeout: Duration = .seconds(30)

    // Keychain constants for Tavily API key storage
    let tavilyKeyService = "com.yourapp.tavily"  // Replace with your app's bundle ID
    let tavilyKeyAccount = "TavilyAPIKey"

    // MARK: - Coalesced saving with optimized debouncing
    var pendingSaveTask: Task<Void, Never>?
    private let saveInterval: Duration = .milliseconds(300) // Optimized for streaming performance
    var saveCount: Int = 0
    private let forceSaveThreshold: Int = 25 // Balance between persistence and performance
    var lastSaveTime: Date = .distantPast

    // MARK: - Foundation Models Session Management

    /// Serializes Foundation Models operations so only one session runs at a time,
    /// preventing "accumulator already completed" errors.
    func withFoundationModelsLock<T>(
        timeout: Duration = .seconds(15),
        operation: @escaping () async throws -> T
    ) async throws -> T {
        await fmGate.acquire()
        do {
            let result = try await withTimeout(timeout, operation: operation)
            await fmGate.release()
            return result
        } catch {
            await fmGate.release()
            throw error
        }
    }

    // MARK: - Initializer
    init(generator: LLMGenerator, context: ModelContext, conversation: Conversation) {
        self.generator = generator
        self.context = context
        self.conversation = conversation

        loadTavilyAPIKey()

        keyChangeCancellable = NotificationCenter.default.publisher(for: NSNotification.Name("TavilyKeyChanged"))
            .sink { [weak self] _ in
                self?.loadTavilyAPIKey()
            }

        modelPipelineResetCancellable = NotificationCenter.default.publisher(for: .modelPipelineWillReset)
            .sink { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.cancelGeneration()
                }
            }
    }

    deinit {
        pendingSaveTask?.cancel()
        currentStreamTask?.cancel()
        keyChangeCancellable?.cancel()
        modelPipelineResetCancellable?.cancel()
    }

    // MARK: - Coalesced Save

    internal func scheduleCoalescedSave() {
        saveCount += 1

        let timeSinceLastSave = Date().timeIntervalSince(lastSaveTime)
        if timeSinceLastSave < 0.15 { // Less than 150ms since last save
            return
        }

        if saveCount >= forceSaveThreshold {
            immediateSave()
            return
        }

        guard pendingSaveTask?.isCancelled != false else {
            return // Task already scheduled, let it complete
        }

        pendingSaveTask = Task { @MainActor [weak self] in
            guard let self = self else { return }
            try? await Task.sleep(for: self.saveInterval)
            guard !Task.isCancelled else { return }
            do {
                try self.context.save()
                self.saveCount = 0
                self.lastSaveTime = Date()
            } catch {
                print("❌ Failed to save context: \(error.localizedDescription)")
                // Retry once more immediately on failure
                try? self.context.save()
            }
            self.pendingSaveTask = nil
        }
    }

    /// Inserts the conversation into the model context if it hasn't been persisted yet.
    /// Called before every save so draft conversations are transparently promoted on first message send.
    private func insertIntoContextIfNeeded() {
        guard conversation.modelContext == nil else { return }
        context.insert(conversation)
    }

    internal func immediateSave() {
        insertIntoContextIfNeeded()
        pendingSaveTask?.cancel()
        pendingSaveTask = nil
        saveCount = 0
        lastSaveTime = Date()
        do {
            try context.save()
        } catch {
            print("Failed to immediately save context: \(error)")
        }
    }

    internal static func mergeSearchInvocations(
        _ liveInvocations: [SearchInvocation],
        preservingAnchorsFrom existingInvocations: [SearchInvocation],
        currentChunkCount: Int
    ) -> [SearchInvocation] {
        guard !liveInvocations.isEmpty else { return [] }

        let existingByID = Dictionary(uniqueKeysWithValues: existingInvocations.map { ($0.id, $0) })

        return liveInvocations.map { invocation in
            if let existing = existingByID[invocation.id] {
                var merged = invocation
                merged.anchorStepNumber = existing.anchorStepNumber ?? invocation.anchorStepNumber
                return merged
            }

            var anchored = invocation
            if anchored.anchorStepNumber == nil {
                anchored.anchorStepNumber = max(0, currentChunkCount)
            }
            return anchored
        }
    }

    private func syncLiveSearchInvocations(into message: Message, from bridge: AppWebSearchToolBridge?) {
        guard let bridge else { return }
        let liveInvocations = bridge.allInvocations
        guard !liveInvocations.isEmpty else { return }

        let currentChunkCount = reasoningChunkCount(for: message.reasoning)
        let merged = Self.mergeSearchInvocations(
            liveInvocations,
            preservingAnchorsFrom: message.searchInvocations ?? [],
            currentChunkCount: currentChunkCount
        )

        message.searchInvocations = merged
    }

    private func reasoningChunkCount(for reasoning: String?) -> Int {
        guard let reasoning else { return 0 }
        let trimmed = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }

        let chunks = trimmed
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        return max(1, chunks.count)
    }

    internal func resolvedReasoningMode(for prompt: String, logContext: String) async -> Bool {
        do {
            return try await shouldUseReasoningForPrompt(prompt)
        } catch {
            print("Unexpected error determining reasoning mode\(logContext.isEmpty ? "" : " for \(logContext)"), using conversation fallback: \(error)")
            return conversation.reasoningMode || conversation.smartReasoningMode
        }
    }

    internal func appendUserMessage(_ text: String) -> (message: Message, needsAutoNaming: Bool) {
        let userMsg = Message(
            role: .user,
            text: text,
            order: nextOrder,
            conversation: conversation,
            isFinal: true
        )
        conversation.messages.append(userMsg)

        let userMessagesCount = conversation.messages.lazy.filter { $0.role == .user }.count
        let isFirstUserMessage = userMessagesCount == 1
        let needsAutoNaming =
            conversation.title == String(localized: "New Chat") &&
            !conversation.hasAutoGeneratedTitle &&
            isFirstUserMessage

        return (userMsg, needsAutoNaming)
    }

    internal func appendAssistantPlaceholder(
        isReasoningMode: Bool,
        searchQuery: String? = nil,
        requiresWebSearch: Bool = false
    ) -> Message {
        let assistantMsg = Message(
            role: .assistant,
            text: "",
            order: nextOrder,
            conversation: conversation,
            isFinal: false,
            isReasoningMode: isReasoningMode,
            searchQuery: searchQuery,
            requiresWebSearch: requiresWebSearch
        )
        conversation.messages.append(assistantMsg)
        conversation.lastUpdated = Date()
        immediateSave()
        return assistantMsg
    }

    internal func finishAutoNamingIfNeeded(_ needsAutoNaming: Bool, userText: String) async {
        guard needsAutoNaming, !Task.isCancelled else { return }
        await generateChatTitle(fromUserMessage: userText)
    }

    // MARK: - Public controls

    func cancelGeneration() {
        // Only cancel the task; let the streaming task perform the unified cleanup.
        currentStreamTask?.cancel()
    }

    // MARK: - Chat flow

    func send(userText: String, forceSearch: Bool = false, disableToolCalls: Bool = false) async {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        await waitForStreamToFinish()

        guard !isGenerating, !Task.isCancelled else { return }

        let shouldUseReasoning = await resolvedReasoningMode(for: trimmed, logContext: "")

        guard !Task.isCancelled else { return }

        let turn = appendUserMessage(trimmed)
        let assistantMsg = appendAssistantPlaceholder(
            isReasoningMode: shouldUseReasoning,
            searchQuery: forceSearch ? trimmed : nil,
            requiresWebSearch: forceSearch
        )

        guard !Task.isCancelled else { return }

        let searchInstruction = forceSearch
            ? "You must call the webSearch tool before answering. Search for current information about this request first, then answer using the tool results."
            : nil
        await streamAssistant(
            into: assistantMsg,
            basedOnHistoryUpTo: assistantMsg.order,
            additionalUserInstruction: searchInstruction,
            disableWebSearch: disableToolCalls
        )

        await finishAutoNamingIfNeeded(turn.needsAutoNaming, userText: trimmed)
    }

    // MARK: - OCR

    func extractOCR(from image: UIImage) async -> String {
        await withTaskCancellationHandler {
            await Task.detached(priority: .userInitiated) { () -> String in
                guard let cg = image.cgImage else {
                    print("Failed to get CGImage from UIImage")
                    return ""
                }

                let request = VNRecognizeTextRequest()
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = true

                request.automaticallyDetectsLanguage = true
                if #available(iOS 16.0, *) {
                    request.revision = VNRecognizeTextRequestRevision3
                }

                let handler = VNImageRequestHandler(cgImage: cg, options: [:])
                do {
                    try handler.perform([request])
                    guard let results = request.results, !results.isEmpty else {
                        print("No text recognition results found")
                        return ""
                    }

                    let texts: [String] = results.compactMap { observation in
                        guard let topCandidate = observation.topCandidates(1).first,
                              topCandidate.confidence > 0.1 else { return nil }
                        return topCandidate.string
                    }

                    let extractedText = texts.joined(separator: "\n")
                    print("OCR extracted \(texts.count) text segments")
                    return extractedText
                } catch {
                    print("OCR processing failed: \(error)")
                    return ""
                }
            }.value
        } onCancel: {
            print("OCR task was cancelled")
        }
    }

    // MARK: - Internals

    internal func waitForStreamToFinish() async {
        if let task = currentStreamTask {
            do {
                try await withTimeout(.seconds(4)) {
                    await task.value
                }
            } catch is GenerationTimeoutError {
                print("⚠️ waitForStreamToFinish timed out after 4 seconds")
                task.cancel()
                print("🛑 Cancelled stale streaming task")
            } catch {
                print("⚠️ waitForStreamToFinish saw unexpected error: \(error.localizedDescription)")
            }
        }
        // Also ensure flags are down (in case a fast-path set them slightly later)
        var attempts = 0
        while (isGenerating || currentStreamTask != nil) && attempts < 100 {
            try? await Task.sleep(for: .milliseconds(40))
            attempts += 1
        }

        if attempts >= 100 {
            if let task = currentStreamTask {
                task.cancel()

                for _ in 0..<10 {
                    try? await Task.sleep(for: .milliseconds(50))
                    if currentStreamTask == nil && !isGenerating {
                        break
                    }
                }
            }

            isGenerating = false
            currentStreamTask = nil
            print("🔄 Force-reset streaming state after cancellation attempt")
        }
    }

    func renumberMessagesByOrder() {
        let sorted = conversation.messages.sorted {
            if $0.order == $1.order {
                return $0.createdAt < $1.createdAt
            }
            return $0.order < $1.order
        }
        for (i, msg) in sorted.enumerated() {
            msg.order = i
        }
    }

    private struct GenerationTimeoutError: LocalizedError {
        var timeout: Duration
        var errorDescription: String? {
            let seconds = Double(timeout.components.seconds) + Double(timeout.components.attoseconds) / 1e18
            return "Generation timed out after \(Int(seconds)) seconds."
        }
    }

    private struct ContextWindowError: LocalizedError {
        var errorDescription: String? {
            return """
            Context window exceeded. The conversation has become too long for the on-device model to process.

            Try one of these solutions:
            • Start a new chat
            • Delete some earlier messages in this conversation
            • Reduce the length of your system prompt if you have one

            Tip: The on-device model works best with focused conversations of 10-20 messages.
            """
        }
    }

    enum StreamOutcome {
        case succeeded
        case failedBeforeOutput
        case failedAfterPartialOutput
        case cancelled
    }

    func withTimeout<T>(_ duration: Duration, operation: @escaping () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                try await Task.sleep(for: duration)
                throw GenerationTimeoutError(timeout: duration)
            }
            let value = try await group.next()
            guard let value = value else {
                throw GenerationTimeoutError(timeout: duration)
            }
            group.cancelAll()
            return value
        }
    }

    private static let contextWindowErrorCodes: Set<Int> = [-1, 400, 413]

    private struct StreamResetState {
        let target: Message
        let targetID: UUID
        let previousText: String
        let previousReasoning: String?
        let previousFinal: String?
        let forceSearchRequired: Bool
        let latestUserQuestion: String
        let questionLogPreview: String
    }

    private enum StreamBackendRequest {
        case mlx(
            manager: MLXModelManager,
            messages: [Chat.Message],
            enableThinking: Bool,
            tools: [MLXToolSpec],
            toolDispatch: (@Sendable (MLXToolCall) async throws -> String)?
        )
        case foundation(
            prompt: String,
            tools: [any FoundationModelTool]
        )
    }

    private struct StreamPreparation {
        let targetID: UUID
        let forceSearchRequired: Bool
        let previousText: String
        let previousReasoning: String?
        let previousFinal: String?
        let questionLogPreview: String
        let webSearchBridge: AppWebSearchToolBridge?
        let backendRequest: StreamBackendRequest
    }

    private struct StreamConsumptionResult {
        var cumulativeText = ""
        var wroteAny = false
        var outcome: StreamOutcome = .succeeded
    }

    private enum StreamEvent {
        case text(String)
        case toolCall(MLXToolCall)
    }

    /// Detects if an error is related to context window/length limits
    func isContextWindowError(_ error: Error) -> Bool {
        let errorString = error.localizedDescription.lowercased()
        let hasContext = errorString.contains("context")
        let hasLimitWord = errorString.contains("length") || errorString.contains("limit") || errorString.contains("window")
        if hasContext && hasLimitWord { return true }
        let nsError = error as NSError
        if Self.contextWindowErrorCodes.contains(nsError.code) &&
           (nsError.domain.contains("LanguageModel") || nsError.domain.contains("FoundationModels")) &&
           hasContext {
            return true
        }
        return false
    }

    // Add optional transient instruction that is appended to the prompt for this run only.
    @discardableResult
    internal func streamAssistant(
        into assistantMessage: Message,
        basedOnHistoryUpTo order: Int,
        additionalUserInstruction: String? = nil,
        disableWebSearch: Bool = false,
        allowNativeImages: Bool = true
    ) async -> StreamOutcome {
        print("🔄 streamAssistant: Starting (order: \(order), messageID: \(assistantMessage.id))")

        guard !Task.isCancelled else {
            print("⚠️ streamAssistant: Task cancelled before starting")
            return .cancelled
        }

        // isGenerating is managed exclusively here to avoid race conditions.
        isGenerating = true
        defer {
            isGenerating = false
        }

        streamingMessageID = assistantMessage.id

        guard let resetState = prepareStreamResetState(
            from: assistantMessage,
            basedOnHistoryUpTo: order
        ) else {
            streamingMessageID = nil
            return .failedBeforeOutput
        }

        guard let preparation = await prepareStream(
            from: resetState,
            basedOnHistoryUpTo: order,
            additionalUserInstruction: additionalUserInstruction,
            disableWebSearch: disableWebSearch,
            allowNativeImages: allowNativeImages
        ) else {
            streamingMessageID = nil
            return .failedBeforeOutput
        }

        let task = Task { @MainActor [weak self] in
            guard let self = self else { return }
            let consumption = await self.performStreaming(preparation: preparation)
            let capturedInvocations = self.captureSearchInvocations(
                from: preparation.webSearchBridge,
                targetID: preparation.targetID
            )
            let outcome = self.finalizeStreaming(
                preparation: preparation,
                consumption: consumption,
                capturedInvocations: capturedInvocations
            )
            self.finishStreamingSession(with: outcome)
        }

        lastStreamOutcome = .succeeded
        currentStreamTask = task
        await task.value
        currentStreamTask = nil
        return lastStreamOutcome
    }

    private func prepareStreamResetState(
        from assistantMessage: Message,
        basedOnHistoryUpTo order: Int
    ) -> StreamResetState? {
        let targetID = assistantMessage.id
        let targetRole = assistantMessage.role
        let previousText = assistantMessage.text
        let previousReasoning = assistantMessage.reasoning
        let previousFinal = assistantMessage.finalAnswer
        let forceSearchRequired = assistantMessage.requiresWebSearch == true ||
            (assistantMessage.searchInvocations == nil && assistantMessage.searchQuery != nil)
        let latestUserQuestion = conversation.messages
            .filter { $0.order < order && $0.role == .user }
            .sortedByOrder
            .last?
            .text ?? ""
        let questionLogPreview = String(
            latestUserQuestion
                .replacingOccurrences(of: "\n", with: " ")
                .prefix(240)
        )

        guard let target = conversation.messages.first(where: { $0.id == targetID }),
              targetRole == .assistant else {
            return nil
        }

        let preservedSources = target.resetForRegeneration(preserveSources: true)
        if let sources = preservedSources, !sources.isEmpty {
            let sourcesBlock = "<sources>\n\(sources)\n</sources>\n\n"
            if target.isReasoningMode {
                target.finalAnswer = sourcesBlock
            } else {
                target.text = sourcesBlock
            }
        }

        conversation.lastUpdated = Date()
        immediateSave()

        return StreamResetState(
            target: target,
            targetID: targetID,
            previousText: previousText,
            previousReasoning: previousReasoning,
            previousFinal: previousFinal,
            forceSearchRequired: forceSearchRequired,
            latestUserQuestion: latestUserQuestion,
            questionLogPreview: questionLogPreview
        )
    }

    private func prepareStream(
        from resetState: StreamResetState,
        basedOnHistoryUpTo order: Int,
        additionalUserInstruction: String?,
        disableWebSearch: Bool,
        allowNativeImages: Bool
    ) async -> StreamPreparation? {
        let promptBridge = ModelBackendBridge.shared
        let toolCallsDisabled = disableWebSearch || UserDefaults.standard.disableToolCalls
        let backendSupportsWebSearchTools = promptBridge.toolCallsAvailableForCurrentBackend
        let autonomousWebSearchAvailable = !toolCallsDisabled && searchService != nil && backendSupportsWebSearchTools
        let webSearchAvailable = autonomousWebSearchAvailable
        let webSearchBridge: AppWebSearchToolBridge? = if webSearchAvailable, let service = searchService {
            AppWebSearchToolBridge(searchService: service)
        } else {
            nil
        }

        logger.notice(
            "streamAssistant tool setup: backend=\(promptBridge.selectedBackend.rawValue, privacy: .public) force_search=\(resetState.forceSearchRequired, privacy: .public) tools_disabled=\(toolCallsDisabled, privacy: .public) backend_tool_support=\(backendSupportsWebSearchTools, privacy: .public) autonomous_search=\(autonomousWebSearchAvailable, privacy: .public) search_service=\(self.searchService != nil, privacy: .public) web_tool_available=\(webSearchAvailable, privacy: .public)"
        )
        logger.notice(
            "streamAssistant question: chars=\(resetState.latestUserQuestion.count, privacy: .public) search_query_present=\(resetState.target.searchQuery != nil, privacy: .public)"
        )

        let foundationModelTools: [any FoundationModelTool] = webSearchBridge.map { [$0.foundationModelTool] } ?? []
        let mlxTools: [MLXToolSpec] = webSearchBridge.map { [$0.mlxToolSpec] } ?? []
        let mlxToolDispatch: (@Sendable (MLXToolCall) async throws -> String)? = if let bridge = webSearchBridge {
            { toolCall in
                try await bridge.dispatchMLXToolCall(toolCall)
            }
        } else {
            nil
        }

        logger.notice(
            "streamAssistant tools prepared: foundation_tools=\(foundationModelTools.count, privacy: .public) mlx_tools=\(mlxTools.count, privacy: .public)"
        )

        if resetState.forceSearchRequired && webSearchBridge == nil {
            let failureMessage: String
            if let backendIssue = promptBridge.toolCallAvailabilityIssueForCurrentBackend {
                failureMessage = "\(backendIssue) Disable forced web search for this response or switch to a supported backend."
            } else if toolCallsDisabled {
                failureMessage = "Web search was explicitly required for this response, but tool calls are disabled. Re-enable tool calls or turn off forced web search."
            } else {
                failureMessage = "Web search was explicitly required for this response, but no search tool is configured. Add a Tavily API key."
            }
            failStreamPreparation(
                target: resetState.target,
                message: failureMessage,
                clearVisibleContent: true
            )
            return nil
        }

        let backendRequest: StreamBackendRequest
        if promptBridge.selectedBackend == .mlx, let manager = promptBridge.modelManager {
            do {
                let messages = try await buildQwenMessages(
                    upToOrderExclusive: order,
                    additionalInstruction: additionalUserInstruction,
                    includeLatestUserImages: allowNativeImages,
                    maxMessages: webSearchAvailable ? 10 : nil,
                    toolsAvailable: webSearchAvailable,
                    forceWebSearch: resetState.forceSearchRequired
                )
                backendRequest = .mlx(
                    manager: manager,
                    messages: messages,
                    enableThinking: resetState.target.isReasoningMode,
                    tools: mlxTools,
                    toolDispatch: mlxToolDispatch
                )
            } catch {
                failStreamPreparation(
                    target: resetState.target,
                    message: "Failed to prepare image for native inference: \(error.localizedDescription)"
                )
                return nil
            }
        } else {
            var prompt = buildPrompt(
                upToOrderExclusive: order,
                currentReasoningActive: resetState.target.isReasoningMode,
                webSearchAvailable: webSearchAvailable,
                forceWebSearchRequired: resetState.forceSearchRequired
            )
            if let instruction = additionalUserInstruction, !instruction.isEmpty {
                prompt += "\n\nUser: \(instruction)"
            }
            if resetState.target.isReasoningMode && !prompt.contains("<thinking>") {
                prompt += "\n\nAssistant: <thinking>"
            }
            backendRequest = .foundation(prompt: prompt, tools: foundationModelTools)
        }

        conversation.lastUpdated = Date()
        immediateSave()

        return StreamPreparation(
            targetID: resetState.targetID,
            forceSearchRequired: resetState.forceSearchRequired,
            previousText: resetState.previousText,
            previousReasoning: resetState.previousReasoning,
            previousFinal: resetState.previousFinal,
            questionLogPreview: resetState.questionLogPreview,
            webSearchBridge: webSearchBridge,
            backendRequest: backendRequest
        )
    }

    private func failStreamPreparation(
        target: Message,
        message: String,
        clearVisibleContent: Bool = false
    ) {
        target.generationError = message
        if clearVisibleContent {
            target.text = ""
            target.finalAnswer = nil
            target.reasoning = nil
        }
        target.markAsComplete()
        conversation.lastUpdated = Date()
        immediateSave()
    }

    private func performStreaming(preparation: StreamPreparation) async -> StreamConsumptionResult {
        guard let target = conversation.messages.first(where: { $0.id == preparation.targetID }) else {
            return StreamConsumptionResult(outcome: .failedBeforeOutput)
        }

        var result = StreamConsumptionResult()
        var lastModelWrite: Date = .distantPast
        var lastRenderedLength = 0
        let activeChunkTimeout = target.isReasoningMode ? reasoningChunkTimeout : chunkTimeout
        let generationStartedAt = Date()
        let backendLabel: String
        let modelName: String?

        switch preparation.backendRequest {
        case .mlx(let manager, _, _, _, _):
            backendLabel = "MLX"
            modelName = manager.currentModel?.displayName
        case .foundation:
            backendLabel = "Apple Intelligence"
            modelName = "Foundation Models"
        }

        target.beginGenerationCapture(
            backend: backendLabel,
            modelName: modelName,
            startedAt: generationStartedAt
        )

        @MainActor
        func renderIfThrottled() {
            guard let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) else { return }
            let hasEnoughDelta = lastRenderedLength == 0 || (result.cumulativeText.count - lastRenderedLength) >= 24
            let now = Date()
            guard now.timeIntervalSince(lastModelWrite) >= 0.12 && hasEnoughDelta else { return }
            syncLiveSearchInvocations(into: liveTarget, from: preparation.webSearchBridge)
            updateMessageWithReasoningContent(liveTarget, fullText: result.cumulativeText, finalize: false)
            lastModelWrite = now
            lastRenderedLength = result.cumulativeText.count
            conversation.lastUpdated = now
            scheduleCoalescedSave()
        }

        do {
            let stream = try await makeGenerationStream(from: preparation)
            var iterator = stream.makeAsyncIterator()

            while !Task.isCancelled {
                let nextEvent: StreamEvent?
                do {
                    nextEvent = try await withTimeout(result.wroteAny ? activeChunkTimeout : firstTokenTimeout) {
                        try await iterator.next()
                    }
                } catch let timeout as GenerationTimeoutError {
                    result.outcome = result.wroteAny ? .failedAfterPartialOutput : .failedBeforeOutput
                    if let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                        liveTarget.generationError = timeout.localizedDescription
                        if result.wroteAny {
                            liveTarget.completeGenerationCapture(rawText: result.cumulativeText)
                        }
                        liveTarget.markAsComplete()
                        conversation.lastUpdated = Date()
                        scheduleCoalescedSave()
                    }
                    break
                }

                guard let nextEvent else { break }

                guard conversation.messages.contains(where: { $0.id == preparation.targetID }) else {
                    break
                }

                switch nextEvent {
                case .text(let newText):
                    guard !newText.isEmpty else { continue }
                    if !appendStreamingChunk(newText, into: &result) {
                        continue
                    }
                    renderIfThrottled()
                case .toolCall:
                    if let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }),
                       liveTarget.isReasoningMode {
                        liveTarget.streamingReasoningPhase = .postToolReasoning
                        syncLiveSearchInvocations(into: liveTarget, from: preparation.webSearchBridge)
                    }
                }
            }

            if result.wroteAny,
               let finalTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                syncLiveSearchInvocations(into: finalTarget, from: preparation.webSearchBridge)
                updateMessageWithReasoningContent(finalTarget, fullText: result.cumulativeText, finalize: true)
                finalTarget.completeGenerationCapture(rawText: result.cumulativeText)
                conversation.lastUpdated = Date()
                scheduleCoalescedSave()
            }

            logger.notice(
                "streamAssistant raw output: chars=\(result.cumulativeText.count, privacy: .public) wrote_any=\(result.wroteAny, privacy: .public)"
            )
        } catch is CancellationError {
            print("⚠️ streaming Task: Cancelled")
            result.outcome = .cancelled
        } catch {
            print("❌ streaming Task: Error - \(error)")
            logger.error(
                "streamAssistant generation error: question_preview_chars=\(preparation.questionLogPreview.count, privacy: .public) partial_chars=\(result.cumulativeText.count, privacy: .public) error=\((error as NSError).localizedDescription, privacy: .public)"
            )
            if let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                let message = isContextWindowError(error)
                    ? ContextWindowError().localizedDescription
                    : (error as NSError).localizedDescription
                liveTarget.generationError = message
                if result.cumulativeText.isEmpty {
                    liveTarget.text = ""
                }
                if !result.cumulativeText.isEmpty {
                    liveTarget.completeGenerationCapture(rawText: result.cumulativeText)
                }
                liveTarget.markAsComplete()
                result.wroteAny = true
                result.outcome = result.cumulativeText.isEmpty ? .failedBeforeOutput : .failedAfterPartialOutput
            }
        }

        return result
    }

    private func makeGenerationStream(from preparation: StreamPreparation) async throws -> AsyncThrowingStream<StreamEvent, Error> {
        switch preparation.backendRequest {
        case .mlx(let manager, let messages, let enableThinking, let tools, let toolDispatch):
            logger.notice("streamAssistant entering MLX generation path")
            while manager.isLoading && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
            }
            guard !Task.isCancelled else { throw CancellationError() }
            guard manager.currentModel != nil else {
                let modelName = manager.currentModel?.name ?? "The model"
                let reason = manager.loadError ?? "The model hasn't finished loading."
                throw NSError(domain: "MLXGeneration", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "\(modelName) isn't ready. \(reason)\n\nTry waiting for it to load, or switch to Apple Intelligence in Settings."
                ])
            }

            return TaskBackedAsyncThrowingStream.make { continuation in
                Task { @MainActor in
                    do {
                        let generationResult = try await manager.generateTextStream(
                            messages: messages,
                            enableThinking: enableThinking,
                            tools: tools,
                            toolDispatch: toolDispatch,
                            onToken: { token in continuation.yield(.text(token)) },
                            onToolCall: { toolCall in continuation.yield(.toolCall(toolCall)) }
                        )
                        self.logger.notice(
                            "streamAssistant MLX generation finished: tool_invocations=\(generationResult.toolInvocationCount, privacy: .public) question_preview_chars=\(preparation.questionLogPreview.count, privacy: .public)"
                        )
                        continuation.finish()
                    } catch is CancellationError {
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        case .foundation(let prompt, let tools):
            logger.notice("streamAssistant entering Foundation Models generation path")
            let textStream = try await withTimeout(firstTokenTimeout) {
                try await self.generator.streamResponse(to: prompt, tools: tools)
            }
            return AsyncThrowingStream { continuation in
                Task {
                    do {
                        for try await chunk in textStream {
                            continuation.yield(.text(chunk))
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    internal static func mergedStreamingChunk(currentText: String, newText: String) -> String? {
        if currentText.contains(newText) && newText.count > 10 {
            return nil
        }
        if !currentText.isEmpty &&
            newText.hasPrefix(currentText) &&
            newText.count > currentText.count {
            return newText
        }
        if !newText.isEmpty && newText.count > 10 && currentText.hasPrefix(newText) {
            return nil
        }

        var overlapLength = 0
        if !currentText.isEmpty {
            let maxCheckLength = min(100, min(currentText.count, newText.count))
            for length in stride(from: maxCheckLength, through: 3, by: -1) {
                if currentText.suffix(length) == newText.prefix(length) {
                    overlapLength = length
                    break
                }
            }
        }

        let delta = String(newText.dropFirst(overlapLength))
        if !delta.isEmpty {
            return currentText + delta
        }
        if overlapLength == 0 && !newText.isEmpty {
            return currentText + newText
        }

        return nil
    }

    private func appendStreamingChunk(_ newText: String, into result: inout StreamConsumptionResult) -> Bool {
        if !result.cumulativeText.isEmpty && newText.count > 50 {
            let combinedTest = result.cumulativeText + newText
            let checkLength = min(newText.count, 200)
            if combinedTest.count >= checkLength * 2 {
                let recentSection = String(combinedTest.suffix(checkLength))
                let earlierSection = String(combinedTest.dropLast(checkLength).suffix(checkLength))
                if recentSection.lowercased() == earlierSection.lowercased() {
                    return false
                }
            }
        }

        if let mergedText = Self.mergedStreamingChunk(
            currentText: result.cumulativeText,
            newText: newText
        ) {
            result.cumulativeText = mergedText
            result.wroteAny = true
            return true
        }

        return false
    }

    private func captureSearchInvocations(
        from webSearchBridge: AppWebSearchToolBridge?,
        targetID: UUID
    ) -> [SearchInvocation] {
        let capturedInvocations = webSearchBridge?.allInvocations ?? []
        logger.notice(
            "streamAssistant search summary: invocations=\(capturedInvocations.count, privacy: .public)"
        )

        guard !capturedInvocations.isEmpty,
              let target = conversation.messages.first(where: { $0.id == targetID }) else {
            return capturedInvocations
        }

        let storedInvocations = Self.mergeSearchInvocations(
            capturedInvocations,
            preservingAnchorsFrom: target.searchInvocations ?? [],
            currentChunkCount: reasoningChunkCount(for: target.reasoning)
        )
        target.searchInvocations = storedInvocations

        let combinedResults = storedInvocations.map { $0.results }.joined(separator: "\n\n")
        let sourcesBlock = "<sources>\n\(combinedResults)\n</sources>\n\n"
        if target.isReasoningMode {
            let existing = target.finalAnswer ?? ""
            if !existing.contains("<sources>") {
                target.finalAnswer = sourcesBlock + existing
            }
        } else if !target.text.contains("<sources>") {
            target.text = sourcesBlock + target.text
        }

        return storedInvocations
    }

    private func finalizeStreaming(
        preparation: StreamPreparation,
        consumption: StreamConsumptionResult,
        capturedInvocations: [SearchInvocation]
    ) -> StreamOutcome {
        var outcome = consumption.outcome

        guard let target = conversation.messages.first(where: { $0.id == preparation.targetID }) else {
            return outcome
        }

        if preparation.forceSearchRequired && capturedInvocations.isEmpty {
            target.text = ""
            target.finalAnswer = nil
            target.reasoning = nil
            target.searchInvocations = nil
            target.generationError = "Search was explicitly required for this response, but the model did not call the webSearch tool."
            outcome = .failedBeforeOutput
        }

        let hasVisibleFinal: Bool = {
            guard let final = target.finalAnswer else { return false }
            let stripped = stripSourcesFromText(final)
            return !stripped.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }()
        let hasVisibleText = !target.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasAnyVisibleContent = hasVisibleText || hasVisibleFinal
        let hasPreviousContent = !preparation.previousText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !(preparation.previousFinal?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

        if outcome == .cancelled && !hasAnyVisibleContent && target.generationError == nil {
            if hasPreviousContent {
                target.text = preparation.previousText
                target.reasoning = preparation.previousReasoning
                target.finalAnswer = preparation.previousFinal
            } else {
                conversation.messages.removeAll { $0.id == target.id }
                context.delete(target)
                conversation.lastUpdated = Date()
                return outcome
            }
        }

        if !hasAnyVisibleContent && target.generationError == nil {
            logger.warning("No visible content after streaming (wroteAny=\(consumption.wroteAny))")
            logger.warning(
                "streamAssistant empty visible output: question_preview_chars=\(preparation.questionLogPreview.count, privacy: .public) raw_chars=\(consumption.cumulativeText.count, privacy: .public) tool_invocations=\(capturedInvocations.count, privacy: .public)"
            )

            if hasPreviousContent {
                target.text = preparation.previousText
                target.reasoning = preparation.previousReasoning
                target.finalAnswer = preparation.previousFinal
            } else {
                target.generationError = "The model returned an empty response. Try regenerating or rephrasing."
                outcome = .failedBeforeOutput
            }
        }

        if !target.text.isEmpty {
            target.text = stripSearchTags(target.text, preserveBoundaries: false)
        }
        if let final = target.finalAnswer, !final.isEmpty {
            target.finalAnswer = stripSearchTags(final, preserveBoundaries: false)
        }

        target.markAsComplete()
        logger.notice(
            "streamAssistant finalized: outcome=\(String(describing: outcome), privacy: .public) error=\(target.generationError ?? "", privacy: .public) final_text_chars=\(target.text.count, privacy: .public)"
        )

        Task { @MainActor in
            let generator = UIImpactFeedbackGenerator(style: .light)
            generator.prepare()
            generator.impactOccurred(intensity: 0.7)
            try? await Task.sleep(for: .milliseconds(80))
            generator.impactOccurred(intensity: 1.0)
        }

        return outcome
    }

    private func finishStreamingSession(with outcome: StreamOutcome) {
        streamingMessageID = nil
        immediateSave()
        currentStreamTask = nil
        lastStreamOutcome = outcome
    }
}
