//
//  ChatViewModel.swift
//  ChatLLM
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

nonisolated private final class TimeoutRace<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var didResume = false
    private var continuation: CheckedContinuation<T, Error>?
    private var operationTask: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?

    func start(
        duration: Duration,
        operation: @escaping @Sendable () async throws -> T,
        continuation: CheckedContinuation<T, Error>
    ) {
        lock.withLock {
            self.continuation = continuation
        }

        let operationTask = Task {
            do {
                let value = try await operation()
                resume(with: .success(value))
            } catch {
                resume(with: .failure(error))
            }
        }

        let timeoutTask = Task {
            do {
                try await Task.sleep(for: duration)
                resume(with: .failure(ChatViewModel.GenerationTimeoutError(timeout: duration)))
            } catch {
                // Cancellation means the operation completed or the caller cancelled.
            }
        }

        let shouldCancelStartedTasks = lock.withLock { () -> Bool in
            guard !didResume else { return true }
            self.operationTask = operationTask
            self.timeoutTask = timeoutTask
            return false
        }
        if shouldCancelStartedTasks {
            operationTask.cancel()
            timeoutTask.cancel()
        }
    }

    func cancel() {
        resume(with: .failure(CancellationError()))
    }

    private func resume(with result: Result<T, Error>) {
        let continuation = lock.withLock { () -> CheckedContinuation<T, Error>? in
            guard !didResume else { return nil }
            didResume = true
            let continuation = self.continuation
            self.continuation = nil
            operationTask?.cancel()
            timeoutTask?.cancel()
            return continuation
        }
        if let continuation {
            continuation.resume(with: result)
        }
    }
}

nonisolated private final class FoundationModelsGateLease: @unchecked Sendable {
    private let gate: FoundationModelsGate
    private let lock = NSLock()
    private var didRelease = false

    init(gate: FoundationModelsGate) {
        self.gate = gate
    }

    func release() async {
        let shouldRelease = lock.withLock {
            guard !didRelease else { return false }
            didRelease = true
            return true
        }
        if shouldRelease {
            await gate.release()
        }
    }
}

@MainActor
final class ChatViewModel: ObservableObject {

    // MARK: - Logging
    let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "ChatViewModel")

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

    internal var nextOrder: Int {
        (conversation.messages.max(by: { $0.order < $1.order })?.order ?? -1) + 1
    }

    private(set) var conversation: Conversation

    // Track the active streaming task so we can cancel it
    internal var currentStreamTask: Task<Void, Never>?
    private var lastStreamOutcome: StreamOutcome = .succeeded
    private var activeGenerationID: UUID?
    private let canPromoteDraft: Bool
    private var didPromoteDraft = false

    // Serializes concurrent Foundation Models sessions via continuation queue.
    private static let fmGate = FoundationModelsGate()

    // MARK: - Timeouts
    // Give the first token longer (model load/warm-up), then enforce a tighter per-chunk timeout.
    private let firstTokenTimeout: Duration = .seconds(30)
    private let chunkTimeout: Duration = .seconds(20)
    private let reasoningChunkTimeout: Duration = .seconds(30)

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
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        await Self.fmGate.acquire()
        let gateLease = FoundationModelsGateLease(gate: Self.fmGate)

        let operationTask = Task<T, Error> {
            do {
                let value = try await operation()
                await gateLease.release()
                return value
            } catch {
                await gateLease.release()
                throw error
            }
        }

        do {
            return try await withTimeout(timeout) {
                try await operationTask.value
            }
        } catch {
            operationTask.cancel()
            await gateLease.release()
            throw error
        }
    }

    // MARK: - Initializer
    init(generator: LLMGenerator, context: ModelContext, conversation: Conversation) {
        self.generator = generator
        self.context = context
        self.conversation = conversation
        self.canPromoteDraft = conversation.modelContext == nil

        loadTavilyAPIKey()

        keyChangeCancellable = NotificationCenter.default.publisher(for: TavilyAPIKeyStore.didChangeNotification)
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

        if let existing = pendingSaveTask, !existing.isCancelled {
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
                self.logger.error("Failed to save context: \((error as NSError).localizedDescription, privacy: .public)")
                // Retry once more immediately on failure
                try? self.context.save()
            }
            self.pendingSaveTask = nil
        }
    }

    /// Inserts the conversation into the model context if it hasn't been persisted yet.
    /// Called before every save so draft conversations are transparently promoted on first message send.
    private func insertIntoContextIfNeeded() {
        guard canPromoteDraft, !didPromoteDraft, conversation.modelContext == nil else { return }
        context.insert(conversation)
        didPromoteDraft = true
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
            logger.error("Failed to immediately save context: \((error as NSError).localizedDescription, privacy: .public)")
        }
    }

    internal static func mergeSearchInvocations(
        _ liveInvocations: [SearchInvocation],
        preservingAnchorsFrom existingInvocations: [SearchInvocation],
        currentChunkCount: Int
    ) -> [SearchInvocation] {
        guard !liveInvocations.isEmpty else { return [] }

        let existingByID = Dictionary(existingInvocations.map { ($0.id, $0) }, uniquingKeysWith: { _, latest in latest })

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
            logger.warning("Error determining reasoning mode\(logContext.isEmpty ? "" : " for \(logContext)"), using conversation fallback: \(error)")
            return conversation.reasoningMode || conversation.smartReasoningMode
        }
    }

    internal func resolvedReasoningMode(for userMessage: Message?, logContext: String) async -> Bool {
        guard let userMsg = userMessage else {
            return conversation.reasoningMode || conversation.smartReasoningMode
        }
        return await resolvedReasoningMode(for: userMsg.text, logContext: logContext)
    }

    internal func appendUserMessage(
        _ text: String,
        forceAutoNaming: Bool = false
    ) -> (message: Message, needsAutoNaming: Bool) {
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
        let needsAutoNaming = forceAutoNaming || (
            conversation.title == String(localized: "New Chat") &&
            !conversation.hasAutoGeneratedTitle &&
            isFirstUserMessage
        )

        return (userMsg, needsAutoNaming)
    }

    internal func prepareUserMessageForSend(
        _ text: String,
        forceAutoNaming: Bool = false,
        replacingMessageID: UUID? = nil
    ) -> (message: Message, needsAutoNaming: Bool)? {
        guard let replacingMessageID else {
            return appendUserMessage(text, forceAutoNaming: forceAutoNaming)
        }

        guard let userMessage = conversation.messages.first(where: {
            $0.id == replacingMessageID && $0.role == .user
        }) else {
            logger.warning("Edited user message disappeared before send")
            return nil
        }

        userMessage.text = text
        userMessage.isFinal = true

        let isFirstUserMessage = !conversation.messages.contains {
            $0.role == .user && $0.order < userMessage.order
        }
        let needsAutoNaming = forceAutoNaming || (
            conversation.title == String(localized: "New Chat") &&
            !conversation.hasAutoGeneratedTitle &&
            isFirstUserMessage
        )
        return (userMessage, needsAutoNaming)
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
        cancelGeneration(at: Date())
    }

    internal func cancelGeneration(at cancelledAt: Date) {
        if let streamingMessageID,
           let streamingMessage = conversation.messages.first(where: { $0.id == streamingMessageID }),
           streamingMessage.reasoningStartedAt != nil,
           streamingMessage.reasoningCompletedAt == nil {
            streamingMessage.completeReasoningCapture(completedAt: cancelledAt)
            conversation.lastUpdated = cancelledAt
            immediateSave()
        }

        activeGenerationID = nil
        isGenerating = false
        streamingMessageID = nil
        currentStreamTask?.cancel()
    }

    @discardableResult
    func cancelGenerationAndWait() async -> Bool {
        cancelGeneration()
        return await waitForStreamToFinish()
    }

    // MARK: - Chat flow

    func send(
        userText: String,
        forceSearch: Bool = false,
        disableToolCalls: Bool = false,
        regenerateTitle: Bool = false,
        editedUserMessageID: UUID? = nil
    ) async {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        guard await waitForStreamToFinish() else { return }

        guard !isGenerating, !Task.isCancelled,
              let generationID = beginGenerationLifecycle() else { return }
        defer { endGenerationLifecycle(generationID) }

        let shouldUseReasoning = await resolvedReasoningMode(for: trimmed, logContext: "")

        guard isGenerationActive(generationID) else { return }

        guard let turn = prepareUserMessageForSend(
            trimmed,
            forceAutoNaming: regenerateTitle,
            replacingMessageID: editedUserMessageID
        ) else { return }
        let assistantMsg = appendAssistantPlaceholder(
            isReasoningMode: shouldUseReasoning,
            searchQuery: forceSearch ? trimmed : nil,
            requiresWebSearch: forceSearch
        )

        guard isGenerationActive(generationID) else { return }

        let searchInstruction = forceSearch
            ? "You must call the webSearch tool before answering. Search for current information about this request first, then answer using the tool results."
            : nil
        await streamAssistant(
            into: assistantMsg,
            basedOnHistoryUpTo: assistantMsg.order,
            additionalUserInstruction: searchInstruction,
            disableWebSearch: disableToolCalls,
            generationID: generationID
        )

        guard isGenerationActive(generationID) else { return }
        await finishAutoNamingIfNeeded(turn.needsAutoNaming, userText: trimmed)
    }

    // MARK: - OCR

    func extractOCR(from image: UIImage) async -> String {
        let logger = self.logger
        return await withTaskCancellationHandler {
            await Task.detached(priority: .userInitiated) { () -> String in
                guard let cg = image.cgImage else {
                    logger.error("Failed to get CGImage from UIImage")
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
                        logger.debug("No text recognition results found")
                        return ""
                    }

                    let texts: [String] = results.compactMap { observation in
                        guard let topCandidate = observation.topCandidates(1).first,
                              topCandidate.confidence > 0.1 else { return nil }
                        return topCandidate.string
                    }

                    let extractedText = texts.joined(separator: "\n")
                    logger.debug("OCR extracted \(texts.count, privacy: .public) text segments")
                    return extractedText
                } catch {
                    logger.error("OCR processing failed: \((error as NSError).localizedDescription, privacy: .public)")
                    return ""
                }
            }.value
        } onCancel: {
            logger.debug("OCR task was cancelled")
        }
    }

    // MARK: - Internals

    internal func waitForStreamToFinish() async -> Bool {
        guard let task = currentStreamTask else { return activeGenerationID == nil }

        do {
            try await withTimeout(.seconds(4)) {
                await task.value
            }
            return true
        } catch is GenerationTimeoutError {
            logger.warning("waitForStreamToFinish timed out; cancelling task")
            task.cancel()
            // Give cancellation a brief window to propagate before we force-reset.
            if (try? await withTimeout(.seconds(1)) {
                await task.value
            }) != nil {
                return true
            }
            return false
        } catch {
            logger.warning("waitForStreamToFinish saw unexpected error: \((error as NSError).localizedDescription, privacy: .public)")
            return false
        }
    }

    @discardableResult
    internal func beginGenerationLifecycle() -> UUID? {
        guard activeGenerationID == nil else { return nil }
        let id = UUID()
        activeGenerationID = id
        isGenerating = true
        return id
    }

    internal func isGenerationActive(_ id: UUID) -> Bool {
        activeGenerationID == id && !Task.isCancelled
    }

    internal func endGenerationLifecycle(_ id: UUID) {
        guard activeGenerationID == id else { return }
        activeGenerationID = nil
        isGenerating = false
        streamingMessageID = nil
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

    fileprivate struct GenerationTimeoutError: LocalizedError {
        var timeout: Duration
        var errorDescription: String? {
            let seconds = Double(timeout.components.seconds) + Double(timeout.components.attoseconds) / 1e18
            return "Generation timed out after \(Int(seconds)) seconds."
        }
    }

    private struct ContextWindowError: LocalizedError {
        let tokenLimit: Int

        var errorDescription: String? {
            return """
            Context window limit reached. This device is currently capped at about \(tokenLimit) tokens for on-device MLX chats.

            Try one of these solutions:
            • Start a new chat or trim earlier messages
            • Lower the Context Window setting
            • Shorten your system prompt or the latest message
            """
        }
    }

    enum StreamOutcome {
        case succeeded
        case failedBeforeOutput
        case failedAfterPartialOutput
        case cancelled
    }

    func withTimeout<T>(_ duration: Duration, operation: @escaping @Sendable () async throws -> T) async throws -> T {
        let race = TimeoutRace<T>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                race.start(duration: duration, operation: operation, continuation: continuation)
            }
        } onCancel: {
            race.cancel()
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
            conversationID: UUID,
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

    private enum StreamEvent: @unchecked Sendable {
        case text(String)
        case toolCall(MLXToolCall)
        case searchActivityChanged
    }

    nonisolated private final class StreamEventIterator: @unchecked Sendable {
        private var iterator: AsyncThrowingStream<StreamEvent, Error>.Iterator

        init(stream: AsyncThrowingStream<StreamEvent, Error>) {
            self.iterator = stream.makeAsyncIterator()
        }

        func next() async throws -> StreamEvent? {
            try await iterator.next()
        }
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

    private func currentContextWindowLimit() -> Int {
        let bridge = ModelBackendBridge.shared
        let model = bridge.selectedModelID.flatMap { bridge.modelManager?.model(withID: $0) } ??
            bridge.modelManager?.currentModel
        let deviceMaximum = MLXDeviceSupportProfile.current.maxContextWindowTokens(for: model)
        return UserDefaults.standard.mlxContextWindowTokens(deviceMaximum: deviceMaximum)
    }

    // Add optional transient instruction that is appended to the prompt for this run only.
    @discardableResult
    internal func streamAssistant(
        into assistantMessage: Message,
        basedOnHistoryUpTo order: Int,
        additionalUserInstruction: String? = nil,
        disableWebSearch: Bool = false,
        allowNativeImages: Bool = true,
        generationID: UUID? = nil
    ) async -> StreamOutcome {
        logger.debug("streamAssistant starting: order=\(order, privacy: .public) message_id=\(assistantMessage.id.uuidString, privacy: .public)")

        let lifecycleID: UUID
        let ownsLifecycle: Bool
        if let generationID {
            guard isGenerationActive(generationID) else { return .cancelled }
            lifecycleID = generationID
            ownsLifecycle = false
        } else {
            guard let newID = beginGenerationLifecycle() else { return .cancelled }
            lifecycleID = newID
            ownsLifecycle = true
        }
        defer {
            if ownsLifecycle {
                endGenerationLifecycle(lifecycleID)
            }
        }

        guard isGenerationActive(lifecycleID) else {
            logger.debug("streamAssistant cancelled before starting")
            return .cancelled
        }

        streamingMessageID = assistantMessage.id

        guard let resetState = prepareStreamResetState(
            from: assistantMessage,
            basedOnHistoryUpTo: order
        ) else {
            streamingMessageID = nil
            return .failedBeforeOutput
        }

        guard isGenerationActive(lifecycleID), let preparation = await prepareStream(
            from: resetState,
            basedOnHistoryUpTo: order,
            additionalUserInstruction: additionalUserInstruction,
            disableWebSearch: disableWebSearch,
            allowNativeImages: allowNativeImages,
            generationID: lifecycleID
        ) else {
            streamingMessageID = nil
            // `currentStreamTask` is not assigned until after preparation, so a
            // cancel landing in this window never reaches `finalizeStreaming`.
            // The placeholder has already been reset and persisted by
            // `prepareStreamResetState`, so tidy it up here instead.
            if !isGenerationActive(lifecycleID) {
                discardOrRestoreResetPlaceholder(resetState)
                return .cancelled
            }
            return .failedBeforeOutput
        }

        let task = Task { @MainActor [weak self] in
            guard let self = self else { return }
            guard self.isGenerationActive(lifecycleID) else { return }
            let consumption = await self.performStreaming(preparation: preparation)
            guard self.isGenerationActive(lifecycleID) || consumption.outcome == .cancelled else { return }
            let capturedInvocations = self.captureSearchInvocations(
                from: preparation.webSearchBridge,
                targetID: preparation.targetID
            )
            let outcome = self.finalizeStreaming(
                preparation: preparation,
                consumption: consumption,
                capturedInvocations: capturedInvocations
            )
            self.finishStreamingSession(with: outcome, generationID: lifecycleID)
        }

        lastStreamOutcome = .succeeded
        currentStreamTask = task
        await task.value
        if activeGenerationID == lifecycleID {
            currentStreamTask = nil
        }
        return lastStreamOutcome
    }

    /// Undoes `prepareStreamResetState` for a turn cancelled before streaming
    /// began. Mirrors the cancellation branch of `finalizeStreaming`: restore
    /// prior content when regenerating, otherwise drop the empty placeholder.
    private func discardOrRestoreResetPlaceholder(_ resetState: StreamResetState) {
        guard let target = conversation.messages.first(where: { $0.id == resetState.targetID }) else {
            return
        }
        guard target.generationError == nil else { return }

        let hasPreviousContent = !resetState.previousText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !(resetState.previousFinal?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

        if hasPreviousContent {
            target.text = resetState.previousText
            target.reasoning = resetState.previousReasoning
            target.finalAnswer = resetState.previousFinal
            target.markAsComplete()
        } else {
            conversation.messages.removeAll { $0.id == target.id }
            context.delete(target)
        }
        conversation.lastUpdated = Date()
        immediateSave()
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
        let forcedSearchQuery = assistantMessage.requiresWebSearch == true ? assistantMessage.searchQuery : nil
        let forcedSearchRequirement = assistantMessage.requiresWebSearch
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

        target.resetForRegeneration()
        target.requiresWebSearch = forcedSearchRequirement
        target.searchQuery = forcedSearchQuery

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
        allowNativeImages: Bool,
        generationID: UUID
    ) async -> StreamPreparation? {
        let promptBridge = ModelBackendBridge.shared
        let selectedBackend = promptBridge.selectedBackend
        let selectedModelID = promptBridge.selectedModelID
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
            "streamAssistant tool setup: backend=\(selectedBackend.rawValue, privacy: .public) force_search=\(resetState.forceSearchRequired, privacy: .public) tools_disabled=\(toolCallsDisabled, privacy: .public) backend_tool_support=\(backendSupportsWebSearchTools, privacy: .public) autonomous_search=\(autonomousWebSearchAvailable, privacy: .public) search_service=\(self.searchService != nil, privacy: .public) web_tool_available=\(webSearchAvailable, privacy: .public)"
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
        guard isGenerationActive(generationID) else { return nil }

        if selectedBackend == .mlx, let manager = promptBridge.modelManager {
            if let selectedModelID, manager.currentModel?.id != selectedModelID && !manager.isLoading {
                failStreamPreparation(
                    target: resetState.target,
                    message: "Selected MLX model is not loaded. Wait for it to load or choose another backend."
                )
                return nil
            }
            do {
                let messages = try await buildQwenMessages(
                    upToOrderExclusive: order,
                    modelIdentity: manager.currentModel?.promptIdentity ?? "MLX model",
                    additionalInstruction: additionalUserInstruction,
                    includeLatestUserImages: allowNativeImages,
                    maxMessages: webSearchAvailable ? 10 : nil,
                    toolsAvailable: webSearchAvailable,
                    forceWebSearch: resetState.forceSearchRequired
                )
                guard isGenerationActive(generationID) else { return nil }
                backendRequest = .mlx(
                    manager: manager,
                    conversationID: conversation.id,
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
                modelIdentity: ModelBackendBridge.Backend.foundationModels.displayName,
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

        guard isGenerationActive(generationID) else { return nil }

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
            target.text = "Generation failed: \(message)"
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
        case .mlx(let manager, _, _, _, _, _):
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
            // Cheapest check first. `String.count` walks the whole string, so measuring
            // the accumulated response on every token was quadratic in output length;
            // `utf8.count` is O(1) and only needed once the time gate has passed.
            let now = Date()
            guard now.timeIntervalSince(lastModelWrite) >= 0.12 else { return }
            let cumulativeLength = result.cumulativeText.utf8.count
            guard lastRenderedLength == 0 || (cumulativeLength - lastRenderedLength) >= 24 else { return }
            guard let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) else { return }
            syncLiveSearchInvocations(into: liveTarget, from: preparation.webSearchBridge)
            updateMessageWithReasoningContent(liveTarget, fullText: result.cumulativeText, finalize: false)
            lastModelWrite = now
            lastRenderedLength = cumulativeLength
            conversation.lastUpdated = now
            scheduleCoalescedSave()
        }

        do {
            let stream = try await makeGenerationStream(from: preparation)
            let iterator = StreamEventIterator(stream: stream)

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
                            liveTarget.completeGenerationCapture(
                                rawText: result.cumulativeText,
                                captureRawText: UserDefaults.standard.bool(
                                    forKey: AppSettingsKeys.developerModeEnabled
                                )
                            )
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
                        liveTarget.postToolReasoningStartCloseTagCount = Self.reasoningCloseTagCount(
                            in: result.cumulativeText
                        )
                        syncLiveSearchInvocations(into: liveTarget, from: preparation.webSearchBridge)
                    }
                case .searchActivityChanged:
                    if let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                        syncLiveSearchInvocations(into: liveTarget, from: preparation.webSearchBridge)
                        conversation.lastUpdated = Date()
                        scheduleCoalescedSave()
                    }
                }
            }

            if Task.isCancelled {
                result.outcome = .cancelled
            }

            if result.wroteAny,
               let finalTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                syncLiveSearchInvocations(into: finalTarget, from: preparation.webSearchBridge)
                updateMessageWithReasoningContent(finalTarget, fullText: result.cumulativeText, finalize: true)
                finalTarget.completeGenerationCapture(
                    rawText: result.cumulativeText,
                    captureRawText: UserDefaults.standard.bool(
                        forKey: AppSettingsKeys.developerModeEnabled
                    )
                )
                conversation.lastUpdated = Date()
                scheduleCoalescedSave()
            }

            logger.notice(
                "streamAssistant raw output: chars=\(result.cumulativeText.count, privacy: .public) wrote_any=\(result.wroteAny, privacy: .public)"
            )
        } catch is CancellationError {
            logger.debug("streamAssistant generation cancelled")
            result.outcome = .cancelled
        } catch {
            logger.error(
                "streamAssistant generation error: question_preview_chars=\(preparation.questionLogPreview.count, privacy: .public) partial_chars=\(result.cumulativeText.count, privacy: .public) error=\((error as NSError).localizedDescription, privacy: .public)"
            )
            if let liveTarget = conversation.messages.first(where: { $0.id == preparation.targetID }) {
                let message = isContextWindowError(error)
                    ? ContextWindowError(tokenLimit: currentContextWindowLimit()).localizedDescription
                    : (error as NSError).localizedDescription
                liveTarget.generationError = message
                if result.cumulativeText.isEmpty {
                    liveTarget.text = ""
                }
                if !result.cumulativeText.isEmpty {
                    liveTarget.completeGenerationCapture(
                        rawText: result.cumulativeText,
                        captureRawText: UserDefaults.standard.bool(
                            forKey: AppSettingsKeys.developerModeEnabled
                        )
                    )
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
        case .mlx(let manager, let conversationID, let messages, let enableThinking, let tools, let toolDispatch):
            logger.notice("streamAssistant entering MLX generation path")
            while manager.isLoading && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
            }
            guard !Task.isCancelled else { throw CancellationError() }
            guard manager.currentModel != nil else {
                let reason = manager.loadError ?? "The model hasn't finished loading."
                throw NSError(domain: "MLXGeneration", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "The selected MLX model isn't ready. \(reason)\n\nTry waiting for it to load, or switch to Apple Intelligence in Settings."
                ])
            }

            return TaskBackedAsyncThrowingStream.make { continuation in
                let logger = self.logger
                let questionLogPreview = preparation.questionLogPreview
                preparation.webSearchBridge?.setInvocationObserver {
                    continuation.yield(.searchActivityChanged)
                }
                return Task.detached(priority: .userInitiated) {
                    defer { preparation.webSearchBridge?.setInvocationObserver(nil) }
                    do {
                        let generationResult = try await manager.generateTextStream(
                            conversationID: conversationID,
                            messages: messages,
                            enableThinking: enableThinking,
                            tools: tools,
                            toolDispatch: toolDispatch,
                            onToken: { token in continuation.yield(.text(token)) },
                            onToolCall: { toolCall in continuation.yield(.toolCall(toolCall)) }
                        )
                        logger.notice(
                            "streamAssistant MLX generation finished: tool_invocations=\(generationResult.toolInvocationCount, privacy: .public) question_preview_chars=\(questionLogPreview.count, privacy: .public)"
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
            return TaskBackedAsyncThrowingStream.make { continuation in
                preparation.webSearchBridge?.setInvocationObserver {
                    continuation.yield(.searchActivityChanged)
                }
                return Task { @MainActor in
                    defer { preparation.webSearchBridge?.setInvocationObserver(nil) }
                    await Self.fmGate.acquire()
                    let gateLease = FoundationModelsGateLease(gate: Self.fmGate)
                    await withTaskCancellationHandler {
                        do {
                            let textStream = try await self.generator.streamResponse(to: prompt, tools: tools)
                            for try await chunk in textStream {
                                continuation.yield(.text(chunk))
                            }
                            await gateLease.release()
                            continuation.finish()
                        } catch is CancellationError {
                            await gateLease.release()
                            continuation.finish()
                        } catch {
                            await gateLease.release()
                            continuation.finish(throwing: error)
                        }
                    } onCancel: {
                        Task {
                            await gateLease.release()
                        }
                    }
                }
            }
        }
    }

    internal static func mergedStreamingChunk(currentText: String, newText: String) -> String? {
        guard !newText.isEmpty else { return nil }

        // Both backend adapters expose deltas: Foundation Models converts its
        // cumulative partials in OnDeviceLLMGenerator, while MLX emits chunks.
        // Content-based de-duplication here cannot distinguish a provider retry
        // from intentional repetition in code, tables, lists, or prose.
        return currentText + newText
    }

    private func appendStreamingChunk(_ newText: String, into result: inout StreamConsumptionResult) -> Bool {
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

        let legacyResults = storedInvocations
            .filter { $0.response == nil && $0.succeeded }
            .map { Self.escapeSourcesPayload($0.userVisibleResults) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        guard !legacyResults.isEmpty else { return storedInvocations }

        let sourcesBlock = "<sources>\n\(legacyResults)\n</sources>\n\n"
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

    private static func escapeSourcesPayload(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<sources>", with: "&lt;sources&gt;", options: [.caseInsensitive])
            .replacingOccurrences(of: "</sources>", with: "&lt;/sources&gt;", options: [.caseInsensitive])
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

        // A failed search is still recorded as an invocation, so presence alone
        // does not prove the answer is grounded -- require one that succeeded.
        if preparation.forceSearchRequired && !capturedInvocations.contains(where: { $0.succeeded }) {
            target.text = ""
            target.finalAnswer = nil
            target.reasoning = nil
            target.searchInvocations = nil
            target.generationError = capturedInvocations.isEmpty
                ? "Search was explicitly required for this response, but the model did not call the webSearch tool."
                : "Search was explicitly required for this response, but every webSearch attempt failed."
            outcome = .failedBeforeOutput
        }

        let hasVisibleFinal: Bool = {
            guard let final = target.finalAnswer else { return false }
            let stripped = stripSourcesFromText(final)
            return !stripped.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }()
        let hasVisibleText = !target.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasVisibleReasoning = !(target.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        let hasAnyVisibleContent = hasVisibleText || hasVisibleFinal || hasVisibleReasoning
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

        if let generationError = target.generationError,
           !hasAnyVisibleContent {
            target.text = "Generation failed: \(generationError)"
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
            await AppHaptics.generationCompleted()
        }

        return outcome
    }

    private func finishStreamingSession(with outcome: StreamOutcome, generationID: UUID) {
        guard activeGenerationID == generationID || outcome == .cancelled else { return }
        streamingMessageID = nil
        immediateSave()
        currentStreamTask = nil
        lastStreamOutcome = outcome
    }
}
