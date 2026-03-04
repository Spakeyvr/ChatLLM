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
import MLXLMCommon

// Resolve ambiguities between MLXLMCommon and SwiftData/FoundationModels
typealias ModelContext = SwiftData.ModelContext
typealias Tool = FoundationModels.Tool

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

    // Thread-safe order calculation to prevent race conditions
    internal var nextOrder: Int {
        let orders = conversation.messages.map { $0.order }
        return (orders.max() ?? -1) + 1
    }

    private(set) var conversation: Conversation

    // Track the active streaming task so we can cancel it
    internal var currentStreamTask: Task<Void, Never>?

    // Serializes concurrent Foundation Models sessions via continuation queue.
    private let fmGate = FoundationModelsGate()

    // MARK: - Timeouts
    // Give the first token longer (model load/warm-up), then enforce a tighter per-chunk timeout.
    private let firstTokenTimeout: Duration = .seconds(30)
    private let chunkTimeout: Duration = .seconds(10)

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
    }

    deinit {
        pendingSaveTask?.cancel()
        currentStreamTask?.cancel()
        keyChangeCancellable?.cancel()
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

        guard pendingSaveTask == nil || pendingSaveTask!.isCancelled else {
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

    internal func immediateSave() {
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

    // MARK: - Public controls

    func cancelGeneration() {
        // Only cancel the task; let the streaming task perform the unified cleanup.
        currentStreamTask?.cancel()
    }

    // MARK: - Chat flow

    func send(userText: String, forceSearch: Bool = false) async {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        await waitForStreamToFinish()

        guard !isGenerating, !Task.isCancelled else { return }

        let shouldUseReasoning: Bool
        do {
            shouldUseReasoning = try await shouldUseReasoningForPrompt(trimmed)
        } catch let reasoningError as ReasoningEvaluationError {
            print("Error determining reasoning mode, using fallback: \(reasoningError.localizedDescription)")
            shouldUseReasoning = reasoningError.fallbackResult
        } catch {
            print("Unexpected error determining reasoning mode, using conversation fallback: \(error)")
            shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
        }

        guard !Task.isCancelled else { return }

        let baseOrder = nextOrder

        let userMsg = Message(role: .user, text: trimmed, order: baseOrder, conversation: conversation, isFinal: true)
        conversation.messages.append(userMsg)

        // Eagerly resolve properties to avoid SwiftData fault errors before async work
        let userMessagesCount = conversation.messages.lazy.filter { $0.role == .user }.count
        let isFirstUserMessage = userMessagesCount == 1
        let needsAutoNaming = conversation.title == String(localized: "New Chat") && !conversation.hasAutoGeneratedTitle && isFirstUserMessage

        // Assistant message starts non-final; becomes final when streaming completes.
        let assistantMsg = Message(
            role: .assistant,
            text: "",
            order: baseOrder + 1,
            conversation: conversation,
            isFinal: false,
            isReasoningMode: shouldUseReasoning,
            searchQuery: forceSearch ? trimmed : nil  // Store the search query if search was used
        )
        conversation.messages.append(assistantMsg)
        conversation.lastUpdated = Date()
        immediateSave()

        guard !Task.isCancelled else { return }

        let searchInstruction = forceSearch ? "Please search for current information about this topic and then answer." : nil
        await streamAssistant(into: assistantMsg, basedOnHistoryUpTo: assistantMsg.order, additionalUserInstruction: searchInstruction)

        if needsAutoNaming && !Task.isCancelled {
            await generateChatTitle(fromUserMessage: trimmed)
        }
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
            await task.value
        }
        // Also ensure flags are down (in case a fast-path set them slightly later)
        var attempts = 0
        while (isGenerating || currentStreamTask != nil) && attempts < 100 {
            try? await Task.sleep(for: .milliseconds(40))
            attempts += 1
        }

        // CRITICAL FIX: If we hit the timeout, properly cancel the task first
        if attempts >= 100 {
            print("⚠️ waitForStreamToFinish timed out after 4 seconds")

            if let task = currentStreamTask {
                task.cancel()
                print("🛑 Cancelled stale streaming task")

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
    internal func streamAssistant(into assistantMessage: Message, basedOnHistoryUpTo order: Int, additionalUserInstruction: String? = nil, disableWebSearch: Bool = false) async {
        print("🔄 streamAssistant: Starting (order: \(order), messageID: \(assistantMessage.id))")

        guard !Task.isCancelled else {
            print("⚠️ streamAssistant: Task cancelled before starting")
            return
        }

        // isGenerating is managed exclusively here to avoid race conditions.
        isGenerating = true
        defer {
            isGenerating = false
        }

        // Eagerly resolve properties to avoid SwiftData fault errors
        let targetID = assistantMessage.id
        let targetRole = assistantMessage.role
        streamingMessageID = targetID

        let previousText = assistantMessage.text
        let previousReasoning = assistantMessage.reasoning
        let previousFinal = assistantMessage.finalAnswer

        guard let targetToReset = conversation.messages.first(where: { $0.id == targetID }),
              targetRole == .assistant else {
            streamingMessageID = nil
            return
        }

        // Preserve sources from previous web search before resetting
        let preservedSources = targetToReset.resetForRegeneration(preserveSources: true)

        if let sources = preservedSources, !sources.isEmpty {
            let sourcesBlock = "<sources>\n\(sources)\n</sources>\n\n"
            if targetToReset.isReasoningMode {
                targetToReset.finalAnswer = sourcesBlock
            } else {
                targetToReset.text = sourcesBlock
            }
        }

        conversation.lastUpdated = Date()
        immediateSave()

        let webSearchAvailable = !disableWebSearch && searchService != nil
        let promptBridge = ModelBackendBridge.shared
        // MLX path: structured messages passed to Qwen3VLProcessor which applies the Jinja template.
        // enable_thinking is controlled via additionalContext, not baked into the message text.
        var mlxMessages: [Chat.Message]? = nil
        var builtPrompt: String  // FM path prompt (also used for debug snapshot on MLX path)
        if promptBridge.selectedBackend == .mlx && promptBridge.modelManager != nil {
            mlxMessages = buildQwenMessages(
                upToOrderExclusive: order,
                additionalInstruction: additionalUserInstruction
            )
            builtPrompt = "MLX:\(mlxMessages!.count) msgs"
        } else {
            // Foundation Models: keep existing plain-text prompt format unchanged.
            builtPrompt = buildPrompt(upToOrderExclusive: order, currentReasoningActive: targetToReset.isReasoningMode, webSearchAvailable: webSearchAvailable)
            if let instruction = additionalUserInstruction, !instruction.isEmpty {
                builtPrompt += "\n\nUser: \(instruction)"
            }
            if targetToReset.isReasoningMode && !builtPrompt.contains("<thinking>") {
                builtPrompt += "\n\nAssistant: <thinking>"
            }
        }
        targetToReset.promptSnapshot = builtPrompt
        conversation.lastUpdated = Date()
        immediateSave()

        // Create native WebSearchTool if Tavily is available and web search is not disabled.
        let webSearchTool: WebSearchTool?
        if !disableWebSearch, let service = searchService {
            webSearchTool = WebSearchTool(searchService: service)
        } else {
            webSearchTool = nil
        }
        let tools: [any Tool] = webSearchTool.map { [$0] } ?? []

        let task = Task { @MainActor [weak self] in
            guard let self = self else { return }
            var cumulativeSoFar = ""
            var wroteAny = false

            do {
                let stream: AsyncThrowingStream<String, Error>
                let bridge = ModelBackendBridge.shared
                if bridge.selectedBackend == .mlx, let manager = bridge.modelManager {
                    // Wait for any in-progress model load to finish before generating
                    while manager.isLoading && !Task.isCancelled {
                        try? await Task.sleep(for: .milliseconds(200))
                    }
                    guard !Task.isCancelled else { throw CancellationError() }
                    if manager.currentModel != nil {
                        let isReasoning = targetToReset.isReasoningMode
                        let msgs = mlxMessages ?? []
                        stream = AsyncThrowingStream { continuation in
                            Task { @MainActor in
                                do {
                                    try await manager.generateTextStream(
                                        messages: msgs,
                                        enableThinking: isReasoning,
                                        onToken: { token in continuation.yield(token) }
                                    )
                                    continuation.finish()
                                } catch {
                                    continuation.finish(throwing: error)
                                }
                            }
                        }
                    } else {
                        // Model failed to load — surface a clear error instead of falling back silently
                        let modelName = manager.currentModel?.name ?? "The model"
                        let reason = manager.loadError ?? "The model hasn't finished loading."
                        throw NSError(domain: "MLXGeneration", code: 1, userInfo: [
                            NSLocalizedDescriptionKey: "\(modelName) isn't ready. \(reason)\n\nTry waiting for it to load, or switch to Apple Intelligence in Settings."
                        ])
                    }
                } else {
                    stream = try await self.withTimeout(self.firstTokenTimeout) {
                        try await self.generator.streamResponse(to: builtPrompt, tools: tools)
                    }
                }
                var iterator = stream.makeAsyncIterator()

                while !Task.isCancelled {
                    let nextChunk: String?
                    do {
                        nextChunk = try await self.withTimeout(wroteAny ? self.chunkTimeout : self.firstTokenTimeout) {
                            try await iterator.next()
                        }
                    } catch let timeout as GenerationTimeoutError {
                        if let target = self.conversation.messages.first(where: { $0.id == targetID }) {
                            target.generationError = timeout.localizedDescription
                            target.markAsComplete()
                            self.conversation.lastUpdated = Date()
                            self.scheduleCoalescedSave()
                        }
                        break
                    }

                    guard let newText = nextChunk, !newText.isEmpty else {
                        if nextChunk == nil { break }
                        continue
                    }

                    guard let target = self.conversation.messages.first(where: { $0.id == targetID }) else { break }

                    // Detect large-scale repetition before appending
                    if !cumulativeSoFar.isEmpty && newText.count > 50 {
                        let combinedTest = cumulativeSoFar + newText
                        let checkLength = min(newText.count, 200)
                        if combinedTest.count >= checkLength * 2 {
                            let recentSection = String(combinedTest.suffix(checkLength))
                            let earlierSection = String(combinedTest.dropLast(checkLength).suffix(checkLength))
                            if recentSection.lowercased() == earlierSection.lowercased() {
                                continue
                            }
                        }
                    }

                    if cumulativeSoFar.contains(newText) && newText.count > 10 {
                        continue
                    }
                    if !cumulativeSoFar.isEmpty && newText.hasPrefix(cumulativeSoFar) && newText.count > cumulativeSoFar.count {
                        cumulativeSoFar = newText
                        self.updateMessageWithReasoningContent(target, fullText: cumulativeSoFar)
                        wroteAny = true
                        self.conversation.lastUpdated = Date()
                        self.scheduleCoalescedSave()
                        continue
                    }
                    if !newText.isEmpty && newText.count > 10 && cumulativeSoFar.hasPrefix(newText) {
                        continue
                    }

                    var overlapLength = 0
                    if !cumulativeSoFar.isEmpty {
                        let maxCheckLength = min(100, min(cumulativeSoFar.count, newText.count))
                        for length in stride(from: maxCheckLength, through: 1, by: -1) {
                            if cumulativeSoFar.suffix(length) == newText.prefix(length) {
                                overlapLength = length
                                break
                            }
                        }
                    }
                    let delta = String(newText.dropFirst(overlapLength))

                    if !delta.isEmpty {
                        cumulativeSoFar += delta
                        self.updateMessageWithReasoningContent(target, fullText: cumulativeSoFar)
                        wroteAny = true
                    } else if overlapLength == 0 && !newText.isEmpty {
                        cumulativeSoFar += newText
                        self.updateMessageWithReasoningContent(target, fullText: cumulativeSoFar)
                        wroteAny = true
                    }
                    self.conversation.lastUpdated = Date()
                    self.scheduleCoalescedSave()
                }
            } catch is CancellationError {
                print("⚠️ streaming Task: Cancelled")
            } catch {
                print("❌ streaming Task: Error - \(error)")
                if let target = self.conversation.messages.first(where: { $0.id == targetID }) {
                    let msg = self.isContextWindowError(error)
                        ? ContextWindowError().localizedDescription
                        : (error as NSError).localizedDescription
                    target.generationError = msg
                    if cumulativeSoFar.isEmpty { target.text = "" }
                    target.markAsComplete()
                    wroteAny = true
                }
            }

            // Inject search sources from native tool (if the model called it)
            if let invocations = webSearchTool?.allInvocations, !invocations.isEmpty {
                if let target = self.conversation.messages.first(where: { $0.id == targetID }) {
                    // Store all invocations on the message for per-search UI
                    target.searchInvocations = invocations

                    // Build combined sources block for backward-compat text extraction
                    let combinedResults = invocations.map { $0.results }.joined(separator: "\n\n")
                    let sourcesBlock = "<sources>\n\(combinedResults)\n</sources>\n\n"
                    if target.isReasoningMode {
                        let existing = target.finalAnswer ?? ""
                        if !existing.contains("<sources>") {
                            target.finalAnswer = sourcesBlock + existing
                        }
                    } else {
                        if !target.text.contains("<sources>") {
                            target.text = sourcesBlock + target.text
                        }
                    }
                }
            }

            // Finalization
            if let target = self.conversation.messages.first(where: { $0.id == targetID }) {
                let hasVisibleFinal: Bool = {
                    guard let final = target.finalAnswer else { return false }
                    let stripped = self.stripSourcesFromText(final)
                    return !stripped.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                }()
                let hasVisibleText = !target.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                let hasAnyVisibleContent = hasVisibleText || hasVisibleFinal

                if !hasAnyVisibleContent && target.generationError == nil {
                    logger.warning("No visible content after streaming (wroteAny=\(wroteAny))")

                    let hasPreviousContent = !previousText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                                            !(previousFinal?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

                    if hasPreviousContent {
                        // Restore previous content — no error needed
                        target.text = previousFinal ?? previousText
                        target.reasoning = previousReasoning
                        target.finalAnswer = previousFinal
                    } else {
                        target.generationError = "The model returned an empty response. Try regenerating or rephrasing."
                    }
                }

                // Strip any residual search tags from stored messages (backward compat)
                if !target.text.isEmpty {
                    target.text = stripSearchTags(target.text, preserveBoundaries: false)
                }
                if let final = target.finalAnswer, !final.isEmpty {
                    target.finalAnswer = stripSearchTags(final, preserveBoundaries: false)
                }

                target.markAsComplete()

                Task { @MainActor in
                    let generator = UIImpactFeedbackGenerator(style: .light)
                    generator.prepare()
                    generator.impactOccurred(intensity: 0.7)
                    try? await Task.sleep(for: .milliseconds(80))
                    generator.impactOccurred(intensity: 1.0)
                }
            }

            self.streamingMessageID = nil
            self.immediateSave()
        }

        currentStreamTask = task
        await task.value
        currentStreamTask = nil
    }
}
