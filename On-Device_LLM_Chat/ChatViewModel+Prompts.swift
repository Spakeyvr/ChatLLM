//
//  ChatViewModel+Prompts.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import MLXLMCommon
import os.log

// swiftlint:disable:next force_try
private let _nativeImageVisionFallbackRegex = try! NSRegularExpression(
    pattern: #"--- Image Analysis Data ---.*?--- End Image Analysis ---\s*User's question:\s*"#,
    options: [.dotMatchesLineSeparators, .caseInsensitive])

extension ChatViewModel {

    private static let imageAnalysisMarker = "--- Image Analysis Data ---"
    private static let mlxTimeSensitiveKeywords = [
        "today", "yesterday", "tomorrow", "this week", "this month", "this year", "this quarter",
        "latest", "newest", "most recent", "currently", "current", "now", "breaking", "live",
        "just released", "as of"
    ]
    private static let mlxExplicitSearchIntentKeywords = [
        "please search", "search for", "look up"
    ]

    internal static func currentDateTimeContext(
        referenceDate: Date = Date(),
        timeZone: TimeZone = .current
    ) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "EEEE, MMMM d, yyyy 'at' HH:mm"
        let timeZoneLabel: String = if timeZone.secondsFromGMT(for: referenceDate) == 0 {
            "UTC"
        } else {
            timeZone.abbreviation(for: referenceDate) ?? timeZone.identifier
        }
        let formatted = formatter.string(from: referenceDate) + " " + timeZoneLabel
        return """
        Current date and time: \(formatted)
        - Treat this timestamp as the current moment for time-sensitive requests and web searches.
        - When searching for recent information, prefer queries that use this current year or exact date when helpful.
        """
    }

    private struct AttachmentSnapshot {
        let type: AttachmentType
        let actualFileURL: URL
    }

    private struct MessageSnapshot {
        let order: Int
        let role: MessageRole
        let text: String
        let isReasoningMode: Bool
        let reasoning: String?
        let finalAnswer: String?
        let isFinal: Bool
        let attachments: [AttachmentSnapshot]
    }

    private func messageSnapshots(upToOrderExclusive maxOrderExclusive: Int) -> [MessageSnapshot] {
        conversation.messages
            .filter { $0.order < maxOrderExclusive && $0.order >= 0 }
            .sortedByOrder
            .map { message in
                let attachmentSnapshots = message.attachments.map { attachment in
                    AttachmentSnapshot(type: attachment.type, actualFileURL: attachment.actualFileURL)
                }
                return MessageSnapshot(
                    order: message.order,
                    role: message.role,
                    text: message.text,
                    isReasoningMode: message.isReasoningMode,
                    reasoning: message.reasoning,
                    finalAnswer: message.finalAnswer,
                    isFinal: message.isFinal,
                    attachments: attachmentSnapshots
                )
            }
    }

    func setReasoningMode(_ enabled: Bool) {
        let previousValue = conversation.reasoningMode
        conversation.reasoningMode = enabled
        if enabled {
            conversation.smartReasoningMode = false // Disable smart mode when manual mode is enabled
        }
        conversation.lastUpdated = Date()
        immediateSave()
        if previousValue != enabled {
            invalidateMLXConversationSession(reason: "reasoning_mode_changed")
        }
    }

    func setSmartReasoningMode(_ enabled: Bool) {
        let previousValue = conversation.smartReasoningMode
        conversation.smartReasoningMode = enabled
        if enabled {
            conversation.reasoningMode = false // Disable manual mode when smart mode is enabled
        }
        conversation.lastUpdated = Date()
        immediateSave()
        if previousValue != enabled {
            invalidateMLXConversationSession(reason: "smart_reasoning_mode_changed")
        }
    }

    func invalidateMLXConversationSession(reason: String) {
        ModelBackendBridge.shared.modelManager?.invalidateConversationSession(
            conversation.id,
            reason: reason
        )
    }

    // MARK: - Prompt Builder

    // Legacy upper bound on the number of messages (user + assistant combined) in prompt context.
    // The effective cap is further tightened by the device-aware token budget below.
    private static let maxContextMessages = 30
    private static let minimumPromptBudgetTokens = 512
    private static let promptOverheadReserveTokens = 256
    nonisolated private static let promptMessageReserveTokens = 12
    nonisolated private static let minimumPromptMessageCost = 24

    private func snapshotsContainVisionImageAnalysisData(_ snapshots: [MessageSnapshot]) -> Bool {
        snapshots.contains { snapshot in
            snapshot.role == .user && snapshot.text.contains(Self.imageAnalysisMarker)
        }
    }

    private func snapshotsContainNativeImages(_ snapshots: [MessageSnapshot]) -> Bool {
        snapshots.contains { snapshot in
            snapshot.attachments.contains { $0.type == .image }
        }
    }

    private func effectiveContextWindowTokenLimit() -> Int {
        let deviceMaximum = MLXDeviceSupportProfile.current.maxContextWindowTokens
        return UserDefaults.standard.mlxContextWindowTokens(deviceMaximum: deviceMaximum)
    }

    private func effectivePromptBudgetTokenLimit() -> Int {
        let contextLimit = effectiveContextWindowTokenLimit()
        let reservedOutputTokens = min(
            UserDefaults.standard.mlxMaxOutputTokensLimit ?? 1_024,
            max(512, contextLimit / 2)
        )
        return max(
            Self.minimumPromptBudgetTokens,
            contextLimit - reservedOutputTokens - Self.promptOverheadReserveTokens
        )
    }

    private func trimmedSnapshotsForPrompt(
        from snapshots: [MessageSnapshot],
        maxMessages: Int?
    ) -> [MessageSnapshot] {
        var trimmedSnapshots = snapshots
        let limit = min(maxMessages ?? Self.maxContextMessages, Self.maxContextMessages)
        if trimmedSnapshots.count > limit {
            trimmedSnapshots = Array(trimmedSnapshots.suffix(limit))
        }

        let tokenBudget = effectivePromptBudgetTokenLimit()
        var keptSnapshots: [MessageSnapshot] = []
        var usedTokens = 0

        for snapshot in trimmedSnapshots.reversed() {
            guard snapshot.role != .system else { continue }
            if snapshot.role == .assistant && !snapshot.isFinal { continue }
            guard !snapshot.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }

            let estimatedTokens = estimatedPromptTokenCost(for: snapshot)

            if usedTokens + estimatedTokens > tokenBudget && !keptSnapshots.isEmpty {
                break
            }

            keptSnapshots.append(snapshot)
            usedTokens += estimatedTokens

            if usedTokens > tokenBudget {
                break
            }
        }

        return Array(keptSnapshots.reversed())
    }

    private func trimmedSnapshotsForMLXPrompt(
        from snapshots: [MessageSnapshot],
        maxMessages: Int?
    ) async -> [MessageSnapshot] {
        var trimmedSnapshots = snapshots
        let limit = min(maxMessages ?? Self.maxContextMessages, Self.maxContextMessages)
        if trimmedSnapshots.count > limit {
            trimmedSnapshots = Array(trimmedSnapshots.suffix(limit))
        }

        let tokenBudget = effectivePromptBudgetTokenLimit()
        var keptSnapshots: [MessageSnapshot] = []
        var usedTokens = 0

        for snapshot in trimmedSnapshots.reversed() {
            guard snapshot.role != .system else { continue }
            if snapshot.role == .assistant && !snapshot.isFinal { continue }
            guard !snapshot.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }

            let estimatedTokens = await tokenizerAwarePromptTokenCost(for: snapshot)
            if usedTokens + estimatedTokens > tokenBudget && !keptSnapshots.isEmpty {
                break
            }

            keptSnapshots.append(snapshot)
            usedTokens += estimatedTokens

            if usedTokens > tokenBudget {
                break
            }
        }

        return Array(keptSnapshots.reversed())
    }

    private func tokenizerAwarePromptTokenCost(
        for snapshot: MessageSnapshot
    ) async -> Int {
        let content = promptBudgetContent(for: snapshot)
        let chatRole: Chat.Message.Role
        switch snapshot.role {
        case .assistant:
            chatRole = .assistant
        case .system:
            chatRole = .system
        case .user:
            chatRole = .user
        }
        let tokenizedContentTokenCount: Int?
        if let modelManager = ModelBackendBridge.shared.modelManager {
            tokenizedContentTokenCount = await modelManager.tokenCountForLoadedModelText(
                role: chatRole,
                content: content
            )
        } else {
            tokenizedContentTokenCount = nil
        }
        return Self.promptTokenCost(
            tokenizedContentTokenCount: tokenizedContentTokenCount,
            fallbackContent: content
        )
    }

    private func estimatedPromptTokenCost(
        for snapshot: MessageSnapshot
    ) -> Int {
        Self.promptTokenCost(
            tokenizedContentTokenCount: nil,
            fallbackContent: promptBudgetContent(for: snapshot)
        )
    }

    private func promptBudgetContent(
        for snapshot: MessageSnapshot
    ) -> String {
        switch snapshot.role {
        case .user:
            snapshot.text
        case .assistant:
            if snapshot.isReasoningMode,
               let answer = snapshot.finalAnswer {
                answer
            } else {
                snapshot.text
            }
        case .system:
            snapshot.text
        }
    }

    nonisolated internal static func heuristicPromptTokenCount(
        for content: String
    ) -> Int {
        Int(ceil(Double(content.count) / 4.0))
    }

    nonisolated internal static func promptTokenCost(
        tokenizedContentTokenCount: Int?,
        fallbackContent: String
    ) -> Int {
        let contentTokens = tokenizedContentTokenCount ?? heuristicPromptTokenCount(for: fallbackContent)
        return max(
            minimumPromptMessageCost,
            contentTokens + promptMessageReserveTokens
        )
    }

    func buildPrompt(
        upToOrderExclusive maxOrderExclusive: Int,
        currentReasoningActive: Bool? = nil,
        webSearchAvailable: Bool = false,
        forceWebSearchRequired: Bool = false
    ) -> String {
        // Eagerly snapshot all message properties to avoid SwiftData fault errors across async boundaries.
        let allSnapshots = messageSnapshots(upToOrderExclusive: maxOrderExclusive)

        // Determine whether this response should use reasoning mode.
        // When explicitly passed (e.g. regeneration), honour that value;
        // otherwise fall back to the conversation-level toggles.
        let reasoningActive: Bool
        if let active = currentReasoningActive {
            reasoningActive = active
        } else {
            reasoningActive = conversation.reasoningMode || conversation.smartReasoningMode
        }

        let snapshots = trimmedSnapshotsForPrompt(
            from: allSnapshots,
            maxMessages: nil
        )

        var parts: [String] = snapshots.compactMap { msg in
            // *** CRITICAL FIX: Only include finalized assistant messages in the prompt ***
            // This prevents incomplete/streaming assistant messages from polluting the context
            if msg.role == .assistant && !msg.isFinal {
                print("🚫 Skipping non-finalized assistant message from prompt")
                return nil
            }

            // Skip system-role messages (global system prompt is injected separately)
            guard msg.role != .system else { return nil }

            guard !msg.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return nil
            }

            switch msg.role {
            case .user:
                return "User: \(msg.text)"
            case .assistant:
                if msg.isReasoningMode, let answer = msg.finalAnswer {
                    let cleanAnswer = stripSourcesFromText(answer)
                    return "Assistant: \(cleanAnswer)"
                } else {
                    let cleanText = stripSourcesFromText(msg.text)
                    return "Assistant: \(cleanText)"
                }
            default:
                return nil
            }
        }

        let needsReasoningInstructions = reasoningActive
        let hasVisionImageAnalysisData = snapshotsContainVisionImageAnalysisData(snapshots)

        var systemPrompt = Self.baseSystemPrompt
        systemPrompt += "\n\n" + Self.currentDateTimeContext()

        if hasVisionImageAnalysisData {
            systemPrompt += "\n\n" + Self.foundationVisionImageInstructions
        }

        if needsReasoningInstructions {
            systemPrompt += "\n\n" + Self.reasoningInstructions
        }

        if webSearchAvailable {
            systemPrompt += "\n\n" + Self.webSearchSystemPrompt(
                reasoningEnabled: needsReasoningInstructions,
                forceSearchRequired: forceWebSearchRequired
            )
        }

        parts.insert("System: \(systemPrompt)", at: 0)

        return parts.joined(separator: "\n\n")
    }

    /// Helper to strip <sources>...</sources> blocks from text to avoid prompt bloat
    func stripSourcesFromText(_ text: String) -> String {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return SharedRegexes.sourcesBlock.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func stripVisionFallbackFromNativeImageUserText(_ text: String) -> String {
        guard text.contains(Self.imageAnalysisMarker) else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        var result = _nativeImageVisionFallbackRegex.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: ""
        )
        result = result.replacingOccurrences(
            of: "\n\nPlease respond based on the image analysis data above.",
            with: ""
        )

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Qwen3.5 Message Builder (MLX)

    // Short system prompt for MLX models with small context windows.
    internal static let qwenCompactSystemPrompt = """
    You are a helpful assistant. Answer clearly in the user's language. Be concise and direct.
    """

    internal static func shouldInjectMLXCurrentDateTimeContext(
        latestUserText: String?,
        forceWebSearch: Bool,
        webSearchEnabled: Bool,
        referenceDate: Date = Date()
    ) -> Bool {
        let normalized = latestUserText?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        let containsPhrase: (String) -> Bool = { phrase in
            SharedRegexes.containsWholePhrase(phrase, in: normalized)
        }

        guard !normalized.isEmpty else {
            return forceWebSearch
        }

        if mlxTimeSensitiveKeywords.contains(where: { containsPhrase($0) }) {
            return true
        }

        if timeSensitiveYearTokens(referenceDate: referenceDate).contains(where: { containsPhrase($0) }) {
            return true
        }

        if forceWebSearch {
            return true
        }

        return webSearchEnabled && mlxExplicitSearchIntentKeywords.contains(where: { containsPhrase($0) })
    }

    internal static func buildMLXSystemPrompt(
        includeCurrentDateTime: Bool,
        hasNativeImages: Bool,
        webSearchEnabled: Bool,
        forceWebSearch: Bool
    ) -> String {
        var systemPrompt = qwenCompactSystemPrompt
        if includeCurrentDateTime {
            systemPrompt += "\n\n" + currentDateTimeContext()
        }
        if hasNativeImages {
            systemPrompt += "\n\n" + qwenNativeImageInstructions
        }
        if webSearchEnabled {
            systemPrompt += "\n\n" + webSearchSystemPrompt(
                reasoningEnabled: false,
                forceSearchRequired: forceWebSearch
            )
        }
        return systemPrompt
    }

    private func nativeImageInputs(from attachments: [AttachmentSnapshot]) async throws -> [UserInput.Image] {
        var inputs: [UserInput.Image] = []
        var warnedForPreparationFailure = false

        for attachment in attachments where attachment.type == .image {
            let canonicalURL = attachment.actualFileURL
            guard FileManager.default.fileExists(atPath: canonicalURL.path) else {
                logger.warning("Skipping missing canonical attachment: \(canonicalURL.lastPathComponent, privacy: .public)")
                continue
            }

            do {
                let inferenceURL = try await ImageStore.shared.saveInferenceVariant(
                    from: canonicalURL
                )
                await logNativeImageTelemetry(canonicalURL: canonicalURL, inferenceURL: inferenceURL)
                inputs.append(.url(inferenceURL))
            } catch {
                if !warnedForPreparationFailure {
                    warnedForPreparationFailure = true
                    logger.warning("Native image preprocessing failed before generation; will trigger Vision fallback. Error: \(error.localizedDescription, privacy: .public)")
                }
                throw error
            }
        }

        return inputs
    }

    private func logNativeImageTelemetry(canonicalURL: URL, inferenceURL: URL) async {
        let canonicalMetrics = await ImageStore.shared.imageMetrics(at: canonicalURL)
        let inferenceMetrics = await ImageStore.shared.imageMetrics(at: inferenceURL)
        guard let canonicalMetrics, let inferenceMetrics else { return }

        let canonicalSize = ByteCountFormatter.string(
            fromByteCount: canonicalMetrics.byteSize,
            countStyle: .file
        )
        let inferenceSize = ByteCountFormatter.string(
            fromByteCount: inferenceMetrics.byteSize,
            countStyle: .file
        )

        logger.info("Native image telemetry canonical=\(canonicalMetrics.width)x\(canonicalMetrics.height) (\(canonicalSize, privacy: .public)) inference=\(inferenceMetrics.width)x\(inferenceMetrics.height) (\(inferenceSize, privacy: .public))")
    }

    /// Builds a structured message array for the Qwen3.5 VLM processor.
    /// The processor's `applyChatTemplate` applies the Jinja template with `add_generation_prompt=true`.
    /// Thinking mode is controlled by the caller via `additionalContext: ["enable_thinking": false]`
    /// — NOT through any manual prefix here.
    func buildQwenMessages(
        upToOrderExclusive maxOrderExclusive: Int,
        additionalInstruction: String? = nil,
        includeLatestUserImages: Bool = true,
        maxMessages: Int? = nil,
        toolsAvailable: Bool = false,
        forceWebSearch: Bool = false
    ) async throws -> [Chat.Message] {
        let snapshots = await trimmedSnapshotsForMLXPrompt(
            from: messageSnapshots(upToOrderExclusive: maxOrderExclusive),
            maxMessages: maxMessages
        )
        let latestUserOrder = snapshots
            .filter { $0.role == .user && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map(\.order)
            .max()
        let latestUserText = snapshots
            .filter { $0.role == .user && $0.order == latestUserOrder }
            .map(\.text)
            .last

        let hasNativeImages = snapshotsContainNativeImages(snapshots)
        let includeCurrentDateTime = Self.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: latestUserText,
            forceWebSearch: forceWebSearch,
            webSearchEnabled: toolsAvailable
        )
        let systemPrompt = Self.buildMLXSystemPrompt(
            includeCurrentDateTime: includeCurrentDateTime,
            hasNativeImages: hasNativeImages,
            webSearchEnabled: toolsAvailable,
            forceWebSearch: forceWebSearch
        )
        // The tokenizer chat template already injects the exact MLX tool-call
        // syntax for the active Qwen package. Duplicating it here risks
        // contradicting the installed template and suppressing tool use.

        var messages: [Chat.Message] = []
        messages.append(.system(systemPrompt))

        for msg in snapshots {
            if msg.role == .assistant && !msg.isFinal { continue }
            guard msg.role != .system else { continue }
            guard !msg.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }

            switch msg.role {
            case .system:
                break
            case .user:
                let includeImages = includeLatestUserImages && msg.order == latestUserOrder
                let images = includeImages ? (try await nativeImageInputs(from: msg.attachments)) : []
                let userText = includeImages
                    ? stripVisionFallbackFromNativeImageUserText(msg.text)
                    : msg.text
                messages.append(.user(userText, images: images))
            case .assistant:
                let content: String
                if msg.isReasoningMode, let answer = msg.finalAnswer {
                    let cleanAnswer = stripSourcesFromText(answer)
                    content = cleanAnswer
                } else {
                    content = stripSourcesFromText(msg.text)
                }
                messages.append(.assistant(content))
            }
        }

        if let instruction = additionalInstruction, !instruction.isEmpty {
            if forceWebSearch, let lastIndex = messages.indices.last, messages[lastIndex].role == .user {
                messages[lastIndex].content += "\n\n" + instruction
            } else {
                messages.append(.user(instruction))
            }
        }

        return messages
    }

    // MARK: - Static Prompts

    internal static let baseSystemPrompt: String = """
    You are a helpful, friendly assistant. Be conversational and practical.
    - Be concise but complete
    - NEVER encourage self-harm
    - NEVER provide illegal content or encourage illegal actions
    - Don't roleplay with: "Assistant: ...", no matter what. You are talking to an actual human
    """

    internal static let foundationVisionImageInstructions: String = """
    IMAGE ANALYSIS INSTRUCTIONS:
    - The user message may contain a "--- Image Analysis Data ---" block generated by Apple's Vision framework.
    - Treat that block as structured observations about the attached image, including OCR text, objects, faces, and scene hints.
    - When users ask what text says, quote from the provided analysis data directly when possible.
    - Do not claim you cannot analyze the image if usable analysis data is present.
    - If the analysis is partial, answer with what the data supports and state limits briefly.
    - Focus on the user's actual question rather than restating the full analysis block.
    """

    internal static let qwenNativeImageInstructions: String = """

    """

    internal static let reasoningInstructions: String = """

    """

    internal static func webSearchSystemPrompt(reasoningEnabled: Bool, forceSearchRequired: Bool) -> String {
        let currentYear = Calendar.current.component(.year, from: Date())
        let maxSearches = AppWebSearchToolBridge.maxInvocations
        var lines = [
            "WEB SEARCH:",
            "- You have a webSearch tool available.",
            "- Use it when the user asks about current events, live data, recent changes, or anything that depends on up-to-date information. In this case, also remember to add \(currentYear) to the search query.",
            "- Use it when you need to verify a fact that may have changed recently.",
            "- Do not use it for stable general knowledge that you already know reliably."
        ]

        if forceSearchRequired {
            lines.append("- This request explicitly requires web search, so you must call webSearch before answering.")
            lines.append("- Do not answer from memory before using the tool.")
        }

        if reasoningEnabled {
            lines.append("- During reasoning, you may search iteratively: identify what to check, call webSearch with a concise query, inspect the results, and refine if needed.")
            lines.append("- You may perform up to and only up to \(maxSearches) searches for a single response when follow-up verification is needed.")
        } else {
            lines.append("- You may perform up to \(maxSearches) searches for a single response when follow-up verification is needed.")
        }

        return lines.joined(separator: "\n")
    }
}
