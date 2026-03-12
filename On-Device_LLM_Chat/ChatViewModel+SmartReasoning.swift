//
//  ChatViewModel+SmartReasoning.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import FoundationModels
import os

// Cached regexes for parseReasoningResponse (compiled once at app launch)
// swiftlint:disable force_try
private let _parseReasoningRegexes: [NSRegularExpression] = [
    try! NSRegularExpression(pattern: #"<thinking>(.*?)</thinking>\s*(?:Final answer:\s*)?(.*?)$"#, options: [.dotMatchesLineSeparators]),
    try! NSRegularExpression(pattern: #"<thinking>(.*?)</thinking>\s*(.*?)$"#, options: [.dotMatchesLineSeparators]),
    try! NSRegularExpression(pattern: #"<think>(.*?)</think>\s*(?:Final answer:\s*)?(.*?)$"#, options: [.dotMatchesLineSeparators]),
    try! NSRegularExpression(pattern: #"<think>(.*?)</think>\s*(.*?)$"#, options: [.dotMatchesLineSeparators]),
]

// Cached regex for stripping <sources>…</sources> in updateMessageWithReasoningContent
private let _sourcesRegexSR = try! NSRegularExpression(
    pattern: #"<sources>(.*?)</sources>"#, options: [.dotMatchesLineSeparators, .caseInsensitive])
// swiftlint:enable force_try

extension ChatViewModel {

    // MARK: - Smart Reasoning

    internal enum ReasoningEvaluationError: LocalizedError {
        case foundationModelsUnavailable(fallbackResult: Bool)
        case foundationModelsError(error: Error, fallbackResult: Bool)

        var errorDescription: String? {
            switch self {
            case .foundationModelsUnavailable(let fallbackResult):
                return "Foundation Models unavailable, using heuristic result: \(fallbackResult)"
            case .foundationModelsError(let error, let fallbackResult):
                return "Foundation Models error (\(error.localizedDescription)), using heuristic result: \(fallbackResult)"
            }
        }

        var fallbackResult: Bool {
            switch self {
            case .foundationModelsUnavailable(let result), .foundationModelsError(_, let result):
                return result
            }
        }
    }

    /// Determines if reasoning mode should be used based on prompt complexity
    internal func shouldUseReasoningForPrompt(_ userText: String) async throws -> Bool {
        let backendBridge = ModelBackendBridge.shared
        guard backendBridge.reasoningAvailable else { return false }

        if conversation.reasoningMode { return true }
        guard conversation.smartReasoningMode else { return false }

        // On MLX backend, skip Foundation Models entirely — loading Apple Intelligence
        // alongside the MLX model causes OOM. Use the heuristic directly.
        if backendBridge.selectedBackend == .mlx {
            return useReasoningHeuristic(userText)
        }

        let result = try await evaluatePromptComplexity(userText)
        return result
    }

    /// Uses Foundation Models to determine if a prompt requires reasoning.
    /// Only called on the Foundation Models backend.
    private func evaluatePromptComplexity(_ userText: String) async throws -> Bool {
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            throw ReasoningEvaluationError.foundationModelsUnavailable(fallbackResult: useReasoningHeuristic(userText))
        }

        let trimmedInput = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedInput.isEmpty, trimmedInput.count <= 2000 else {
            throw ReasoningEvaluationError.foundationModelsUnavailable(fallbackResult: useReasoningHeuristic(userText))
        }

        do {
            let instructions = """
            You are an AI assistant evaluator. Your job is to determine if a user's request requires step-by-step reasoning or if it can be answered directly.

            IMPORTANT: Be helpful and interpret user intent. Typos and minor errors should not affect your decision.

            Respond with ONLY "YES" if the request requires reasoning (complex problem-solving, multi-step analysis, mathematical calculations, code debugging, logical deduction, architecture decisions, etc.).

            Respond with ONLY "NO" if the request can be answered directly (simple questions, factual lookups, basic explanations, casual conversation, straightforward how-to questions, etc.).

            Be conservative - only use reasoning when it genuinely adds value for complex, multi-step problems.
            """

            let response = try await withFoundationModelsLock(timeout: .seconds(10)) {
                let session = LanguageModelSession(instructions: instructions)
                return try await session.respond(to: "User request: \"\(trimmedInput)\"")
            }

            let decision = response.content.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
            let shouldUseReasoning = decision == "YES"
            logger.info("Smart reasoning decision computed: request_chars=\(trimmedInput.count, privacy: .public) result=\(shouldUseReasoning ? "YES" : "NO", privacy: .public)")
            return shouldUseReasoning
        } catch {
            logger.error("Error evaluating prompt complexity: \(error.localizedDescription)")
            throw ReasoningEvaluationError.foundationModelsError(error: error, fallbackResult: useReasoningHeuristic(userText))
        }
    }

    /// Fallback heuristic for determining if reasoning is needed when Foundation Models unavailable
    private func useReasoningHeuristic(_ userText: String) -> Bool {
        let lowercased = userText.lowercased()

        // Simple questions that definitely don't need reasoning
        let simplePatterns = [
            "what is", "what are", "who is", "who was", "when was", "when is",
            "hello", "hi ", "hey ", "thanks", "thank you", "goodbye", "bye"
        ]

        let isSimpleQuestion = simplePatterns.contains { lowercased.hasPrefix($0) }
        if isSimpleQuestion && userText.count < 50 {
            print("Heuristic: Simple question detected, no reasoning needed")
            return false
        }

        // Keywords that often indicate complex tasks requiring reasoning
        let complexityKeywords = [
            "debug", "fix", "optimize", "improve", "refactor", "architecture",
            "design pattern", "algorithm", "performance", "error", "issue", "problem",
            "compare", "analyze", "evaluate", "trade-off", "pros and cons",
            "best practice", "approach", "strategy", "implementation",
            "why does", "why is", "how should", "what's wrong", "what's the best",
            "step by step", "walk through", "explain how", "show me how"
        ]

        // Mathematical or logical indicators
        let logicalKeywords = [
            "calculate", "solve", "equation", "formula", "logic", "reasoning",
            "proof", "derive", "determine", "find the", "compute"
        ]

        // Code-related complexity indicators
        let codeKeywords = [
            "swift", "swiftui", "uikit", "appkit", "xcode", "ios", "macos",
            "function", "class", "struct", "protocol", "extension", "enum",
            "async", "await", "combine", "coredata", "swiftdata"
        ]

        // Check for complexity indicators
        let hasComplexityKeyword = complexityKeywords.contains { lowercased.contains($0) }
        let hasLogicalKeyword = logicalKeywords.contains { lowercased.contains($0) }
        let hasCodeKeyword = codeKeywords.contains { lowercased.contains($0) }

        // Additional heuristics
        let hasQuestionWords = lowercased.contains("how") || lowercased.contains("why") || lowercased.contains("what")
        let isLongQuery = userText.count > 100
        let hasMultipleSentences = userText.components(separatedBy: CharacterSet(charactersIn: ".!?")).count > 2

        // Combine heuristics - be somewhat conservative
        if hasComplexityKeyword || hasLogicalKeyword {
            print("Heuristic: Complexity/logical keywords found, reasoning needed")
            return true
        }

        if hasCodeKeyword && (hasQuestionWords || isLongQuery) {
            print("Heuristic: Code-related complex question, reasoning needed")
            return true
        }

        if isLongQuery && hasMultipleSentences && hasQuestionWords {
            print("Heuristic: Long multi-sentence question, reasoning needed")
            return true
        }

        print("Heuristic: No complexity indicators found, no reasoning needed")
        return false
    }

    // MARK: - Reasoning Response Parsing

    private func parseReasoningResponse(_ text: String) -> (reasoning: String?, finalAnswer: String?) {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            return (nil, nil)
        }

        // DEBUG: Log what we're parsing to diagnose short answer issues
        if trimmedText.count < 200 {
            print("🔍 parseReasoningResponse input chars: \(trimmedText.count)")
        }

        // Check all tag variants (Qwen3.5 uses <think>, older paths use <thinking>)
        let hasOpenThinking  = trimmedText.range(of: "<thinking>",  options: .caseInsensitive) != nil
        let hasCloseThinking = trimmedText.range(of: "</thinking>", options: .caseInsensitive) != nil
        let hasOpenThink     = trimmedText.range(of: "<think>",     options: .caseInsensitive) != nil
        let hasCloseThink    = trimmedText.range(of: "</think>",    options: .caseInsensitive) != nil

        // In-progress: opening tag present but closing tag not yet arrived
        let inProgressTag: String? = (hasOpenThinking && !hasCloseThinking) ? "<thinking>" :
                                     (hasOpenThink    && !hasCloseThink)    ? "<think>"    : nil
        if let openTag = inProgressTag,
           let openRange = trimmedText.range(of: openTag, options: .caseInsensitive) {
            let after = String(trimmedText[openRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (after.isEmpty ? nil : after, nil)
        }

        // MLX path: opening <think> was in the prompt prefix so only </think> appears in output.
        // Split at the first </think> — everything before is reasoning, everything after is the answer.
        if !hasOpenThinking && !hasOpenThink && hasCloseThink,
           let closeRange = trimmedText.range(of: "</think>", options: .caseInsensitive) {
            let reasoning = String(trimmedText[..<closeRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let answer    = String(trimmedText[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, answer.isEmpty ? nil : answer)
        }

        // No recognized tags → plain answer
        guard (hasOpenThinking && hasCloseThinking) || (hasOpenThink && hasCloseThink) else {
            return (nil, trimmedText)
        }

        // Full pattern match for both tag variants
        for regex in _parseReasoningRegexes {
            let range = NSRange(trimmedText.startIndex..<trimmedText.endIndex, in: trimmedText)

            if let match = regex.firstMatch(in: trimmedText, options: [], range: range),
               match.numberOfRanges >= 3 {

                var reasoning: String? = nil
                var finalAnswer: String? = nil

                // Safely extract reasoning
                let reasoningRange = match.range(at: 1)
                if reasoningRange.location != NSNotFound,
                   let swiftRange = Range(reasoningRange, in: trimmedText) {
                    reasoning = String(trimmedText[swiftRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if reasoning?.isEmpty == true { reasoning = nil }
                }

                // Safely extract final answer
                let answerRange = match.range(at: 2)
                if answerRange.location != NSNotFound,
                   let swiftRange = Range(answerRange, in: trimmedText) {
                    finalAnswer = String(trimmedText[swiftRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if finalAnswer?.isEmpty == true { finalAnswer = nil }
                }

                // DEBUG: Log extracted values for short responses
                if (reasoning?.count ?? 0) < 200 || (finalAnswer?.count ?? 0) < 100 {
                    print("🔍 parseReasoningResponse extracted:")
                    print("   - reasoning chars: \(reasoning?.count ?? 0)")
                    print("   - finalAnswer chars: \(finalAnswer?.count ?? 0)")
                }

                return (reasoning, finalAnswer)
            }
        }

        // If all patterns failed, treat as final answer
        return (nil, trimmedText)
    }

    internal static func parseReasoningResponseForSearchSessionText(_ text: String) -> (reasoning: String?, finalAnswer: String?) {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            return (nil, nil)
        }

        let closeTag = "</think>"
        let altCloseTag = "</thinking>"
        let lastThinkClose = trimmedText.range(of: closeTag, options: [.caseInsensitive, .backwards])
        let lastThinkingClose = trimmedText.range(of: altCloseTag, options: [.caseInsensitive, .backwards])
        let lastCloseRange: Range<String.Index>? = {
            switch (lastThinkClose, lastThinkingClose) {
            case let (.some(a), .some(b)):
                return a.lowerBound > b.lowerBound ? a : b
            case let (.some(a), nil):
                return a
            case let (nil, .some(b)):
                return b
            case (nil, nil):
                return nil
            }
        }()

        guard let closeRange = lastCloseRange else {
            let reasoningOnly = trimmedText
                .replacingOccurrences(of: "<think>", with: "", options: .caseInsensitive)
                .replacingOccurrences(of: "<thinking>", with: "", options: .caseInsensitive)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoningOnly.isEmpty ? nil : reasoningOnly, nil)
        }

        let beforeClose = String(trimmedText[..<closeRange.lowerBound])
            .replacingOccurrences(of: "<think>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "<thinking>", with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let afterClose = String(trimmedText[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)

        let paragraphs = afterClose
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var reasoningParts: [String] = beforeClose.isEmpty ? [] : [beforeClose]
        var answerParts: [String] = []

        for paragraph in paragraphs {
            if answerParts.isEmpty && looksLikeReasoningContinuation(paragraph) {
                reasoningParts.append(paragraph)
            } else {
                answerParts.append(paragraph)
            }
        }

        let reasoning = reasoningParts.joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
        var answer = answerParts.joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
        if answer.lowercased().hasPrefix("final answer:") {
            answer = String(answer.dropFirst("final answer:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return (reasoning.isEmpty ? nil : reasoning, answer.isEmpty ? nil : answer)
    }

    private func parseReasoningResponseForSearchSession(_ text: String) -> (reasoning: String?, finalAnswer: String?) {
        Self.parseReasoningResponseForSearchSessionText(text)
    }

    private static func looksLikeReasoningContinuation(_ text: String) -> Bool {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }

        let reasoningPrefixes = [
            "let me", "i should", "i need to", "i'll", "i will", "now i",
            "i should check", "i should search", "i need to verify", "i need to check",
            "wait,", "hold on,", "before i answer"
        ]
        if reasoningPrefixes.contains(where: { normalized.hasPrefix($0) }) {
            return true
        }

        if normalized.contains("search") && normalized.contains("let me") {
            return true
        }

        if normalized.contains("search again") || normalized.contains("another search") {
            return true
        }

        return false
    }

    private static func looksLikeAnswerStartDuringStreaming(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        let normalized = trimmed.lowercased()
        if looksLikeReasoningContinuation(trimmed) {
            return false
        }

        if normalized.hasPrefix("final answer:") ||
            normalized.hasPrefix("based on") ||
            normalized.hasPrefix("according to") ||
            normalized.hasPrefix("the answer is") ||
            normalized.hasPrefix("in summary") ||
            normalized.hasPrefix("here's") ||
            normalized.hasPrefix("here is") {
            return true
        }

        if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("1. ") || trimmed.hasPrefix("• ") {
            return true
        }

        if trimmed.contains("\n\n") {
            return true
        }

        if normalized.hasPrefix("i ") || normalized.hasPrefix("i'm ") || normalized.hasPrefix("i’ve ") || normalized.hasPrefix("i've ") {
            return false
        }

        return trimmed.count >= 48
    }

    // made internal so it can be called from extensions in other files
    internal func updateMessageWithReasoningContent(_ message: Message, fullText: String, finalize: Bool = false) {
        // DEBUG: Log if we're updating with empty/whitespace-only content
        if fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("⚠️ updateMessageWithReasoningContent called with empty/whitespace-only text")
            print("   - Original text length: \(fullText.count)")
            print("   - Message ID: \(message.id)")
            print("   - isReasoningMode: \(message.isReasoningMode)")
            return // Don't update with empty content
        }

        // 1) During streaming, keep updates lightweight to avoid UI lag on long outputs.
        // Full cleanup is done on final flush.
        let cleanedText: String
        if finalize {
            let noSearchTags = stripSearchTags(fullText)
            cleanedText = cleanGlitchedText(noSearchTags)
        } else {
            cleanedText = fullText
        }

        // 3) Preserve or extract any <sources> from either the existing message content or this update
        let existingCarrier = message.isReasoningMode ? (message.finalAnswer ?? message.text) : message.text
        let existingSources = extractSources(from: existingCarrier)
        let newSources = extractSources(from: cleanedText)
        let chosenSources = newSources ?? existingSources
        let sourcesBlock = chosenSources.map { "<sources>\n\($0)\n</sources>\n\n" } ?? ""

        // 4) Remove <sources> from the cleanedText that we will display
        let visiblePortion = {
            if let s = newSources, !s.isEmpty {
                // Remove the first <sources> block occurrence
                let re = _sourcesRegexSR
                let ns = cleanedText as NSString
                let range = NSRange(location: 0, length: ns.length)
                return re.stringByReplacingMatches(in: cleanedText, options: [], range: range, withTemplate: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return cleanedText
        }()

        if message.isReasoningMode {
            // MLX reasoning path: the <think> opening tag is baked into the prompt prefix,
            // so the streamed output starts with raw thinking content and ends with </think>.
            // Before </think> arrives all content is in-progress reasoning with no tags at all.
            let isMLX = ModelBackendBridge.shared.selectedBackend == .mlx
            if isMLX {
                let hasClose = visiblePortion.contains("</think>") || visiblePortion.contains("</thinking>")
                if !hasClose {
                    // Still thinking – show in the reasoning bubble, nothing in the answer yet.
                    let r = visiblePortion.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !r.isEmpty {
                        message.reasoning = r
                        message.updateReasoningSteps()
                    }
                    return
                }
            }

            let hasLiveSearches = !((message.searchInvocations ?? []).isEmpty)
            let parsed = if isMLX && hasLiveSearches {
                parseReasoningResponseForSearchSession(visiblePortion)
            } else {
                parseReasoningResponse(visiblePortion)
            }

            if isMLX && hasLiveSearches && !finalize {
                let phase = message.streamingReasoningPhase ?? .initialThinking
                if phase == .postToolReasoning {
                    if let answer = parsed.finalAnswer,
                       Self.looksLikeAnswerStartDuringStreaming(answer) {
                        message.streamingReasoningPhase = .finalAnswer
                    } else {
                        let provisionalReasoning = (
                            parsed.reasoning ??
                            visiblePortion
                                .replacingOccurrences(of: "</think>", with: "", options: .caseInsensitive)
                                .replacingOccurrences(of: "</thinking>", with: "", options: .caseInsensitive)
                        )
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !provisionalReasoning.isEmpty {
                            message.reasoning = provisionalReasoning
                            message.updateReasoningSteps()
                        }
                        message.text = ""
                        return
                    }
                }
            }

            // Always update reasoning if present
            if let reasoning = parsed.reasoning {
                message.reasoning = reasoning
                // Parse reasoning into structured steps for step-by-step UI
                message.updateReasoningSteps()
            }

            // Only update final answer (and visible text) when we actually have one
            if let answer = parsed.finalAnswer {
                // Prepend hidden sources (if any) into finalAnswer so UI can find them
                let stored = sourcesBlock + answer
                message.finalAnswer = stored
                message.text = answer // Keep raw visible text for non-reasoning fallbacks
                message.streamingReasoningPhase = .finalAnswer
            } else {
                // While only reasoning is present, keep the visible text empty or last known final answer
                if parsed.reasoning == nil && visiblePortion.contains("<thinking>") {
                    // Keep raw text for UI fallback parsing
                    message.text = visiblePortion
                } else if parsed.reasoning == nil && !visiblePortion.isEmpty {
                    // No reasoning detected, but we have content (likely an error message or plain text)
                    // Store it directly in text so it's visible
                    message.text = visiblePortion
                } else {
                    // Extract visible text from finalAnswer (if any) without sources
                    if let currentFinal = message.finalAnswer {
                        message.text = stripSourcesFromText(currentFinal)
                    } else {
                        message.text = ""
                    }
                }
                // If we have sources already but no final answer yet, ensure they're retained in finalAnswer
                if chosenSources != nil && (message.finalAnswer ?? "").isEmpty {
                    message.finalAnswer = sourcesBlock
                }
            }
        } else {
            // Non-reasoning: prepend hidden sources into the stored text so UI can extract and show button
            let stored = sourcesBlock + visiblePortion
            message.text = stored
        }
    }
}
