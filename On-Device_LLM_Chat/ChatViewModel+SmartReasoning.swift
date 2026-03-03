//
//  ChatViewModel+SmartReasoning.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import FoundationModels
import os

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
        // If manual reasoning mode is explicitly enabled, always use it
        if conversation.reasoningMode {
            return true
        }

        // If smart reasoning mode is not enabled, don't use reasoning
        guard conversation.smartReasoningMode else {
            return false
        }

        // Use Foundation Models to evaluate if reasoning is needed
        let result = try await evaluatePromptComplexity(userText)
        return result
    }

    /// Uses Foundation Models to determine if a prompt requires reasoning
    private func evaluatePromptComplexity(_ userText: String) async throws -> Bool {
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            // Throw an error so calling methods can handle fallback appropriately
            throw ReasoningEvaluationError.foundationModelsUnavailable(fallbackResult: useReasoningHeuristic(userText))
        }

        // Add input validation to prevent issues
        let trimmedInput = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedInput.isEmpty, trimmedInput.count <= 2000 else {
            logger.debug("Input too long or empty for complexity evaluation (\(trimmedInput.count) chars), using heuristic")
            throw ReasoningEvaluationError.foundationModelsUnavailable(fallbackResult: useReasoningHeuristic(userText))
        }

        do {
            let instructions = """
            You are an AI assistant evaluator. Your job is to determine if a user's request requires step-by-step reasoning or if it can be answered directly.

            IMPORTANT: Be helpful and interpret user intent. Typos and minor errors should not affect your decision.

            Respond with ONLY "YES" if the request requires reasoning (complex problem-solving, multi-step analysis, mathematical calculations, code debugging, logical deduction, architecture decisions, etc.).

            Respond with ONLY "NO" if the request can be answered directly (simple questions, factual lookups, basic explanations, casual conversation, straightforward how-to questions, etc.).

            Examples that NEED reasoning:
            - "How do I optimize this Swift code for better performance?"
            - "What's wrong with my algorithm logic?"
            - "Help me debug this SwiftUI layout issue"
            - "Explain the trade-offs between different architectural patterns"
            - "How should I structure my app's data model?"
            - "Why is my app crashing when I do X?"

            Examples that DON'T need reasoning:
            - "What is SwiftUI?"
            - "How do I create a button in SwiftUI?"
            - "What's the syntax for a for loop in Swift?"
            - "What is OpenAI?"
            - "Hello, how are you?"
            - "Can you help me?"
            - "Show me an example of..."

            Be conservative - only use reasoning when it genuinely adds value for complex, multi-step problems.
            """

            // CRITICAL FIX: Use the lock to prevent concurrent Foundation Models sessions
            // Increased timeout from 5s to 10s for better reliability under load
            let response = try await withFoundationModelsLock(timeout: .seconds(10)) {
                // Create session inside the lock
                let session = LanguageModelSession(instructions: instructions)

                // BUG FIX: Removed unnecessary 100ms delay that added latency to every message
                // The lock already prevents concurrent access, making this delay redundant

                return try await session.respond(to: "User request: \"\(trimmedInput)\"")
            }

            let decision = response.content.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
            let shouldUseReasoning = decision == "YES"

            logger.info("Smart reasoning decision for '\(String(trimmedInput.prefix(50)))...': \(shouldUseReasoning ? "YES" : "NO")")
            return shouldUseReasoning

        } catch {
            logger.error("Error evaluating prompt complexity: \(error.localizedDescription)")
            // Throw a specialized error with heuristic fallback
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
            print("🔍 parseReasoningResponse input: '\(trimmedText)'")
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
        let patterns = [
            #"<thinking>(.*?)</thinking>\s*(?:Final answer:\s*)?(.*?)$"#,
            #"<thinking>(.*?)</thinking>\s*(.*?)$"#,
            #"<think>(.*?)</think>\s*(?:Final answer:\s*)?(.*?)$"#,
            #"<think>(.*?)</think>\s*(.*?)$"#,
        ]

        for pattern in patterns {
            do {
                let regex = try NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
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
                        print("   - reasoning: '\(reasoning ?? "nil")'")
                        print("   - finalAnswer: '\(finalAnswer ?? "nil")'")
                    }

                    return (reasoning, finalAnswer)
                }
            } catch {
                continue
            }
        }

        // If all patterns failed, treat as final answer
        return (nil, trimmedText)
    }

    // made internal so it can be called from extensions in other files
    internal func updateMessageWithReasoningContent(_ message: Message, fullText: String) {
        // DEBUG: Log if we're updating with empty/whitespace-only content
        if fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("⚠️ updateMessageWithReasoningContent called with empty/whitespace-only text")
            print("   - Original text length: \(fullText.count)")
            print("   - Message ID: \(message.id)")
            print("   - isReasoningMode: \(message.isReasoningMode)")
            return // Don't update with empty content
        }

        // 1) Strip any <search> scaffolding variants from the incoming text
        let noSearchTags = stripSearchTags(fullText)

        // 2) Clean glitches
        let cleanedText = cleanGlitchedText(noSearchTags)

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
                let pattern = #"<sources>(.*?)</sources>"#
                if let re = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) {
                    let ns = cleanedText as NSString
                    let range = NSRange(location: 0, length: ns.length)
                    return re.stringByReplacingMatches(in: cleanedText, options: [], range: range, withTemplate: "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
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

            let parsed = parseReasoningResponse(visiblePortion)

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
