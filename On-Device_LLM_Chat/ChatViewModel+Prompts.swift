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
private let _sourcesRegex = try! NSRegularExpression(
    pattern: #"<sources>(.*?)</sources>"#,
    options: [.dotMatchesLineSeparators, .caseInsensitive])
// swiftlint:disable:next force_try
private let _nativeImageVisionFallbackRegex = try! NSRegularExpression(
    pattern: #"--- Image Analysis Data ---.*?--- End Image Analysis ---\s*User's question:\s*"#,
    options: [.dotMatchesLineSeparators, .caseInsensitive])

extension ChatViewModel {

    private static let imageAnalysisMarker = "--- Image Analysis Data ---"

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
        conversation.reasoningMode = enabled
        if enabled {
            conversation.smartReasoningMode = false // Disable smart mode when manual mode is enabled
        }
        conversation.lastUpdated = Date()
        immediateSave()
    }

    func setSmartReasoningMode(_ enabled: Bool) {
        conversation.smartReasoningMode = enabled
        if enabled {
            conversation.reasoningMode = false // Disable manual mode when smart mode is enabled
        }
        conversation.lastUpdated = Date()
        immediateSave()
    }

    // MARK: - Prompt Builder

    // Maximum number of messages (user + assistant combined) to include in the prompt context.
    // Older messages are silently dropped to prevent OOM on long conversations.
    private static let maxContextMessages = 30

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

    func buildPrompt(
        upToOrderExclusive maxOrderExclusive: Int,
        currentReasoningActive: Bool? = nil,
        webSearchAvailable: Bool = false,
        forceWebSearchRequired: Bool = false
    ) -> String {
        // Eagerly snapshot all message properties to avoid SwiftData fault errors across async boundaries.
        var snapshots = messageSnapshots(upToOrderExclusive: maxOrderExclusive)
        // Sliding window: keep only the most recent N messages to cap context size.
        if snapshots.count > Self.maxContextMessages {
            snapshots = Array(snapshots.suffix(Self.maxContextMessages))
        }

        // Determine whether this response should use reasoning mode.
        // When explicitly passed (e.g. regeneration), honour that value;
        // otherwise fall back to the conversation-level toggles.
        let reasoningActive: Bool
        if let active = currentReasoningActive {
            reasoningActive = active
        } else {
            reasoningActive = conversation.reasoningMode || conversation.smartReasoningMode
        }

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
                if msg.isReasoningMode, let reasoning = msg.reasoning, let answer = msg.finalAnswer {
                    let cleanAnswer = stripSourcesFromText(answer)
                    // Only include <thinking> tags when reasoning is still active,
                    // otherwise the model continues reasoning after toggle-off.
                    if reasoningActive {
                        return "Assistant: <thinking>\n\(reasoning)\n</thinking>\n\n\(cleanAnswer)"
                    } else {
                        return "Assistant: \(cleanAnswer)"
                    }
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
        return _sourcesRegex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
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
        var snapshots = messageSnapshots(upToOrderExclusive: maxOrderExclusive)
        // Sliding window: cap context to avoid OOM from accumulated KV cache / token pressure.
        let limit = maxMessages ?? Self.maxContextMessages
        if snapshots.count > limit {
            snapshots = Array(snapshots.suffix(limit))
        }
        let latestUserOrder = snapshots
            .filter { $0.role == .user && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map(\.order)
            .max()

        let hasNativeImages = snapshotsContainNativeImages(snapshots)

        var systemPrompt = Self.qwenCompactSystemPrompt
        systemPrompt += "\n\n" + Self.currentDateTimeContext()
        if hasNativeImages {
            systemPrompt += "\n\n" + Self.qwenNativeImageInstructions
        }
        if toolsAvailable {
            systemPrompt += "\n\n" + Self.webSearchSystemPrompt(
                reasoningEnabled: false,
                forceSearchRequired: forceWebSearch
            )
            // The tokenizer chat template already injects the exact MLX tool-call
            // syntax for the active Qwen package. Duplicating it here risks
            // contradicting the installed template and suppressing tool use.
        }

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
                if msg.isReasoningMode, let reasoning = msg.reasoning, let answer = msg.finalAnswer {
                    let cleanAnswer = stripSourcesFromText(answer)
                    content = "<think>\n\(reasoning)\n</think>\n\n\(cleanAnswer)"
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
    - Answer questions directly and helpfully
    - Answer in the user's language
    - Answer casually and conversationally with the user
    - If you see a typo or unclear request, interpret the user's intent and respond accordingly
    - Don't apologize for typos or minor errors - just answer what the user likely meant
    - Be concise but complete
    - You can do math. When numbers start getting large though, always say that your answer may not be correct and to double check.
    - NEVER encourage self-harm
    - NEVER provide illegal content or encourage illegal actions
    - Don't repeat yourself too often when casually talking
    - Don't roleplay with: "Assistant: ...", no matter what. You are talking to an actual human
    - Do NOT wrap your answer in <answer> tags or any other XML tags
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
    NATIVE IMAGE INSTRUCTIONS:
    - Image attachments are provided directly to you as native multimodal inputs.
    - Inspect the image itself; do not assume there will be a separate Vision-analysis text block unless the user message explicitly contains one.
    - Read visible text from the image when asked, and describe visual details directly and concretely.
    - Focus on the user's actual question about the image.
    """

    internal static let reasoningInstructions: String = """
    REASONING MODE INSTRUCTIONS:
    1. First, write your thinking process inside <thinking> tags:
       <thinking>
       - Break down the problem
       - Consider different approaches
       - Work through the logic step-by-step
       - This is your private workspace - the user will see this as "View Reasoning"
       </thinking>

    2. After </thinking>, write your ACTUAL ANSWER for the user:
       - This is what the user will see as your main response
       - Provide the complete, final answer here
       - Include all the information, examples, code, lists, etc. that the user asked for
       - Don't just write a summary or meta-comment - give the FULL answer

    Do NOT wrap your answer in <answer> tags or any other XML tags. Write plain text after </thinking>.

    Example (CORRECT):
    <thinking>
    The user wants 5 story ideas. I should brainstorm different genres: sci-fi, fantasy, mystery, etc.
    Let me come up with creative concepts for each.
    </thinking>

    Here are 5 story ideas:

    1. **Time-Traveling Librarian**: A librarian discovers...
    2. **AI Awakening**: In a future where...
    [...complete list of 5 ideas with full descriptions ...]
"""

    internal static func webSearchSystemPrompt(reasoningEnabled: Bool, forceSearchRequired: Bool) -> String {
        var lines = [
            "WEB SEARCH:",
            "- You have a webSearch tool available.",
            "- Use it when the user asks about current events, live data, recent changes, or anything that depends on up-to-date information. In this case, also remember to add 2026 to the search query.",
            "- Use it when you need to verify a fact that may have changed recently.",
            "- Do not use it for stable general knowledge that you already know reliably."
        ]

        if forceSearchRequired {
            lines.append("- This request explicitly requires web search, so you must call webSearch before answering.")
            lines.append("- Do not answer from memory before using the tool.")
        }

        if reasoningEnabled {
            lines.append("- During reasoning, you may search iteratively: identify what to check, call webSearch with a concise query, inspect the results, and refine if needed.")
            lines.append("- You may perform up to and only up to 2 searches for a single response when follow-up verification is needed.")
        } else {
            lines.append("- You may perform up to 2 searches for a single response when follow-up verification is needed.")
        }

        return lines.joined(separator: "\n")
    }
}
