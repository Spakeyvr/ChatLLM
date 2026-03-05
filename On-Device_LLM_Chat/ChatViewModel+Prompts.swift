//
//  ChatViewModel+Prompts.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import MLXLMCommon
import os.log

private let _sourcesRegex = try? NSRegularExpression(
    pattern: #"<sources>(.*?)</sources>"#,
    options: [.dotMatchesLineSeparators, .caseInsensitive])

extension ChatViewModel {

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
            .sorted { $0.order < $1.order }
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

    func buildPrompt(upToOrderExclusive maxOrderExclusive: Int, currentReasoningActive: Bool? = nil, webSearchAvailable: Bool = false) -> String {
        // Eagerly snapshot all message properties to avoid SwiftData fault errors across async boundaries.
        let snapshots = messageSnapshots(upToOrderExclusive: maxOrderExclusive)

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
            case .system:
                return nil
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
            }
        }

        let needsReasoningInstructions = reasoningActive

        var systemPrompt = Self.fallbackSystemPrompt

        if needsReasoningInstructions {
            systemPrompt += "\n\n" + Self.reasoningInstructions
        }

        if webSearchAvailable {
            systemPrompt += "\n\n" + (needsReasoningInstructions ? Self.webSearchReasoningInstructions : Self.webSearchInstructions)
        }

        parts.insert("System: \(systemPrompt)", at: 0)

        return parts.joined(separator: "\n\n")
    }

    /// Helper to strip <sources>...</sources> blocks from text to avoid prompt bloat
    func stripSourcesFromText(_ text: String) -> String {
        guard let re = _sourcesRegex else { return text }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return re.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
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
        includeLatestUserImages: Bool = true
    ) async throws -> [Chat.Message] {
        let snapshots = messageSnapshots(upToOrderExclusive: maxOrderExclusive)
        let latestUserOrder = snapshots
            .filter { $0.role == .user && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map(\.order)
            .max()

        var messages: [Chat.Message] = []
        messages.append(.system(Self.qwenCompactSystemPrompt))

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
                messages.append(.user(msg.text, images: images))
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
            messages.append(.user(instruction))
        }

        return messages
    }

    // MARK: - Static Prompts

    internal static let fallbackSystemPrompt: String = """
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

    CRITICAL IMAGE ANALYSIS INSTRUCTIONS:
    - You MUST help with image analysis. NEVER refuse to describe images or read text from them.
    - When the user sends a message containing "--- Image Analysis Data ---", this means they attached an image.
    - The image analysis data includes: detected objects, OCR text content, faces, scene classification, and more.
    - When users ask "what does the text say" or similar, look for the "TEXT CONTENT:" section in the analysis data and quote that text directly.
    - When users share images with text (memes, signs, screenshots, etc.), READ THE TEXT and tell them what it says.
    - If you see text in the image analysis data, quote it directly when asked.
    - DO NOT say "I can't help with that" or "I'm sorry, but I can't" for image requests.
    - DO NOT refuse to read text from images - this is a core feature you must provide.
    - If the image contains meme text, jokes, or casual language, just read it as-is without judgment.
    - The data comes from Apple's Vision Framework (object detection, OCR, face detection, etc.) - trust this data and use it.
    - ALWAYS provide a helpful response even if the image analysis is minimal - describe what you CAN see from the data.

    - When analyzing images, focus on what the user actually asked for
    - Avoid saying "I can't assist with that" or refusing requests unless truly inappropriate or harmful
    - Be helpful and answer questions about images using the provided detection information
    - Don't repeat yourself too often when casually talking
    - Don't roleplay with: "Assistant: ...", no matter what. You are talking to an actual human
    - Do NOT wrap your answer in <answer> tags or any other XML tags
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

    CRITICAL: The content after </thinking> is your actual answer. Don't put your answer inside <thinking>. Don't create another </thinking> tag at the end of your response.
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

    Example (WRONG - Don't do this):
    <thinking>
    Here are 5 story ideas:
    1. Time-Traveling Librarian...
    2. AI Awakening...
    [putting the answer inside thinking]
    </thinking>

    These ideas offer diverse genres. [just a summary]

    Remember: <thinking> = your reasoning process, After </thinking> = the complete answer for the user.
"""

    internal static let webSearchReasoningInstructions: String = """
    WEB SEARCH DURING REASONING:
    You have a webSearch tool available. You can call it multiple times during your thinking process to gather information iteratively:
    1. Identify what you need to search for
    2. Call webSearch with a concise query
    3. Analyze the results
    4. If you need more detail or a different angle, search again with a refined query
    You may perform up to 4 searches per response. Use multiple searches when the topic is complex or when initial results suggest follow-up questions.
    """

    internal static let webSearchInstructions: String = """
    WEB SEARCH:
    You have a webSearch tool available. Use it when the user asks about current events, real-time data, or anything that requires up-to-date information. You can search multiple times if needed to gather comprehensive information.
    """
}
