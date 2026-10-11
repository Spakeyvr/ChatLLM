//
//  LLMGenerator.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import FoundationModels
import OSLog

/// Keep trusted instructions and conversation roles separate from user text.
/// Serializing roles into the prompt teaches the model to echo that serialization.
struct LLMRequest: Sendable, Equatable {
    struct Turn: Sendable, Equatable {
        enum Role: Sendable { case user, assistant }
        let role: Role
        let content: String
        /// Images attached to this turn and sent to the model as pixels.
        var imageURLs: [URL] = []
    }

    var instructions: String = ""
    var history: [Turn] = []
    var prompt: String
    /// Images attached to `prompt` and sent to the model as pixels.
    var imageURLs: [URL] = []
}

protocol LLMGenerator {
    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error>

    /// Streams like `streamResponse(to:tools:)` and, once the response
    /// completes, reports the token counts the model measured for it.
    func streamResponse(
        to request: LLMRequest,
        tools: [any FoundationModelTool],
        onMetrics: @escaping @Sendable (GenerationMetrics) -> Void
    ) async throws -> AsyncThrowingStream<String, Error>
}

extension LLMGenerator {
    /// Generators that cannot measure usage stream without reporting it.
    func streamResponse(
        to request: LLMRequest,
        tools: [any FoundationModelTool],
        onMetrics: @escaping @Sendable (GenerationMetrics) -> Void
    ) async throws -> AsyncThrowingStream<String, Error> {
        try await streamResponse(to: request, tools: tools)
    }
}

extension LLMRequest {
    static let reasoningInstructions = """
    Reasoning Mode Instructions:
    - Think step-by-step only inside <thinking> ... </thinking>.
    - If you encounter typos or unclear requests, interpret the user's intent.
    - Don't refuse to answer due to minor errors - work with what the user likely meant.
    - Keep the content inside <thinking> concise, factual, and directly relevant.
    - After </thinking>, write the final answer clearly in plain text without referencing the tags.
    - Do NOT wrap your answer in <answer> tags or any other XML tags. Write plain text after </thinking>.
    - Prefer bullet points or short paragraphs; be direct and avoid repetition.
    - If code is needed, use idiomatic Swift for Apple platforms and include only necessary parts.
    - If unsure, state assumptions briefly and proceed.
    - Make sure your final answer is formal and complete.

    """
}

nonisolated private func makeSafetyBlockedError() -> NSError {
    NSError(domain: "OnDeviceLLMGenerator", code: 3,
            userInfo: [NSLocalizedDescriptionKey: String(localized: "Apple's safety system blocked this response. Try rephrasing your request.", bundle: .appLocalized)])
}

final class OnDeviceLLMGenerator: LLMGenerator {

    nonisolated private static let logger = Logger(subsystem: "Nevio.ChatLLM", category: "OnDeviceLLMGenerator")

    nonisolated private static let safetyKeywords = ["unsafe", "safety", "content filter", "violates"]

    private func isAvailable() -> Bool {
        SystemLanguageModel.default.availability == .available
    }

    nonisolated private func isSafetyModerationError(_ error: Error) -> Bool {
        let message = (error as NSError).localizedDescription.lowercased()
        return Self.safetyKeywords.contains(where: message.contains)
    }

    static func makeSession(for request: LLMRequest, tools: [any FoundationModelTool]) -> LanguageModelSession {
        let session = LanguageModelSession(tools: tools, instructions: request.instructions)
        guard !request.history.isEmpty else { return session }

        // Retain the framework-generated instructions entry, including tool schemas.
        var imageNumber = 0
        let history: [Transcript.Entry] = request.history.map { turn in
            let images: [Transcript.Segment] = turn.imageURLs.map { url in
                imageNumber += 1
                return .attachment(.init(
                    content: .image(.init(imageURL: url)),
                    label: imageLabel(imageNumber)
                ))
            }
            let segments = images + [.text(.init(content: turn.content))]
            switch turn.role {
            case .user:
                return .prompt(.init(segments: segments))
            case .assistant:
                return .response(.init(assetIDs: [], segments: segments))
            }
        }
        let transcript = Transcript(entries: Array(session.transcript) + history)
        return LanguageModelSession(tools: tools, transcript: transcript)
    }

    /// Images go before the text so the question reads as being about them.
    /// Labels continue the numbering of images earlier in the conversation.
    static func makePrompt(for request: LLMRequest) -> Prompt {
        let earlierImageCount = request.history.reduce(0) { $0 + $1.imageURLs.count }
        return Prompt {
            request.imageURLs.enumerated().map { index, url in
                Attachment(imageURL: url).label(imageLabel(earlierImageCount + index + 1))
            }
            request.prompt
        }
    }

    /// Numbered in conversation order, so "image-1" is the oldest image sent.
    nonisolated static func imageLabel(_ number: Int) -> String {
        "image-\(number)"
    }

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        try await streamResponse(to: request, tools: tools, onMetrics: { _ in })
    }

    func streamResponse(
        to request: LLMRequest,
        tools: [any FoundationModelTool],
        onMetrics: @escaping @Sendable (GenerationMetrics) -> Void
    ) async throws -> AsyncThrowingStream<String, Error> {
        guard isAvailable() else {
            throw NSError(domain: "OnDeviceLLMGenerator", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: String(localized: "On‑device model unavailable on this device.", bundle: .appLocalized)])
        }
        let session = Self.makeSession(for: request, tools: tools)
        let prompt = Self.makePrompt(for: request)

        return TaskBackedAsyncThrowingStream.make { continuation in
            Task {
                do {
                    let clock = ContinuousClock()
                    let startedAt = clock.now
                    var firstPartialAt: ContinuousClock.Instant?
                    var lastYieldedUTF8Count = 0
                    var iterator = session.streamResponse(to: prompt).makeAsyncIterator()
                    while let partial = try await iterator.next() {
                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }
                        if firstPartialAt == nil { firstPartialAt = clock.now }
                        let newContent = partial.content
                        if let delta = Self.incrementalDelta(
                            from: newContent,
                            afterUTF8Count: &lastYieldedUTF8Count
                        ) {
                            continuation.yield(delta)
                        }
                    }
                    let usage = session.usage
                    let metrics = Self.generationMetrics(
                        inputTokens: usage.input.totalTokenCount,
                        cachedInputTokens: usage.input.cachedTokenCount,
                        outputTokens: usage.output.totalTokenCount,
                        timeToFirstToken: firstPartialAt.map { Self.seconds(startedAt.duration(to: $0)) },
                        totalTime: Self.seconds(startedAt.duration(to: clock.now))
                    )
                    Self.logger.notice(
                        "Foundation Models usage: input=\(metrics.promptTokenCount, privacy: .public) cached=\(metrics.cachedPromptTokenCount, privacy: .public) output=\(metrics.outputTokenCount, privacy: .public) images=\(request.imageURLs.count, privacy: .public)"
                    )
                    onMetrics(metrics)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    if self.isSafetyModerationError(error) {
                        continuation.finish(throwing: makeSafetyBlockedError())
                        return
                    }
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Maps the session's measured usage onto the stats MLX reports. The
    /// framework exposes no prefill timing, so time to first token stands in
    /// for prompt time, and decoding is the rest of the response.
    nonisolated static func generationMetrics(
        inputTokens: Int,
        cachedInputTokens: Int,
        outputTokens: Int,
        timeToFirstToken: TimeInterval?,
        totalTime: TimeInterval
    ) -> GenerationMetrics {
        let promptTime = timeToFirstToken ?? totalTime
        return GenerationMetrics(
            promptTokenCount: inputTokens,
            cachedPromptTokenCount: cachedInputTokens,
            promptTime: promptTime,
            outputTokenCount: outputTokens,
            decodeTime: max(0, totalTime - promptTime),
            timeToFirstToken: timeToFirstToken
        )
    }

    nonisolated private static func seconds(_ duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds) + TimeInterval(components.attoseconds) / 1e18
    }

    nonisolated static func incrementalDelta(
        from accumulatedContent: String,
        afterUTF8Count lastYieldedUTF8Count: inout Int
    ) -> String? {
        let utf8 = accumulatedContent.utf8
        guard utf8.count > lastYieldedUTF8Count else { return nil }

        let deltaStart = utf8.index(utf8.startIndex, offsetBy: lastYieldedUTF8Count)
        let delta = String(decoding: utf8[deltaStart...], as: UTF8.self)
        lastYieldedUTF8Count = utf8.count
        return delta.isEmpty ? nil : delta
    }
}
