//
//  LLMGenerator.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import FoundationModels

/// Keep trusted instructions and conversation roles separate from user text.
/// Serializing roles into the prompt teaches the model to echo that serialization.
struct LLMRequest: Sendable, Equatable {
    struct Turn: Sendable, Equatable {
        enum Role: Sendable { case user, assistant }
        let role: Role
        let content: String
    }

    var instructions: String = ""
    var history: [Turn] = []
    var prompt: String
}

protocol LLMGenerator {
    func isAvailable() -> Bool
    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String
    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error>
}

extension LLMGenerator {
    func respond(to prompt: String) async throws -> String {
        try await respond(to: LLMRequest(prompt: prompt), tools: [])
    }
    func streamResponse(to prompt: String) async throws -> AsyncThrowingStream<String, Error> {
        try await streamResponse(to: LLMRequest(prompt: prompt), tools: [])
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
            userInfo: [NSLocalizedDescriptionKey: "Apple's safety system blocked this response. Try rephrasing your request."])
}

final class OnDeviceLLMGenerator: LLMGenerator {

    nonisolated private static let safetyKeywords = ["unsafe", "safety", "content filter", "violates"]

    func isAvailable() -> Bool {
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
        let history: [Transcript.Entry] = request.history.map { turn in
            let segments: [Transcript.Segment] = [.text(.init(content: turn.content))]
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

    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String {
        guard isAvailable() else {
            throw NSError(domain: "OnDeviceLLMGenerator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "On‑device model unavailable on this device."])
        }
        let session = Self.makeSession(for: request, tools: tools)
        do {
            let response = try await session.respond(to: request.prompt)
            return response.content
        } catch {
            if isSafetyModerationError(error) { throw makeSafetyBlockedError() }
            throw error
        }
    }

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        guard isAvailable() else {
            throw NSError(domain: "OnDeviceLLMGenerator", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "On‑device model unavailable on this device."])
        }
        let session = Self.makeSession(for: request, tools: tools)

        return TaskBackedAsyncThrowingStream.make { continuation in
            Task {
                do {
                    var lastYieldedUTF8Count = 0
                    var iterator = session.streamResponse(to: request.prompt).makeAsyncIterator()
                    while let partial = try await iterator.next() {
                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }
                        let newContent = partial.content
                        if let delta = Self.incrementalDelta(
                            from: newContent,
                            afterUTF8Count: &lastYieldedUTF8Count
                        ) {
                            continuation.yield(delta)
                        }
                    }
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
