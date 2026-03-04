//
//  LLMGenerator.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import FoundationModels

protocol LLMGenerator {
    func isAvailable() -> Bool
    func respond(to prompt: String, tools: [any Tool]) async throws -> String
    func streamResponse(to prompt: String, tools: [any Tool]) async throws -> AsyncThrowingStream<String, Error>
    func cleanupForRegeneration()
}

extension LLMGenerator {
    func respond(to prompt: String) async throws -> String {
        try await respond(to: prompt, tools: [])
    }
    func streamResponse(to prompt: String) async throws -> AsyncThrowingStream<String, Error> {
        try await streamResponse(to: prompt, tools: [])
    }
}

private func buildEnhancedPrompt(for original: String) -> String {
    guard original.contains("<thinking>"),
          !original.contains("Reasoning Mode Instructions:") else {
        return original
    }
    let preamble = """
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
    return preamble + original
}

private func makeSafetyBlockedError() -> NSError {
    NSError(domain: "OnDeviceLLMGenerator", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "Apple's safety system blocked this response. Try rephrasing your request."])
}

final class OnDeviceLLMGenerator: LLMGenerator {

    func isAvailable() -> Bool {
        SystemLanguageModel.default.availability == .available
    }

    private func isSafetyModerationError(_ error: Error) -> Bool {
        let message = (error as NSError).localizedDescription.lowercased()
        return message.contains("unsafe") ||
               message.contains("safety") ||
               message.contains("content filter") ||
               message.contains("violates")
    }

    func respond(to prompt: String, tools: [any Tool]) async throws -> String {
        guard isAvailable() else {
            throw NSError(domain: "OnDeviceLLMGenerator", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "On‑device model unavailable on this device."])
        }
        let session = tools.isEmpty ? LanguageModelSession() : LanguageModelSession(tools: tools)
        let effectivePrompt = buildEnhancedPrompt(for: prompt)
        do {
            let response = try await session.respond(to: effectivePrompt)
            return response.content
        } catch {
            if isSafetyModerationError(error) { throw makeSafetyBlockedError() }
            throw error
        }
    }

    func streamResponse(to prompt: String, tools: [any Tool]) async throws -> AsyncThrowingStream<String, Error> {
        guard isAvailable() else {
            throw NSError(domain: "OnDeviceLLMGenerator", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "On‑device model unavailable on this device."])
        }
        let session = tools.isEmpty ? LanguageModelSession() : LanguageModelSession(tools: tools)
        let effectivePrompt = buildEnhancedPrompt(for: prompt)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var lastYieldedLength = 0
                    var iterator = session.streamResponse(to: effectivePrompt).makeAsyncIterator()
                    while let partial = try await iterator.next() {
                        if Task.isCancelled { continuation.finish(); return }
                        let newContent = partial.content
                        if newContent.count > lastYieldedLength {
                            let delta = String(newContent.dropFirst(lastYieldedLength))
                            if !delta.isEmpty {
                                continuation.yield(delta)
                                lastYieldedLength = newContent.count
                            }
                        }
                    }
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

    func cleanupForRegeneration() {
        // Session is created fresh per call; no shared state to clean up.
    }
}
