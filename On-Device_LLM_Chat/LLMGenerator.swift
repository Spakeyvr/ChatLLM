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

// Shared prompt enhancement — avoids duplicating the same logic in OnDeviceLLMGenerator and MockGenerator.
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

final class OnDeviceLLMGenerator: LLMGenerator {

    func isAvailable() -> Bool {
        SystemLanguageModel.default.availability == .available
    }

    private func isSafetyModerationError(_ error: Error) -> Bool {
        let ns = error as NSError
        let message = ns.localizedDescription.lowercased()
        if message.contains("unsafe") ||
           message.contains("safety") ||
           message.contains("content filter") ||
           message.contains("violates") {
            return true
        }
        if ns.domain.contains("FoundationModels") || ns.domain.contains("LanguageModel") {
            return true
        }
        return false
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
            if isSafetyModerationError(error) { return "" }
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
                        continuation.finish()
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

final class MockGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to prompt: String, tools: [any Tool]) async throws -> String {
        let effectivePrompt = buildEnhancedPrompt(for: prompt)
        if effectivePrompt.contains("<thinking>") {
            return """
            <thinking>
            1) Identify the user's goal and constraints.
            2) Outline a concise plan with the minimum steps to succeed.
            3) Verify assumptions and edge cases.
            </thinking>

            Answer:
            Mock response to: \(String(prompt.prefix(60)))
            """
        }
        return "Mock response to: \(String(prompt.prefix(60)))"
    }

    func streamResponse(to prompt: String, tools: [any Tool]) async throws -> AsyncThrowingStream<String, Error> {
        let effectivePrompt = buildEnhancedPrompt(for: prompt)
        let isReasoningMode = effectivePrompt.contains("<thinking>")

        let parts: [String]
        if isReasoningMode {
            parts = [
                "<thinking>\n",
                "1) Identify the user's goal and constraints.\n",
                "2) Outline a concise plan with the minimum steps to succeed.\n",
                "3) Verify assumptions and edge cases.\n",
                "</thinking>\n\n",
                "Answer:\n",
                "Mock ", "streamed ", "response ", "to: ", String(prompt.prefix(40))
            ]
        } else {
            parts = ["Mock ", "streamed ", "response ", "to: ", String(prompt.prefix(40))]
        }

        var iterator = parts.makeIterator()
        return AsyncThrowingStream { continuation in
            Task {
                while let next = iterator.next() {
                    if Task.isCancelled { continuation.finish(); return }
                    try? await Task.sleep(nanoseconds: 150_000_000)
                    continuation.yield(next)
                }
                continuation.finish()
            }
        }
    }

    func cleanupForRegeneration() {}
}
