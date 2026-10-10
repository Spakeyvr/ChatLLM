//
//  MLXPromptPrefixCache.swift
//  ChatLLM
//
//  Persists the KV cache of a prompt's system turn so a fresh session only
//  prefills what follows it. The system prompt (plus tool definitions when web
//  search is on) is several hundred tokens and identical across new chats and
//  across Qwen's per-iteration tool-loop sessions, so restoring it from disk
//  removes most of their prefill without keeping anything resident.
//

import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import OSLog
import Tokenizers

/// A restored or freshly prefilled prefix cache, plus what it cost to obtain.
nonisolated struct MLXPromptPrefix: @unchecked Sendable {
    /// Per-layer cache holding the system turn. Owned by the session it is
    /// handed to; it is never shared between sessions.
    let cache: [KVCache]
    /// The rest of the rendered prompt, to be prefilled on top of `cache`.
    let remainingInput: LMInput
    /// Tokens restored from disk without running the model.
    let restoredTokenCount: Int
    /// Tokens prefilled now because no saved prefix matched.
    let prefilledTokenCount: Int
    let prefillTime: TimeInterval
}

nonisolated struct MLXPromptPrefixCache: Sendable {
    /// Each entry is roughly 40-60 MB on disk for Qwen 3.5 4B; four cover the
    /// variants a user moves between (web search on or off, with or without
    /// the date line) for the loaded model.
    static let maximumEntries = 4
    /// Below this, restoring from disk saves too little to be worth a file.
    static let minimumPrefixTokens = 64
    private static let prefixTokenCountMetadataKey = "chatllm.prefixTokenCount"

    let directory: URL
    private let logger: Logger

    init(directory: URL = Self.defaultDirectory, subsystem: String) {
        self.directory = directory
        self.logger = Logger(subsystem: subsystem, category: "MLXPromptPrefixCache")
    }

    static var defaultDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLXPromptPrefixCache", isDirectory: true)
    }

    // MARK: - Eligibility

    /// Text-only prompts that open with a system turn. Media turns use a
    /// single-pass prefill whose positions depend on the whole sequence.
    static func isEligible(messages: [Chat.Message]) -> Bool {
        guard messages.count >= 2, messages[0].role == .system else { return false }
        return messages.allSatisfy { $0.images.isEmpty && $0.videos.isEmpty }
    }

    /// Length of the shared token prefix between the full prompt and renders
    /// of the stable system prompt followed by different probe messages.
    /// Taking the shortest match stops the prefix before any token that
    /// depends on the conversation, including the first token of the next
    /// message. Returns nil when the prefix is too short to be worth a file or
    /// leaves nothing to prefill.
    static func prefixTokenCount(full: [Int32], probes: [[Int32]]) -> Int? {
        guard !probes.isEmpty else { return nil }
        let count = probes.map { probe in
            zip(full, probe).prefix { $0 == $1 }.count
        }.min() ?? 0
        guard count >= minimumPrefixTokens, count < full.count else { return nil }
        return count
    }

    /// Identifies a prefix by model, exact token sequence, and the cache layout
    /// the generation parameters produce, so a restored cache always matches
    /// what `newCache(parameters:)` would have built.
    static func entryKey(modelID: String, prefixTokens: ArraySlice<Int32>, parameters: GenerateParameters) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(modelID.utf8))
        hasher.update(data: Data("|maxKV=\(parameters.maxKVSize ?? -1)".utf8))
        hasher.update(data: Data("|compression=\(String(describing: parameters.resolvedCacheCompression))|".utf8))
        Array(prefixTokens).withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Lookup

    /// Returns a cache holding the stable system turn of `messages` and the
    /// remaining prompt input, restoring the cache from disk when a matching
    /// entry exists and otherwise prefilling and saving it. Returns nil when
    /// the request is not eligible or anything fails, in which case the caller
    /// prefills the full prompt as before.
    ///
    /// - Parameter stableSystemPrompt: the leading part of the system message
    ///   that does not change between requests (it excludes the date line).
    ///   The cached prefix ends where the rendered prompt departs from it.
    func prefix(
        container: ModelContainer,
        modelID: String,
        messages: [Chat.Message],
        stableSystemPrompt: String?,
        tools: [ToolSpec]?,
        processing: UserInput.Processing,
        additionalContext: [String: any Sendable]?,
        parameters: GenerateParameters
    ) async -> MLXPromptPrefix? {
        guard Self.isEligible(messages: messages) else { return nil }
        let stableSystemPrompt = stableSystemPrompt ?? messages[0].content
        guard messages[0].content.hasPrefix(stableSystemPrompt) else { return nil }
        let promptMessages = SendableMessages(messages: messages)
        let tools = tools.map(SendableTools.init)
        let cacheDirectory = directory
        let logger = logger

        do {
            return try await container.perform { context in
                func render(_ chat: [Chat.Message]) async throws -> LMInput {
                    try await context.processor.prepare(
                        input: UserInput(
                            chat: chat,
                            processing: processing,
                            tools: tools?.specs,
                            additionalContext: additionalContext
                        )
                    )
                }
                func tokens(_ input: LMInput) -> [Int32] {
                    input.text.tokens.asType(.int32).reshaped(-1).asArray(Int32.self)
                }

                let fullInput = try await render(promptMessages.messages)
                let fullTokens = tokens(fullInput)
                var probes: [[Int32]] = []
                for probe in ["A", "B"] {
                    probes.append(tokens(try await render([.system(stableSystemPrompt), .user(probe)])))
                }
                guard let prefixCount = Self.prefixTokenCount(full: fullTokens, probes: probes) else {
                    logger.notice("MLX prompt prefix skipped: no stable prefix of useful length")
                    return nil
                }

                let key = Self.entryKey(
                    modelID: modelID,
                    prefixTokens: fullTokens[..<prefixCount],
                    parameters: parameters
                )
                let url = cacheDirectory.appendingPathComponent("\(key).safetensors")
                let expectedLayout = context.model.newCache(parameters: parameters)
                let remainingInput = LMInput(
                    text: LMInput.Text(tokens: fullInput.text.tokens[.ellipsis, prefixCount...])
                )

                if let restored = Self.restore(
                    from: url, prefixCount: prefixCount, expectedLayout: expectedLayout, logger: logger
                ) {
                    logger.notice("MLX prompt prefix restored: tokens=\(prefixCount, privacy: .public)")
                    return MLXPromptPrefix(
                        cache: restored,
                        remainingInput: remainingInput,
                        restoredTokenCount: prefixCount,
                        prefilledTokenCount: 0,
                        prefillTime: 0
                    )
                }

                let startedAt = Date()
                let cache = expectedLayout
                let prefixInput = LMInput(
                    text: LMInput.Text(tokens: fullInput.text.tokens[.ellipsis, ..<prefixCount])
                )
                switch try context.model.prepare(
                    prefixInput, cache: cache, windowSize: parameters.prefillStepSize
                ) {
                case .tokens(let remaining):
                    // LLM `prepare` leaves its last chunk for the token
                    // iterator; run it so the cache covers the whole prefix.
                    if remaining.tokens.size > 0 {
                        _ = context.model(remaining[text: .newAxis], cache: cache, state: nil)
                    }
                case .logits:
                    break
                }
                eval(cache)
                let prefillTime = Date().timeIntervalSince(startedAt)

                Self.save(cache, to: url, prefixCount: prefixCount, in: cacheDirectory, logger: logger)
                logger.notice(
                    "MLX prompt prefix prefilled: tokens=\(prefixCount, privacy: .public) seconds=\(String(format: "%.3f", prefillTime), privacy: .public)"
                )
                return MLXPromptPrefix(
                    cache: cache,
                    remainingInput: remainingInput,
                    restoredTokenCount: 0,
                    prefilledTokenCount: prefixCount,
                    prefillTime: prefillTime
                )
            }
        } catch {
            logger.error("MLX prompt prefix unavailable: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    // Text-only messages and JSON-like tool specs are only read inside the
    // model container's serialized `perform`.
    private struct SendableMessages: @unchecked Sendable {
        let messages: [Chat.Message]
    }

    private struct SendableTools: @unchecked Sendable {
        let specs: [ToolSpec]
    }

    // MARK: - Storage

    private static func restore(
        from url: URL,
        prefixCount: Int,
        expectedLayout: [KVCache],
        logger: Logger
    ) -> [KVCache]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let (cache, metadata) = try loadPromptCache(url: url)
            let layoutMatches = cache.count == expectedLayout.count
                && zip(cache, expectedLayout).allSatisfy { type(of: $0) == type(of: $1) }
            guard layoutMatches,
                  metadata[prefixTokenCountMetadataKey] == String(prefixCount) else {
                logger.notice("MLX prompt prefix discarded: saved layout no longer matches")
                try? FileManager.default.removeItem(at: url)
                return nil
            }
            // Keeps least-recently-used eviction honest.
            try? FileManager.default.setAttributes(
                [.modificationDate: Date()], ofItemAtPath: url.path
            )
            return cache
        } catch {
            logger.error("MLX prompt prefix restore failed: \(error.localizedDescription, privacy: .public)")
            try? FileManager.default.removeItem(at: url)
            return nil
        }
    }

    private static func save(
        _ cache: [KVCache],
        to url: URL,
        prefixCount: Int,
        in directory: URL,
        logger: Logger
    ) {
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            // Write beside the destination and move into place so a reader never
            // sees a partially written entry.
            let temporaryURL = directory.appendingPathComponent("\(UUID().uuidString).partial.safetensors")
            try savePromptCache(
                url: temporaryURL,
                cache: cache,
                metadata: [prefixTokenCountMetadataKey: String(prefixCount)]
            )
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
            try fileManager.moveItem(at: temporaryURL, to: url)
            evictOldEntries(in: directory)
        } catch {
            logger.error("MLX prompt prefix save failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private static func evictOldEntries(in directory: URL) {
        let fileManager = FileManager.default
        guard let entries = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        let sorted = entries.sorted { lhs, rhs in
            let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lhsDate > rhsDate
        }
        for entry in sorted.dropFirst(maximumEntries) {
            try? fileManager.removeItem(at: entry)
        }
    }
}
