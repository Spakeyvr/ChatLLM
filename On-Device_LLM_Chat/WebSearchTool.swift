//
//  WebSearchTool.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 02/28/26.
//

import Foundation
import FoundationModels
import MLXLMCommon
import Tokenizers
import OSLog

typealias FoundationModelTool = FoundationModels.Tool
typealias MLXToolSpec = Tokenizers.ToolSpec
typealias MLXToolCall = MLXLMCommon.ToolCall

// MARK: - Search Invocation (shared between WebSearchTool & Message)

struct SearchInvocation: Codable, Sendable, Identifiable {
    var id: UUID
    var query: String
    var results: String

    nonisolated init(id: UUID = UUID(), query: String, results: String) {
        self.id = id
        self.query = query
        self.results = results
    }
}

final class AppWebSearchToolBridge: @unchecked Sendable {

    static let toolName = "webSearch"
    static let toolDescription = "Search the web for current information. Use for breaking news, today's events, real-time data, current prices or versions, or when the user explicitly asks to search."

    struct MLXArguments: Codable, Sendable {
        var query: String
    }

    private let searchService: TavilySearchService
    private static let maxInvocations = 4
    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "WebSearchTool")

    // Thread-safe storage for captured results (written in `call`, read on @MainActor).
    private let lock = NSLock()
    private var _invocations: [SearchInvocation] = []

    // Backward-compat accessors (return last invocation)
    var lastSearchQuery: String? {
        lock.withLock { _invocations.last?.query }
    }

    var lastSearchResults: String? {
        lock.withLock { _invocations.last?.results }
    }

    var allInvocations: [SearchInvocation] {
        lock.withLock { _invocations }
    }

    init(searchService: TavilySearchService) {
        self.searchService = searchService
    }

    var foundationModelTool: WebSearchTool {
        WebSearchTool(bridge: self)
    }

    var mlxToolSpec: MLXToolSpec {
        makeMLXTool().schema
    }

    func dispatchMLXToolCall(_ toolCall: MLXToolCall) async throws -> String {
        let rawArguments = String(describing: toolCall.function.arguments)
        logger.notice(
            "MLX tool call received: name=\(toolCall.function.name, privacy: .public) arguments=\(rawArguments, privacy: .public)"
        )
        do {
            let result = try await toolCall.execute(with: makeMLXTool())
            logger.notice(
                "MLX tool call executed: name=\(toolCall.function.name, privacy: .public) result_chars=\(result.count, privacy: .public)"
            )
            return result
        } catch {
            logger.error(
                "MLX tool call execution failed: name=\(toolCall.function.name, privacy: .public) arguments=\(rawArguments, privacy: .public) error=\(error.localizedDescription, privacy: .public)"
            )
            throw error
        }
    }

    func executeSearch(query: String) async throws -> String {
        let count = lock.withLock { _invocations.count }
        if count >= Self.maxInvocations {
            logger.warning("Search skipped: invocation limit reached (\(count, privacy: .public))")
            return "[Search limit reached — maximum \(Self.maxInvocations) searches per response.]"
        }

        logger.notice("Search started: query=\(query, privacy: .public) prior_invocations=\(count, privacy: .public)")
        let start = Date()
        let results = try await searchService.search(query: query, maxResults: 3)
        let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)
        let truncated = results.count > 3500
            ? Self.truncateAtWordBoundary(results, maxChars: 3500)
            : results

        let invocation = SearchInvocation(query: query, results: truncated)
        lock.withLock {
            _invocations.append(invocation)
        }

        logger.notice(
            "Search finished: query=\(query, privacy: .public) chars=\(truncated.count, privacy: .public) elapsed_ms=\(elapsedMs, privacy: .public)"
        )

        return truncated
    }

    private func makeMLXTool() -> MLXLMCommon.Tool<MLXArguments, String> {
        MLXLMCommon.Tool(
            name: Self.toolName,
            description: Self.toolDescription,
            parameters: [
                .required("query", type: .string, description: "A concise search query (3-7 words)")
            ]
        ) { [self] arguments in
            try await executeSearch(query: arguments.query)
        }
    }

    // MARK: - Helpers

    private static func truncateAtWordBoundary(_ text: String, maxChars: Int) -> String {
        guard text.count > maxChars else { return text }
        let truncated = String(text.prefix(maxChars))
        if let lastSpace = truncated.lastIndex(of: " ") {
            return String(truncated[..<lastSpace]) + "...\n\n[... results truncated to fit context ...]"
        }
        return truncated + "...\n\n[... results truncated to fit context ...]"
    }
}

/// Native FoundationModels `Tool` that wraps the shared web-search bridge.
/// The framework executes `call(arguments:)` and feeds the result back into
/// the generation — all within a single streaming session.

final class WebSearchTool: FoundationModelTool, @unchecked Sendable {

    let name = AppWebSearchToolBridge.toolName
    let description = AppWebSearchToolBridge.toolDescription

    @Generable
    struct Arguments: Sendable {
        @Guide(description: "A concise search query (3-7 words)")
        var query: String
    }

    private let bridge: AppWebSearchToolBridge

    init(bridge: AppWebSearchToolBridge) {
        self.bridge = bridge
    }

    func call(arguments: Arguments) async throws -> String {
        try await bridge.executeSearch(query: arguments.query)
    }
}
