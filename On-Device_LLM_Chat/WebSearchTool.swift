//
//  WebSearchTool.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 02/28/26.
//

import Foundation
import FoundationModels

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

/// Native FoundationModels `Tool` that wraps `TavilySearchService`.
/// The on-device model autonomously decides when to invoke this tool;
/// the framework executes `call(arguments:)` and feeds the result
/// back into the generation — all within a single streaming session.

final class WebSearchTool: Tool, @unchecked Sendable {

    let name = "webSearch"
    let description = "Search the web for current information. Use for breaking news, today's events, real-time data, current prices/versions, or when the user explicitly asks to search."

    @Generable
    struct Arguments: Sendable {
        @Guide(description: "A concise search query (3-7 words)")
        var query: String
    }

    private let searchService: TavilySearchService
    private static let maxInvocations = 4

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

    func call(arguments: Arguments) async throws -> String {
        let count = lock.withLock { _invocations.count }
        if count >= Self.maxInvocations {
            return "[Search limit reached — maximum \(Self.maxInvocations) searches per response.]"
        }

        let results = try await searchService.search(query: arguments.query, maxResults: 3)
        let truncated = results.count > 3500
            ? Self.truncateAtWordBoundary(results, maxChars: 3500)
            : results

        let invocation = SearchInvocation(query: arguments.query, results: truncated)
        lock.withLock {
            _invocations.append(invocation)
        }

        return truncated
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

