//
//  ChatViewModel+WebSearch.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import os

// swiftlint:disable force_try
private let _extractSourcesRegex = try! NSRegularExpression(
    pattern: #"<sources>(.*?)</sources>"#,
    options: [.dotMatchesLineSeparators, .caseInsensitive])
// swiftlint:enable force_try

extension ChatViewModel {

    // MARK: - Web Search Helpers

    internal static func timeSensitiveYearTokens(referenceDate: Date = Date()) -> [String] {
        let currentYear = Calendar.current.component(.year, from: referenceDate)
        return [
            String(currentYear - 1),
            String(currentYear),
            String(currentYear + 1)
        ]
    }

    // Cached regex patterns for stripping legacy search tags from stored messages.
    static let searchPatterns: [NSRegularExpression] = {
        let patterns = [
            #"<search>(.*?)</search>"#,
            #"<search\s+query\s*=\s*"(.*?)"\s*/>"#,
            #"\[search\](.*?)\[/search\]"#,
            #"(?im)^\s*SEARCH:\s*(.+?)\s*$"#,
            #"<tool:search>(.*?)</tool:search>"#
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: [.dotMatchesLineSeparators, .caseInsensitive]) }
    }()

    func requiresWebSearch(for userText: String) -> Bool {
        let q = userText.lowercased()

        if q.contains("please search") || q.contains("search for") || q.contains("look up") {
            return true
        }

        let timeSpecificKeywords = [
            "today", "yesterday", "this week", "this month", "this year", "now", "currently",
            "latest", "newest", "most recent", "just released", "breaking", "live",
            "this quarter"
        ]

        let contextualKeywords = [
            "version", "price", "cost", "release", "update", "announced", "news"
        ]

        let generalKnowledgePatterns = [
            "what does", "what is", "what are", "define", "meaning of", "stands for",
            "how to", "explain", "difference between", "advantages of", "disadvantages of"
        ]

        let mathPatterns = [
            "calculate", "plus", "minus", "times", "divided", "multiply", "add", "subtract",
            "=", "+", "-", "*", "/", "math", "equation"
        ]

        for pattern in generalKnowledgePatterns {
            if q.contains(pattern) {
                let hasTimeSpecific = timeSpecificKeywords.contains { q.contains($0) }
                if !hasTimeSpecific {
                    return false
                }
            }
        }

        for pattern in mathPatterns {
            if q.contains(pattern) {
                return false
            }
        }

        if timeSpecificKeywords.contains(where: { q.contains($0) }) ||
            Self.timeSensitiveYearTokens().contains(where: { q.contains($0) }) {
            return true
        }

        let hasContextualKeyword = contextualKeywords.contains { q.contains($0) }
        if hasContextualKeyword {
            let productKeywords = ["ios", "iphone", "ipad", "mac", "xcode", "swift", "apple"]
            let hasProductContext = productKeywords.contains { q.contains($0) }

            if hasProductContext && (q.contains("version") || q.contains("price") || q.contains("release")) {
                return true
            }
        }

        let eventKeywords = ["won", "winner", "champion", "results", "score"]
        if eventKeywords.contains(where: { q.contains($0) }) {
            return true
        }

        return false
    }

    // Extract the first <sources>...</sources> block content (without the tags)
    func extractSources(from text: String) -> String? {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = _extractSourcesRegex.firstMatch(in: text, options: [], range: range),
              match.numberOfRanges >= 2 else { return nil }
        let inner = match.range(at: 1)
        guard inner.location != NSNotFound else { return nil }
        return ns.substring(with: inner).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Strip legacy <search> tags from stored messages (backward compat).
    func stripSearchTags(_ text: String, preserveBoundaries: Bool = true) -> String {
        var working = text

        for re in Self.searchPatterns {
            let ns = working as NSString
            let range = NSRange(location: 0, length: ns.length)
            working = re.stringByReplacingMatches(in: working, options: [], range: range, withTemplate: "")
        }

        let collapsed = working.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        return preserveBoundaries ? collapsed : collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
