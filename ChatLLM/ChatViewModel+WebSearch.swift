//
//  ChatViewModel+WebSearch.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import os

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

    // Combined regex for stripping legacy search tags from stored messages (single-pass).
    // swiftlint:disable:next force_try
    static let combinedSearchPattern: NSRegularExpression = try! NSRegularExpression(
        pattern: [
            #"<search>(.*?)</search>"#,
            #"<search\s+query\s*=\s*"(.*?)"\s*/>"#,
            #"\[search\](.*?)\[/search\]"#,
            #"(?im)^\s*SEARCH:\s*(.+?)\s*$"#,
            #"<tool:search>(.*?)</tool:search>"#
        ].joined(separator: "|"),
        options: [.dotMatchesLineSeparators, .caseInsensitive]
    )

    func requiresWebSearch(for userText: String) -> Bool {
        let q = userText.lowercased()
        let containsPhrase: (String) -> Bool = { phrase in
            SharedRegexes.containsWholePhrase(phrase, in: q)
        }

        if containsPhrase("please search") || containsPhrase("search for") || containsPhrase("look up") {
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
            if containsPhrase(pattern) {
                let hasTimeSpecific = timeSpecificKeywords.contains { containsPhrase($0) }
                if !hasTimeSpecific {
                    return false
                }
            }
        }

        for pattern in mathPatterns {
            if containsPhrase(pattern) {
                return false
            }
        }

        if timeSpecificKeywords.contains(where: { containsPhrase($0) }) ||
            Self.timeSensitiveYearTokens().contains(where: { containsPhrase($0) }) {
            return true
        }

        let hasContextualKeyword = contextualKeywords.contains { containsPhrase($0) }
        if hasContextualKeyword {
            let productKeywords = ["ios", "iphone", "ipad", "mac", "xcode", "swift", "apple"]
            let hasProductContext = productKeywords.contains { containsPhrase($0) }

            if hasProductContext &&
                (containsPhrase("version") || containsPhrase("price") || containsPhrase("release")) {
                return true
            }
        }

        let eventKeywords = ["won", "winner", "champion", "results", "score"]
        if eventKeywords.contains(where: { containsPhrase($0) }) {
            return true
        }

        return false
    }

    // Extract the first <sources>...</sources> block content (without the tags)
    func extractSources(from text: String) -> String? {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = SharedRegexes.sourcesBlock.firstMatch(in: text, options: [], range: range),
              match.numberOfRanges >= 2 else { return nil }
        let inner = match.range(at: 1)
        guard inner.location != NSNotFound else { return nil }
        return ns.substring(with: inner).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Strip legacy <search> tags from stored messages (backward compat, single-pass).
    func stripSearchTags(_ text: String, preserveBoundaries: Bool = true) -> String {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        let working = Self.combinedSearchPattern.stringByReplacingMatches(
            in: text, options: [], range: range, withTemplate: "")
        let collapsed = working.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        return preserveBoundaries ? collapsed : collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
