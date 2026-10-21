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
