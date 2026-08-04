//
//  SharedRegexes.swift
//  On-Device_LLM_Chat
//

import Foundation

// swiftlint:disable force_try
enum SharedRegexes {
    /// Matches `<sources>...</sources>` blocks (case-insensitive, multiline).
    static let sourcesBlock = try! NSRegularExpression(
        pattern: #"<sources>(.*?)</sources>"#,
        options: [.dotMatchesLineSeparators, .caseInsensitive])

    /// Collapses multiple blank lines into a single newline.
    static let multipleBlankLines = try! NSRegularExpression(
        pattern: #"[ \t]*\n[ \t]*\n+"#, options: [])

    /// Collapses runs of 2+ whitespace characters into a single space.
    static let excessiveWhitespace = try! NSRegularExpression(
        pattern: #"\s{2,}"#, options: [])

    /// Phrase patterns are drawn from fixed keyword lists and reused constantly
    /// (`requiresWebSearch` alone probes ~60 phrases per message), so the compiled
    /// regexes are cached instead of rebuilt on every call.
    private static let wholePhraseCacheLock = NSLock()
    nonisolated(unsafe) private static var wholePhraseCache: [String: NSRegularExpression] = [:]

    private static func wholePhraseRegex(for phrase: String) -> NSRegularExpression? {
        wholePhraseCacheLock.lock()
        defer { wholePhraseCacheLock.unlock() }

        if let cached = wholePhraseCache[phrase] {
            return cached
        }
        let escaped = NSRegularExpression.escapedPattern(for: phrase)
        let pattern = #"(?<![\p{L}\p{N}])\#(escaped)(?![\p{L}\p{N}])"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        wholePhraseCache[phrase] = regex
        return regex
    }

    static func containsWholePhrase(_ phrase: String, in text: String) -> Bool {
        guard let regex = wholePhraseRegex(for: phrase) else {
            return false
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }
}
// swiftlint:enable force_try
