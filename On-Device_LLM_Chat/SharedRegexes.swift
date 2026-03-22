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
}
// swiftlint:enable force_try
