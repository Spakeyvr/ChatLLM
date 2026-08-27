//
//  ChatViewModel+TextCleaning.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation

// MARK: - Cached regexes (compiled once at startup)
private let _answerTagRegex = try? NSRegularExpression(pattern: #"<answer>([\s\S]*?)</answer>"#, options: [.caseInsensitive])
private let _leadingMarkdownRegex = try? NSRegularExpression(pattern: #"(?s)^\s*(\*\*|\*|[-]{2,}|#{1,6})\s*\n+"#, options: [])
private let _typoFixRegexes: [(NSRegularExpression, String)] = [
    (#"(?i)\bios\b"#, "iOS"),
    (#"(?i)(?<![\/\.])\bsecurityreleases\b"#, "security releases"),
    (#"(?i)\bversionhistory\b"#, "version history"),
].compactMap { (p, r) in (try? NSRegularExpression(pattern: p, options: [])).map { ($0, r) } }

extension ChatViewModel {

    // MARK: - Text Cleaning

    /// Cleans glitched/repetitive text patterns from LLM output
    static func cleanGlitchedText(_ text: String) -> String {
        // IMPORTANT: Don't clean very short responses (like "29." or "Yes.")
        // as the cleaning logic can accidentally mangle them
        guard text.count > 50 else { return text }

        var cleaned = text

        // Strip <answer>...</answer> tags, keeping inner content
        if cleaned.contains("<answer>") || cleaned.contains("<Answer>") || cleaned.contains("<ANSWER>") {
            if let re = _answerTagRegex {
                let ns = cleaned as NSString
                let range = NSRange(location: 0, length: ns.length)
                let stripped = re.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "$1")
                if stripped != cleaned {
                    cleaned = stripped.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }

        // Remove stray leading Markdown-only lines like "**", "*", "---", or "###"
        if let re = _leadingMarkdownRegex {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            let result = re.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
            if result != cleaned {
                cleaned = result
            }
        }

        // Conservative normalizations for frequent LLM typos; skips URLs.
        for (regex, replacement) in _typoFixRegexes {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            let result = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: replacement)
            if result != cleaned {
                cleaned = result
            }
        }

        // Strip Unicode whitespace artifacts that can appear in web/tool-injected content
        let normalized = cleaned
            .replacingOccurrences(of: "\u{00A0}", with: " ")   // NBSP -> space
            .replacingOccurrences(of: "\u{200B}", with: "")    // Zero-width space
            .replacingOccurrences(of: "\u{00AD}", with: "")    // Soft hyphen

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
