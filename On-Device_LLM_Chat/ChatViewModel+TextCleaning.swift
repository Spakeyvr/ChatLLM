//
//  ChatViewModel+TextCleaning.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation

// MARK: - Cached regexes (compiled once at startup)
private let _answerTagRegex = try? NSRegularExpression(pattern: #"<answer>([\s\S]*?)</answer>"#, options: [.caseInsensitive])
private let _leadingMarkdownRegex = try? NSRegularExpression(pattern: #"(?s)^\s*(\*\*|\*|[-]{2,}|#{1,6})\s*\n+"#, options: [])
private let _wordBoundaryRegexes: [(NSRegularExpression, String)] = [
    ("to([A-Z])", "to $1"), ("be([A-Z])", "be $1"), ("is([A-Z])", "is $1"),
    ("as([A-Z])", "as $1"), ("in([A-Z])", "in $1"), ("of([A-Z])", "of $1"),
].compactMap { (p, r) in (try? NSRegularExpression(pattern: p, options: [])).map { ($0, r) } }
private let _typoFixRegexes: [(NSRegularExpression, String)] = [
    (#"(?i)\bios\b"#, "iOS"),
    (#"(?i)(?<![\/\.])\bsecurityreleases\b"#, "security releases"),
    (#"(?i)\bversionhistory\b"#, "version history"),
].compactMap { (p, r) in (try? NSRegularExpression(pattern: p, options: [])).map { ($0, r) } }

extension ChatViewModel {

    // MARK: - Text Cleaning

    /// Cleans glitched/repetitive text patterns from LLM output
    func cleanGlitchedText(_ text: String) -> String {
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
                print("🧹 Removed leading Markdown artifact")
                cleaned = result
            }
        }

        // Fix missing-space glitches like "appears toThe"
        for (regex, replacement) in _wordBoundaryRegexes {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            let result = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: replacement)
            if result != cleaned {
                print("🧹 Fixed word boundary glitch")
                cleaned = result
            }
        }

        // NOTE: Removed risky connector-word spacing fix that caused inside-word splits like "F or".

        // Conservative normalizations for frequent LLM typos; skips URLs.
        for (regex, replacement) in _typoFixRegexes {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            let result = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: replacement)
            if result != cleaned {
                print("🧹 Normalized common typo pattern")
                cleaned = result
            }
        }

        // Detect exact character-level repetitions like "hello worldhello world".
        // Only scan the tail (newly added text) — any earlier repetitions were caught on prior tokens.
        for windowSize in stride(from: min(50, cleaned.count / 2), through: 5, by: -1) {
            let tailLength = min(cleaned.count, windowSize * 2 + 100)
            let tailOffset = cleaned.count - tailLength
            var prevIdx = cleaned.index(cleaned.startIndex, offsetBy: tailOffset)
            var nextIdx = cleaned.index(prevIdx, offsetBy: 1)
            let limit = tailLength - windowSize - 1
            guard limit > 0 else { continue }
            for _ in 0..<limit {
                guard let prevEnd = cleaned.index(prevIdx, offsetBy: windowSize, limitedBy: cleaned.endIndex),
                      let nextEnd = cleaned.index(nextIdx, offsetBy: windowSize, limitedBy: cleaned.endIndex),
                      nextEnd <= cleaned.endIndex else { break }
                if cleaned[prevIdx..<prevEnd] == cleaned[nextIdx..<nextEnd] {
                    print("🧹 Cleaning repetition (window \(windowSize))")
                    cleaned.removeSubrange(prevIdx..<prevEnd)
                    return cleanGlitchedText(cleaned)
                }
                prevIdx = cleaned.index(after: prevIdx)
                nextIdx = cleaned.index(after: nextIdx)
            }
        }

        // Fix mid-word cuts like "I can'I can't"
        let words = cleaned.components(separatedBy: .whitespaces)
        var fixedWords: [String] = []
        var i = 0
        while i < words.count {
            let word = words[i]
            if i + 1 < words.count {
                let nextWord = words[i + 1]
                if word.count < 4 && word.count > 0 && nextWord.count > word.count && nextWord.hasPrefix(word) {
                    print("🧹 Fixing mid-word glitch")
                    fixedWords.append(nextWord)
                    i += 2
                    continue
                }
            }
            fixedWords.append(word)
            i += 1
        }
        let result = fixedWords.joined(separator: " ")

        // Strip Unicode whitespace artifacts that can appear in web/tool-injected content
        let normalized = result
            .replacingOccurrences(of: "\u{00A0}", with: " ")   // NBSP -> space
            .replacingOccurrences(of: "\u{200B}", with: "")    // Zero-width space
            .replacingOccurrences(of: "\u{00AD}", with: "")    // Soft hyphen

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
