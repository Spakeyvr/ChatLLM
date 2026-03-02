//
//  ChatViewModel+TextCleaning.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation

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
            let answerPattern = #"<answer>([\s\S]*?)</answer>"#
            if let re = try? NSRegularExpression(pattern: answerPattern, options: [.caseInsensitive]) {
                let ns = cleaned as NSString
                let range = NSRange(location: 0, length: ns.length)
                let stripped = re.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "$1")
                if stripped != cleaned {
                    cleaned = stripped.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }

        let paragraphs = cleaned.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if paragraphs.count > 1 {
            var seenParagraphs: Set<String> = []
            var uniqueParagraphs: [String] = []

            for paragraph in paragraphs {
                let normalized = paragraph.lowercased()
                if !seenParagraphs.contains(normalized) {
                    seenParagraphs.insert(normalized)
                    uniqueParagraphs.append(paragraph)
                } else {
                    print("🧹 Removing duplicate paragraph: '\(String(paragraph.prefix(50)))...'")
                }
            }

            if uniqueParagraphs.count < paragraphs.count {
                cleaned = uniqueParagraphs.joined(separator: "\n\n")
                print("🧹 Removed \(paragraphs.count - uniqueParagraphs.count) duplicate paragraph(s)")
            }
        }

        // Remove stray leading Markdown-only lines like "**", "*", "---", or "###"
        if let re = try? NSRegularExpression(pattern: #"(?s)^\s*(\*\*|\*|[-]{2,}|#{1,6})\s*\n+"#, options: []) {
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            let result = re.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
            if result != cleaned {
                print("🧹 Removed leading Markdown artifact")
                cleaned = result
            }
        }

        // Fix missing-space glitches like "appears toThe"
        let patterns = [
            ("to([A-Z])", "to $1"),
            ("be([A-Z])", "be $1"),
            ("is([A-Z])", "is $1"),
            ("as([A-Z])", "as $1"),
            ("in([A-Z])", "in $1"),
            ("of([A-Z])", "of $1"),
        ]

        for (pattern, replacement) in patterns {
            do {
                let regex = try NSRegularExpression(pattern: pattern, options: [])
                let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
                let result = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: replacement)
                if result != cleaned {
                    print("🧹 Fixed word boundary glitch: '\(pattern)'")
                    cleaned = result
                }
            } catch {
                continue
            }
        }

        // NOTE: Removed risky connector-word spacing fix that caused inside-word splits like "F or".

        // Conservative normalizations for frequent LLM typos; skips URLs.
        do {
            let fixes: [(String, String)] = [
                (#"(?i)\bios\b"#, "iOS"),
                // "securityreleases" not after "/" or "." to avoid mangling URLs
                (#"(?i)(?<![\/\.])\bsecurityreleases\b"#, "security releases"),
                (#"(?i)\bversionhistory\b"#, "version history")
            ]
            for (pattern, replacement) in fixes {
                let regex = try NSRegularExpression(pattern: pattern, options: [])
                let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
                let result = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: replacement)
                if result != cleaned {
                    print("🧹 Normalized '\(pattern)' -> '\(replacement)'")
                    cleaned = result
                }
            }
        } catch {
            // ignore
        }

        // Detect exact character-level repetitions like "hello worldhello world"
        for windowSize in stride(from: min(50, cleaned.count / 2), through: 5, by: -1) {
            let chunks = stride(from: 0, to: cleaned.count - windowSize, by: 1).map { i in
                let start = cleaned.index(cleaned.startIndex, offsetBy: i)
                let end = cleaned.index(start, offsetBy: windowSize)
                return String(cleaned[start..<end])
            }
            for i in 0..<(chunks.count - 1) {
                if chunks[i] == chunks[i + 1] {
                    let startIdx = cleaned.index(cleaned.startIndex, offsetBy: i + windowSize)
                    guard let endIdx = cleaned.index(startIdx, offsetBy: windowSize, limitedBy: cleaned.endIndex) else { continue }
                    if endIdx <= cleaned.endIndex {
                        print("🧹 Cleaning repetition: '\(chunks[i])'")
                        cleaned.removeSubrange(startIdx..<endIdx)
                        return cleanGlitchedText(cleaned)
                    }
                }
            }
        }

        // Remove duplicate sentence fragments
        let sentences = cleaned.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var seenSentences: Set<String> = []
        var uniqueSentences: [String] = []
        for sentence in sentences {
            let normalized = sentence.lowercased()
            var isDuplicate = false
            for seen in seenSentences {
                if normalized.count > 15 && seen.count > 15 {
                    if normalized.hasPrefix(seen) || seen.hasPrefix(normalized) {
                        isDuplicate = true
                        print("🧹 Removing duplicate sentence fragment: '\(String(sentence.prefix(30)))...)'")
                        break
                    }
                }
            }
            if !isDuplicate && !seenSentences.contains(normalized) {
                seenSentences.insert(normalized)
                uniqueSentences.append(sentence)
            }
        }
        if uniqueSentences.count < sentences.count {
            cleaned = uniqueSentences.joined(separator: ". ")
            if cleaned.count > 10 &&
               !cleaned.hasSuffix(".") &&
               !cleaned.hasSuffix("!") &&
               !cleaned.hasSuffix("?") {
                cleaned += "."
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
                    print("🧹 Fixing mid-word glitch: '\(word)' -> '\(nextWord)'")
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
