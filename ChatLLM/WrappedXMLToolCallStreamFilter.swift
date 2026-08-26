//
//  WrappedXMLToolCallStreamFilter.swift
//  ChatLLM
//
//  Filters wrapped XML tool-call markup out of streamed MLX output so the UI
//  only sees user-visible assistant text while tool invocations are dispatched
//  separately.
//

import Foundation

actor WrappedXMLToolCallStreamFilter {
    private static let toolMarkupMarkers = [
        "<tool_call>",
        "<function=",
        "<parameter=",
        "</parameter>",
        "</function>",
        "</tool_call>"
    ]

    private var pendingText = ""
    private var suppressingToolMarkup = false
    private var suppressedMarkup = ""

    func consume(_ chunk: String) -> String? {
        guard !chunk.isEmpty else { return nil }

        if suppressingToolMarkup {
            suppressedMarkup += chunk
            return resumeAfterClosedToolCallIfPossible()
        }

        pendingText += chunk

        if let markerRange = Self.firstMarkerRange(in: pendingText) {
            let visiblePrefix = String(pendingText[..<markerRange.lowerBound])
            suppressedMarkup = String(pendingText[markerRange.lowerBound...])
            pendingText = ""
            suppressingToolMarkup = true
            let resumedText = resumeAfterClosedToolCallIfPossible() ?? ""
            let visibleText = visiblePrefix + resumedText
            return visibleText.isEmpty ? nil : visibleText
        }

        let safePrefixLength = Self.safePrefixLength(in: pendingText)
        guard safePrefixLength > 0 else {
            return nil
        }

        let safeEnd = pendingText.index(pendingText.startIndex, offsetBy: safePrefixLength)
        let safeChunk = String(pendingText[..<safeEnd])
        pendingText = String(pendingText[safeEnd...])
        return safeChunk.isEmpty ? nil : safeChunk
    }

    func didDispatchToolCall() {
        pendingText = ""
        suppressedMarkup = ""
        suppressingToolMarkup = false
    }

    func finish() -> String? {
        defer {
            pendingText = ""
            suppressedMarkup = ""
            suppressingToolMarkup = false
        }

        if suppressingToolMarkup {
            if let recovered = resumeAfterClosedToolCallIfPossible(), !recovered.isEmpty {
                return recovered
            }

            // Some tokenizer templates omit the outer </tool_call> wrapper.
            // Once </function> has closed, content after it is no longer part
            // of the arguments and is safe to return instead of swallowing the
            // remainder of the assistant response.
            if let functionEnd = suppressedMarkup.range(
                of: "</function>",
                options: [.caseInsensitive, .backwards]
            ) {
                var trailing = String(suppressedMarkup[functionEnd.upperBound...])
                if let wrapperEnd = trailing.range(
                    of: "</tool_call>",
                    options: [.caseInsensitive, .anchored]
                ) {
                    trailing.removeSubrange(wrapperEnd)
                }
                return trailing.isEmpty ? nil : trailing
            }

            return nil
        }

        return pendingText.isEmpty ? nil : pendingText
    }

    /// Ends suppression as soon as a complete wrapper is observed, even when
    /// the model framework fails to emit a corresponding `.toolCall` event.
    /// Any following assistant text is fed back through the normal marker
    /// boundary logic so partial markers remain buffered safely.
    private func resumeAfterClosedToolCallIfPossible() -> String? {
        guard let closingRange = suppressedMarkup.range(
            of: "</tool_call>",
            options: [.caseInsensitive]
        ) else {
            return nil
        }

        let trailing = String(suppressedMarkup[closingRange.upperBound...])
        suppressedMarkup = ""
        suppressingToolMarkup = false
        guard !trailing.isEmpty else { return nil }
        return consume(trailing)
    }

    private static func firstMarkerRange(in text: String) -> Range<String.Index>? {
        toolMarkupMarkers
            .compactMap { marker in text.range(of: marker) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    private static func safePrefixLength(in text: String) -> Int {
        guard text.contains("<") else {
            return text.count
        }

        let suffixLength = longestMarkerPrefixSuffix(in: text)
        return text.count - suffixLength
    }

    private static func longestMarkerPrefixSuffix(in text: String) -> Int {
        let maxSuffixLength = min(
            text.count,
            toolMarkupMarkers.map(\.count).max() ?? 0
        )
        guard maxSuffixLength > 0 else {
            return 0
        }

        for suffixLength in stride(from: maxSuffixLength, through: 1, by: -1) {
            let suffixStart = text.index(text.endIndex, offsetBy: -suffixLength)
            let suffix = String(text[suffixStart...])
            if toolMarkupMarkers.contains(where: { $0.hasPrefix(suffix) }) {
                return suffixLength
            }
        }

        return 0
    }
}
